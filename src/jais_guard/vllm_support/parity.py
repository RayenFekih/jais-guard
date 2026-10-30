"""Run each reference in a fresh process, then fail closed on probability/class drift."""

import json
import time
from pathlib import Path

import numpy as np

from jais_guard.benchmarks.latency import summary, write_new
from jais_guard.data.test_workflow import sha
from jais_guard.finetuning.data import encode_rows
from jais_guard.serving.hf import cuda_required, frozen_inputs, read_json

from .artifacts import checked_export, checked_plan, seal, verify_hashes
from .runtime import OfflineClassifier, audit_worker, probabilities


def inputs(root, key):
    from transformers import AutoTokenizer

    plan, artifact, _ = checked_export(root, key)
    item = plan["models"][key]
    cfg = item["metadata"]["config"]
    if sha(cfg["validation_path"]) != cfg["validation_sha256"]:
        raise ValueError("Validation export changed")
    validation = [json.loads(s) for s in Path(cfg["validation_path"]).read_text().splitlines()]
    rows = [dict(r, id="validation:" + r["id"], category="validation") for r in validation]
    rows += [dict(r, id="benchmark:" + r["id"], category="benchmark") for r in frozen_inputs(plan)]
    rows += [
        {"id": f"probe:{i}", "category": category, "input_text": text}
        for i, (category, text) in enumerate(plan["probes"])
    ]
    original = AutoTokenizer.from_pretrained(
        Path(item["checkpoint"]) / "tokenizer", local_files_only=True, trust_remote_code=True
    )
    exported = AutoTokenizer.from_pretrained(
        artifact, local_files_only=True, trust_remote_code=True
    )
    original.padding_side = exported.padding_side = "right"
    a = encode_rows(rows, original, cfg["max_length"])
    b = encode_rows(rows, exported, cfg["max_length"])
    if a != b:
        raise ValueError("Export changed tokenizer IDs, special tokens or input lengths")
    return [{**r, "token_ids": e["input_ids"]} for r, e in zip(rows, a, strict=True)]


def collect(root, key, backend):
    import torch

    from jais_guard.finetuning.data import collate

    cuda_required()
    plan = checked_plan(root)
    rows = inputs(root, key)
    folder = Path(root) / "parity" / key / backend
    folder.mkdir(parents=True, exist_ok=False)
    item = plan["models"][key]
    start = time.perf_counter()
    if backend == "vllm":
        classifier = OfflineClassifier(root, key)
        loading = classifier.loading
    elif backend in ("hf-original", "hf-fp16"):
        from jais_guard.finetuning.model import load_checkpoint, seed_everything

        seed_everything(item["seed"])
        model, tokenizer, _ = load_checkpoint(item["checkpoint"], "cuda")
        model.eval().requires_grad_(False)
        model.backbone.gradient_checkpointing_disable()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if backend == "hf-fp16":
            model.backbone.to(torch.float16)
            dtype = torch.float16
        torch.cuda.synchronize()
        loading = {
            "dtype": str(dtype),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    else:
        raise ValueError("Unknown parity backend")
    load_seconds = time.perf_counter() - start
    records = []
    with (folder / "predictions.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            if backend == "vllm":
                probs = classifier.predict_tokens(row["token_ids"])
            else:
                encoded = [
                    {"input_ids": row["token_ids"], "attention_mask": [1] * len(row["token_ids"])}
                ]
                batch = collate(encoded, tokenizer, "cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    logits = model(**batch).float()
                probs = probabilities(logits.softmax(-1)[0].cpu().tolist())
            record = {k: row[k] for k in ("id", "category", "token_ids")}
            record["probabilities"] = probs
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{key} {backend}: {i + 1}/{len(rows)}", flush=True)
    if backend == "vllm":
        classifier.close()
    write_new(
        folder / "completion.json",
        {
            "status": "complete",
            "plan_identity": plan["identity"],
            "backend": backend,
            "model_key": key,
            "samples": len(rows),
            "load_seconds": load_seconds,
            "loading": loading,
            "predictions_sha256": sha(folder / "predictions.jsonl"),
            "export_identity": read_json(Path(root) / "exports" / key / "export.json")["identity"],
        },
    )


def compare_probabilities(reference, candidate, threshold, policy):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Incomplete parity records")
    for a, b in zip(reference, candidate, strict=True):
        if any(a[k] != b[k] for k in ("id", "category", "token_ids")):
            raise ValueError("Parity IDs/order/tokenization differ")
    a = np.array([probabilities(r["probabilities"]) for r in reference])
    b = np.array([probabilities(r["probabilities"]) for r in candidate])
    differences = np.abs(a - b)
    locked_matches = (a[:, 1] >= threshold) == (b[:, 1] >= threshold)
    default_matches = (a[:, 1] >= 0.5) == (b[:, 1] >= 0.5)
    maximum, mean = float(differences.max()), float(differences.mean())
    return {
        "passed": bool(
            maximum <= policy["max_abs_probability_difference"]
            and mean <= policy["mean_abs_probability_difference"]
            and locked_matches.all()
            and default_matches.all()
        ),
        "samples": len(reference),
        "max_abs_probability_difference": maximum,
        "mean_abs_probability_difference": mean,
        "locked_prediction_mismatches": int((~locked_matches).sum()),
        "default_prediction_mismatches": int((~default_matches).sum()),
        "mismatch_ids": [
            r["id"]
            for r, ok in zip(reference, locked_matches & default_matches, strict=True)
            if not ok
        ],
    }


def acceptance_reference(plan):
    return plan.get("acceptance", {}).get("reference", "hf-original")


def acceptance_passes(plan, comparisons):
    if acceptance_reference(plan) == "hf-fp16":
        return comparisons["hf-fp16_vs_vllm"]["passed"]
    return all(value["passed"] for value in comparisons.values())


def verify(root, key):
    plan, _, receipt = checked_export(root, key)
    folder = Path(root) / "parity" / key
    outputs, files = {}, {}
    expected = inputs(root, key)
    for backend in ("hf-original", "hf-fp16", "vllm"):
        completion_path = folder / backend / "completion.json"
        completion = read_json(completion_path)
        path = folder / backend / "predictions.jsonl"
        if (
            completion["status"] != "complete"
            or completion["plan_identity"] != plan["identity"]
            or completion["export_identity"] != receipt["identity"]
            or completion["samples"] != len(expected)
            or completion["predictions_sha256"] != sha(path)
        ):
            raise ValueError("Parity reference incomplete or changed")
        if backend == "vllm":
            audit_worker([completion["loading"]], receipt)
        outputs[backend] = [json.loads(s) for s in path.read_text().splitlines()]
        if len(outputs[backend]) != len(expected):
            raise ValueError("Parity sample count changed")
        for a, b in zip(outputs[backend], expected, strict=True):
            if any(a[k] != b[k] for k in ("id", "category", "token_ids")):
                raise ValueError("Parity reference does not cover all locked inputs")
        files[str(path.resolve())] = sha(path)
        files[str(completion_path.resolve())] = sha(completion_path)
    policy = plan["parity_policy"]
    comparison = {
        f"{a}_vs_{b}": compare_probabilities(outputs[a], outputs[b], receipt["threshold"], policy)
        for a, b in (("hf-original", "hf-fp16"), ("hf-fp16", "vllm"), ("hf-original", "vllm"))
    }
    baseline = acceptance_reference(plan)
    categories = sorted({r["category"] for r in expected})
    by_category = {
        category: compare_probabilities(
            [r for r in outputs[baseline] if r["category"] == category],
            [r for r in outputs["vllm"] if r["category"] == category],
            receipt["threshold"],
            policy,
        )
        for category in categories
    }
    result = {
        "by_category": by_category,
        "passed": acceptance_passes(plan, comparison),
        "acceptance_reference": baseline,
        "bf16_diagnostic_only": baseline == "hf-fp16",
        "trained_head_verified": True,
        "plan_identity": plan["identity"],
        "export_identity": receipt["identity"],
        "model_key": key,
        "threshold": receipt["threshold"],
        "policy": policy,
        "comparisons": comparison,
        "files": files,
    }
    result["identity"] = seal(result)
    write_new(folder / "gate.json", result)
    return result


def checked_gate(root, key):
    plan, _, receipt = checked_export(root, key)
    gate = read_json(Path(root) / "parity" / key / "gate.json")
    if (
        gate["plan_identity"] != plan["identity"]
        or gate["export_identity"] != receipt["identity"]
        or seal({k: v for k, v in gate.items() if k != "identity"}) != gate["identity"]
    ):
        raise ValueError("Prediction parity has not passed; benchmarking refused")
    verify_hashes(gate["files"])
    return gate


def require_parity(root, key):
    gate = checked_gate(root, key)
    if not gate["passed"]:
        raise ValueError("Prediction parity has not passed; benchmarking refused")
    return gate


def benchmark_offline(root, key, backend="vllm"):
    import torch
    from transformers import AutoTokenizer

    require_parity(root, key)
    plan, artifact, _ = checked_export(root, key)
    folder = Path(root) / "offline" / key / backend
    folder.mkdir(parents=True, exist_ok=False)
    rows = [r for r in inputs(root, key) if r["category"] == "benchmark"]
    if backend == "vllm":
        classifier = OfflineClassifier(root, key)
        tokenizer = AutoTokenizer.from_pretrained(
            artifact, local_files_only=True, trust_remote_code=True
        )

        def predict(row):
            encoded = encode_rows(
                [row], tokenizer, plan["models"][key]["metadata"]["config"]["max_length"]
            )
            return classifier.predict_tokens(encoded[0]["input_ids"])

        load_seconds, loading = classifier.load_seconds, classifier.loading
        memory = classifier.memory
        reset = classifier.reset_peak
    elif backend in ("hf-original", "hf-fp16"):
        from jais_guard.serving.hf import Predictor

        from .fp16 import FP16Predictor

        factory = FP16Predictor if backend == "hf-fp16" else Predictor
        classifier = factory(read_json(Path(plan["source_manifest"]) / "manifest.json"), key)

        def predict(row):
            result = classifier.predict(row["input_text"])
            return [result["probabilities"][name] for name in ("VALID", "GIBBERISH")]

        load_seconds, loading = classifier.load_seconds, classifier.load_memory
        memory = classifier.memory
        reset = torch.cuda.reset_peak_memory_stats
    else:
        raise ValueError("Offline benchmark backend must be vllm, hf-original or hf-fp16")
    for row in rows[:3]:
        predict(row)
    if backend.startswith("hf-"):
        torch.cuda.synchronize()
    reset()
    observations = []
    start_loop = time.perf_counter()
    for row in rows:
        start = time.perf_counter()
        probs = predict(row)
        observations.append(
            {
                "id": row["id"],
                "category": row["category"],
                "token_ids": row["token_ids"],
                "probabilities": probs,
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
        )
    elapsed = time.perf_counter() - start_loop
    write_new(folder / "samples.json", observations)
    reference_backend = backend if backend.startswith("hf-") else acceptance_reference(plan)
    reference_path = Path(root) / "parity" / key / reference_backend / "predictions.jsonl"
    reference = {r["id"]: r for r in map(json.loads, reference_path.read_text().splitlines())}
    match = compare_probabilities(
        [reference[r["id"]] for r in rows],
        observations,
        plan["models"][key]["threshold"],
        plan["parity_policy"],
    )
    write_new(folder / "response_parity.json", match)
    if not match["passed"]:
        raise ValueError("Offline benchmark changed classifications; performance withheld")
    peak_memory = memory()
    if backend == "vllm":
        classifier.close()
    write_new(
        folder / "metrics.json",
        {
            "model_key": key,
            "backend": backend,
            "scope": "Full local prediction, includes tokenizer and transfers, "
            "batch size 1, no HTTP",
            "plan_identity": plan["identity"],
            "latency": summary([r["latency_ms"] for r in observations], elapsed),
            "model_load_seconds": load_seconds,
            "loading": loading,
            "memory": peak_memory,
            "warmup_inputs": 3,
            "response_parity": match,
        },
    )
