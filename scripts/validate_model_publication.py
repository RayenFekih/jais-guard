"""Read-only GPU reproduction checks for compact model staging.

Requires authentic local references. Writes only new validation outputs, never
historical plans, checkpoints, predictions or metrics. One model per process.
"""

# Immutable numeric receipts and validation-report expressions may be long.
# ruff: noqa: E501


def main():
    import argparse
    import hashlib
    import json
    import os
    import pathlib
    import sys
    import time

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/jaisguard-model-publication/hf-modules")
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--precision", choices=["bf16", "fp16"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--package")
    p.add_argument(
        "--quality-batch2",
        action="store_true",
        help="Reproduce the separate 400-input BF16 batch-two quality run",
    )
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    if args.quality_batch2:
        if args.precision != "bf16":
            raise ValueError("Historical quality reproduction requires BF16")
        return validate_quality(args, root)
    import numpy as np
    import torch

    from jais_guard.finetuning.data import collate, encode_rows
    from jais_guard.finetuning.model import load_checkpoint, seed_everything

    v = root / "results/vllm_serving/jaisguard-v4"
    plan = json.loads((v / "plan.json").read_text())
    item = plan["models"][args.model]

    def sha(path):
        with pathlib.Path(path).open("rb") as s:
            return hashlib.file_digest(s, "sha256").hexdigest()

    for name, expected in item["locked_files"].items():
        assert sha(name) == expected, name
    reference_folder = (
        v / "parity" / args.model / ("hf-original" if args.precision == "bf16" else "hf-fp16")
    )
    completion = json.loads((reference_folder / "completion.json").read_text())
    refpath = reference_folder / "predictions.jsonl"
    assert sha(refpath) == completion["predictions_sha256"]
    refs = [json.loads(s) for s in refpath.read_text().splitlines()]
    assert len(refs) == 658
    cfg = item["metadata"]["config"]
    assert sha(cfg["validation_path"]) == cfg["validation_sha256"]
    validation = [
        json.loads(s) for s in pathlib.Path(cfg["validation_path"]).read_text().splitlines()
    ]
    test = [
        json.loads(s) for s in (pathlib.Path(plan["test"]) / "test.jsonl").read_text().splitlines()
    ]
    assert sha(pathlib.Path(plan["test"]) / "test.jsonl") == plan["test_sha256"]
    rows = (
        [dict(r, id="validation:" + r["id"], category="validation") for r in validation]
        + [dict(r, id="benchmark:" + r["id"], category="benchmark") for r in test]
        + [
            {"id": f"probe:{i}", "category": c, "input_text": t}
            for i, (c, t) in enumerate(plan["probes"])
        ]
    )
    assert torch.cuda.is_available()
    seed_everything(item["seed"])
    started = time.perf_counter()
    if args.package:
        import importlib.util

        folder = pathlib.Path(args.package).resolve()
        spec = importlib.util.spec_from_file_location(
            "jaisguard_model", folder / "jaisguard_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        classifier = module.JaisGuardClassifier.from_pretrained(
            folder, device="cuda", precision=args.precision, local_files_only=True
        )
        model = classifier.model
        tokenizer = classifier.tokenizer
    else:
        model, tokenizer, saved = load_checkpoint(item["checkpoint"], "cuda")
        assert saved == item["metadata"]
        model.eval().requires_grad_(False)
        model.backbone.gradient_checkpointing_disable()
        if args.precision == "fp16":
            model.backbone.to(torch.float16)
    assert next(model.head.parameters()).dtype == torch.float32
    encoded = encode_rows(rows, tokenizer, 64)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    outputs = []
    for i, (row, tokens, ref) in enumerate(zip(rows, encoded, refs, strict=True)):
        assert (row["id"], row["category"], tokens["input_ids"]) == (
            ref["id"],
            ref["category"],
            ref["token_ids"],
        )
        if args.package:
            result = classifier.predict(row["input_text"])
            values = [result["probabilities"][k] for k in ["VALID", "GIBBERISH"]]
            assert result["threshold"] == item["threshold"] and result["class_id"] == int(
                values[1] >= item["threshold"]
            )
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                logits = model(**collate([tokens], tokenizer, "cuda")).float()
            values = logits.softmax(-1)[0].cpu().tolist()
        outputs.append(values)
        if (i + 1) % 200 == 0:
            print(
                args.model,
                args.precision,
                "package" if args.package else "refactored",
                i + 1,
                flush=True,
            )
    a = np.array([r["probabilities"] for r in refs])
    b = np.array(outputs)
    delta = np.abs(a - b)
    result = {
        "model": args.model,
        "precision": args.precision,
        "implementation": "staged-package" if args.package else "refactored",
        "samples": len(rows),
        "reference_sha256": sha(refpath),
        "max_abs_probability_difference": float(delta.max()),
        "mean_abs_probability_difference": float(delta.mean()),
        "default_mismatches": int(((a[:, 1] >= 0.5) != (b[:, 1] >= 0.5)).sum()),
        "locked_mismatches": int(
            ((a[:, 1] >= item["threshold"]) != (b[:, 1] >= item["threshold"])).sum()
        ),
        "threshold": item["threshold"],
        "head_dtype": str(next(model.head.parameters()).dtype),
        "backbone_dtype": str(next(model.backbone.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "seconds": time.perf_counter() - started,
    }
    # Exact same-precision reproduction gate, deliberately stricter than cross-engine vLLM.
    result["passed"] = (
        result["max_abs_probability_difference"] <= 1e-6
        and result["default_mismatches"] == 0
        and result["locked_mismatches"] == 0
    )
    if args.precision == "fp16":
        vr = [
            json.loads(s)
            for s in (v / "parity" / args.model / "vllm/predictions.jsonl").read_text().splitlines()
        ]
        c = np.array([r["probabilities"] for r in vr])
        diff = np.abs(b - c)
        result["historical_vllm_comparison"] = {
            "max_abs_probability_difference": float(diff.max()),
            "mean_abs_probability_difference": float(diff.mean()),
            "default_mismatches": int(((b[:, 1] >= 0.5) != (c[:, 1] >= 0.5)).sum()),
            "locked_mismatches": int(
                ((b[:, 1] >= item["threshold"]) != (c[:, 1] >= item["threshold"])).sum()
            ),
            "claim": "comparison with trusted HISTORICAL vLLM observations; engine not rerun",
        }
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as s:
        json.dump(result, s, indent=2)
        s.write("\n")
    with path.with_suffix(".probabilities.json").open("x") as s:
        json.dump(outputs, s)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit("Probability/class reproduction failed; release blocked")


def validate_quality(args, root):
    import hashlib
    import importlib.util
    import json
    import os
    import pathlib
    import sys
    import time

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/jaisguard-model-publication/hf-modules")
    import numpy as np

    from jais_guard.finetuning.model import load_checkpoint, predict_rows, seed_everything

    plan = json.loads((root / "results/vllm_serving/jaisguard-v4/plan.json").read_text())
    item = plan["models"][args.model]
    test = pathlib.Path(plan["test"]) / "test.jsonl"
    assert hashlib.sha256(test.read_bytes()).hexdigest() == plan["test_sha256"]
    rows = [json.loads(s) for s in test.read_text().splitlines()]
    seed_everything(item["seed"])
    start = time.perf_counter()
    if args.package:
        folder = pathlib.Path(args.package)
        spec = importlib.util.spec_from_file_location(
            "jaisguard_model", folder / "jaisguard_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        classifier = module.JaisGuardClassifier.from_pretrained(
            folder, device="cuda", precision="bf16", local_files_only=True
        )
        model, tokenizer = classifier.model, classifier.tokenizer
    else:
        model, tokenizer, _ = load_checkpoint(item["checkpoint"], "cuda")
        model.eval().requires_grad_(False)
    for _ in range(3):
        predict_rows(model, tokenizer, rows[:2], 64, "cuda", 2)
    scores, _ = predict_rows(model, tokenizer, rows, 64, "cuda", 2)
    refpath = (
        root
        / f"results/version_comparison/jaisguard-v1-v2-v1/benchmark/{item['approach']}-{item['seed']}/predictions.jsonl"
    )
    refs = [json.loads(s) for s in refpath.read_text().splitlines()]
    assert [r["id"] for r in refs] == [r["id"] for r in rows]
    a = np.array([r["p_gibberish"] for r in refs])
    b = np.array(scores)
    diff = np.abs(a - b)
    result = {
        "model": args.model,
        "precision": "bf16",
        "batch_size": 2,
        "samples": 400,
        "implementation": "staged-package" if args.package else "refactored",
        "max_abs_probability_difference": float(diff.max()),
        "mean_abs_probability_difference": float(diff.mean()),
        "default_mismatches": int(((a >= 0.5) != (b >= 0.5)).sum()),
        "locked_mismatches": int(((a >= item["threshold"]) != (b >= item["threshold"])).sum()),
        "reference_sha256": hashlib.sha256(refpath.read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - start,
    }
    result["passed"] = (
        result["max_abs_probability_difference"] <= 1e-6
        and result["default_mismatches"] == result["locked_mismatches"] == 0
    )
    with pathlib.Path(args.output).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
    assert result["passed"]


if __name__ == "__main__":
    main()
