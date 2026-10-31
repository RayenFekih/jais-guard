"""Offline quality snapshots and new GPU/HTTP latency; never calls Jev or trains."""

import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from jais_guard.serving.hf import (
    Predictor,
    checked_manifest,
    cuda_required,
    frozen_inputs,
    read_json,
)

COMPARISON = Path("results/version_comparison/jaisguard-v1-v2-v1")
JEV = Path("results/jev/test400-jev1140-v1/reports/2026-10-30T093300.501767_0000-077607bc")
MEASURES = (
    "accuracy",
    "macro_f1",
    "gibberish_precision",
    "gibberish_recall",
    "valid_false_rejection_rate",
    "valid_acceptance_rate",
    "invalid",
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_validation(entries):
    """One representative per architecture/version; loss then numeric seed breaks ties."""
    if not entries or any(
        not math.isfinite(e["metadata"]["metadata"]["validation_loss"]) for e in entries
    ):
        raise ValueError("Missing validation loss")
    return min(entries, key=lambda e: (e["metadata"]["metadata"]["validation_loss"], e["seed"]))


def prepare(directory, comparison=COMPARISON, jev=JEV):
    from jais_guard.data.test_workflow import sha
    from jais_guard.finetuning.evaluation import metrics
    from jais_guard.finetuning.version_comparison import (
        metric_match,
        verify_files,
    )

    directory, comparison, jev = Path(directory), Path(comparison), Path(jev)
    if directory.exists():
        raise ValueError("Use a new versioned directory; historical output is never overwritten")
    from jais_guard.finetuning.final_test import locked_registry

    protocol = read_json(comparison / "protocol.json")
    lock = read_json(comparison / "threshold_lock.json")
    if sha(comparison / "threshold_lock.json") != protocol["threshold_lock_sha256"]:
        raise ValueError("Historical selection lock changed")
    verify_files(lock["files"])
    code_drift = {}
    for name, expected in protocol["files"].items():
        actual = sha(name)
        if actual != expected:
            # Old quality is recomputed from predictions below, using current metrics.
            # Preserve the old source hash; never rewrite an old lock or source file.
            if (
                Path(name).resolve()
                != Path(__file__).parents[1].joinpath("finetuning", "data.py").resolve()
            ):
                raise ValueError(f"Historical artifact changed: {name}")
            code_drift[name] = {"historical_sha256": expected, "current_sha256": actual}
    rows, frozen, _ = locked_registry(protocol["test"])
    if frozen["test_sha256"] != protocol["test_sha256"]:
        raise ValueError("Historical test identity changed")
    # Choose representatives before opening any historical test metrics.
    models, selected = {}, {}
    files = {
        str((comparison / n).resolve()): sha(comparison / n)
        for n in ("protocol.json", "threshold_lock.json")
    }
    for version, entries in (("v1", protocol["v1_registry"]), ("v2", lock["checkpoints"])):
        for approach in ("frozen", "lora"):
            candidates = [e for e in entries if e["approach"] == approach]
            for item in candidates:
                verify_files(item["locked_files"])
                history_path = Path(item["checkpoint"]).parent / "history.json"
                history = read_json(history_path)
                best = min(history, key=lambda e: e["validation_loss"])
                if best != item["metadata"]["metadata"]:
                    raise ValueError("Checkpoint is not first minimum validation-loss epoch")
                threshold = read_json(item["threshold_path"])
                if (
                    threshold["selection_data"] != "validation"
                    or threshold["exploratory"] != item["threshold"]
                ):
                    raise ValueError("Threshold is not validation locked")
                key = f"{version}-{approach}-{item['seed']}"
                models[key] = {**item, "version": version}
                files.update(item["locked_files"])
                files[str(history_path.resolve())] = sha(history_path)
            representative = select_validation(candidates)
            selected[f"{version}/{approach}"] = f"{version}-{approach}-{representative['seed']}"
    baseline_path = Path(protocol["baseline_results"]) / "validation_settings.json"
    settings = read_json(baseline_path)
    verify_files(settings["files"])
    models["untuned"] = {"approach": "untuned", "seed": 42, "settings": settings}
    files.update(settings["files"])
    files[str(baseline_path.resolve())] = sha(baseline_path)
    # Revalidate cached classification quality on the same IDs/text/labels.
    from jais_guard.finetuning.version_comparison import cached_metrics

    completion = read_json(comparison / "benchmark/completion.json")
    if completion["status"] != "complete" or completion["test_sha256"] != protocol["test_sha256"]:
        raise ValueError("Historical comparison incomplete")
    archived = read_json(comparison / "benchmark/metrics.json")
    groups = {}
    grouped = {}
    seen = set()
    for entry in archived["per_seed"]:
        version, approach, seed = entry["version"], entry["approach"], entry["seed"]
        identity = (version, approach, seed)
        if identity in seen:
            raise ValueError("Duplicate historical model")
        seen.add(identity)
        baseline = version == "untuned"
        source = (
            comparison / "benchmark" / f"{approach}-{seed}"
            if version == "v2"
            else Path(entry["source_directory"])
        )
        predictions = [
            json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()
        ]
        threshold = None if baseline else models[f"{version}-{approach}-{seed}"]["threshold"]
        measured, _ = cached_metrics(
            rows, predictions, threshold, read_json(source / "metrics.json"), baseline=baseline
        )
        for mode, values in measured.items():
            metric_match(entry["metrics"][mode], values)
            grouped.setdefault(f"{version}/{approach}/{mode}", []).append(values)
        for name in ("predictions.jsonl", "metrics.json"):
            files[str((source / name).resolve())] = sha(source / name)
    expected = {
        (v, a, seed) for v in ("v1", "v2") for a in ("frozen", "lora") for seed in (42, 43, 44)
    } | {("untuned", "untuned_prompt", 42)}
    if seen != expected:
        raise ValueError("Expected all twelve classifiers and the untuned baseline")
    for key, values in grouped.items():
        groups[key] = {}
        for measure in MEASURES:
            numbers = [v[measure] for v in values]
            result = {
                "mean": float(np.mean(numbers)),
                "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
            }
            if measure != "invalid":
                metric_match(archived["groups"][key][measure], result)
            groups[key][measure] = result
    for name in ("metrics.json", "completion.json"):
        files[str((comparison / "benchmark" / name).resolve())] = sha(
            comparison / "benchmark" / name
        )
    jev_metrics = read_json(jev / "metrics.json")
    predictions = [json.loads(s) for s in (jev / "predictions.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in predictions}
    if len(predictions) != 400 or len(by_id) != 400 or set(by_id) != {r["id"] for r in rows}:
        raise ValueError("Jev must cover the same 400 IDs")
    for row in rows:
        old = by_id[row["id"]]
        if any(old[k] != row[k] for k in ("input_text", "label")):
            raise ValueError("Jev input/label mismatch")
        if old["model"] != "jev-1.14.0":
            raise ValueError("Unexpected Jev model")
    measured = metrics(
        [int(r["label"] == "GIBBERISH") for r in rows],
        [by_id[r["id"]]["prediction_id"] for r in rows],
    )
    metric_match(jev_metrics["metrics"], measured)
    if not jev_metrics["complete"]:
        raise ValueError("Incomplete Jev report")
    groups["jev/hosted/zero_shot"] = {k: {"mean": measured[k], "std": None} for k in MEASURES}
    for name in ("metrics.json", "predictions.jsonl"):
        files[str((jev / name).resolve())] = sha(jev / name)
    files.update(lock["files"])
    files.update(
        {
            p: code_drift[p]["current_sha256"] if p in code_drift else h
            for p, h in protocol["files"].items()
        }
    )
    quality = {
        "scope": jev_metrics["scope"],
        "groups": groups,
        "per_seed": archived["per_seed"],
        "jev_latency": jev_metrics["latency"]["successful_request"],
        "jev_latency_policy": jev_metrics["latency_policy"],
        "std_ddof": 1,
    }
    directory.mkdir(parents=True)
    write_new(directory / "quality.json", quality)
    manifest = {
        "historical_source_drift": code_drift,
        "models": models,
        "selected": selected,
        "default_model": selected["v2/lora"],
        "selection_rule": "Minimum archived validation loss, ties by seed; no test selection",
        "test": protocol["test"],
        "test_sha256": protocol["test_sha256"],
        "files": files,
        "quality_sha256": sha(directory / "quality.json"),
        "warmup_inputs": 3,
        "batch_size": 1,
        "source_code": {
            str(p): sha(p)
            for p in (Path(__file__), Path(__file__).parents[1].joinpath("serving", "hf.py"))
        },
    }
    manifest["files"].update(manifest["source_code"])
    manifest["identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_new(directory / "manifest.json", manifest)
    return manifest


def summary(values, elapsed_seconds=None):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected finite positive latency samples")
    seconds = elapsed_seconds if elapsed_seconds is not None else float(values.sum() / 1000)
    return {
        "samples": len(values),
        "mean_ms": float(values.mean()),
        **{f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "throughput_per_second": len(values) / seconds,
        "elapsed_seconds": seconds,
    }


def benchmark_gpu(directory, model_key, execution="gpu"):
    import torch

    cuda_required()
    directory = Path(directory)
    manifest = checked_manifest(directory)
    rows = frozen_inputs(manifest)
    output = directory / execution / model_key
    output.mkdir(parents=True, exist_ok=False)
    predictor = Predictor(manifest, model_key)
    for i in range(manifest["warmup_inputs"]):
        predictor.predict(rows[i]["input_text"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    start = time.perf_counter()
    with (output / "samples.jsonl").open("x") as stream:
        for row in rows:
            result = predictor.predict(row["input_text"])
            record = {"id": row["id"], **result}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "status": "complete",
        "model_key": model_key,
        "manifest_identity": manifest["identity"],
        "batch_size": 1,
        "warmup_inputs": manifest["warmup_inputs"],
        "environment": predictor.info(),
        "memory": predictor.memory(),
        "python": platform.python_version(),
        "latency": summary([r["inference_ms"] for r in records]),
        "model_only": summary([r["model_ms"] for r in records]),
        "loop_throughput_per_second": len(rows) / elapsed,
        "model_only_scope": "Synchronized forward for classifiers; original model.generate "
        "for untuned "
        "including decoding loop and CPU scheduling; excludes prompt/tokenization and parsing.",
        "scope": "Synchronized local prediction: original prompt/tokenization, transfers, "
        "forward or generation, softmax or parsing, CPU result. No HTTP. "
        "Loading/warmup excluded. Includes CPU scheduling, not kernel-only.",
        "test_sha256": manifest["test_sha256"],
    }
    checked_manifest(directory)
    write_new(output / "metrics.json", result)
    return result


def benchmark_http(directory, model_key, url="http://127.0.0.1:8000", execution="http"):
    import httpx

    if url not in ("http://127.0.0.1:8000", "http://localhost:8000"):
        raise ValueError("HTTP benchmark is restricted to the local demo on port 8000")
    directory = Path(directory)
    manifest = checked_manifest(directory)
    rows = frozen_inputs(manifest)
    output = directory / execution / model_key
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with httpx.Client(base_url=url, timeout=180, trust_env=False) as client:
        response = client.get("/health")
        response.raise_for_status()
        health = response.json()
        if health["model_key"] != model_key or health["manifest_identity"] != manifest["identity"]:
            raise ValueError("Server is running a different model or manifest")
        for row in rows[: manifest["warmup_inputs"]]:
            client.post("/predict", json={"text": row["input_text"]}).raise_for_status()
        start = time.perf_counter()
        with (output / "samples.jsonl").open("x") as stream:
            for row in rows:
                begin = time.perf_counter()
                response = client.post("/predict", json={"text": row["input_text"]})
                elapsed = (time.perf_counter() - begin) * 1000
                response.raise_for_status()  # No retries or silently dropped failures.
                body = response.json()
                if body["model_key"] != model_key:
                    raise ValueError("Model changed during HTTP benchmark")
                record = {"id": row["id"], "http_ms": elapsed, **body}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        total = time.perf_counter() - start
    result = {
        "status": "complete",
        "model_key": model_key,
        "manifest_identity": manifest["identity"],
        "test_sha256": manifest["test_sha256"],
        "concurrency": 1,
        "warmup_inputs": manifest["warmup_inputs"],
        "environment": health,
        "latency": summary([r["http_ms"] for r in records], total),
        "server_inference": summary([r["inference_ms"] for r in records]),
        "scope": "Sequential HTTP round-trip including JSON, loopback, FastAPI dispatch, "
        "queue and prediction; reused httpx connection; loading/warmup excluded.",
    }
    checked_manifest(directory)
    write_new(output / "metrics.json", result)
    return result


def report(directory, execution="report", gpu="gpu", http="http"):
    directory = Path(directory)
    manifest = checked_manifest(directory)
    quality = read_json(directory / "quality.json")
    output = directory / execution
    output.mkdir(exist_ok=False)
    latency = {}
    for scope, folder in (("local_gpu", gpu), ("local_http", http)):
        for key in manifest["models"]:
            path = directory / folder / key / "metrics.json"
            if path.exists():
                value = read_json(path)
                if (
                    value["status"] != "complete"
                    or value["manifest_identity"] != manifest["identity"]
                    or value["test_sha256"] != manifest["test_sha256"]
                    or value["latency"]["samples"] != 400
                ):
                    raise ValueError("Incompatible/incomplete latency output")
                latency[f"{scope}/{key}"] = value
    records = []
    for group, values in quality["groups"].items():
        v, a, mode = group.split("/")
        key = "untuned" if v == "untuned" else manifest["selected"].get(f"{v}/{a}")
        row = {
            "group": group,
            "latency_checkpoint": key,
            "quality_seeds": 3 if v in ("v1", "v2") else 1,
        }
        for measure in MEASURES:
            row[measure] = values[measure]["mean"]
            row[f"{measure}_std"] = values[measure]["std"]
        for scope in ("local_gpu", "local_http"):
            measurement = latency.get(f"{scope}/{key}")
            for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_per_second"):
                row[f"{scope}_{field}"] = measurement["latency"][field] if measurement else None
        if v == "jev":
            for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
                row[f"jev_remote_{field}"] = quality["jev_latency"][field]
        records.append(row)
    columns = list(dict.fromkeys(k for row in records for k in row))
    with (output / "comparison.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    latency_rows = []
    for name, value in latency.items():
        latency_rows.append(
            {
                "measurement": name,
                **value["latency"],
                "model_load_seconds": value["environment"]["model_load_seconds"],
                **value.get("memory", {}),
                **{f"loading_{k}": v for k, v in value["environment"]["loading_memory"].items()},
                **{f"model_only_{k}": v for k, v in value.get("model_only", {}).items()},
            }
        )
    if latency_rows:
        with (output / "latency.csv").open("x") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(k for row in latency_rows for k in row))
            )
            writer.writeheader()
            writer.writerows(latency_rows)
    write_new(
        output / "metrics.json",
        {
            "quality": quality,
            "latency": latency,
            "comparison": records,
            "selected": manifest["selected"],
        },
    )

    def fmt(value):
        return "pending" if value is None else f"{value:.2f}"

    lines = [
        "# JaisGuard quality and latency",
        "",
        quality["scope"],
        "",
        "Quality reuses all archived predictions and locked operating points. Three-seed "
        "means ± sample SD describe repeated models on the same 400 inputs, not independent "
        "test samples. INVALID stays -1 and counts as wrong; VALID FRR counts explicit "
        "GIBBERISH only, so acceptance need not equal 1 − FRR.",
        "",
        "Latency in this quality table uses one representative checkpoint per architecture "
        "and iteration, selected by minimum validation loss (ties by seed). All measured "
        "seed timings appear below. Default and locked thresholds share one forward pass; "
        "quality is never recomputed from the latency runs.",
        "",
        "| Model / operating point | Accuracy | Macro F1 | G precision | G recall | VALID FRR | "
        "VALID acceptance | INVALID / 400 | GPU mean ms | HTTP mean ms | Remote mean ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        cells = []
        for measure in MEASURES:
            val, std = row[measure], row[f"{measure}_std"]
            cells.append(f"{val:.4f}" + (f" ± {std:.4f}" if std is not None else ""))
        lines.append(
            "| "
            + row["group"]
            + " | "
            + " | ".join(cells)
            + " | "
            + " | ".join(
                fmt(row.get(f"{s}_mean_ms")) for s in ("local_gpu", "local_http", "jev_remote")
            )
            + " |"
        )
    lines += [
        "",
        "## Local latency measurements",
        "",
        "Batch size 1; three warmup inputs; 400 sequential measured inputs in frozen order. "
        "GPU latency includes preprocessing, transfers and classification/generation with "
        "CUDA synchronization. HTTP is a separate run and includes local transport and "
        "server overhead. Loading is excluded from both. Throughput uses summed prediction "
        "time for GPU and full request-loop time for HTTP. Raw samples and server durations "
        "are saved separately. Memory is process CUDA allocator memory, not total board VRAM.",
        "",
        "| Scope / model | Mean ms | p50 | p95 | p99 | inputs/s | Load s | "
        "Inference peak allocated MiB | Inference peak reserved MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in latency.items():
        m, env = value["latency"], value["environment"]
        mem = value.get("memory", {})
        lines.append(
            f"| {name} | "
            + " | ".join(
                fmt(m[k])
                for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_per_second")
            )
            + f" | {env['model_load_seconds']:.3f} | "
            + " | ".join(
                fmt(mem[k] / 2**20) if k in mem else "see GPU run"
                for k in ("peak_allocated_bytes", "peak_reserved_bytes")
            )
            + " |"
        )
    lines += [
        "",
        "## Model-only timing",
        "",
        "Classifier forward only; untuned model.generate includes the decoding loop and CPU "
        "scheduling. These exclude prompt/tokenization, transfers outside the model call, "
        "softmax and parsing. They are not CUDA-kernel-only durations.",
        "",
        "| Model | Mean ms | p50 | p95 | p99 | inputs/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in latency.items():
        if "model_only" in value:
            m = value["model_only"]
            lines.append(
                "| "
                + name
                + " | "
                + " | ".join(
                    fmt(m[k])
                    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_per_second")
                )
                + " |"
            )
    if manifest.get("historical_source_drift"):
        lines += [
            "",
            "Historical source drift is recorded in manifest.json. The old data loader "
            "source differs from its historical lock; all non-code historical artifacts "
            "were hash-verified and current metrics reproduced cached prediction quality. "
            "The new manifest pins the current inference source. No historical locks were edited.",
        ]
    jev = quality["jev_latency"]
    lines += [
        "",
        f"Jev 1.14.0 existing remote HTTP: mean {jev['mean_ms']:.2f} ms, "
        f"p50 {jev['p50_ms']:.2f}, p95 {jev['p95_ms']:.2f}, p99 {jev['p99_ms']:.2f}.",
        "",
        "Jev includes remote network overhead; Jais is served locally on the RTX 4050. "
        "These are different deployment paths, not a controlled model-speed comparison. "
        "Jev had no warmup; local timings use warmup and connection reuse. No Jev requests "
        "were rerun. Untuned Jais uses its original zero-shot generative prompt; classifiers "
        "use a trained head. Generation has variable output length and can produce INVALID. "
        "Cache state, precision, GPU clock/temperature and other processes affect latency. "
        "Model loading may benefit from the filesystem cache.",
        "",
        "Missing local measurements are **pending**, never inferred from historical batch timings.",
        "",
        "## Validation-only representative checkpoints",
        "",
    ]
    for name, key in manifest["selected"].items():
        item = manifest["models"][key]
        lines.append(
            f"- {name}: {key}, epoch {item['metadata']['metadata']['epoch']}, "
            f"validation loss {item['metadata']['metadata']['validation_loss']:.8f}, "
            f"locked threshold {item['threshold']}; `{item['checkpoint']}`"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    charts(output, records, latency, quality["jev_latency"])
    return output


def charts(output, records, latency, jev):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["group"].replace("/validation_locked", "/locked") for r in records]
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(records))
    for offset, metric, title in (
        (-0.2, "macro_f1", "Macro F1"),
        (0.2, "gibberish_recall", "G recall"),
    ):
        ax.bar(
            x + offset,
            [r[metric] for r in records],
            0.4,
            label=title,
            yerr=[r[f"{metric}_std"] or 0 for r in records],
            capsize=3,
        )
    ax.set(ylim=(0, 1.1), ylabel="Score", title="Seen Arabic benchmark: 400 inputs; quality reused")
    ax.set_xticks(x, labels, rotation=40, ha="right")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"quality.{ext}", dpi=180)
    plt.close(fig)
    names = list(latency) + ["Jev remote HTTP"]
    means = [v["latency"]["mean_ms"] for v in latency.values()] + [jev["mean_ms"]]
    p95s = [v["latency"]["p95_ms"] for v in latency.values()] + [jev["p95_ms"]]
    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.35)))
    y = np.arange(len(names))
    ax.barh(y, means, label="Mean")
    ax.scatter(p95s, y, color="black", marker="|", label="p95")
    ax.set_yticks(y, names)
    ax.set(xlabel="Latency (ms)", title="Local GPU / local HTTP / remote HTTP — different scopes")
    if not latency:
        ax.text(
            0.5,
            0.85,
            "Local GPU and HTTP measurements pending",
            transform=ax.transAxes,
            ha="center",
        )
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"latency.{ext}", dpi=180)
    plt.close(fig)
