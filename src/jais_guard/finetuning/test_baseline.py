"""Untuned final-test baseline; reuse validation behavior, never rerun classifiers."""

import gc
import hashlib
import inspect
import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from jais_guard.baselines.prompts import (
    RECOVERABLE_INSTRUCTION,
    RECOVERABLE_PROMPT_VERSION,
    build_prompt,
)
from jais_guard.data.test_workflow import sha

from .evaluation import metrics, write_csv, write_json
from .final_test import DESCRIPTION, locked_registry

INVALID_POLICY = (
    "INVALID (-1): wrong for accuracy; false negative for its true class in recall/F1; "
    "never silently assigned a binary label. VALID FRR counts only explicit GIBBERISH; "
    "valid acceptance and invalid counts additionally expose failures."
)
MARKER = "<!-- JAISGUARD_UNTUNED_TEST_BASELINE -->"


def validation_settings(source, frozen):
    """Read original local MLflow provenance and refuse unverified settings changes."""
    source = Path(source).resolve()
    rid = source.name
    if not re.fullmatch(r"[0-9a-f]{32}", rid):
        raise ValueError("Expected the original validation baseline MLflow run directory")
    with sqlite3.connect(f"file:{source.parent / 'mlflow.db'}?mode=ro", uri=True) as db:
        run = db.execute(
            "SELECT status,artifact_uri,experiment_id FROM runs WHERE run_uuid=?", (rid,)
        ).fetchone()
        params = dict(db.execute("SELECT key,value FROM params WHERE run_uuid=?", (rid,)))
    if run is None or run[0] != "FINISHED":
        raise ValueError("Original validation baseline run is missing or unfinished")
    uri = urlparse(run[1])
    if uri.scheme != "mlflow-artifacts" or uri.netloc:
        raise ValueError("Cannot resolve original local baseline artifact store safely")
    artifact = source.parent / "artifacts" / uri.path.lstrip("/")
    if (
        artifact.resolve()
        != (source.parent / "artifacts" / str(run[2]) / rid / "artifacts").resolve()
    ):
        raise ValueError("Unexpected validation artifact mapping")
    current_path = next(
        Path(p)
        for p in frozen["selection_lock"]["prompt_files"]
        if Path(p).name == "prompt_baseline.json"
    )
    current = json.loads(current_path.read_text())
    baseline = json.loads((source / "baseline_config.json").read_text())
    prompt_hash = hashlib.sha256(
        (
            RECOVERABLE_PROMPT_VERSION
            + current["variant"]
            + RECOVERABLE_INSTRUCTION
            + inspect.getsource(build_prompt)
        ).encode()
    ).hexdigest()
    if (
        baseline != {**current, "prompt_sha256": prompt_hash}
        or baseline["prompt_version"] != RECOVERABLE_PROMPT_VERSION
        or baseline["variant"] != "recoverable_v1"
    ):
        raise ValueError("Validation baseline prompt or decoding configuration changed")
    config = json.loads((source / "config.json").read_text())
    reference = frozen["selection_lock"]["checkpoints"][0]
    for key in ("model", "revision", "train_sha256", "validation_sha256"):
        if config[key] != reference[key]:
            raise ValueError(f"Validation baseline model/dataset mismatch: {key}")
    expected = {
        **baseline,
        "approach": "untuned_prompt",
        "model": config["model"],
        "revision": config["revision"],
        "do_sample": False,
        "batch_size": 1,
        "dtype": "torch.float16",
        "seed": 42,
    }
    if any(params.get(k) != str(v) for k, v in expected.items()):
        raise ValueError("Original MLflow baseline parameters do not match archived settings")
    old_metrics = json.loads((source / "evaluation/metrics.json").read_text())
    if (
        old_metrics["default"]["examples"] != 200
        or old_metrics["unparseable_policy"] != INVALID_POLICY
    ):
        raise ValueError("Cannot establish the original invalid-output evaluation policy")
    paths = [
        current_path,
        source / "baseline_config.json",
        source / "config.json",
        source / "evaluation/metrics.json",
        artifact / "generation_config.json",
    ]
    generation = json.loads(paths[-1].read_text())
    return {
        "baseline": baseline,
        "model": config["model"],
        "revision": config["revision"],
        "generation_config": generation,
        "validation_run_id": rid,
        "validation_experiment_id": str(run[2]),
        "validation_params": params,
        "files": {str(p): sha(p) for p in paths},
        "invalid_policy": INVALID_POLICY,
    }


def comparison_snapshot(directory, frozen, registry):
    directory = Path(directory).resolve()
    completion = json.loads((directory / "completion.json").read_text())
    preflight = json.loads((directory / "preflight.json").read_text())
    values = json.loads((directory / "metrics.json").read_text())
    if (
        completion.get("status") != "complete"
        or completion.get("models_evaluated") != 6
        or completion.get("test_sha256") != frozen["test_sha256"]
        or preflight["test_sha256"] != frozen["test_sha256"]
    ):
        raise ValueError("Comparison is not a completed six-classifier evaluation of this test")
    runs = values["runs"]
    if len(runs) != 6 or {(r["approach"], r["seed"]) for r in runs} != {
        (r["approach"], r["seed"]) for r in registry
    }:
        raise ValueError("Comparison does not contain exactly the six locked classifiers")
    mapped = {(r["approach"], r["seed"]): r for r in registry}
    for run in runs:
        item = mapped[(run["approach"], run["seed"])]
        if (
            run["threshold_validation_locked"] != item["threshold"]
            or run["metrics"]["default"]["threshold"] != 0.5
            or run["metrics"]["validation_locked"]["threshold"] != item["threshold"]
        ):
            raise ValueError("Comparison thresholds disagree with frozen selections")
    text = (directory / "comparison.md").read_text()
    if MARKER in text:
        raise ValueError("A baseline row is already attached; classifiers will not be rerun")
    if text.count("\n## Across seeds\n") != 1:
        raise ValueError("Unrecognized comparison report format")
    return {
        "directory": str(directory),
        "metrics": values,
        "files": {str(p): sha(p) for p in directory.rglob("*") if p.is_file()},
    }


def prompt_preflight(rows, settings):
    from transformers import AutoTokenizer

    cfg = settings["baseline"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"],
        revision=settings["revision"],
        code_revision=settings["revision"],
        trust_remote_code=True,
        local_files_only=True,
    )
    lengths = []
    for row in rows:
        prompt = build_prompt(
            tokenizer, row["input_text"], cfg["reasoning_effort"], False, cfg["variant"]
        )
        n = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n + cfg["max_tokens"] > cfg["max_model_len"]:
            raise ValueError(f"Baseline context budget exceeded: {row['id']}")
        lengths.append(n)
    return {
        "max_input_tokens": max(lengths),
        "max_new_tokens": cfg["max_tokens"],
        "max_model_len": cfg["max_model_len"],
        "truncation": False,
    }


def classify_timed(classifier, text, variant):
    """Observe generate() without changing its arguments, model or parsing behavior."""
    import torch

    original = classifier.model.generate
    generation_ms = []

    def observe(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        generation_ms.append((time.perf_counter() - start) * 1000)
        return result

    classifier.model.generate = observe
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = classifier.classify(text, variant=variant)
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) * 1000
    finally:
        classifier.model.generate = original
    if len(generation_ms) != 1:
        raise ValueError("Expected exactly one generation call per input")
    return {
        **{k: v for k, v in result.items() if k != "prediction"},
        "prediction_id": {"VALID": 0, "GIBBERISH": 1}.get(result["prediction"], -1),
        "latency_ms": total_ms,
        "generation_ms": generation_ms[0],
    }


def infer_baseline(rows, settings, output):
    import torch

    from jais_guard.baselines.inference import BaselineClassifier, BaselineConfig

    from .model import seed_everything

    classifier = None
    cfg = settings["baseline"]
    seed_everything(42)
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        classifier = BaselineClassifier(
            BaselineConfig(
                settings["model"],
                settings["revision"],
                cfg["reasoning_effort"],
                cfg["max_tokens"],
                cfg["max_model_len"],
                local_files_only=True,
            )
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - begin
        if classifier.device != "cuda" or str(classifier.model.dtype) != "torch.float16":
            raise ValueError("Baseline device/dtype differs from validation")
        if classifier.model.generation_config.to_dict() != settings["generation_config"]:
            raise ValueError("Pinned model generation configuration differs from validation")
        # Warm up on the same first input without inspecting its label. Not counted as a prediction.
        classify_timed(classifier, rows[0]["input_text"], cfg["variant"])
        records = []
        with (output / "progress.jsonl").open("x") as stream:
            for row in rows:
                result = classify_timed(classifier, row["input_text"], cfg["variant"])
                records.append({**row, **result})
                stream.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
                stream.flush()
                print(f"untuned baseline {len(records)}/{len(rows)}", flush=True)
        timing = {
            "model_load_seconds": load_seconds,
            "warmup_inputs": 1,
            "batch_size": 1,
            "dtype": str(classifier.model.dtype),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "latency_ms: synchronized classify incl prompt/tokenize/H2D/generate/parse; "
            "generation_ms: synchronized model.generate incl decoding loop and CPU "
            "scheduling, not CUDA-kernel-only. Loading/warmup excluded from summaries.",
        }
        for name in ("latency_ms", "generation_ms"):
            values = [r[name] for r in records]
            timing.update(
                {
                    f"{name}_mean": float(np.mean(values)),
                    f"{name}_p50": float(np.percentile(values, 50)),
                    f"{name}_p95": float(np.percentile(values, 95)),
                }
            )
        return records, timing
    finally:
        del classifier
        gc.collect()
        torch.cuda.empty_cache()


def baseline_report(output, records):
    truth = [int(r["label"] == "GIBBERISH") for r in records]
    predictions = [r["prediction_id"] for r in records]
    report = {
        "default": metrics(truth, predictions),
        "unparseable_policy": INVALID_POLICY,
        "parsing_status_counts": dict(Counter(r["parse_status"] for r in records)),
    }
    report["default"]["parsing_success_rate"] = report["parsing_status_counts"].get(
        "success", 0
    ) / len(records)
    report["default"]["valid_invalid_outputs"] = sum(
        p == -1 and y == 0 for y, p in zip(truth, predictions, strict=True)
    )
    errors = [r for r, y, p in zip(records, truth, predictions, strict=True) if y != p]
    for name, items in (("predictions", records), ("errors", errors)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in items)
        )
        keys = (
            "id",
            "input_text",
            "label",
            "prediction_id",
            "raw_output",
            "parse_status",
            "finish_reason",
            "input_tokens",
            "generated_tokens",
            "latency_ms",
            "generation_ms",
        )
        write_csv(output / f"{name}.csv", [{k: r.get(k) for k in keys} for r in items])
    write_json(output / "metrics.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    matrix = np.array(report["default"]["confusion"])
    ax.imshow(matrix, cmap="Blues")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["VALID", "GIBBERISH", "INVALID"],
        yticks=[0, 1],
        yticklabels=["VALID", "GIBBERISH"],
        xlabel="Prediction",
        ylabel="Truth",
    )
    fig.tight_layout()
    fig.savefig(output / "confusion.png")
    plt.close(fig)
    return report


def attach_comparison(comparison, output, report, timing, run_id):
    """Insert one row, retain all classifier bytes/metrics, back up the original report."""
    target = Path(comparison["directory"]) / "comparison.md"
    for path, expected in comparison["files"].items():
        if sha(path) != expected:
            raise ValueError(f"Classifier result changed since baseline preflight: {path}")
    original = target.read_text()
    if MARKER in original:
        raise ValueError("Baseline already attached")
    m = report["default"]
    row = (
        "| untuned_prompt | 42 | zero_shot | n/a | "
        + " | ".join(
            f"{m[k]:.4f}"
            for k in (
                "accuracy",
                "macro_f1",
                "gibberish_recall",
                "valid_false_rejection_rate",
                "valid_acceptance_rate",
            )
        )
        + " |\n"
    )
    anchor = "\n## Across seeds\n"
    updated = original.replace(anchor, row + anchor, 1)
    section = (
        f"\n## Untuned zero-shot baseline\n\n{MARKER}\n\n"
        f"Prompt: {RECOVERABLE_PROMPT_VERSION}. One seed-42 baseline, not a three-seed mean.\n\n"
        f"Parsing success: {m['parsing_success_rate']:.2%}; invalid outputs: {m['invalid']}; "
        f"invalid VALID inputs: {m['valid_invalid_outputs']}.\n\n{INVALID_POLICY}\n\n"
        f"Mean classify latency: {timing['latency_ms_mean']:.3f} ms; "
        f"mean generation time: {timing['generation_ms_mean']:.3f} ms/input. "
        f"Model loading: {timing['model_load_seconds']:.3f} s. Batch size 1; warmup excluded.\n\n"
        f"Baseline artifacts: {output.resolve()}; MLflow run: {run_id}. "
        "The six classifier results and their across-seed statistics are unchanged.\n"
    )
    updated += section
    (output / "classifier_comparison.original.md").write_bytes(target.read_bytes())
    (output / "comparison.md").write_text(updated)
    write_json(
        output / "comparison_with_baseline.json",
        {
            **comparison["metrics"],
            "untuned_prompt_baseline": {"run_id": run_id, "metrics": report, "timing": timing},
        },
    )
    temporary = target.with_name(".comparison-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(updated)
        if sha(target) != comparison["files"][str(target)]:
            raise ValueError("Comparison changed while attaching baseline")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def run(args):
    """Baseline-only entrypoint called before any classifier execution branch."""
    from datetime import UTC, datetime

    rows, frozen, registry = locked_registry(args.test)
    settings = validation_settings(args.validation_baseline, frozen)
    comparison = comparison_snapshot(args.comparison, frozen, registry)
    tokens = prompt_preflight(rows, settings)
    protected = {
        str(args.test.resolve() / name): sha(args.test / name)
        for name in ("test.jsonl", "manifest.json", "freeze.json")
    }
    protected.update(settings["files"])
    preflight = {
        "status": "baseline-verified-no-inference",
        "model": settings["model"],
        "revision": settings["revision"],
        "baseline": settings["baseline"],
        "validation_run_id": settings["validation_run_id"],
        "tokens": tokens,
        "comparison": comparison["directory"],
        "classifiers_to_rerun": 0,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.check_only:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable in Codex: no baseline inference or MLflow writes. "
            "Run --baseline-only in your WSL terminal. CPU inference is disabled."
        )
    import mlflow

    from .tracking import configure, environment, log_metrics

    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Baseline must create a new run, not resume an existing MLflow run")
    mlflow = configure(
        {"tracking_uri": args.tracking_uri, "experiment": "JaisGuard/general-arabic-v1"}
    )
    source_run = mlflow.tracking.MlflowClient().get_run(settings["validation_run_id"])
    if (
        source_run.info.status != "FINISHED"
        or str(source_run.info.experiment_id) != settings["validation_experiment_id"]
        or any(source_run.data.params.get(k) != v for k, v in settings["validation_params"].items())
    ):
        raise ValueError("Tracking server does not contain the original validation baseline")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_root.resolve() / f"untuned-baseline-{stamp}"
    from jais_guard.data.annotations import ROOT

    if any(
        output == p or p in output.parents
        for p in (ROOT / "data", ROOT / "results/fine_tuning", Path(comparison["directory"]))
    ):
        raise ValueError("Baseline output must be separate from datasets and existing results")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", preflight)
    write_json(output / "validation_settings.json", settings)
    env = environment()
    props = torch.cuda.get_device_properties(0)
    env.update(
        {
            "batch_size": 1,
            "dtype": "torch.float16",
            "warmup_inputs": 1,
            "gpu_total_memory_bytes": props.total_memory,
            "models_concurrent": 1,
        }
    )
    write_json(output / "environment.json", env)
    tags = {
        "evaluation_stage": "independent_test",
        "approach": "untuned_prompt",
        "dataset_version": "test-400-v1",
        "constructed_benchmark": "true",
        "threshold_tuning": "false",
    }
    with mlflow.start_run(run_name=f"independent-test-untuned-{stamp}", tags=tags) as active:
        mlflow.log_params(
            {
                **settings["baseline"],
                "model": settings["model"],
                "revision": settings["revision"],
                "seed": 42,
                "do_sample": False,
                "batch_size": 1,
                "source_validation_run_id": settings["validation_run_id"],
                "test_sha256": frozen["test_sha256"],
            }
        )
        import pandas as pd

        dataset = mlflow.data.from_pandas(
            pd.DataFrame([{k: r[k] for k in ("id", "input_text", "label")} for r in rows]),
            source=str(args.test.resolve() / "test.jsonl"),
            name="test-400-v1",
            targets="label",
        )
        mlflow.log_input(dataset, context="testing")
        mlflow.log_text(RECOVERABLE_INSTRUCTION, "baseline_instruction.txt")
        mlflow.log_dict(frozen, "frozen_protocol.json")
        mlflow.log_text(DESCRIPTION, "benchmark_scope.txt")
        records, timing = infer_baseline(rows, settings, output)
        report = baseline_report(output, records)
        write_json(output / "timing.json", timing)
        locked_registry(args.test)
        for path, expected in protected.items():
            if sha(path) != expected:
                raise ValueError(f"Protected baseline/dataset file changed: {path}")
        write_json(
            output / "preservation.json",
            {"verified": protected, "classifier_files_before": comparison["files"]},
        )
        log_metrics(mlflow, "default", report["default"])
        log_metrics(mlflow, "latency", timing)
        mlflow.log_metrics({f"parsing/{k}": v for k, v in report["parsing_status_counts"].items()})
        mlflow.log_artifacts(str(output), artifact_path="final_test_baseline")
        target = attach_comparison(comparison, output, report, timing, active.info.run_id)
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "classifiers_rerun": 0,
                "run_id": active.info.run_id,
                "comparison_report": str(target),
                "test_sha256": frozen["test_sha256"],
            },
        )
        mlflow.log_artifacts(str(output), artifact_path="final_test_baseline")
    print(f"Baseline results: {output}\nUpdated comparison: {target}", flush=True)
