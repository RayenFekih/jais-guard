"""Final independent-test evaluation. CUDA only; defaults to all six locked classifiers."""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")


def source_runs(client, registry, experiment_id):
    """Read-only provenance resolution; never resumes or changes a training run."""
    identities = {}
    for item in registry:
        parent = item["source_parent_run_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", parent):
            raise ValueError("Checkpoint directory does not identify its original MLflow parent")
        run = client.get_run(parent)
        if run.info.status != "FINISHED" or str(run.info.experiment_id) != str(experiment_id):
            raise ValueError("Selected training parent is unfinished or in another experiment")
        children = client.search_runs(
            [str(experiment_id)], filter_string=f"tags.`mlflow.parentRunId` = '{parent}'"
        )
        matches = [
            r
            for r in children
            if r.info.status == "FINISHED"
            and r.data.params.get("approach") == item["approach"]
            and r.data.params.get("seed") == str(item["seed"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "Cannot uniquely resolve selected checkpoint to a finished training run"
            )
        identities[f"{item['approach']}-{item['seed']}"] = {
            "parent_run_id": parent,
            "training_run_id": matches[0].info.run_id,
        }
    return identities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data/exports/test-test-400-v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/final_test")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run original untuned baseline only; never rerun classifiers",
    )
    parser.add_argument("--comparison", type=Path, help="Completed six-classifier result directory")
    parser.add_argument(
        "--validation-baseline", type=Path, help="Original validation baseline artifact directory"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify locks and tokenize; no models or MLflow writes",
    )
    args = parser.parse_args(argv)
    if args.baseline_only:
        if args.comparison is None or args.validation_baseline is None:
            parser.error("--baseline-only requires --comparison and --validation-baseline")
        from jais_guard.finetuning.test_baseline import run

        return run(args)
    if args.comparison is not None or args.validation_baseline is not None:
        parser.error("Baseline arguments require --baseline-only; no classifiers were started")
    if args.warmup_batches < 1:
        parser.error("At least one warmup batch is required")
    from jais_guard.finetuning.final_test import locked_registry, token_preflight
    from jais_guard.test_workflow import sha

    rows, frozen, registry = locked_registry(args.test)
    tokens = token_preflight(rows, registry)
    preflight = {
        "status": "verified-no-inference",
        "test": str(args.test.resolve()),
        "test_sha256": frozen["test_sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": sha(args.test / "freeze.json"),
        "counts": {"VALID": 188, "GIBBERISH": 212},
        "checkpoints": [
            {k: r[k] for k in ("approach", "seed", "checkpoint", "threshold")} for r in registry
        ],
        "token_budgets": tokens,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.check_only:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable here. No inference or MLflow runs started. "
            "Run this same command in your CUDA-capable WSL terminal; "
            "CPU inference is intentionally unsupported."
        )
    import mlflow

    from jais_guard.finetuning.evaluation import write_json
    from jais_guard.finetuning.final_test import (
        DESCRIPTION,
        fixed_report,
        summarize,
        timed_classifier,
    )
    from jais_guard.finetuning.tracking import configure, environment, log_metrics

    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Final evaluation must create new runs, not resume an existing MLflow run")
    mlflow = configure(
        {"tracking_uri": args.tracking_uri, "experiment": "JaisGuard/general-arabic-v1"}
    )
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("JaisGuard/general-arabic-v1")
    provenance = source_runs(client, registry, experiment.experiment_id)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_root.resolve() / stamp
    # Do not allow output within the frozen export or an existing training run.
    protected = [args.test.resolve(), ROOT / "data", ROOT / "results/fine_tuning"]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError(
            "Output must be separate from datasets and historical experiment artifacts"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", preflight)
    write_json(output / "source_runs.json", provenance)
    write_json(output / "frozen_protocol.json", frozen)
    env = environment()
    props = torch.cuda.get_device_properties(0)
    env.update(
        {
            "platform": sys.platform,
            "python": sys.version,
            "device": "cuda:0",
            "gpu_total_memory_bytes": props.total_memory,
            "gpu_compute_capability": [props.major, props.minor],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "attention": "sdpa",
            "padding": "right; longest in batch",
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "offline_model_cache": True,
            "models_concurrent": 1,
            "baseline_included": False,
            "argv": sys.argv,
        }
    )
    write_json(output / "environment.json", env)
    tags = {
        "evaluation_stage": "independent_test",
        "dataset_version": "test-400-v1",
        "constructed_benchmark": "true",
        "threshold_tuning": "false",
    }
    results, joined = (
        [],
        {r["id"]: {**r, "truth_id": int(r["label"] == "GIBBERISH"), "runs": []} for r in rows},
    )
    with mlflow.start_run(run_name=f"independent-test-{stamp}", tags=tags) as parent:
        write_json(
            output / "mlflow.json",
            {"run_id": parent.info.run_id, "tracking_uri": args.tracking_uri},
        )
        mlflow.log_params(
            {
                "test_sha256": frozen["test_sha256"],
                "examples": len(rows),
                "batch_size": args.batch_size,
                "warmup_batches": args.warmup_batches,
                "protocol": "all six locked; default and validation_locked; no test tuning",
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
        for item in registry:
            name = f"{item['approach']}-{item['seed']}"
            with mlflow.start_run(
                run_name=f"independent-test-{name}", nested=True, tags=tags
            ) as child:
                mlflow.log_params(
                    {
                        "approach": item["approach"],
                        "seed": item["seed"],
                        "threshold_default": 0.5,
                        "threshold_validation_locked": item["threshold"],
                        "checkpoint": item["checkpoint"],
                        "test_sha256": frozen["test_sha256"],
                        **provenance[name],
                    }
                )
                mlflow.log_input(dataset, context="testing")
                scores, enriched, timing = timed_classifier(
                    item, rows, args.batch_size, args.warmup_batches
                )
                report, records = fixed_report(output / name, enriched, scores, item["threshold"])
                write_json(output / name / "timing.json", timing)
                write_json(output / name / "locked_checkpoint.json", item)
                mlflow.log_dict(env, "environment.json")
                mlflow.log_dict(frozen, "frozen_protocol.json")
                for mode in ("default", "validation_locked"):
                    log_metrics(mlflow, mode, report[mode])
                log_metrics(mlflow, "latency", timing)
                mlflow.log_artifacts(str(output / name), artifact_path="final_test")
                results.append(
                    {
                        "approach": item["approach"],
                        "seed": item["seed"],
                        "threshold_validation_locked": item["threshold"],
                        "run_id": child.info.run_id,
                        "source_runs": provenance[name],
                        "metrics": report,
                        "timing": timing,
                    }
                )
                for record in records:
                    joined[record["id"]]["runs"].append(
                        {
                            "approach": item["approach"],
                            "seed": item["seed"],
                            **{
                                k: record[k]
                                for k in (
                                    "p_gibberish",
                                    "prediction_default",
                                    "prediction_validation_locked",
                                )
                            },
                        }
                    )
                print(f"Completed {name}; model released; artifacts: {output / name}", flush=True)
        # Verify pinned artifacts again after inference.
        locked_registry(args.test)
        preserved = {}
        expected_inputs = {
            "test.jsonl": preflight["test_sha256"],
            "manifest.json": preflight["manifest_sha256"],
            "freeze.json": preflight["freeze_sha256"],
        }
        for name, expected in expected_inputs.items():
            if sha(args.test / name) != expected:
                raise ValueError(f"Frozen dataset changed during evaluation: {name}")
            preserved[str(args.test.resolve() / name)] = expected
        for item in registry:
            for path, expected in item["locked_files"].items():
                if sha(path) != expected:
                    raise ValueError(f"Locked artifact changed during evaluation: {path}")
                preserved[path] = expected
        write_json(output / "preservation.json", {"verified": preserved})
        summary = summarize(output, results, list(joined.values()))
        write_json(output / "predictions.json", list(joined.values()))
        for group, values in summary.items():
            for measure, stats in values.items():
                mlflow.log_metrics({f"{group}/{measure}/{stat}": v for stat, v in stats.items()})
        mlflow.log_text(DESCRIPTION, "benchmark_scope.txt")
        mlflow.log_artifacts(str(output), artifact_path="final_test")
        write_json(
            output / "completion.json",
            {"status": "complete", "models_evaluated": 6, "test_sha256": frozen["test_sha256"]},
        )
        mlflow.log_artifact(str(output / "completion.json"), artifact_path="final_test")
    print(f"Final comparison report: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
