"""Prepare a validation-only v2 lock, then evaluate regression on the seen 400-row benchmark."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "evaluate"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/finetuning-iteration2.json")
    parser.add_argument("--mlflow-db", type=Path, default=ROOT / "results/fine_tuning/mlflow.db")
    parser.add_argument("--test", type=Path, default=ROOT / "data/exports/test-test-400-v1")
    parser.add_argument(
        "--v1-results", type=Path, default=ROOT / "results/final_test/20261029T135604Z-0dd2abb5"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=ROOT / "results/final_test/untuned-baseline-20261029T143426Z-199eaaa4",
    )
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--execution-name",
        default="benchmark",
        help="New subdirectory; no resuming or overwriting predictions",
    )
    args = parser.parse_args(argv)
    from jais_guard.finetuning.version_comparison import checked_protocol, prepare

    directory = args.directory.resolve()
    if args.action == "prepare":
        print(
            json.dumps(
                prepare(
                    directory,
                    args.config,
                    args.mlflow_db,
                    args.test,
                    args.v1_results,
                    args.baseline_results,
                ),
                indent=2,
            )
        )
        return
    protocol, lock, rows = checked_protocol(directory)
    if args.action == "verify":
        print(
            json.dumps(
                {
                    "status": "verified-no-inference",
                    "scope": protocol["scope"],
                    "models_to_evaluate": 6,
                    "old_models_to_rerun": 0,
                    "test_sha256": protocol["test_sha256"],
                },
                indent=2,
            )
        )
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable in Codex. No v2 benchmark inference or MLflow runs started. "
            "Use this evaluate command in WSL; CPU inference is disabled."
        )
    if Path(args.execution_name).name != args.execution_name or args.execution_name in {".", ".."}:
        raise ValueError("Execution name must be a simple new directory name")
    output = directory / args.execution_name
    if output.exists():
        raise ValueError("Execution already exists; never overwrite or read partial v2 predictions")
    import mlflow

    from jais_guard.finetuning.evaluation import write_json
    from jais_guard.finetuning.final_test import fixed_report, timed_classifier
    from jais_guard.finetuning.tracking import configure, environment, log_metrics
    from jais_guard.finetuning.version_comparison import (
        SCOPE,
        comparison_report,
        load_reused,
    )

    if os.environ.get("MLFLOW_RUN_ID") or mlflow.active_run() is not None:
        raise ValueError("Version comparison must create new MLflow runs, never resume old ones")
    reused, _ = load_reused(
        rows,
        {"test_sha256": protocol["test_sha256"]},
        protocol["v1_registry"],
        protocol["v1_results"],
        protocol["baseline_results"],
    )
    mlflow = configure({"tracking_uri": args.tracking_uri, "experiment": protocol["experiment"]})
    client = mlflow.tracking.MlflowClient()
    for item in lock["checkpoints"]:
        source = client.get_run(item["source_runs"]["training_run_id"])
        if (
            source.info.status != "FINISHED"
            or source.data.params.get("approach") != item["approach"]
            or source.data.params.get("seed") != str(item["seed"])
        ):
            raise ValueError("Tracking server does not contain the selected v2 training runs")
    output.mkdir()
    write_json(output / "threshold_lock.json", lock)
    write_json(output / "protocol.json", protocol)
    env = environment()
    env.update(
        {
            "batch_size": protocol["batch_size"],
            "warmup_batches": protocol["warmup_batches"],
            "models_concurrent": 1,
            "benchmark_previously_examined": True,
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    )
    write_json(output / "environment.json", env)
    tags = {
        "evaluation_stage": "version_regression",
        "previously_examined_benchmark": "true",
        "constructed_benchmark": "true",
        "dataset_version": "test-400-v1",
        "threshold_tuning": "false",
    }
    results = list(reused)
    with mlflow.start_run(run_name="jaisguard-v1-v2-regression", tags=tags) as parent:
        write_json(
            output / "mlflow.json",
            {"run_id": parent.info.run_id, "tracking_uri": args.tracking_uri},
        )
        mlflow.log_text(SCOPE, "benchmark_scope.txt")
        mlflow.log_params(
            {
                "test_sha256": protocol["test_sha256"],
                "old_models_rerun": 0,
                "v2_models": 6,
                "batch_size": protocol["batch_size"],
                "threshold_lock_sha256": protocol["threshold_lock_sha256"],
            }
        )
        import pandas as pd

        dataset = mlflow.data.from_pandas(
            pd.DataFrame([{k: r[k] for k in ("id", "input_text", "label")} for r in rows]),
            source=str(Path(protocol["test"]) / "test.jsonl"),
            name="test-400-v1",
            targets="label",
        )
        mlflow.log_input(dataset, context="version_regression")
        for item in lock["checkpoints"]:
            name = f"{item['approach']}-{item['seed']}"
            with mlflow.start_run(
                run_name=f"v2-regression-{name}", nested=True, tags=tags
            ) as child:
                mlflow.log_params(
                    {
                        "version": "v2",
                        "approach": item["approach"],
                        "seed": item["seed"],
                        "selected_epoch": item["selected_epoch"],
                        "threshold_default": 0.5,
                        "threshold_validation_locked": item["threshold"],
                        **item["source_runs"],
                    }
                )
                mlflow.log_input(dataset, context="version_regression")
                scores, enriched, timing = timed_classifier(
                    item, rows, protocol["batch_size"], protocol["warmup_batches"]
                )
                measured, records = fixed_report(output / name, enriched, scores, item["threshold"])
                write_json(output / name / "timing.json", timing)
                write_json(output / name / "environment.json", env)
                for mode in ("default", "validation_locked"):
                    log_metrics(mlflow, mode, measured[mode])
                log_metrics(mlflow, "latency", timing)
                mlflow.log_artifacts(str(output / name), artifact_path="regression")
                results.append(
                    {
                        "version": "v2",
                        "approach": item["approach"],
                        "seed": item["seed"],
                        "metrics": measured,
                        "records": records,
                        "timing": timing,
                        "environment": env,
                        "source_runs": item["source_runs"],
                        "run_id": child.info.run_id,
                    }
                )
                print(f"Finished v2 {name}; model released", flush=True)
        checked_protocol(
            directory
        )  # Original weights, datasets, validation/cached results unchanged.
        comparison_report(output, rows, results)
        mlflow.log_artifacts(str(output), artifact_path="regression")
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "scope": SCOPE,
                "v2_models_evaluated": 6,
                "cached_models_reused": 7,
                "test_sha256": protocol["test_sha256"],
            },
        )
        mlflow.log_artifact(str(output / "completion.json"), artifact_path="regression")
    print(f"Version comparison report: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
