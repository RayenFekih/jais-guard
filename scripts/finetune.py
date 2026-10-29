"""Explicit entrypoint: inspect, GPU smoke, train, baseline, compare, or predict."""

import argparse
import gc
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def configuration(path):
    config = json.loads(Path(path).read_text())
    for field in ("train_path", "validation_path", "output_dir"):
        config[field] = str((ROOT / config[field]).resolve())
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/finetuning.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Read-only export integrity and token-length checks")
    smoke = sub.add_parser(
        "smoke", help="GPU-only disposable forward/backward checks, not training"
    )
    smoke.add_argument("--approach", choices=["frozen", "lora"], required=True)
    smoke.add_argument("--microbatch", type=int, choices=[1, 2], default=2)
    train = sub.add_parser("train", help="Full experiment; launch only after separate approval")
    train.add_argument("--approach", choices=["frozen", "lora"], required=True)
    train.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    train.add_argument("--microbatch", type=int, choices=[1, 2], default=2)
    baseline = sub.add_parser("baseline", help="Full 200-row fixed prompt evaluation")
    baseline.add_argument("--baseline-config", default=str(ROOT / "configs/prompt_baseline.json"))
    baseline.add_argument("--genai", action="store_true", help="Optional cached-output scoring")
    compare = sub.add_parser("compare", help="Summarize saved runs; never loads a model")
    compare.add_argument("directories", nargs="+")
    predict = sub.add_parser("predict", help="Reload a selected checkpoint for inference")
    predict.add_argument("checkpoint")
    predict.add_argument("text")
    predict.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = configuration(args.config)
    if args.command == "compare":
        return compare_runs(args.directories)
    if args.command == "predict":
        from jais_guard.finetuning.model import load_checkpoint, predict_rows

        model, tokenizer, saved = load_checkpoint(args.checkpoint, args.device)
        score, _ = predict_rows(
            model,
            tokenizer,
            [{"id": "input", "input_text": args.text}],
            saved["config"]["max_length"],
            args.device,
        )
        print(json.dumps({"p_gibberish": score[0], "prediction_default": int(score[0] >= 0.5)}))
        return
    from jais_guard.finetuning.data import encode_rows, load_datasets

    train_rows, val, tm, vm = load_datasets(config)
    if args.command == "inspect":
        from collections import Counter

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"],
            revision=config["revision"],
            trust_remote_code=True,
            local_files_only=True,
        )
        for name, rows in (("train", train_rows), ("validation", val)):
            lengths = sorted(
                len(r["input_ids"]) for r in encode_rows(rows, tokenizer, config["max_length"])
            )
            print(
                json.dumps(
                    {
                        "partition": name,
                        "rows": len(rows),
                        "labels": dict(Counter(r["label"] for r in rows)),
                        "label_origins": dict(Counter(r["label_origin"] for r in rows)),
                        "tokens_median": lengths[len(lengths) // 2],
                        "tokens_p95": lengths[int(len(lengths) * 0.95)],
                        "tokens_max": max(lengths),
                    },
                    ensure_ascii=False,
                )
            )
        print("PASS: hashes, labels, partitions, scripts, token budgets, and separation")
        return
    if args.command == "smoke":
        from jais_guard.finetuning.smoke import gpu_smoke

        return gpu_smoke(config, args.approach, args.microbatch, train_rows, val)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Run in your WSL terminal; do not change drivers here.")
    if args.command == "train":
        config["lora"]["microbatch"] = args.microbatch
        config["lora"]["accumulation"] = 16 // args.microbatch
        config["feature_batch_size"] = args.microbatch
    from jais_guard.finetuning.tracking import configure, log_setup

    mlflow = configure(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    name = f"{args.command}-{getattr(args, 'approach', 'recoverable_v1')}-{stamp}"
    with mlflow.start_run(run_name=name) as parent:
        directory = Path(config["output_dir"]) / parent.info.run_id
        directory.mkdir(parents=True, exist_ok=False)
        log_setup(mlflow, config, directory, train_rows, val, (tm, vm))
        if args.command == "train":
            from jais_guard.finetuning.model import load_classifier, seed_everything
            from jais_guard.finetuning.training import train_one

            seed_everything(args.seed)
            torch.cuda.reset_peak_memory_stats()
            with mlflow.start_run(run_name=f"{args.approach}-seed-{args.seed}", nested=True):
                mlflow.log_params(
                    {
                        "approach": args.approach,
                        "seed": args.seed,
                        "microbatch": args.microbatch,
                        "config": json.dumps(config),
                    }
                )
                child_dir = directory / f"{args.approach}-{args.seed}"
                child_dir.mkdir()
                log_setup(mlflow, config, child_dir, train_rows, val, (tm, vm))
                model, tokenizer = load_classifier(config, args.approach, "cuda", args.seed)
                train_one(model, tokenizer, train_rows, val, config, args.seed, child_dir, mlflow)
        else:
            baseline_run(config, val, directory, mlflow, args.genai, args.baseline_config)
        mlflow.log_artifacts(str(directory), artifact_path="run_files")
        print(f"Run artifacts: {directory}")


def baseline_run(config, val, directory, mlflow, genai=False, baseline_config=None):
    import torch

    from jais_guard.finetuning.evaluation import save_report, write_json
    from jais_guard.finetuning.model import seed_everything
    from jais_guard.finetuning.tracking import log_genai, log_metrics
    from jais_guard.inference import BaselineClassifier, BaselineConfig
    from jais_guard.prompts import RECOVERABLE_INSTRUCTION, RECOVERABLE_PROMPT_VERSION, build_prompt

    seed_everything(42)
    import hashlib
    import inspect
    from collections import Counter

    baseline = json.loads(
        Path(baseline_config or ROOT / "configs/prompt_baseline.json").read_text()
    )
    if (
        baseline["prompt_version"] != RECOVERABLE_PROMPT_VERSION
        or baseline["variant"] != "recoverable_v1"
    ):
        raise ValueError("Unknown prompt version; define a new version explicitly")
    baseline["prompt_sha256"] = hashlib.sha256(
        (
            baseline["prompt_version"]
            + baseline["variant"]
            + RECOVERABLE_INSTRUCTION
            + inspect.getsource(build_prompt)
        ).encode()
    ).hexdigest()
    classifier = BaselineClassifier(
        BaselineConfig(
            config["model"],
            config["revision"],
            reasoning_effort=baseline["reasoning_effort"],
            max_tokens=baseline["max_tokens"],
            max_model_len=baseline["max_model_len"],
            local_files_only=True,
        )
    )
    write_json(directory / "baseline_config.json", baseline)
    mlflow.log_dict(baseline, "baseline_config.json")
    mlflow.log_params(
        {
            "approach": "untuned_prompt",
            **baseline,
            "model": config["model"],
            "revision": config["revision"],
            "do_sample": False,
            "batch_size": 1,
            "dtype": str(classifier.model.dtype),
            "seed": 42,
        }
    )
    mlflow.log_text(RECOVERABLE_INSTRUCTION, "baseline_instruction.txt")
    mlflow.log_dict(classifier.model.generation_config.to_dict(), "generation_config.json")
    results = []
    for row in val:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = classifier.classify(row["input_text"], variant=baseline["variant"])
        torch.cuda.synchronize()
        results.append(
            {
                **row,
                **{k: v for k, v in result.items() if k != "prediction"},
                "prediction_id": {"VALID": 0, "GIBBERISH": 1}.get(result["prediction"], -1),
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
        )
        # Incremental save is confined to this new run directory.
        write_json(directory / "baseline_progress.json", results)
        print(f"baseline {len(results)}/{len(val)}", flush=True)
    report = save_report(
        directory / "evaluation", results, predictions=[r["prediction_id"] for r in results]
    )
    parsing = dict(Counter(r["parse_status"] for r in results))
    report["default"]["parsing_success_rate"] = parsing.get("success", 0) / len(results)
    report["parsing_status_counts"] = parsing
    report["unparseable_policy"] = (
        "INVALID (-1): wrong for accuracy; false negative for its true class in recall/F1; "
        "never silently assigned a binary label. VALID FRR counts only explicit GIBBERISH; "
        "valid acceptance and invalid counts additionally expose failures."
    )
    write_json(directory / "evaluation/metrics.json", report)
    report_path = directory / "evaluation/report.md"
    report_path.write_text(
        report_path.read_text().replace(
            "Validation-selected results; no independent test estimate.",
            "Fixed zero-shot baseline evaluated on validation; no independent test estimate.",
        )
    )
    with report_path.open("a") as stream:
        stream.write(
            f"\n## Prompt baseline\n\nPrompt: {baseline['prompt_version']}\n\n"
            f"Parsing success: {report['default']['parsing_success_rate']:.1%}\n\n"
            f"Statuses: {parsing}\n\n{report['unparseable_policy']}\n"
        )
    log_metrics(mlflow, "default", report["default"])
    mlflow.log_metrics({f"parsing/{k}": v for k, v in parsing.items()})
    if genai:
        records = [
            json.loads(line)
            for line in (directory / "evaluation/predictions.jsonl").read_text().splitlines()
        ]
        try:
            log_genai(mlflow, records)
        except Exception as exc:
            # Auxiliary inspection must not invalidate the completed standard report.
            mlflow.log_text(str(exc), "genai_optional_failure.txt")
    del classifier
    gc.collect()
    torch.cuda.empty_cache()


def compare_runs(directories):
    import numpy as np

    from jais_guard.finetuning.evaluation import write_csv, write_json

    collected = []
    groups = {}
    dataset_identity = None
    for directory in directories:
        path = Path(directory)
        run_config = json.loads((path / "config.json").read_text())
        identity = (run_config["train_sha256"], run_config["validation_sha256"])
        if dataset_identity is not None and identity != dataset_identity:
            raise ValueError("Cannot compare runs with different dataset hashes")
        dataset_identity = identity
        report = json.loads((path / "evaluation/metrics.json").read_text())
        saved = path / "best/classifier.json"
        metadata = (
            json.loads(saved.read_text()) if saved.exists() else {"approach": "prompt", "seed": 42}
        )
        for mode in ("default", "exploratory"):
            if mode not in report:
                continue
            entry = {
                "directory": str(path),
                "approach": metadata["approach"],
                "seed": metadata["seed"],
                "mode": mode,
                **{k: v for k, v in report[mode].items() if isinstance(v, (int, float))},
            }
            collected.append(entry)
            groups.setdefault((entry["approach"], mode), []).append(entry)
    summary = {}
    for (approach, mode), entries in groups.items():
        if len({r["seed"] for r in entries}) != len(entries):
            raise ValueError("Duplicate seeds: supply one selected run per approach/seed")
        summary[f"{approach}/{mode}"] = {
            "seeds": [r["seed"] for r in entries],
            "runs": len(entries),
            "complete_three_seeds": {r["seed"] for r in entries} == {42, 43, 44},
            "metrics": {
                k: {
                    "mean": float(np.mean([r[k] for r in entries])),
                    "std": float(np.std([r[k] for r in entries])),
                }
                for k in ("accuracy", "macro_f1", "gibberish_recall", "valid_false_rejection_rate")
            },
        }
    import uuid

    output = ROOT / "results/fine_tuning" / f"comparison-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    write_csv(
        output / "runs.csv",
        [
            {k: r.get(k) for k in sorted(set().union(*(e.keys() for e in collected)))}
            for r in collected
        ],
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}; validation-selected estimates, no independent test performance.")


if __name__ == "__main__":
    main()
