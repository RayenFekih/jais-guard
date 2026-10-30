"""Opt-in paid Jev smoke/evaluation; offline comparison reporting. No automatic API calls."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jais_guard.benchmarks import jev  # noqa: E402
from jais_guard.benchmarks.jev_reporting import cached_comparison, save_report  # noqa: E402
from jais_guard.data.annotations import arabic_script  # noqa: E402


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    stages = cli.add_subparsers(dest="stage", required=True)
    for name in ("smoke", "evaluate", "report"):
        stage = stages.add_parser(name)
        stage.add_argument("--run-dir", type=Path, required=True)
        if name == "report":
            stage.add_argument("--allow-incomplete", action="store_true")
            continue
        stage.add_argument("--question", type=Path, default=ROOT / "configs/jev_benchmark.json")
        stage.add_argument("--model", default="jev-latest")
        stage.add_argument("--concurrency", type=int, default=1)
        stage.add_argument("--max-retries", type=int, default=3)
        stage.add_argument("--timeout-seconds", type=float, default=30)
        stage.add_argument("--backoff-seconds", type=float, default=1)
        stage.add_argument("--max-backoff-seconds", type=float, default=30)
        execute = stage.add_mutually_exclusive_group()
        execute.add_argument("--execute-paid", action="store_true")
        execute.add_argument(
            "--dry-run",
            action="store_true",
            help="Default: validate locally, no API calls or output writes",
        )
        stage.add_argument("--retry-failed", action="store_true")
        stage.add_argument(
            "--retry-uncertain",
            action="store_true",
            help="May repeat a billed request interrupted before checkpoint commit",
        )
        if name == "smoke":
            stage.add_argument("--text", action="append", required=True)
        else:
            stage.add_argument("--test", type=Path, default=ROOT / "data/exports/test-test-400-v1")
            stage.add_argument(
                "--comparison",
                type=Path,
                default=ROOT / "results/version_comparison/jaisguard-v1-v2-v1/benchmark",
            )
            stage.add_argument(
                "--allow-alias",
                action="store_true",
                help="Explicitly accept weaker reproducibility; any detected "
                "returned-model drift stops execution",
            )
    return cli


def safe_directory(directory):
    resolved = directory.resolve()
    # Dedicated new directory only; never turn a dataset/experiment directory into an API run.
    protected = [ROOT / "data", ROOT / "configs", ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    historical_results = resolved.is_relative_to(ROOT / "results") and not resolved.is_relative_to(
        ROOT / "results/jev"
    )
    if (
        historical_results
        or resolved == ROOT
        or any(resolved == p or p in resolved.parents for p in protected)
    ):
        raise ValueError("Use a separate Jev result directory, e.g. results/jev/test400-jev1140-v1")
    if resolved.exists() and any(resolved.iterdir()) and not (resolved / "manifest.json").exists():
        raise ValueError("Output directory is nonempty without a Jev manifest; refusing reuse")


def prepare(args):
    safe_directory(args.run_dir)
    if not 1 <= args.concurrency <= 16 or not 0 <= args.max_retries <= 10:
        raise ValueError("Concurrency must be 1..16 and max retries 0..10")
    if (
        not 0 < args.timeout_seconds <= 300
        or not 0 < args.backoff_seconds <= args.max_backoff_seconds <= 300
    ):
        raise ValueError("Timeout/backoff must be positive and bounded by 300 seconds")
    if args.model != "jev-latest" and not jev.PINNED.fullmatch(args.model):
        raise ValueError("Use jev-latest for smoke, or a documented pinned jev-X.Y.Z version")
    question = jev.load_question(args.question)
    settings = {
        k: getattr(args, k)
        for k in (
            "concurrency",
            "max_retries",
            "timeout_seconds",
            "backoff_seconds",
            "max_backoff_seconds",
        )
    }
    for name in ("timeout_seconds", "backoff_seconds", "max_backoff_seconds"):
        settings[name] = float(settings[name])
    if args.stage == "smoke":
        if len(args.text) > 20 or any(
            not text.strip() or not arabic_script(text) for text in args.text
        ):
            raise ValueError("Smoke accepts 1..20 manually supplied Arabic-script inputs")
        rows = [{"id": f"smoke-{i + 1}", "input_text": text} for i, text in enumerate(args.text)]
        config = jev.manifest_config("smoke", rows, question, args.model, settings)
    else:
        if args.execute_paid and not jev.PINNED.fullmatch(args.model) and not args.allow_alias:
            raise ValueError(
                "Smoke jev-latest first, then pass its documented pinned model ID; "
                "--allow-alias explicitly accepts alias risk"
            )
        settings["allow_alias"] = args.allow_alias
        rows = jev.frozen_rows(args.test)
        comparison = cached_comparison(args.comparison, rows, jev.sha(args.test / "test.jsonl"))
        config = jev.manifest_config(
            "evaluate", rows, question, args.model, settings, args.test, args.comparison
        )
        config["comparison_files"] = comparison["files"]
    statuses = {}
    if (args.run_dir / "manifest.json").exists():
        manifest, examples, _ = jev.read_ledger(args.run_dir)
        if manifest["config"] != config:
            raise ValueError(
                "Existing run configuration changed; resume with identical settings "
                "or use a new directory"
            )
        statuses = dict(jev.Counter(e["status"] for e in examples))
    return config, rows, statuses


def main(argv=None):
    args = parser().parse_args(argv)
    # HTTP debug loggers are never enabled by this CLI; headers and raw errors are not logged.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        if args.stage == "report":
            print(save_report(args.run_dir, args.allow_incomplete))
            return 0
        config, rows, statuses = prepare(args)
        pending = (
            len(rows) if not statuses else statuses.get("pending", 0) + statuses.get("retrying", 0)
        )
        if args.retry_failed:
            pending += statuses.get("failed", 0)
        if args.retry_uncertain:
            pending += statuses.get("uncertain", 0) + statuses.get("in_flight", 0)
        plan = {
            "stage": args.stage,
            "paid_execution": args.execute_paid,
            "model": args.model,
            "question_version": config["question"]["question_version"],
            "protocol_sha256": jev.digest(config),
            "examples": len(rows),
            "existing_statuses": statuses,
            "successful_requests_skipped": statuses.get("success", 0),
            "queued_examples": pending,
            "maximum_new_attempts": pending * (args.max_retries + 1),
            "concurrency": args.concurrency,
            "output": str(args.run_dir),
            "cost": "Usage-dependent; no calls or token-count requests are made by "
            "dry-run. See documented pricing snapshot in config.",
        }
        print(json.dumps(plan, indent=2))
        if not args.execute_paid:
            print(
                "DRY RUN: no API calls, credential access, or output writes. "
                "Explicit --execute-paid is required."
            )
            return 0
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "Set TYPESAFE_API_KEY in your WSL environment; never put credentials "
                "in CLI arguments"
            )
        result = asyncio.run(
            jev.execute(
                args.run_dir,
                config,
                rows,
                key,
                retry_failed=args.retry_failed,
                retry_uncertain=args.retry_uncertain,
            )
        )
        print(json.dumps({"statuses": result}, indent=2))
        if args.stage == "smoke":
            _, examples, _ = jev.read_ledger(args.run_dir)
            for record in examples:
                print(
                    json.dumps(
                        {
                            "id": record["id"],
                            "status": record["status"],
                            "response": json.loads(record["response_json"])
                            if record["response_json"]
                            else None,
                        },
                        ensure_ascii=False,
                    )
                )
            print(
                "For the benchmark use the returned documented pinned model ID in a "
                "NEW run directory. Smoke has no labels or accuracy metric."
            )
        return 0 if result.get("success", 0) == len(rows) else 2
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "Interrupted. Committed successes are preserved; in-flight requests "
            "need explicit --retry-uncertain before resending.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
