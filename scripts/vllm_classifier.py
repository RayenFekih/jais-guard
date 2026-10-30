"""Prepare/export exact JaisGuard models, run gated parity and benchmark real vLLM."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


def main():
    from jais_guard.vllm_support.artifacts import (
        DEFAULT_ROOT,
        check_export_structure,
        checked_plan,
        export_classifier,
        inspect_installation,
        prepare,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "inspect",
            "prepare",
            "list",
            "export",
            "check-export",
            "collect",
            "verify-parity",
            "offline",
            "http-suite",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model")
    parser.add_argument("--report-name", default="report")
    parser.add_argument("--backend", choices=("hf-original", "hf-fp16", "vllm"))
    args = parser.parse_args()
    os.environ.setdefault("HF_MODULES_CACHE", str(args.root.resolve() / "hf-modules-cache"))
    if args.action == "inspect":
        result = inspect_installation()
    elif args.action == "prepare":
        result = prepare(args.root)
    elif args.action == "list":
        print("\n".join(checked_plan(args.root)["models"]))
        return
    elif args.action in ("http-suite", "report"):
        from jais_guard.vllm_support.benchmark import report, run_suite

        result = (
            run_suite(args.root)
            if args.action == "http-suite"
            else str(report(args.root, args.report_name))
        )
    else:
        if args.model not in checked_plan(args.root)["models"]:
            parser.error("--model must be one of the keys from list")
        if args.action == "export":
            result = export_classifier(args.root, args.model)
        elif args.action == "check-export":
            result = check_export_structure(args.root, args.model)
        else:
            from jais_guard.vllm_support.parity import benchmark_offline, collect, verify

            if args.action == "collect":
                if args.backend is None:
                    parser.error("collect requires --backend")
                result = collect(args.root, args.model, args.backend)
            elif args.action == "offline":
                result = benchmark_offline(args.root, args.model, args.backend or "vllm")
            else:
                result = verify(args.root, args.model)
                if not result["passed"]:
                    print(json.dumps(result, indent=2))
                    raise SystemExit("Parity failed; benchmarking remains blocked")
    if result is not None:
        if args.action == "prepare":
            result = {"identity": result["identity"], "models": list(result["models"])}
        elif args.action == "export":
            result = {k: v for k, v in result.items() if k != "files"}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
