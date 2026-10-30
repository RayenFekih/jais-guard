"""Prepare, measure and report all JaisGuard approaches without changing old experiments."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    from jais_guard.benchmarks.latency import benchmark_gpu, benchmark_http, prepare, report
    from jais_guard.serving.hf import DEFAULT_DIRECTORY, checked_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "list", "gpu", "http", "report", "verify"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--model", help="One exact key from list; one resident model per process")
    parser.add_argument("--execution", help="New output subdirectory; existing output refused")
    parser.add_argument("--gpu-execution", default="gpu")
    parser.add_argument("--http-execution", default="http")
    args = parser.parse_args()
    for name in (args.execution, args.gpu_execution, args.http_execution):
        if name is not None and (Path(name).name != name or name in (".", "..")):
            parser.error("Execution must be a simple directory name")
    if args.action == "prepare":
        result = prepare(args.directory)
        print(
            json.dumps({"selected": result["selected"], "models": list(result["models"])}, indent=2)
        )
    elif args.action == "list":
        print("\n".join(checked_manifest(args.directory)["models"]))
    elif args.action == "verify":
        from jais_guard.serving.hf import frozen_inputs

        manifest = checked_manifest(args.directory)
        print(
            f"Verified {len(manifest['models'])} local models; "
            f"{len(frozen_inputs(manifest))} inputs"
        )
    elif args.action == "report":
        print(
            report(
                args.directory, args.execution or "report", args.gpu_execution, args.http_execution
            )
        )
    else:
        manifest = checked_manifest(args.directory)
        if args.model not in manifest["models"]:
            parser.error("--model must be an exact key from list")
        function = benchmark_gpu if args.action == "gpu" else benchmark_http
        print(
            json.dumps(
                function(args.directory, args.model, execution=args.execution or args.action),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
