"""Serve the real asynchronous vLLM classifier, or the original HF app for comparison."""

import argparse
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
    import uvicorn

    from jais_guard.vllm_support.artifacts import DEFAULT_ROOT
    from jais_guard.vllm_support.server import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="vllm")
    args = parser.parse_args()
    os.environ.setdefault("HF_MODULES_CACHE", str(args.root.resolve() / "hf-modules-cache"))
    uvicorn.run(
        create_app(args.root, args.model, args.backend), host="127.0.0.1", port=8001, workers=1
    )


if __name__ == "__main__":
    main()
