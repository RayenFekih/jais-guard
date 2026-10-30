"""Create a new attempt for the TorchVersion IPC repair, reusing verified HF references."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/vllm_serving/jaisguard-v2")
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "results/vllm_serving/jaisguard-v3"
    )
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(args.destination.resolve() / "hf-modules-cache"))
    from jais_guard.vllm_support.serialization_repair import repair

    print(json.dumps(repair(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
