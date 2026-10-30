"""Run sequential HTTP benchmarks, one server/model at a time; preserve every attempt."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from jais_guard.serving import DEFAULT_DIRECTORY, checked_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--execution", default="http")
    args = parser.parse_args()
    if Path(args.execution).name != args.execution or args.execution in (".", ".."):
        parser.error("Execution must be a simple directory name")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Port 8000 is occupied; stop the existing demo server first")
    directory = args.directory.resolve()
    manifest = checked_manifest(directory)
    logs = directory / f"{args.execution}-server-logs"
    logs.mkdir(exist_ok=False)
    for key in manifest["models"]:
        print(f"HTTP benchmark: {key}", flush=True)
        env = {
            **os.environ,
            "JAISGUARD_DIRECTORY": str(directory),
            "JAISGUARD_MODEL": key,
            "HF_HUB_OFFLINE": "1",
        }
        with (logs / f"{key}.log").open("x") as log:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/serve_jaisguard.py")],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 240
                with httpx.Client(timeout=2, trust_env=False) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited; inspect {log.name}")
                        try:
                            response = client.get("http://127.0.0.1:8000/health")
                            if response.status_code == 200:
                                health = response.json()
                                if (
                                    health["model_key"] != key
                                    or health["manifest_identity"] != manifest["identity"]
                                ):
                                    raise RuntimeError("Unexpected server on port 8000")
                                break
                        except httpx.TransportError:
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError(f"Startup timed out; inspect {log.name}")
                        time.sleep(0.5)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/benchmark_latency.py"),
                        "http",
                        "--directory",
                        str(directory),
                        "--model",
                        key,
                        "--execution",
                        args.execution,
                    ],
                    cwd=ROOT,
                    check=True,
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
