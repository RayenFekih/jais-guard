"""Read-only verification of the explicitly reviewed benchmark evidence package."""

import hashlib
import json
from pathlib import Path

import numpy as np


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "allowlist.json").read_text())
    for name, record in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Evidence path escapes package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    metrics = json.loads((directory / "vllm/metrics.json").read_text())
    observations = json.loads((directory / "vllm/http-timings.json").read_text())
    if len(metrics["http"]) != 16 or len(observations) != 16:
        raise ValueError("Incomplete HTTP suite")
    for row in metrics["http"]:
        key = f"{row['model']}/{row['backend']}/c{row['concurrency']}"
        values = observations[key]
        expected = [row[k] for k in ("p50_ms", "p95_ms", "p99_ms")]
        if (
            len(values) != 400
            or not np.isfinite(values).all()
            or not np.isclose(np.mean(values), row["mean_ms"])
            or not np.allclose(np.percentile(values, [50, 95, 99]), expected)
            or not np.isclose(400 / row["elapsed_seconds"], row["throughput_per_second"])
        ):
            raise ValueError(f"HTTP aggregate mismatch: {key}")
    gates = json.loads((directory / "vllm/parity-summary.json").read_text())
    if {gate["model_key"] for gate in gates} != {"v2-frozen-43", "v2-lora-44"} or len(gates) != 2:
        raise ValueError("Incomplete parity gates")
    for gate in gates:
        fp16 = gate["comparisons"]["hf-fp16_vs_vllm"]
        if (
            not gate["passed"]
            or not gate["trained_head_verified"]
            or fp16["locked_prediction_mismatches"] != 0
            or fp16["default_prediction_mismatches"] != 0
            or fp16["max_abs_probability_difference"] > 0.02
            or fp16["mean_abs_probability_difference"] > 0.005
        ):
            raise ValueError("FP16 parity gate failed")
    return {
        "files": len(manifest["files"]),
        "http_workloads": 16,
        "requests": 6400,
        "fp16_gates": len(gates),
    }
