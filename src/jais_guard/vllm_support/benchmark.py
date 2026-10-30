"""Matched sequential/concurrent HTTP measurements, gated by probability parity."""

import asyncio
import csv
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import httpx

from jais_guard.benchmarks.latency import summary, write_new
from jais_guard.serving.hf import read_json

from .artifacts import checked_plan
from .parity import (
    acceptance_reference,
    checked_gate,
    compare_probabilities,
    inputs,
    require_parity,
)

ROOT = Path(__file__).resolve().parents[3]


async def http_run(root, key, backend, concurrency):
    plan = checked_plan(root)
    require_parity(root, key)
    if concurrency not in (1, 4, 8, 16) or backend not in ("hf", "vllm"):
        raise ValueError("Unsupported benchmark configuration")
    folder = Path(root) / "http" / key / backend / f"c{concurrency}"
    folder.mkdir(parents=True, exist_ok=False)
    rows = [r for r in inputs(root, key) if r["category"] == "benchmark"]
    reference_path = Path(root) / "parity" / key / acceptance_reference(plan) / "predictions.jsonl"
    reference = {r["id"]: r for r in map(json.loads, reference_path.read_text().splitlines())}
    endpoint = "/predict" if backend == "hf" else "/classify"
    field = "text" if backend == "hf" else "input"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8001", timeout=180, limits=limits, trust_env=False
    ) as client:
        response = await client.get("/benchmark/info")
        response.raise_for_status()
        server = response.json()
        if (
            server["backend"] != backend
            or server["model_key"] != key
            or server["plan_identity"] != plan["identity"]
        ):
            raise ValueError("Wrong server/model/plan")
        for row in rows[:3]:
            (await client.post(endpoint, json={field: row["input_text"]})).raise_for_status()
        (await client.post("/benchmark/reset")).raise_for_status()
        pending = iter(enumerate(rows))
        samples = [None] * len(rows)

        async def worker():
            for index, row in pending:
                start = time.perf_counter()
                try:
                    response = await client.post(endpoint, json={field: row["input_text"]})
                    elapsed = (time.perf_counter() - start) * 1000
                    response.raise_for_status()
                    body = response.json()
                    if body["model_key"] != key or body["threshold"] != server["threshold"]:
                        raise ValueError("Response model or threshold changed")
                    probs = [body["probabilities"][name] for name in ("VALID", "GIBBERISH")]
                    if body["class_id"] != int(probs[1] >= server["threshold"]):
                        raise ValueError("Response did not apply locked threshold")
                    samples[index] = {
                        "id": row["id"],
                        "category": row["category"],
                        "token_ids": row["token_ids"],
                        "probabilities": probs,
                        "http_ms": elapsed,
                        "server_inference_ms": body["inference_ms"],
                    }
                except Exception as exc:
                    samples[index] = {
                        "id": row["id"],
                        "error": str(exc),
                        "http_ms": (time.perf_counter() - start) * 1000,
                    }

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
        response = await client.get("/benchmark/memory")
        response.raise_for_status()
        memory = response.json()
    write_new(folder / "samples.json", samples)
    failed = [s for s in samples if s is None or "error" in s]
    if failed:
        write_new(folder / "failure.json", {"failed_requests": len(failed), "samples": failed})
        raise RuntimeError("HTTP failures recorded; no successful timing report created")
    match = compare_probabilities(
        [reference[r["id"]] for r in rows], samples, server["threshold"], plan["parity_policy"]
    )
    write_new(folder / "response_parity.json", match)
    if not match["passed"]:
        raise ValueError("Concurrent HTTP predictions differ; performance result withheld")
    result = {
        "model_key": key,
        "backend": backend,
        "concurrency": concurrency,
        "plan_identity": plan["identity"],
        "server": server,
        "memory": memory,
        "latency": summary([r["http_ms"] for r in samples], elapsed),
        "server_inference": summary([r["server_inference_ms"] for r in samples]),
        "response_parity": match,
        "warmup_inputs": 3,
        "batch_size_per_request": 1,
        "scope": "400 closed-loop HTTP requests, one input each; connection reuse. "
        "vLLM may continuously batch concurrent inputs; HF serializes GPU forward.",
    }
    write_new(folder / "metrics.json", result)
    return result


def run_suite(root):
    plan = checked_plan(root)
    for key in plan["models"]:
        require_parity(root, key)
    logs = Path(root) / "http-server-logs"
    logs.mkdir(exist_ok=False)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 8001)) == 0:
            raise RuntimeError("Port 8001 is occupied; stop the existing demo first")
    for key in plan["models"]:
        for backend in ("hf", "vllm"):
            python = ROOT / (".venv" if backend == "hf" else ".venv-vllm") / "bin/python"
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "HF_HUB_OFFLINE": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_NO_USAGE_STATS": "1",
            }
            with (logs / f"{key}-{backend}.log").open("x") as log:
                process = subprocess.Popen(
                    [
                        str(python),
                        str(ROOT / "scripts/serve_vllm_classifier.py"),
                        "--root",
                        str(Path(root).resolve()),
                        "--model",
                        key,
                        "--backend",
                        backend,
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                start = time.perf_counter()
                try:
                    deadline = time.monotonic() + 300
                    with httpx.Client(timeout=2, trust_env=False) as client:
                        while True:
                            if process.poll() is not None:
                                raise RuntimeError(f"Server exited; inspect {log.name}")
                            try:
                                response = client.get("http://127.0.0.1:8001/benchmark/info")
                                if response.status_code == 200:
                                    info = response.json()
                                    if (
                                        info["model_key"] != key
                                        or info["backend"] != backend
                                        or info["plan_identity"] != plan["identity"]
                                    ):
                                        raise RuntimeError("Unexpected server identity")
                                    break
                            except httpx.TransportError:
                                pass
                            if time.monotonic() > deadline:
                                raise RuntimeError(f"Server startup timed out; inspect {log.name}")
                            time.sleep(0.5)
                    startup = time.perf_counter() - start
                    write_new(
                        logs / f"{key}-{backend}-startup.json",
                        {"process_to_ready_seconds": startup, "server": info},
                    )
                    for concurrency in (1, 4, 8, 16):
                        print(f"Benchmark {key} {backend} c={concurrency}", flush=True)
                        result = asyncio.run(http_run(root, key, backend, concurrency))
                        print(json.dumps(result["latency"]), flush=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    # Server cleanup must complete before another GPU model is started.


def report(root, name="report"):
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Report name must be a simple directory name")
    plan = checked_plan(root)
    folder = Path(root) / name
    folder.mkdir(exist_ok=False)
    lines = [
        "# JaisGuard: vLLM versus original Hugging Face/FastAPI",
        "",
        "Same validation-selected trained checkpoints, exact saved thresholds, label order "
        "VALID=0 / GIBBERISH=1 and final nonpadding token. "
        "Historical quality results are unchanged. "
        "No Jev API requests. A CPU export alone is not successful conversion: "
        "GPU parity is required.",
        "",
        "Acceptance: HF FP16 versus vLLM FP16, verified trained head and bias, zero default "
        "and locked-threshold mismatches, max probability difference ≤0.02 and mean ≤0.005. "
        "Original BF16 differences are preserved as nonblocking diagnostics by user authorization.",
        "",
        "## Parity",
        "",
    ]
    results = []
    offline_results = []
    for key, item in plan["models"].items():
        path = Path(root) / "parity" / key / "gate.json"
        gate = checked_gate(root, key) if path.exists() else None
        lines += [f"### {key}", "", f"Locked threshold: {item['threshold']}.", ""]
        if gate is None:
            lines += ["**PENDING GPU PARITY — no conversion or performance claim.**", ""]
        else:
            for name, m in gate["comparisons"].items():
                lines.append(
                    f"- {name}: max absolute probability difference "
                    f"{m['max_abs_probability_difference']:.8f}, mean "
                    f"{m['mean_abs_probability_difference']:.8f}, locked mismatches "
                    f"{m['locked_prediction_mismatches']}, default mismatches "
                    f"{m['default_prediction_mismatches']}; within_strict_tolerance={m['passed']} "
                    "(BF16 rows are diagnostic)."
                )
        if gate is not None:
            for category, m in gate.get("by_category", {}).items():
                lines.append(
                    f"- {category}: {m['samples']} inputs, max/mean probability difference "
                    f"{m['max_abs_probability_difference']:.8f}/"
                    f"{m['mean_abs_probability_difference']:.8f}, "
                    f"locked mismatches={m['locked_prediction_mismatches']}."
                )
        for backend in ("hf-original", "hf-fp16", "vllm"):
            offline_path = Path(root) / "offline" / key / backend / "metrics.json"
            if offline_path.exists():
                if gate is None or not gate["passed"]:
                    raise ValueError("Offline measurements without valid parity")
                m = read_json(offline_path)
                if (
                    m["plan_identity"] != plan["identity"]
                    or not m["response_parity"]["passed"]
                    or m["latency"]["samples"] != 400
                ):
                    raise ValueError("Incompatible offline measurements")
                offline_results.append(m)
        for backend in ("hf", "vllm"):
            for concurrency in (1, 4, 8, 16):
                path = Path(root) / "http" / key / backend / f"c{concurrency}" / "metrics.json"
                if not path.exists():
                    continue
                if gate is None or not gate["passed"]:
                    raise ValueError("Metrics without valid parity gate")
                data = read_json(path)
                if (
                    data["plan_identity"] != plan["identity"]
                    or not data["response_parity"]["passed"]
                ):
                    raise ValueError("Incompatible benchmark or response parity failed")
                startup = read_json(
                    Path(root) / "http-server-logs" / f"{key}-{backend}-startup.json"
                )
                result = {
                    "model": key,
                    "backend": backend,
                    "concurrency": concurrency,
                    **data["latency"],
                    "model_load_seconds": data["server"]["model_load_seconds"],
                    "process_to_ready_seconds": startup["process_to_ready_seconds"],
                    "peak_allocated_bytes": data["memory"]["peak_allocated_bytes"],
                    "peak_reserved_bytes": data["memory"]["peak_reserved_bytes"],
                }
                results.append(result)
    lines += [
        "",
        "## HTTP latency",
        "",
        "| Checkpoint | Backend | Concurrent requests | Mean ms | p50 | p95 | p99 | requests/s | "
        "Load s | Ready s | Peak allocated MiB | Peak reserved MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['backend']} | {r['concurrency']} | "
            + " | ".join(
                f"{r[k]:.3f}"
                for k in (
                    "mean_ms",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "throughput_per_second",
                    "model_load_seconds",
                    "process_to_ready_seconds",
                )
            )
            + f" | {r['peak_allocated_bytes'] / 2**20:.1f} | "
            + f"{r['peak_reserved_bytes'] / 2**20:.1f} |"
        )
    lines += [
        "",
        f"Completed HTTP workloads: {len(results)}/{len(plan['models']) * 8}. "
        "Missing workloads remain pending; partial results are not a completed suite.",
    ]
    lines += [
        "",
        "## Standalone batch-1 local inference",
        "",
        "Includes original input tokenization and transfers; no HTTP. "
        "vLLM uses LLM.classify, HF uses the existing Predictor. "
        "Both must preserve the same locked classifications.",
        "",
        "| Model | Backend | Mean ms | p50 | p95 | p99 | requests/s | Load s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in offline_results:
        m = item["latency"]
        lines.append(
            f"| {item['model_key']} | {item['backend']} | "
            + " | ".join(
                f"{m[k]:.3f}"
                for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_per_second")
            )
            + f" | {item['model_load_seconds']:.3f} |"
        )
    if not offline_results:
        lines += ["", "Standalone GPU measurements pending."]
    if not results:
        lines += ["", "**GPU/HTTP timings pending WSL execution.**"]
    lines += [
        "",
        "## Measurement scope and limitations",
        "",
        "Each concurrency setting sends the same 400 frozen inputs once, one input per request, "
        "using a reused HTTP connection pool and three warmup inputs. Throughput is completed "
        "requests / wall time; mean and linear p50/p95/p99 summarize request durations. "
        "vLLM can batch up to 16 concurrent requests; "
        "the original HF GPU lock serializes inference. "
        "Every HTTP result must pass probability/class parity against the HF FP16 reference. "
        "The existing FastAPI app and GPU lock are reused with an FP16-only predictor adaptation; "
        "the original BF16 implementation remains unchanged.",
        "",
        "Loading is separate from warm inference; ready time includes process startup, integrity "
        "checks, tokenizer and engine initialization. Peak memory is CUDA allocated/reserved "
        "memory in the actual GPU worker, with resident weights included; not total board usage. "
        "Reserved memory can retain loading/warmup allocations. "
        "vLLM reserves more memory by design. "
        "Only one model is resident at a time. These local HTTP results exclude remote networking. "
        "Existing Jev results include remote network overhead and are not rerun or relabeled.",
        "",
        "FP16 serving equivalence uses the existing tolerances: maximum absolute "
        "difference ≤0.02 and mean ≤0.005 across both probabilities, with zero classification "
        "mismatches at default and saved thresholds. Exceeding either limit blocks performance "
        "claims against the HF FP16 reference. BF16 differences do not block this authorized "
        "engineering experiment and remain reported above. "
        "Thresholds and tolerances are unchanged.",
        "",
        "vLLM 0.32.0 uses its supported Transformers backend with vLLM attention/linear layers, "
        "LAST pooler and the verified original FP32 head weight and bias. `classifier.bias` "
        "avoids the installed LegacyMixin rule that drops `score.bias`. LoRA uses a verified "
        "CPU FP32 static merge into a new export; runtime LoRA classification support is not "
        "assumed. Merging may change floating-point rounding; parity remains mandatory.",
        "",
        "This is the previously examined, constructed Arabic benchmark, not new quality evaluation "
        "or production traffic. Hardware, clocks, thermal state, dtype, kernel/backend versions "
        "and caches affect latency. No test-based model/threshold selection or silent HF fallback.",
        "",
    ]
    (folder / "comparison.md").write_text("\n".join(lines))
    write_new(
        folder / "metrics.json",
        {"plan_identity": plan["identity"], "http": results, "offline": offline_results},
    )
    if results:
        with (folder / "comparison.csv").open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    comparison_charts(folder, results)
    return folder


def comparison_charts(folder, rows):
    """Only plot measured workloads; pending results are never invented."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.8 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for backend in ("hf", "vllm"):
            group = sorted(
                (r for r in rows if r["model"] == model and r["backend"] == backend),
                key=lambda r: r["concurrency"],
            )
            if not group:
                continue
            label = "HF/FastAPI FP16" if backend == "hf" else "vLLM FP16"
            for j, (field, ylabel) in enumerate(
                (
                    ("throughput_per_second", "Requests / second"),
                    ("p95_ms", "p95 HTTP latency (ms)"),
                )
            ):
                axes[i, j].plot(
                    [r["concurrency"] for r in group],
                    [r[field] for r in group],
                    marker="o",
                    label=label,
                )
                axes[i, j].set(
                    title=model, xlabel="HTTP concurrency", ylabel=ylabel, xticks=[1, 4, 8, 16]
                )
                axes[i, j].legend()
                axes[i, j].grid(alpha=0.2)
    fig.suptitle("Same frozen 400 inputs; trained FP32 heads; FP16 backbones")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"latency-throughput.{suffix}", dpi=180)
    plt.close(fig)
    # Startup/memory are separate from warm request latency.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    representatives = [r for r in rows if r["concurrency"] == 1]
    labels = [r["model"] + " / " + r["backend"] for r in representatives]
    axes[0].barh(labels, [r["process_to_ready_seconds"] for r in representatives])
    axes[0].set(xlabel="Process to ready (seconds)", title="Startup")
    axes[1].barh(labels, [r["peak_reserved_bytes"] / 2**30 for r in representatives])
    axes[1].set(xlabel="Peak CUDA reserved GiB (concurrency 1)", title="GPU allocator memory")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"startup-memory.{suffix}", dpi=180)
    plt.close(fig)
