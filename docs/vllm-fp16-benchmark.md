# Approved JaisGuard FP16 serving experiment

This supersedes the BF16-blocking acceptance rule for the new experiment only.
Results live in **`results/vllm_serving/jaisguard-v4`**. The prior v2/v3 attempts,
BF16 findings, checkpoints, dataset, thresholds and historical benchmarks remain
unchanged. Neither environment was reinstalled or upgraded.

Acceptance requires the actual trained FP32 head **and bias** hashes to match,
zero default/validation-locked classification mismatches against **HF FP16**, and
maximum/mean absolute probability differences ≤ **0.02 / 0.005**. These numerical
tolerances are unchanged. BF16 comparisons remain recorded as diagnostics and do
not block this user-authorized experiment. No threshold or model selection occurs.

The frozen `v2-frozen-43` observations are reused with verified source hashes,
unchanged token IDs, and provenance receipts. Its new FP16 gate passes; all
original BF16 disagreements remain in its gate/report. No frozen GPU parity
rerun is necessary. LoRA `v2-lora-44` must still pass the same checks.

The matched comparison reuses the existing FastAPI app, Predictor lifecycle,
request validation, classification head, saved threshold and serialized GPU lock.
A separate `FP16Predictor` adapter converts the backbone to FP16 and selects FP16
autocast; the trained head stays FP32. The original BF16 implementation is not
edited. This matches the collected HF FP16 reference and avoids comparing FP16
vLLM against a differently rounded BF16 server. vLLM executes its actual pooling
engine with the verified trained head and statically merged LoRA weights.

## Exact remaining WSL commands

The v4 preparation and frozen acceptance check are already complete. Stop any
existing GPU model server, then run the following in Bash. Explicit interpreter
paths select the two existing environments regardless of shell activation.

```bash
cd /home/rayen/projects/fine-tuning
set -euo pipefail
experiment=results/vllm_serving/jaisguard-v4

# LoRA: original BF16 diagnostics, HF FP16 reference, then real vLLM FP16.
.venv/bin/python scripts/vllm_classifier.py collect \
  --root "$experiment" --model v2-lora-44 --backend hf-original
.venv/bin/python scripts/vllm_classifier.py collect \
  --root "$experiment" --model v2-lora-44 --backend hf-fp16
.venv-vllm/bin/python scripts/vllm_classifier.py collect \
  --root "$experiment" --model v2-lora-44 --backend vllm
.venv/bin/python scripts/vllm_classifier.py verify-parity \
  --root "$experiment" --model v2-lora-44

# Both classifiers: batch-1 local inference, with one model loaded at a time.
while IFS= read -r model_key; do
  .venv/bin/python scripts/vllm_classifier.py offline \
    --root "$experiment" --model "$model_key" --backend hf-fp16
  .venv-vllm/bin/python scripts/vllm_classifier.py offline \
    --root "$experiment" --model "$model_key" --backend vllm
done < <(.venv/bin/python scripts/vllm_classifier.py list --root "$experiment")

# Both classifiers and engines, sequentially: HTTP concurrency 1, 4, 8 and 16.
.venv/bin/python scripts/vllm_classifier.py http-suite --root "$experiment"

# Final report, machine-readable metrics and PNG/SVG charts.
.venv/bin/python scripts/vllm_classifier.py report --root "$experiment"
```

A failing LoRA FP16 gate stops these commands before performance benchmarks. BF16
numerical or classification differences alone do not stop them. Each HTTP workload
also compares its 400 responses against HF FP16, so concurrency cannot silently
introduce new classification mismatches. Failed requests are preserved and stop
successful performance reporting. There are no retries or omitted observations.

Every GPU stage runs in a fresh process. The HTTP suite starts/stops a single
server at a time on port 8001; it refuses an occupied port. No more than one 0.9B
model is loaded on the GPU. Settings remain FP16, eager execution, max engine
length 1024, 75% GPU memory utilization, and at most 16 continuously batched
requests. The original classifier's 64-token policy is unchanged.

## Outputs

```text
results/vllm_serving/jaisguard-v4/
  upgrade.json                        provenance and reused observations
  plan.json                           authorized acceptance rule and source locks
  parity/<model>/gate.json             FP16 acceptance plus preserved BF16 diagnostics
  offline/<model>/<backend>/           400 batch-1 observations and GPU timing
  http/<model>/hf/c1,c4,c8,c16/         existing FastAPI app with FP16 predictor
  http/<model>/vllm/c1,c4,c8,c16/       real async vLLM classifier
  http-server-logs/                    startup logs and process-to-ready measurements
  report-prepared/                    frozen accepted; LoRA/GPU timings pending
  report/
    comparison.md                     final comparison
    comparison.csv                    per-model/backend/concurrency statistics
    metrics.json                      complete measured statistics
    latency-throughput.png/.svg       throughput and p95 versus concurrency
    startup-memory.png/.svg           startup and concurrency-1 GPU reserved memory
```

Mean, p50, p95, p99, requests/sec, startup/loading time, and peak allocated/reserved
GPU memory are recorded. HTTP request latency includes transport, queueing and
serialization; standalone timing excludes HTTP. Loading and three warmup inputs
are excluded from steady-state summaries. Throughput is 400 completed requests
per measured wall time; percentile interpolation is NumPy's linear default.
Allocator memory includes resident weights and retained reservations, not other
processes or total board usage. Model loading and process-to-ready times are
reported separately. The frozen 400 inputs and their order are preserved at each
concurrency setting. This is an engineering experiment on a seen constructed
benchmark; original BF16 quality results remain historical results.

Files are created exclusively. Do not repeat completed commands over an existing
output directory or edit a locked plan. Preserve partial attempts if a run fails.
For future reproduction into an unused version, the preparation utility is:

```bash
.venv/bin/python scripts/prepare_vllm_fp16_experiment.py \
  --source results/vllm_serving/jaisguard-v3 \
  --destination results/vllm_serving/jaisguard-v5
```

Pass the new `--root` to all subsequent commands. Preparation copies and verifies
existing exports and completed observations, recomputes acceptance under the new
explicit policy, and does not execute GPU inference. Source-version checks reject
unrelated changes rather than silently relabeling results.

## CPU verification

```bash
.venv/bin/python -m pytest tests/test_vllm_serving.py tests/test_latency_serving.py -q
.venv/bin/ruff check src/jais_guard/vllm_support \
  scripts/prepare_vllm_fp16_experiment.py tests/test_vllm_serving.py
```

Tests include acceptance despite BF16 differences, rejection of FP16 mismatches,
rejection of a wrong trained head, preservation of FP32 head weights during the
FP16 precision adaptation, concurrent HTTP checks and actual chart generation.
GPU LoRA equivalence and measured performance must come from WSL; no timings are
invented from CPU tests.
