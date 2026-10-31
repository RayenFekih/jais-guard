# Local inference and comprehensive latency comparison

> Historical experiment protocol. Completed local evidence supersedes pending-run
> language below. Source-locked replay requires the original revision and artifacts;
> see [reproducibility](reproducibility.md) and [public workflows](public-workflows.md).

This workflow reuses the completed Jev 1.14.0, untuned Jais, frozen v1/v2 and
LoRA v1/v2 classification results on the frozen 400 human-reviewed Arabic inputs.
It makes **no Jev API requests** and performs no training, threshold search,
label edits or historical-result writes.

The prepared results directory is `results/latency_comparison/jaisguard-all-v2`.
`manifest.json` pins the dataset, checkpoints, thresholds, cached predictions,
metrics and inference code. `quality.json` preserves per-seed metrics and
three-seed means/sample SD (`ddof=1`). `report-cpu/` contains the initial Markdown,
CSV, JSON, PNG and SVG comparisons; local latency is explicitly pending.

All 12 classifiers (three seeds × frozen/LoRA × v1/v2) and untuned Jais have model
keys. The suite benchmarks all 13 separately, one resident GPU model per process.
The consolidated quality table uses a representative latency checkpoint for each
architecture/version: minimum archived validation loss, ties by numeric seed.
Epochs remain the first minimum validation-loss epoch in each training history.
The API default is **v2 LoRA seed 44, epoch 4**, validation loss 0.1711179167,
locked threshold **0.942931056022644**. This selection uses no benchmark quality.

The saved original iteration-two config was absent; it was recovered from saved
checkpoint metadata with an **exact match to the old lock's SHA-256**. The old
protocol also pins an earlier data-loader source hash. The new manifest records
that source drift, pins the current source, and verifies all non-code artifacts.
Cached predictions reproduce the archived quality metrics under the current
metric function. Historical protocol files are preserved unchanged.

## Exact WSL commands

Run from the existing project environment with its cached pinned Hugging Face
model and the `finetune` extra (PEFT). No dependency upgrades are needed. Close
other GPU workloads before benchmarking. Do not run the standalone benchmark
while a demo server holds another model in VRAM.

Verify the prepared snapshot and inspect model keys:

```bash
cd /home/rayen/projects/fine-tuning
export HF_HUB_OFFLINE=1
uv run --no-sync python scripts/benchmark_latency.py verify
uv run --no-sync python scripts/benchmark_latency.py list
nvidia-smi
```

Run the batch-1 GPU benchmark for **all 13 local models**, sequentially. Each
command exits before the next model loads, releasing its GPU allocation:

```bash
cd /home/rayen/projects/fine-tuning
set -euo pipefail
export HF_HUB_OFFLINE=1
while IFS= read -r model_key; do
  uv run --no-sync python scripts/benchmark_latency.py gpu --model "$model_key"
done < <(uv run --no-sync python scripts/benchmark_latency.py list)
```

Run **all 13 HTTP benchmarks**. Stop any existing server on port 8000 first. This
helper starts one single-worker server, checks its model/manifest identity,
executes 400 sequential httpx requests after warmup, then stops that server before
starting the next. Server logs are preserved. No concurrency/load test is implied.

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/run_http_latency.py
```

Generate the final consolidated comparison with all available measurements:

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/benchmark_latency.py report --execution report-final
```

Open `results/latency_comparison/jaisguard-all-v2/report-final/comparison.md`.
The directory also contains `comparison.csv` (quality and representative latency),
`latency.csv` (every measured seed/scope, including model-only timing, loading and
VRAM), `metrics.json` (full quality and timing), and `quality.png/.svg` and
`latency.png/.svg`. PNGs are suitable for sharing; SVGs suit a technical README.
Full per-input observations are in `gpu/<model>/samples.jsonl` and
`http/<model>/samples.jsonl`.

Each output directory is created exclusively. A failed attempt remains for
inspection and has no completed `metrics.json`. For another complete attempt,
use new names, for example `--execution gpu-retry-1` on every GPU command and
`--execution http-retry-1` on `run_http_latency.py`, then:

```bash
uv run --no-sync python scripts/benchmark_latency.py report \
  --execution report-retry-1 \
  --gpu-execution gpu-retry-1 --http-execution http-retry-1
```

Do not edit the code after preparation: its hashes are locked. After an intentional
implementation change, prepare a **new** directory and pass `--directory` to all
commands (set `JAISGUARD_DIRECTORY` for a manually started server):

```bash
uv run --no-sync python scripts/benchmark_latency.py prepare \
  --directory results/latency_comparison/jaisguard-all-new-version
```

## Minimal API demo

Start the selected v2 LoRA model in terminal 1:

```bash
cd /home/rayen/projects/fine-tuning
export JAISGUARD_DIRECTORY=results/latency_comparison/jaisguard-all-v2
export JAISGUARD_MODEL=v2-lora-44
uv run --no-sync python scripts/serve_jaisguard.py
```

In terminal 2:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"كيف يمكنني تعلم اللغة العربية؟"}'
uv run --no-sync python scripts/benchmark_latency.py http --model v2-lora-44 \
  --execution http-demo
```

Change `JAISGUARD_MODEL` to `untuned`, `v2-frozen-43`, or any exact key printed by
`list`; restart the process to switch models. FastAPI lifespan loads once, then
reuses the resident model. A lock serializes GPU predictions. Use one worker;
multiple workers load multiple models and can exhaust the 6 GB card.

`POST /predict` accepts `{"text": "..."}` and returns `class_id`, `class`,
`probabilities`, `threshold`, `inference_ms`, `model_ms` and `model_key`.
Classifiers return probabilities for VALID=0 and GIBBERISH=1; decisions use
`p_gibberish >= saved_threshold`. Untuned generation uses its **original saved
recoverable_v1 prompt**, reasoning/decoding settings and strict parser; its
probabilities/threshold are `null`, and unparseable output remains INVALID=-1.
Its generated text is included as `raw_output`. Empty, unsupported-script or
over-budget inputs return HTTP 422; no silent truncation or label substitution.
`GET /health` reports readiness, model identity, loading time and hardware.

## Timing and interpretation

- All local measurements use the exact frozen 400 inputs, original order, batch
  size 1, three unscored warmup inputs, cached pinned weights and CUDA
  synchronization. Loading is measured separately before warmup.
- `model_ms` measures the classifier forward only, or untuned `model.generate`
  including its decoding loop/CPU scheduling. These are not kernel-only times.
- `inference_ms` measures the complete local prediction: original prompt or input
  tokenization, transfers, forward/generation, probability calculation or parsing,
  and CPU output. Timers exclude lock wait. Instrumentation adds synchronization.
- HTTP round-trip time includes local JSON/loopback/server scheduling, lock wait
  and inference. It uses sequential requests, a persistent httpx client, no
  retries, and raises on failure rather than dropping failed samples. Server
  inference durations are saved separately from HTTP durations.
- Mean, p50, p95, p99 use NumPy linear percentiles. GPU throughput is 400 divided
  by summed prediction time; full loop throughput is also saved. HTTP throughput
  uses full measured request-loop wall time. Neither is a concurrency capacity test.
- CUDA memory records peak allocated and reserved bytes separately during loading
  and post-warmup inference, including resident weights. Reserved memory can
  retain allocations from loading/warmup; it is not incremental request memory.
  This allocator accounting excludes other processes/driver overhead and is not
  a measurement of all board VRAM. Hardware, dtype, CUDA/PyTorch versions and
  total device memory are saved. Cache, temperature and power state affect timings.
- Jev's existing remote API latency is mean **217.52 ms**, p50 **203.41 ms**, p95
  **321.96 ms**, p99 **358.48 ms**. It includes remote network overhead; Jais is
  served locally. Jev had no warmup and includes initial connection overhead.
  No remote-versus-local hardware speedup or equal-deployment claim is justified.
- Default and validation-locked operating points retain separate quality rows.
  They share classifier latency because threshold comparison adds negligible work.
  Quality summaries use all three seeds; representative latency is labeled
  separately, and every seed's measured latency remains available.
- INVALID stays wrong for accuracy and true-class recall/F1. VALID false rejection
  counts explicit GIBBERISH; acceptance and invalid counts expose other failures.
  This is a previously examined, constructed benchmark, not production traffic or
  a new independent generalization estimate. No winner is selected from test data.

## CPU verification

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python -m pytest tests/test_latency_serving.py -q
uv run --no-sync ruff check src/jais_guard/serving.py \
  src/jais_guard/latency_benchmark.py scripts/serve_jaisguard.py \
  scripts/benchmark_latency.py scripts/run_http_latency.py tests/test_latency_serving.py
```

CPU tests cover validation-only selection, ties, threshold boundaries, reuse of a
single loaded predictor, API startup/health/validation/INVALID handling, artifact
tampering, CUDA refusal, warmup versus measured observations, model-only timing,
sequential HTTP requests, server identity, failures and exclusive output creation.
They use fake models/transports and do not perform CPU Jais inference.

Codex verification: all **10 new CPU tests passed**, Ruff passed, and real-artifact
preparation/verification succeeded for all 13 local models and 400 inputs. A
broader related-test run produced **72 passes, 1 failure and 3 setup errors**.
Those four issues are in the existing version-comparison tests: the unchanged
`finetuning/data.py` loader hard-codes v1 class counts and rejects their v2
fixtures. The new snapshot workflow does not use that training-data loader.
GPU and live-server measurements remain for WSL execution.
