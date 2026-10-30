# Real vLLM serving for the trained JaisGuard classifiers

> Historical experiment protocol. Completed local evidence supersedes pending-run
> language below. Source-locked replay requires the original revision and artifacts;
> see [reproducibility](reproducibility.md) and [public workflows](public-workflows.md).

This is a separate implementation using the **existing `.venv-vllm`, vLLM
0.32.0**, without installing or upgrading anything. The original `.venv`, model
checkpoints, human labels, quality results, and Hugging Face/FastAPI code remain
unchanged. No Jev requests are made.

The versioned directory is `results/vllm_serving/jaisguard-v2`. Its CPU exports
and structure checks are prepared. **GPU prediction parity is not yet verified;
these exports are not claimed to be successful conversions.** The server and
benchmarks refuse to run until the corresponding parity gate passes.

## What actually executes

The standalone implementation calls `vllm.LLM.classify()`. The asynchronous
server exposes a minimal, threshold-aware `POST /classify` using
`AsyncLLM.encode(..., PoolingParams(task="classify", use_activation=True))`.
The vLLM Transformers backend replaces backbone attention/linear execution with
vLLM layers, then uses its `LAST` pooler and the actual trained classification
head. It does **not** run the project's Hugging Face classifier forward function
inside a wrapper. This follows the installed pooling/classification APIs; see
[vLLM's pooling documentation](https://docs.vllm.ai/en/stable/models/pooling_models/).

The minimal endpoint supports one string input per request, not the full OpenAI
classification protocol. Native vLLM classification responses normally label by
argmax; our endpoint instead applies the **exact saved validation threshold** and
also exposes `default_class_id`. It still obtains probabilities from the real
vLLM classification engine.

The benchmark's `hf` server reuses `jais_guard.serving.create_app` and its existing
`Predictor`, with benchmark diagnostics added on separate endpoints. Its existing
serialization lock is preserved. This measures the current server, rather than
silently replacing it with a different Hugging Face implementation.

## Selection and weight preservation

| Checkpoint | Epoch | Threshold | Selection |
|---|---:|---:|---|
| `v2-frozen-43` | 5 | 0.9563547372817993 | Previously locked minimum validation loss |
| `v2-lora-44` | 4 | 0.942931056022644 | Previously locked minimum validation loss |

The choices are inherited from the existing validation-only serving manifest.
No benchmark accuracy, seed search, new threshold search, or retraining occurs.

Exports stream the pinned base safetensors on CPU into small shards. Frozen base
weights retain their original stored dtype. LoRA uses the saved adapter matrices
and standard PEFT formula `W + (alpha / rank) * B @ A`, evaluated in FP32 one
projection at a time. Unsupported variants (DoRA, RS-LoRA, rank/alpha patterns,
extra trained modules, adapter biases, transposed layout) are rejected. The CPU
unit test independently compares every merged projection with PEFT's
`merge_and_unload(safe_merge=True)`.

The real LoRA export merges all **56 q/v projections**. This static merge is the
least-invasive supported deployment path: native runtime-LoRA classification
compatibility is **not assumed or claimed**. Original adapters remain unchanged.

The exporter copies the actual saved `head.safetensors` weight **and bias** into
`classifier.weight` and `classifier.bias`, without quantizing either. This naming
is deliberate: the installed vLLM 0.32.0 Transformers `LegacyMixin` drops
`score.bias`. Generic conversion could otherwise silently lose JaisGuard's bias.
The exported architecture explicitly defines the biased FP32 head. Its small
head wrapper accommodates vLLM's singleton sequence dimension and returns two
FP32 logits. Softmax and label order remain VALID=0 / GIBBERISH=1.

At engine startup, a supported `worker_extension_cls` inspects the **actual GPU
worker**, verifies the original head tensor hashes and FP32 dtype, and confirms
that vLLM modules are present. Failure stops startup. Creating a meta model to
discover the architecture does not authorize an uninitialized head for serving.
Export/config/shard hashes, base revision, original model hashes and source code
are recorded in immutable receipts.

`LAST` uses the last actual token of the unpadded input. The original tokenizer
is copied, and original/exported token IDs must match for every parity input.
The text is not wrapped in a generative prompt; there is no truncation, label
substitution or replacement with a yes/no language-model head.

## Conservative RTX 4050 settings

The plan pins these settings, based on the previously successful configuration:

```python
model_impl = "transformers"
runner = "pooling"
convert = "classify"
trust_remote_code = True
dtype = "float16"
enforce_eager = True
max_model_len = 1024
gpu_memory_utilization = 0.75
max_num_seqs = 16
max_num_batched_tokens = 1024
enable_prefix_caching = False
enable_chunked_prefill = False
```

Pooling is explicitly `LAST`, with softmax activation and FP32 head computation.
The classifier's original **64-token input policy** is enforced before submission;
1024 is the engine capacity, not permission to change that policy. At HTTP
concurrency 16, vLLM may continuously batch up to 16 single-input requests. The
Hugging Face app keeps its original serialized GPU forward behavior.

One process/model is loaded at a time. Stop other GPU servers and workloads before
running these commands. The HTTP suite uses port **8001** and refuses an occupied
port. It starts and shuts down one backend before starting the next. vLLM uses
spawn workers and a writable cache inside the versioned results directory; no
new Python environment is created.

## Exact WSL commands: parity first

The prepared exports are already present. Inspect the installed interface receipt
and model keys:

```bash
cd /home/rayen/projects/fine-tuning
.venv/bin/python scripts/vllm_classifier.py inspect
.venv/bin/python scripts/vllm_classifier.py list
nvidia-smi
```

Run three **separate GPU processes** per selected checkpoint, in order. The
original reference uses the same BF16 policy as the existing implementation on
the RTX 4050. The second reference uses FP16 to help distinguish precision drift
from export/backend drift. The third uses actual vLLM FP16 classification.

```bash
cd /home/rayen/projects/fine-tuning
set -euo pipefail
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NO_USAGE_STATS=1

while IFS= read -r model_key; do
  .venv/bin/python scripts/vllm_classifier.py collect \
    --model "$model_key" --backend hf-original
  .venv/bin/python scripts/vllm_classifier.py collect \
    --model "$model_key" --backend hf-fp16
  .venv-vllm/bin/python scripts/vllm_classifier.py collect \
    --model "$model_key" --backend vllm
  .venv/bin/python scripts/vllm_classifier.py verify-parity --model "$model_key"
done < <(.venv/bin/python scripts/vllm_classifier.py list)
```

Each process exits before another model is loaded. Offline vLLM workers are
explicitly shut down after collection. The parity set contains **658 inputs**:
250 frozen validation inputs, the same 400 benchmark inputs, and eight fixed
coverage probes for clean Arabic, dialects, recoverable corruption and gibberish.
Probes are coverage categories, not new human labels or quality measurements.

The policy is frozen before GPU execution:

- Maximum absolute probability difference ≤ **0.02** across both class probabilities.
- Mean absolute probability difference ≤ **0.005**.
- **Zero classification mismatches**, both at 0.5 and at the saved threshold.
- Matching input IDs/order/token IDs, label ordering and head hashes.

The gate compares original HF versus FP16 HF, FP16 HF versus vLLM, and original
HF versus vLLM. It records maximum/mean differences, mismatch counts/IDs and
category-specific summaries. Both probability tolerances and exact decisions
must pass; a small numerical difference is insufficient if it crosses a
threshold. No tolerance is automatically relaxed after observing results.

If parity fails, inspect `parity/<model>/gate.json` and the three prediction files.
Performance stays blocked. To produce a report of that failure:

```bash
.venv/bin/python scripts/vllm_classifier.py report --report-name report-parity-failure
```

## Exact WSL commands: standalone and HTTP benchmarks

Only run after both parity gates pass. Measure batch-1 standalone inference for
both engines, including tokenization/transfers and excluding HTTP:

```bash
cd /home/rayen/projects/fine-tuning
set -euo pipefail

while IFS= read -r model_key; do
  .venv/bin/python scripts/vllm_classifier.py offline \
    --model "$model_key" --backend hf-original
  .venv-vllm/bin/python scripts/vllm_classifier.py offline \
    --model "$model_key" --backend vllm
done < <(.venv/bin/python scripts/vllm_classifier.py list)
```

Run the matched HTTP suite at concurrency **1, 4, 8 and 16**, for frozen and LoRA,
with both Hugging Face and vLLM:

```bash
.venv/bin/python scripts/vllm_classifier.py http-suite
.venv/bin/python scripts/vllm_classifier.py report
```

The final report is:

```text
results/vllm_serving/jaisguard-v2/report/comparison.md
```

The HTTP suite sends exactly 400 inputs per backend/checkpoint/concurrency setting,
with one text per request and three warmup inputs. A persistent httpx client uses
closed-loop worker counts of 1/4/8/16. It records request latency separately from
the server-reported inference duration. Failed requests remain recorded and prevent
a successful metrics file; there are no retries or silently dropped samples.

Each concurrent response is checked again against the original HF parity outputs.
If dynamic batching changes a locked or default decision, that workload's
performance result is withheld. Standalone measurements receive the same check.

The report includes mean, linear p50/p95/p99, requests/sec, model loading and
process-to-ready times, and CUDA peak allocated/reserved memory. Worker allocator
peaks include resident weights; they exclude driver/other-process memory and are
not total board VRAM. Reserved memory may retain startup/warmup allocations.
Standalone and HTTP scopes stay separate. Loading/ready time is never included in
steady-state inference latency. Cached historical quality is not recomputed or
replaced by these parity/timing runs.

## Manual server demo

After parity passes, start the frozen model:

```bash
cd /home/rayen/projects/fine-tuning
.venv-vllm/bin/python scripts/serve_vllm_classifier.py \
  --model v2-frozen-43 --backend vllm
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/classify \
  -H 'Content-Type: application/json' \
  -d '{"input":"شنوة نعمل باش نتعلم البرمجة"}'
```

The response includes `class_id`, `class`, both class probabilities,
`default_class_id`, the locked threshold, model identity and inference duration.
Empty, unsupported-script and over-budget text receives HTTP 422. The model is
loaded at startup and reused asynchronously across requests.

Stop that server before starting LoRA:

```bash
.venv-vllm/bin/python scripts/serve_vllm_classifier.py \
  --model v2-lora-44 --backend vllm
```

For the original Hugging Face app on the same checkpoint, use `.venv` and
`--backend hf`; its original request route is `/predict` with `{"text": "..."}`.
`/benchmark/info`, `/benchmark/reset` and `/benchmark/memory` are local diagnostic
endpoints used by the suite. They do not perform classification in place of vLLM.

## Artifacts, reproducibility and retries

```text
plan.json                            source/version/selection/engine/tolerance locks
exports/<model>/                     full standalone sharded classifier artifact
  export.json                        exact trained head hashes and merge provenance
structure-checks/<model>.json         CPU/meta and tokenizer checks, no parity claim
parity/<model>/<backend>/             observations and completion receipts
parity/<model>/gate.json              probability and prediction parity gate
offline/<model>/<backend>/           standalone batch-1 observations and metrics
http/<model>/<backend>/c1,c4,c8,c16/   request samples, response parity and metrics
http-server-logs/                     server output and process-to-ready measurements
report-pending/                      CPU-only report, GPU measurements pending
report/                              final Markdown, CSV and JSON comparison
```

Every attempt uses exclusive output creation. Do not overwrite partial attempts,
edit an export, or change code after preparation. Hash checks will reject changed
inputs. For an intentional change or a new attempt, prepare a new directory, then
pass that same `--root` to **all** subsequent commands:

```bash
cd /home/rayen/projects/fine-tuning
new_results=results/vllm_serving/jaisguard-v3
.venv/bin/python scripts/vllm_classifier.py prepare --root "$new_results"

while IFS= read -r model_key; do
  .venv/bin/python scripts/vllm_classifier.py export \
    --root "$new_results" --model "$model_key"
  .venv-vllm/bin/python scripts/vllm_classifier.py check-export \
    --root "$new_results" --model "$model_key"
done < <(.venv/bin/python scripts/vllm_classifier.py list --root "$new_results")
```

Exports use CPU streaming and do not need CUDA. `check-export` performs meta-device
architecture discovery under the installed Transformers version and verifies
identical original/exported tokenization for all 658 inputs. It does not execute
model inference or establish GPU parity. Original checkpoints are never modified.

## CPU validation performed in Codex

```bash
.venv/bin/python -m pytest tests/test_vllm_serving.py tests/test_latency_serving.py -q
.venv/bin/ruff check src/jais_guard/vllm_support scripts/vllm_classifier.py \
  scripts/serve_vllm_classifier.py tests/test_vllm_serving.py
```

Tests cover trained head bias/FP32 math, left/right nonpadding selection, vLLM head
reshape, independent PEFT merge equivalence, source preservation, probability
ordering, threshold boundaries, worker audits, fail-closed parity, actual offline
and async engine API contracts with fake engines, startup reuse, HTTP concurrency
1/4/8/16 over the same 400 observations, and reports with failed/pending parity.
No GPU inference or speed claim is made from CPU tests.

Codex results: **25 CPU tests passed**, Ruff passed, both real exports completed,
and both structure checks succeeded under `.venv-vllm` Transformers 5.19.0 with
658 identical tokenized inputs (maximum 55 tokens). The real LoRA artifact has 56
merged projections. See `cpu-verification.json` and `structure-checks/` for the
receipts. The pinned upstream K2 model emitted `cache_position` docstring messages
during meta discovery; both checks completed successfully without editing that
source. GPU parity and performance remain pending.

## Recovery from the TorchVersion IPC failure

The first WSL run loaded the vLLM model successfully, then its diagnostics RPC
failed because PyTorch exposes `torch.__version__` as a `TorchVersion` subclass.
vLLM's strict Msgpack encoder rejects that object. Worker diagnostics now return
`str(torch.__version__)`; no insecure serialization flag or dependency change is
needed. The installed vLLM encoder was CPU-tested: the old value reproduces the
failure and the plain string succeeds.

`results/vllm_serving/jaisguard-v3` is the recovery attempt. It contains byte-identical
copies of the exports and the completed frozen-model `hf-original` and `hf-fp16`
references, with verified provenance. The failed vLLM attempt and the entire v2
directory remain preserved. Source validation permits only this exact one-line
runtime metadata change; it rejects unrelated changes. No HF GPU inference was
repeated during recovery. The recovery utility is
`scripts/repair_vllm_serialization.py`; do not rerun it over an existing destination.

Stop the hung run with Ctrl+C, then collect the missing frozen vLLM observations:

```bash
cd /home/rayen/projects/fine-tuning
.venv-vllm/bin/python scripts/vllm_classifier.py collect \
  --root results/vllm_serving/jaisguard-v3 \
  --model v2-frozen-43 --backend vllm
```

Use `--root results/vllm_serving/jaisguard-v3` for subsequent commands. Do not rerun
the original loop against v2: its source hash intentionally describes the old,
failed implementation.

There is also a separate, already measured precision issue: the completed frozen
HF FP16 reference differs from original BF16 by maximum absolute probability
0.05112457 and mean 0.00422282, with **one locked-threshold and three default
classification mismatches** on 658 inputs. Thus the existing FP16 parity policy
cannot pass even after startup is fixed. The missing vLLM collection is diagnostic;
do not run performance benchmarks until prediction parity is resolved. Thresholds,
labels and tolerances have not been changed. This is precision/backend validation,
not a new quality result or model-selection step.

Recovery validation: **27 CPU tests passed**, including strict primitive worker
metadata and preservation of exports/completed HF references during migration.
