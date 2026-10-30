# JaisGuard classification experiments

> Historical experiment protocol. Completed local evidence supersedes pending-run
> language below. Source-locked replay requires the original revision and artifacts;
> see [reproducibility](reproducibility.md) and [public workflows](public-workflows.md).

Implementation is ready for local smoke checks. **No full frozen-head, LoRA, or
200-example prompt-baseline experiment has been launched.** GPU checks belong in
your WSL terminal: Codex's CUDA visibility does not describe your local GPU.

## Setup and read-only inspection

From `/home/rayen/projects/fine-tuning`:

```bash
uv sync --locked --extra finetune --extra annotation
uv run --no-sync python scripts/finetune.py inspect
```

`annotation` preserves the existing Streamlit tooling; omit it if unneeded. Use
`--no-sync` in the following commands to keep this selected environment intact.
MLflow 3.17.2 and PEFT 0.21.2 are locked. Torch 2.14.1 and Transformers 5.20.2 are
unchanged. MLflow needs Pydantic 2, which conflicts with main-environment legacy
vLLM 0.2.5; that dependency is now isolated in the mutually exclusive
`vllm-legacy` extra. The separate `.venv-vllm` environment and scripts are untouched.
The solver selected PyArrow 25.0.1, compatible with datasets and MLflow.

Expected inspection output includes:

```text
train: 700 rows, 359 VALID / 341 GIBBERISH, max 60 tokens
validation: 200 rows, 99 VALID / 101 GIBBERISH, max 29 tokens
PASS: hashes, labels, partitions, scripts, token budgets, and separation
```

The actual first two output lines are JSON. Training retains its 567 human labels
and 133 audited provisional labels. Validation retains all 200 human labels.
The pipeline uses final `label`, never the proposal in `intended_label`.
No annotation database or religious benchmark is opened by these commands.

All model loading uses the pinned, already-cached revision
`e51941b37091287fee4934e3f8a799e0d3fef37f`, offline. A cache miss is an error;
this pipeline does not download a different model or silently use CPU for training.
No driver or Torch reinstall is part of setup.

## GPU smoke checks — run these now in WSL

Close other GPU-heavy applications first. Each smoke command is a separate
process; no MLflow server is needed for these checks.

```bash
nvidia-smi
uv run --no-sync python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0)); print("BF16:", torch.cuda.is_bf16_supported())'
uv run --no-sync python scripts/finetune.py smoke --approach frozen
uv run --no-sync python scripts/finetune.py smoke --approach lora --microbatch 2
```

Expected device output is `CUDA: True`, your RTX 4050 name, and normally
`BF16: True`. Each successful smoke check ends approximately as follows; measured
memory and losses will vary:

```text
PASS: frozen, microbatch=2, seq=64, trainable=3,074, free=<measured> GiB
PASS: finite gradients, frozen weights, optimizer updates, padding, checkpoint reload
.../results/fine_tuning/smoke/frozen-<unique-id>
```

```text
PASS: lora, microbatch=2, seq=64, trainable=1,264,642, free=<measured> GiB
PASS: finite gradients, frozen weights, optimizer updates, padding, checkpoint reload
.../results/fine_tuning/smoke/lora-<unique-id>
```

The script loads the backbone without the language-model head, uses SDPA and BF16
(or FP16 plus scaling), pads to the worst-case configured length of 64, and runs
**two disposable optimizer updates on training examples only**. It verifies finite
head/adapter gradients, frozen parameter versions, actual trainable updates,
padding invariance, and checkpoint serialization/reload. LoRA uses non-reentrant
gradient checkpointing. Disposable weights are deleted; the report is saved to
`smoke.json` in a unique directory. There is no validation-label update, epoch loop,
or call into the full training runner.

`smoke.json` includes hardware/software details, allocated/reserved peaks, free
VRAM, precision, counts, checks and dataset hashes. A pass requires at least 1 GiB
free at the measured optimizer/forward stage. It is a feasibility gate, not a
throughput benchmark or a guarantee against every external GPU-memory change.

If batch 2 runs out of memory or reports `INSUFFICIENT_HEADROOM`:

```bash
uv run --no-sync python scripts/finetune.py smoke --approach lora --microbatch 1
```

The same override is supported for frozen smoke and its later feature extraction. A batch-1 LoRA full run later uses
accumulation 16 instead of 8, keeping effective batch size 16. If batch 1 also
fails, stop and inspect the measured failure; do not launch full runs or change
drivers/Torch. Missing CUDA here means investigate the terminal/environment
selection first. A missing pinned cache file needs that exact cache, not a model
revision change. Send the two smoke reports back before approving full experiments.

## CPU tests and tracking

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q
uv run --no-sync ruff check src/jais_guard/finetuning scripts/finetune.py tests/test_finetuning.py
```

New tests train only tiny, randomly initialized K2 Horizon models on synthetic
fixture rows in pytest temporary directories. They do not train the pretrained
Jais model or run full experiments. They cover both approaches, frozen parameters,
LoRA gradient flow, padding, save/reload, default/tuned metrics, malformed baseline
outputs, dataset integrity, and local SQLite MLflow artifacts and pyfunc reload.
Tests use a temporary Hugging Face code cache because Codex's home cache is read-only.

Start tracking in a second WSL terminal only when needed:

```bash
mkdir -p results/fine_tuning
MLFLOW_ENABLE_TELEMETRY=false uv run --no-sync mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////home/rayen/projects/fine-tuning/results/fine_tuning/mlflow.db \
  --serve-artifacts \
  --artifacts-destination /home/rayen/projects/fine-tuning/results/fine_tuning/artifacts
```

Open http://127.0.0.1:5000. The default config permits only localhost HTTP tracking
or local SQLite URIs. There are no cloud credentials, LLM judges or paid APIs.
Standard run tracking is required. Optional `baseline --genai` scores cached
predictions with deterministic custom MLflow GenAI scorers; its failure is logged
without discarding standard results or repeating inference. This optional path
is not required for the experiment comparison.

## Experiment behavior and artifacts

`configs/finetuning.json` is the source of truth. Both classifiers pool the final
hidden state at the last non-padding token, then apply a FP32 linear 1536→2 head.
Raw Arabic-script inputs have no chat template. Right padding, attention masks,
max length 64, no truncation, no KV cache, no vocabulary projection. Unsupported
scripts raise an error rather than becoming GIBBERISH. Label IDs: VALID=0,
GIBBERISH=1. The default decision rule is `p_gibberish >= 0.5`.

| Setting | Frozen + head | LoRA + head |
|---|---|---|
| Seeds | 42, 43, 44 | 42, 43, 44 |
| Trainable | Head only | Head and q/v adapters in all 28 layers |
| Learning rates | Head 1e-3 | Head 1e-3, adapter 1e-4 |
| LoRA | None | Rank 8, alpha 16, dropout .05, no bias |
| Batch | Cached CPU FP32 features, batch 32 | Microbatch 2 × accumulation 8 |
| Max epochs / patience | 30 / 5 | 8 / 2 |
| Selection | Lowest validation cross-entropy | Lowest validation cross-entropy |

AdamW weight decay .01 except biases, linear decay with 10% warmup, clipping at 1.
Frozen backbone is always in eval mode; its features are extracted in batches of
2 and cached with dataset/model/tokenizer/precision/pooling identity. Its training
is head-only. LoRA's backbone is frozen except adapters. FP32 head and adapters
retain optimizer state; the backbone uses BF16 where supported. Seeded RNG,
CUBLAS configuration, deterministic-algorithm warnings, versions and Git status
are recorded; bitwise equivalence across devices/versions is not promised.

Each invocation creates a unique parent run; a training child represents one
approach and seed. Dataset versions/manifests/hashes and inputs are linked,
including the provisional-label origins. Configs, environment, step loss/LR,
epoch validation curves and metrics, peak VRAM and timing are logged explicitly.
Each child directory contains:

- `best/` and `last/`: head safetensors, LoRA adapter if relevant, tokenizer,
  classifier metadata and optimizer/scheduler/scaler/RNG state. No duplicated base weights.
- `history.json`, `threshold.json`, and `logged_model.json` with the MLflow model URI.
- `evaluation/`: metrics/subgroups JSON, predictions and errors JSONL/CSV,
  confusion PNGs, complete `thresholds.csv`, threshold trade-off PNG and `report.md`.

MLflow models bundle project code, tokenizer, head and adapter, and require the
exact base revision in the local Hugging Face cache. They are local inference
packages, not self-contained copies of the 1B backbone. The checkpoint loader and
MLflow pyfunc expose default probabilities/predictions; pyfunc additionally exposes
the exploratory prediction. Training state is saved for inspection/recovery;
there is no automatic resume CLI in this first implementation.

Every selected checkpoint reports accuracy, macro F1, per-class precision/recall/F1,
gibberish recall, VALID false-rejection counts/rate, valid acceptance, and confusion
counts. FRR includes a Wilson 95% interval; the 99 VALID examples provide limited
precision. Error tables retain text, logits, scores, provenance and metadata.
Subgroups show support and default/exploratory results by subtype, dialect, source,
label origin and length; proposed dialect metadata is not promoted to human truth.

Threshold selection happens after checkpoint selection. Save every score breakpoint
plus all-reject, accept-all and .5 operating points. The exploratory point maximizes
gibberish recall at observed FRR ≤.02 (at most 1/99), breaking ties by fewer false
rejections and then higher threshold. **This is not a production requirement.**
Always retain the .5 metrics and inspect the entire trade-off. All results are
validation-selected; there is no independent general-purpose test estimate.

The current baseline uses the versioned `recoverable_v1` prompt, low effort, greedy
FP16 CUDA generation, 64 output tokens and a 1024-token budget. See
[evaluation phase](evaluation-phase.md) for its prompt version and parsing policy.
Historical `explicit_user` prompting remains available to the earlier diagnostic scripts. Prompt and raw
responses, parsing failures and latency are preserved in the new run. Invalid
responses reduce accuracy and true-class recall; confusion matrices include an
INVALID column. Explicit false rejection excludes invalids, while valid acceptance
penalizes them. No threshold tuning applies to this categorical baseline.

## Later execution — do not run until full experiments are approved

After smoke reports pass, run one process at a time with the local MLflow server
running. For each seed 42, 43 and 44, run these explicitly (replace the seed):

```bash
uv run --no-sync python scripts/finetune.py train --approach frozen --seed 42
uv run --no-sync python scripts/finetune.py train --approach lora --seed 42
```

Complete the three frozen runs first, then the three LoRA runs. If only LoRA
microbatch 1 passed smoke, add `--microbatch 1` to its training commands. The runner
never automatically launches other seeds, another approach or baseline evaluation.
Then run the prompt baseline once, separately:

```bash
uv run --no-sync python scripts/finetune.py baseline
```

Compare selected run directories (training child folders and baseline parent folder):

```bash
uv run --no-sync python scripts/finetune.py compare <frozen-42-dir> <frozen-43-dir> <frozen-44-dir> <lora-42-dir> <lora-43-dir> <lora-44-dir> <baseline-dir>
```

This produces a new comparison folder with per-seed metrics and mean/std, and
flags missing three-seed coverage. It does not select a winning seed or rerun
inference. All writes are under new fine-tuning run/cache directories. Existing
annotations, exports, religious benchmark and historical results remain untouched.
