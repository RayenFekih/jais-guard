# Final independent test evaluation

> Historical experiment protocol. Completed local evidence supersedes pending-run
> language below. Source-locked replay requires the original revision and artifacts;
> see [reproducibility](reproducibility.md) and [public workflows](public-workflows.md).

The dedicated runner evaluates the six preselected hidden-state classifiers on the
frozen `test-400-v1` export: 400 human-reviewed examples, 188 VALID / 212 GIBBERISH.
This is a **human-reviewed, constructed Arabic-input benchmark**, not observed
production traffic. Its authoring source, designed class balance, simulated errors,
length imbalance and unverified dialect hints limit generalization.

## WSL commands

Use the existing uv environment. If the existing local MLflow server is not already
running, start it in a separate WSL terminal:

```bash
cd /home/rayen/projects/fine-tuning
MLFLOW_ENABLE_TELEMETRY=false uv run --no-sync mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////home/rayen/projects/fine-tuning/results/fine_tuning/mlflow.db \
  --serve-artifacts \
  --artifacts-destination /home/rayen/projects/fine-tuning/results/fine_tuning/artifacts
```

The UI is http://127.0.0.1:5000. Run the complete classifier benchmark in another
WSL terminal:

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/evaluate_test.py \
  --test data/exports/test-test-400-v1 \
  --tracking-uri http://127.0.0.1:5000 \
  --batch-size 2 --warmup-batches 3
```

The command evaluates frozen seeds 42/43/44, then LoRA seeds 42/43/44, sequentially.
It prints each completed classifier and the final comparison report path. It does
not retrain, optimize thresholds, rank/select checkpoints, or run the prompt baseline.
Use the baseline-only command below to add the original untuned prompt baseline
without rerunning these classifiers. Its validation experiment remains unchanged.
Run no other GPU model process concurrently.

For CPU-only verification without inference or MLflow writes:

```bash
uv run --no-sync python scripts/evaluate_test.py --check-only
```

This checks all frozen data and locked model hashes, resolves exactly six mappings,
validates the saved thresholds and tokenizes all inputs with the saved tokenizers.
No truncation is allowed: the real preflight found a maximum of **55 tokens** under
the locked **64-token** budget. Normal execution has a mandatory CUDA gate; there
is no CPU inference fallback. All six finished training parent/child MLflow run
identities must resolve uniquely before a model is loaded. Failure stops the run.

## Locked operating points

Each classifier is scored at 0.5 and at the exact value in its hash-locked
`threshold.json`. These thresholds were selected on validation, not test.

| Approach | Seed | Validation-selected threshold |
|---|---:|---:|
| frozen | 42 | 0.7881332635879517 |
| frozen | 43 | 0.9268962740898132 |
| frozen | 44 | 0.9815510511398315 |
| lora | 42 | 0.872642457485199 |
| lora | 43 | 0.8631153106689453 |
| lora | 44 | 0.6232026219367981 |

Predictions use `p_gibberish >= threshold`. The validation FRR constraint remains
exploratory; its observed test performance is reported without changing thresholds.
The runner uses the shared metric function directly. It deliberately does **not**
call `save_report` or `threshold_report`, which select thresholds on their input data.
No test sweep, checkpoint selection, seed selection or hyperparameter optimization
is supported. Inference receives only IDs and input text; human labels are used
only after scoring to compute descriptive metrics and errors.

## Outputs and tracking

Every invocation creates a new directory, never overwriting earlier outputs:

```text
results/final_test/<UTC timestamp>-<unique suffix>/
  comparison.md                 consolidated per-seed and mean/std results
  metrics.json                  all operating-point and subgroup metrics
  error_analysis.json           errors across classifiers, VALID first
  predictions.json              all examples with six sets of predictions
  preflight.json                exact checkpoint/threshold/token mapping
  source_runs.json               original MLflow run identities, read-only
  frozen_protocol.json           copy of the frozen selection protocol
  environment.json              hardware, versions and execution settings
  preservation.json             verified original artifact hashes
  completion.json               written only after all six complete
  frozen-42/ ... lora-44/
    metrics.json
    predictions.jsonl / predictions.csv
    errors.jsonl / errors.csv
    confusion_default.png
    confusion_validation_locked.png
    timing.json
    locked_checkpoint.json
```

If execution fails, a directory may contain partial results. Do not treat it as a
completed comparison without `completion.json` and all six classifier runs. Fix an
environmental failure and rerun into a new output directory; do not select settings
from partial test accuracy. Do not use these results for further tuning.

Metrics include accuracy, macro F1, per-class precision/recall/F1/support, gibberish
recall, VALID false-rejection and acceptance rates, confusion matrices and the
existing descriptive VALID-FRR Wilson interval. Confusion rows are human
VALID/GIBBERISH; columns are predicted VALID/GIBBERISH/INVALID. Trained classifiers
always produce binary predictions; the INVALID column is zero. Errors and subgroups
include proposed subtype/dialect and input length, with the metadata caveats above.
Across-seed summaries use sample standard deviation (`ddof=1`) over the same three
seeds and test records, not independent sample confidence intervals or model ranking.

The existing `JaisGuard/general-arabic-v1` experiment receives **seven new runs**:
one comparison parent plus six nested classifier evaluations. Tags include
`evaluation_stage=independent_test`, `constructed_benchmark=true`,
`dataset_version=test-400-v1`, and `threshold_tuning=false`. The test dataset is logged
with context `testing`. Original training run IDs are recorded as provenance;
existing runs are never resumed or edited. Predictions, errors, confusion images,
metrics, timing, environment and protocol are logged as standard MLflow artifacts.
No judges or GenAI infrastructure are required.

## Timing and GPU memory

Only one model is loaded at a time, then released with garbage collection and CUDA
cache cleanup before the next. Weights are inference-only; no optimizer is created.
Loading is measured separately, with CUDA synchronization. Three warmup batches
are excluded. Steady-state timing measures synchronized batches end to end,
including tokenizer work, padding, host/device transfers, forward pass, softmax and
returning CPU scores. It excludes loading, warmup, report generation and MLflow.
It is **not kernel-only latency**. Per-example timings are amortized batch times,
not independently measured single-request latency. Use batch size 1 for request
latency measurements; the default batch size 2 matches the existing GPU smoke setup.

Mean/p50/p95 batch latency, throughput, amortized latency, load time and peak
allocated GPU memory are saved. Reports also record GPU name, total VRAM, compute
capability, PyTorch/CUDA/package versions, model dtype, seed, SDPA, token budget,
padding, TF32 settings and warmup/batch sizes. Loading time can benefit from OS/model
caches; model order is fixed rather than randomized. These timings describe the
recorded execution environment, not a production-service SLA.

## Preservation and verification

The frozen reader validates test/manifest checksums and all selected checkpoint,
threshold and prompt hashes before inference. Registry validation additionally
checks identity, label mapping, pooling, training/validation hashes and every
saved tokenizer/adapter file consumed by the loader. Missing, duplicate, mismatched
or ambiguous mappings fail closed. Checks run again after inference, including the
original freeze-receipt hash. Only new output directories and new MLflow records
are written; no dataset, annotation, historical candidate, weight or protocol lock
is changed. Freeze-time lexical and parent-family independence checks remain in
effect; they cannot establish absence of all semantic paraphrases or pretraining
contamination.

Focused CPU checks:

```bash
uv run --no-sync python -m pytest tests/test_final_test.py -q
uv run --no-sync ruff check --target-version py311 \
  scripts/evaluate_test.py src/jais_guard/finetuning/final_test.py tests/test_final_test.py
```

Tests use synthetic fixture rows, fake scores/tiny CPU objects and a temporary local
MLflow store. They never evaluate the real test set. Codex's execution environment
reports CUDA unavailable; real final-test scores and comparison artifacts must be
produced by the WSL command above, not inferred from the CPU tests.


## Add the original untuned baseline only

For the completed comparison and original validation baseline currently present in
this repository, run the following in WSL with the same local MLflow server running:

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/evaluate_test.py --baseline-only \
  --test data/exports/test-test-400-v1 \
  --comparison results/final_test/20261029T135604Z-0dd2abb5 \
  --validation-baseline results/fine_tuning/39bbc7a2796340ad93a414dd2d6ad3d8 \
  --tracking-uri http://127.0.0.1:5000
```

Add `--check-only` for CPU-only integrity, provenance and prompt-budget verification.
Baseline-only mode never invokes the classifier evaluation branch. Supplying
baseline paths without `--baseline-only` fails rather than launching six models.
It requires the explicit completed comparison directory and validation-baseline
run directory; it does not choose a report or baseline based on performance.

The original model revision, prompt version/hash, `low` reasoning effort,
64-output-token budget, 1024-token context limit, greedy generation, batch size 1,
FP16 dtype and seed 42 are checked against the original validation run's saved
configuration and local read-only MLflow records. The original archived generation
configuration is checked against the loaded pinned model before inference; a
mismatch stops execution. The original `BaselineClassifier.classify` and strict
parser are reused without modification. Prompt files and models remain hash-locked.
No validation examples or labels enter the prompt.

All 400 fully formatted prompts pass the real CPU preflight: maximum 321 input
tokens plus 64 output tokens is within 1024; no truncation is used. CUDA is required,
and there is no CPU-inference fallback. No drivers or libraries are reinstalled.

Malformed, ambiguous, missing or truncated classifications remain INVALID (-1).
They are wrong for accuracy and false negatives for their true class in recall/F1.
VALID false rejection counts only explicit GIBBERISH, while VALID acceptance and
invalid-VALID counts separately expose abstentions. This is the exact validation
baseline policy, not a new binary fallback. Metrics include parsing success,
status counts, invalid count, per-class metrics and the 2-by-3 confusion matrix.

A new standalone MLflow run is tagged `evaluation_stage=independent_test` and
`approach=untuned_prompt`. It logs the test dataset in context `testing`, the
original validation run ID, prompt/configuration, raw outputs, parsing statuses,
predictions, errors, confusion image, metrics, timing, environment and provenance.
Existing training, validation and classifier-test MLflow runs are not resumed or
edited. Only one untuned model is loaded, then released.

Baseline artifacts are written to a new sibling directory:

```text
results/final_test/untuned-baseline-<timestamp>-<suffix>/
  predictions.jsonl / predictions.csv
  errors.jsonl / errors.csv
  metrics.json
  confusion.png
  timing.json
  progress.jsonl
  preflight.json
  validation_settings.json
  environment.json
  preservation.json
  classifier_comparison.original.md
  comparison.md
  comparison_with_baseline.json
  completion.json
```

Each prediction records synchronized end-to-end `latency_ms` and `generation_ms`.
The latter observes the unchanged `model.generate` invocation, including its
Python decoding loop and scheduling; it is not kernel-only latency. Model loading
is timed separately. One first-input warmup is discarded from timing summaries;
all 400 inputs are subsequently classified and retained. No human label is used
by inference. Mean/p50/p95 times exclude loading and warmup.

After all 400 predictions, metrics and preservation checks succeed, the command
inserts one baseline row into the original `comparison.md` and adds a baseline
parsing/latency section. Its original bytes are backed up in the new baseline
output directory. Classifier rows, per-seed/across-seed metrics, predictions and
all other classifier result files remain unchanged. The existing comparison
`metrics.json` is not rewritten; `comparison_with_baseline.json` is a new combined
copy containing the unchanged classifier results plus a separate baseline entry.
The updated report is logged to the new baseline run, leaving historical MLflow
artifacts unchanged. A report marker prevents accidental duplicate baseline runs.
A partially failed baseline directory has no completion receipt and is not a
completed benchmark; the command does not automatically resume or select results.

Focused CPU tests use only synthetic fixtures and temporary MLflow stores:

```bash
uv run --no-sync python -m pytest tests/test_final_baseline.py tests/test_final_test.py -q
uv run --no-sync ruff check --target-version py311 scripts/evaluate_test.py \
  src/jais_guard/finetuning/test_baseline.py tests/test_final_baseline.py
```
