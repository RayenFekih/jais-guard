# JaisGuard v1 versus v2 benchmark regression

This workflow compares untuned Jais 3, frozen v1, LoRA v1, frozen v2 and LoRA v2 on
exactly the same frozen 400 records (188 VALID / 212 GIBBERISH). This is a
**previously examined, constructed benchmark used for version-to-version regression
analysis**, not an untouched independent estimate for iteration 2. No test-based
winner, seed, checkpoint or threshold selection is performed.

## Prepared version and WSL execution

The validation-only lock and compatibility checks have already been created:

```text
results/version_comparison/jaisguard-v1-v2-v1/
  threshold_lock.json
  validation_comparison.md
  validation_comparison.json
  validation/frozen-42/ ... validation/lora-44/
  protocol.json
  preprocessing.json
  reused_metrics.json
```

Read `validation_comparison.md` before running the benchmark. Preparation validated
all six v2 source runs, selected epochs, training/validation hashes, saved validation
predictions, report metrics and validation-selected thresholds. It copied their
reports and created `threshold_lock.json` **before** loading benchmark results.
No v2 benchmark predictions exist from the Codex execution.

With the existing local MLflow server running, use this exact WSL command:

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/compare_versions.py evaluate \
  --directory results/version_comparison/jaisguard-v1-v2-v1 \
  --tracking-uri http://127.0.0.1:5000
```

Only the six **iteration-two** classifiers are loaded, one at a time. No training,
CPU inference, old classifier rerun or untuned generation occurs. Settings reuse
the old classifier batch size 2, three warmup batches, right padding, no truncation
and max length 64. Every checkpoint/threshold/artifact and the frozen dataset are
hash-verified again before model loading and after all six complete.

The prepared directory and code are immutable inputs. `prepare` refuses an existing
directory, and `evaluate` refuses existing output predictions. Do not edit locks,
source artifacts or pinned inference/metric code after preparation. A failure or
ambiguity stops execution rather than choosing another model/threshold. If an
execution fails for an environmental reason, preserve it; `--execution-name
benchmark-retry-1` can create a separate new attempt with the same locked inputs.
All six v2 runs are required; no resume or partial-result selection is supported.

The final report will be:

```text
results/version_comparison/jaisguard-v1-v2-v1/benchmark/comparison.md
```

Other new output files include:

```text
benchmark/
  metrics.json                     five groups, both classifier operating points
  confusions/*.png                 baseline and mean-per-seed confusion matrices
  paired_predictions.jsonl/.csv    4800 matched architecture/seed/mode decisions
  paired_summary.json              corrected/new/remaining errors and subgroups
  frozen-42/ ... lora-44/          new v2 predictions/errors/metrics/confusions/timing
  environment.json                 actual new inference environment
  protocol.json / threshold_lock.json
  mlflow.json
  completion.json
```

A completed execution requires `completion.json`. Partial folders are not a final
comparison. No existing v1/baseline reports, predictions, epochs, model weights,
training exports, annotations or test protocol files are modified.

For CPU-only verification of the already prepared version:

```bash
uv run --no-sync python scripts/compare_versions.py verify \
  --directory results/version_comparison/jaisguard-v1-v2-v1
```

For a future explicitly new comparison version, preparation can be repeated into
an unused directory (no benchmark inference):

```bash
uv run --no-sync python scripts/compare_versions.py prepare \
  --directory results/version_comparison/jaisguard-v1-v2-new-protocol \
  --config configs/finetuning-iteration2.json \
  --mlflow-db results/fine_tuning/mlflow.db \
  --test data/exports/test-test-400-v1 \
  --v1-results results/final_test/20261029T135604Z-0dd2abb5 \
  --baseline-results results/final_test/untuned-baseline-20261029T143426Z-199eaaa4
```

Do not rerun preparation over the existing version. The default v1/baseline paths
above were inspected and verified; the program does not choose historical results
based on accuracy or silently select the newest/best test run.

## Verified iteration-two selections

| Architecture | Seed | Selected epoch | Default | Validation-selected threshold |
|---|---:|---:|---:|---:|
| Frozen | 42 | 6 | 0.5 | 0.9511277079582214 |
| Frozen | 43 | 5 | 0.5 | 0.9563547372817993 |
| Frozen | 44 | 5 | 0.5 | 0.9604379534721375 |
| LoRA | 42 | 2 | 0.5 | 0.9913899302482605 |
| LoRA | 43 | 2 | 0.5 | 0.9511184692382812 |
| LoRA | 44 | 4 | 0.5 | 0.942931056022644 |

Each selected metadata record equals the **first minimum validation-loss epoch**
in its ordered training history. Saved logits reproduce the selected validation
loss within numerical tolerance, saved probabilities agree with those logits,
validation IDs/labels/text agree with the approved 250-row export, and archived
metrics reproduce using the common metric function. The exact saved exploratory
threshold must equal the validation report's threshold, with selection source
`validation`. No new threshold sweep is run, even on validation.

The model, revision, seed, pooling, label mapping and complete saved configuration
must match iteration two. Training and validation input hashes must match both
checkpoint metadata and the original finished MLflow training child. Exactly one
best checkpoint per architecture/seed is required; ambiguity is an error. All six
seeds are locked before any v2 benchmark prediction can be read or generated.

## Cached v1 and baseline compatibility

The original frozen test's immutable reader verifies data/manifest/checkpoint and
prompt hashes. V1 predictions are loaded from the completed first evaluation;
untuned predictions are loaded from the completed standalone zero-shot run.
All 400 IDs must be unique and exactly cover the frozen set. Raw input text,
labels, numeric truth, family, subtype, dialect and partition must agree exactly.
Predictions are aligned by ID, not assumed to occupy matching file positions.

Default/tuned decisions must reproduce `p_gibberish >= locked_threshold`. Current
shared metrics must reproduce the archived metrics; metric incompatibility stops
execution. Untuned raw outputs are reparsed with the original strict parser and
finish status. Its prompt/revision settings, original invalid-output policy and
parsing counts must match the archived run. INVALID (-1) remains wrong for accuracy
and a false negative for its true class; VALID false rejection counts only explicit
GIBBERISH, while acceptance and invalid counts separately expose failures.

CPU tokenization verified identical encoded classifier inputs for all twelve
checkpoints: saved tokenizers, special tokens, unchanged text, max length 64,
right padding and no truncation. Existing v1 input-token counts agree. The untuned
baseline intentionally uses its **original chat wrapper** rather than hidden-state
classifier preprocessing; that exact rendered prompt and input-token count are
verified for every cached example. Raw inputs, IDs and human labels remain the
same across all five methods.

## Report interpretation

Accuracy, macro F1, gibberish recall, VALID false-rejection/acceptance, per-class
metrics and confusion matrices are included. Fine-tuned metrics are means and
sample standard deviations (`ddof=1`) over three seeds; the single untuned run has
no invented across-seed SD. Default and validation-selected results stay separate.
Per-seed confusion matrices remain available; group means represent average counts
per seed, while sums count the same 400 examples three times.

Latency reuses existing measurements for v1 and untuned Jais. V2 is measured with
the same classifier batch/warmup settings and timing helper. Loading is separate
from steady-state inference. Classifier times include tokenization/transfers and
are amortized per input; baseline timings use batch 1 and actual per-input
classification/generation times. Hardware, software, precision, cache effects and
separate execution sessions limit direct speed comparisons. Cached timing is not
presented as a new matched measurement. Original and new environments and complete
per-seed timing metadata are retained in the comparison artifacts.

Paired records match architecture **and seed** across versions at both operating
points. They identify corrected errors, new errors, remaining errors and cases
correct in both. Separate groups cover human-VALID recoverable corruption and
human-VALID Tunisian/Egyptian/Levantine/Gulf inputs, plus every subtype/dialect and
per-seed breakdown. Proposed dialect/subtype annotations are not independently
verified semantic sublabels. Validation-selected comparisons combine model changes
with differences in the two validation-selected operating thresholds; default-0.5
comparisons retain a common decision threshold.

Three-seed pooled paired counts represent 1200 decisions on 400 inputs, not 1200
independent inputs. All 4800 architecture/seed/mode pairs are saved, and the report
highlights VALID false rejection transitions. No significance, causal, winner or
production-traffic claim is made. The original test influenced iteration-two task
and data decisions; the report explicitly labels this as **seen-benchmark regression
analysis**.

## MLflow and CPU checks

New runs use experiment **`JaisGuard/version-regression-v1-v2`**: one comparison
parent plus six v2 classifier children. Tags include
`evaluation_stage=version_regression`, `previously_examined_benchmark=true`,
`constructed_benchmark=true`, `threshold_tuning=false`. Existing experiment/run
records are never resumed or edited. Cached sources are recorded as provenance,
not duplicated inference runs. The test is logged with context `version_regression`.

```bash
uv run --no-sync python -m pytest tests/test_version_comparison.py -q
uv run --no-sync ruff check --target-version py311 scripts/compare_versions.py \
  src/jais_guard/finetuning/version_comparison.py tests/test_version_comparison.py
```

The focused tests use synthetic data, temporary local MLflow stores and fake
inference. They exercise selection integrity, metric/input compatibility, cache
preservation, paired transitions, CUDA refusal and orchestration that invokes only
six v2 models. No CPU Jais inference or real iteration-two benchmark evaluation is
performed in Codex. The real-data preparation and verification passed; GPU results
must be produced by the WSL command above.
