# Validation prompt baseline and independent test preparation

## Audit and scope

The repository already had pinned Transformers baseline loading, official Jais
chat formatting and strict label parsing, a `finetune.py baseline` command, shared
classification metrics/reports, local MLflow tracking, and a blind Streamlit
annotation app with human-only test export support. Six completed supervised
reports are present under `results/fine_tuning/` (frozen/LoRA × seeds 42/43/44).
Those checkpoints, thresholds and reports are not changed by this phase.

The new work extends these components: a versioned recoverable-meaning prompt,
64-token generation budget, parse diagnostics/success rate, and a preparation
workflow with a separate test annotation DB and guarded final freeze. No test
collection, generation, human labeling or test evaluation is performed automatically.
The candidate checker and freeze commands do not import any model library.

## Run the untuned baseline in WSL

Codex's execution environment cannot see CUDA. The baseline command was attempted
and stopped at its CUDA gate, before model loading or MLflow run creation. No
actual validation baseline metrics are claimed. CPU tests use fake inference to
exercise the runner, real report generation and local SQLite MLflow logging.

Start the existing local MLflow service if it is not already running:

```bash
cd /home/rayen/projects/fine-tuning
mkdir -p results/fine_tuning
MLFLOW_ENABLE_TELEMETRY=false uv run --no-sync mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:////home/rayen/projects/fine-tuning/results/fine_tuning/mlflow.db \
  --serve-artifacts \
  --artifacts-destination /home/rayen/projects/fine-tuning/results/fine_tuning/artifacts
```

Open http://127.0.0.1:5000. In another terminal:

```bash
cd /home/rayen/projects/fine-tuning
uv run --no-sync python scripts/finetune.py inspect
uv run --no-sync python scripts/finetune.py baseline
```

This launches **one original untuned model**, pinned to revision
`e51941b37091287fee4934e3f8a799e0d3fef37f`, with no classifier checkpoint/adapters.
Only the unchanged validation-200-v1 export is evaluated; its SHA-256 is checked.
The existing runner also records training-export provenance, but training examples
are never inserted into prompts or used by this inference path. It never reads
the independent test candidates or evaluates the religious benchmark.

The prompt version is `recoverable-meaning-zero-shot-v1` in
`configs/prompt_baseline.json`. It states the same recoverable-meaning policy shown
in the annotation app: meaningful intent, dialects, typos, short/incomplete requests,
metaphor and factual inaccuracies remain VALID; relevance, safety and answerability
do not determine the class. It contains no labeled examples. Historical prompt
constants and old results remain available unchanged.

Generation uses the official chat template with low reasoning effort, greedy
sampling disabled, FP16 CUDA, batch 1, a 64-new-token cap and 1024-token total budget.
No manual reasoning-close prefill or constrained decoding is used. The small cap
may cause truncation; that is recorded rather than hidden. Keep this configuration
fixed for the first evaluation. A subsequent budget/prompt comparison must be a
new, explicitly versioned run, not a silent retry selected for better accuracy.

Expected console progress is `baseline 1/200` through `baseline 200/200`, followed
by `Run artifacts: .../results/fine_tuning/<new-run-id>`. The run belongs to the
existing `JaisGuard/general-arabic-v1` experiment. It does not relaunch training.

### Parsing and results

Only exact uppercase VALID/GIBBERISH after the expected reasoning-close tag and
normal terminal EOS are accepted. Truncation, missing reasoning close, missing
classification, multiple labels and other malformed answers retain prediction
`-1` (INVALID), with a separate `parse_status`. Reasoning text is never mined for a
label. Invalid outputs count as incorrect for accuracy and false negatives for
their true class in per-class recall and macro F1. They are not silently mapped to
VALID or GIBBERISH. Explicit VALID false rejection counts VALID→GIBBERISH only;
valid acceptance and invalid counts also expose unusable responses.

The new run contains:

- `baseline_config.json`, prompt version/hash, full instruction, generation config,
  model revision, dataset manifests/hashes, environment and Git status.
- `baseline_progress.json`: incremental original rows, rendered prompts, outputs,
  parsing status, token counts and latency.
- `evaluation/report.md`, `metrics.json`, `predictions.jsonl`/`.csv`, `errors.jsonl`/`.csv`,
  and `confusion_default.png`.

MLflow logs accuracy, macro F1, per-class precision/recall, gibberish recall, VALID
FRR/acceptance, parse success rate and per-status counts. No probability threshold
is invented for this categorical baseline. Metrics are computed by the same
shared evaluator used for supervised runs; no LLM judge is involved.

## Independent test: prepare before collecting

Initialize once, **before** drafting or collecting test examples:

```bash
uv run --no-sync python scripts/prepare_test.py init
```

This creates `data/test400_v1/plan.json`, `selection_lock.json`, and an isolated
`annotations.sqlite3`. It locks the six selected checkpoint/head/adapter files,
their already-selected validation thresholds, model/dataset identities, the prompt
files and the current train/validation candidate reference inventory. Hashing is
read-only. It refuses missing/ambiguous experiments or an existing protocol folder.
The default evaluation policy includes all six models at .5 and their previously
selected validation thresholds; do not choose models or thresholds using test data.
No evaluation command is added in this phase.

The protocol can be initialized using `--directory` for a different workspace,
but this implementation deliberately targets the first version `test-400-v1`.
Future data/protocol corrections need a new reviewed version/workflow, not editing
the frozen v1 snapshot in place.

### Candidate creation and coverage

Use the existing candidate schema and provenance conventions. The proposed design
is 200 VALID / 200 GIBBERISH, with meaningful coverage of MSA 80, Tunisian 30,
Egyptian 30, Levantine 30 and Gulf 30. These are **proposal quotas**. Human judgments
may change the final balance; never relabel an example to meet a quota. Gibberish
need not be assigned an invented dialect. Difficulty and dialect hints remain
proposed metadata until human confirmation.

Acquire fresh, naturally written Arabic messages from sources whose rights can be
verified under the existing admission policy. Reserve unused source records and
contributor/author groups. Include everyday domains, informal messages, spelling
mistakes, incomplete requests, Arabic-script loanwords and recoverable corruption.
Author additional challenging meaningful/unrecoverable cases with independently
reserved test-only construction families and runs. For deterministic failures,
record fresh seeds and recipes, including recoverable transformations as well as
unrecoverable ones. Derive only from fresh test-only source material, never from
train/validation examples or their retired originals. Do not include both a parent
and its transformed child in the 400 active candidates.

Do not let one class consist entirely of public messages and the other entirely
of generated noise. Keep realistic length/format ranges overlapping across both
classes, match recoverable/unrecoverable corruptions, and inspect source-method,
class and character-length tables. Preserve truthful source/generator attribution;
do not rename methods to make the table appear balanced. Include hard invented-word,
corruption and incoherence cases alongside harder VALID cases. No model predictions,
including the prompt baseline, may be used to label or curate candidates.

Prepare `data/test400_v1/draft.jsonl` externally, one JSON object per row. Required
fields reuse `validate_candidate` from the annotation backend:

```text
id                  stable opaque ID (e.g. T-<sha256(version + family + text)[:12]>)
input_text          original Arabic-script text, never normalized in storage
partition           "test"
intended_label      "VALID" or "GIBBERISH"; a proposal, never ground truth
intended_label_status "provisional"
review_status       "pending-human-review"
source_method       "public", "codex", or "python", truthfully assigned
subtype, dialect, domain, family_id
provenance          source, source_id, license, attribution, rights_evidence
                    public: also pinned revision and rights_status="verified"
                    retain author_group/contributor_group and modifications if known
generation          codex: batch_id and reserved run_id
                    python: deterministic seed, recipe and reserved run_id
                    retain parent_id and recipe for transformations
```

Save authoring instructions/output/batch metadata or original source receipts and
rights evidence next to the draft. Preserve source IDs and author groups verbatim;
changing names to evade overlap checks is prohibited. Reuse the existing approved
source cards/selection conventions, but verify rights for every new source.
Generated content remains subject to publication review; do not claim it was
naturally written by people. This workflow supplies checks and review, not a
bulk generator that fabricates 400 labels.

**Code-switching:** Arabic-script borrowed words remain in scope. Keep Latin-script
code-switching/Arabizi in `data/test400_v1/out_of_scope.jsonl`, with provenance and
`scope_status="unsupported-script"`. It is separate and unscored, never part of the
400. Main candidate import rejects mixed-script inputs instead of assigning them
GIBBERISH. No binary ground truth is required for that supplement in v1.

### Check and import

```bash
uv run --no-sync python scripts/prepare_test.py check data/test400_v1/draft.jsonl
uv run --no-sync python scripts/prepare_test.py import data/test400_v1/draft.jsonl
```

`check` prints a JSON quality report. It checks exactly 400 records, existing schema,
explicit provisional/pending status, stable/unique IDs, Arabic-script scope,
normalized text overlap, family overlap, suspicious near-duplicates, source IDs,
author groups, parent reuse, generation batches/runs, proposed dialect coverage,
class balance and method/length confounding. References include **all stored**
train/validation records (including retired originals) and the religious benchmark
as read-only contamination references. No religious data is used for tuning.

Exact/normalized duplicates and cross-partition overlap block import. Within-test
near-duplicates are flagged for human resolution by the existing importer.
Proposed coverage gaps and source/length shortcuts block import so they can be
addressed before blind review. Imported candidates and their receipt are immutable;
the checker is intentionally conservative. Do all drafting/replacement before
import; this first implementation has no automatic replacement/rebalancing command.

### Blind review

```bash
JAISGUARD_ANNOTATION_DB="$PWD/data/test400_v1/annotations.sqlite3" \
  uv run --no-sync python -m streamlit run apps/annotate.py
```

Select **test → Review Queue**. The existing app hides source/intended-label hints
while reviewing. Human labels and revision history go only into this separate DB.
Resolve every flag, especially near-duplicates and borderline recoverability.
Proposed labels are never copied into human annotation columns. Do not use the
app's generic export/import controls for this collection; its generic export is
disabled for this workflow. Use the guarded CLI freeze instead.

### Freeze and verify

After all 400 final human labels and flags are resolved:

```bash
uv run --no-sync python scripts/prepare_test.py freeze
uv run --no-sync python scripts/prepare_test.py verify data/exports/test-test-400-v1
```

Freeze reruns separation/identity checks, verifies the immutable imported snapshot
against the DB, verifies all locked models/thresholds/prompts, checks final-label
source/length distributions, and requires all 400 examples to be active and human
reviewed. If final human labels introduce a confounding warning, inspect it and
supply a factual `--confounding-note "..."`; do not change labels to hide it.
Partial/provisional exports are not accepted. Existing version folders are never
overwritten. Output:

```text
data/exports/test-test-400-v1/test.jsonl
data/exports/test-test-400-v1/manifest.json
data/exports/test-test-400-v1/freeze.json
```

`freeze.json` records hashes, the locked evaluation protocol, final coverage and
review notes. Future evaluators must call `read_frozen_test`, which rejects missing
freeze receipts, tampering, incomplete review and changed locked artifacts. Do not
train, tune, select checkpoints/thresholds, or evaluate test candidates before
this point. The six completed validation experiments remain unchanged.

## Tests and limits

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' uv run --no-sync python -m pytest -q
uv run --no-sync ruff check src/jais_guard/prompts.py src/jais_guard/inference.py \
  src/jais_guard/test_workflow.py src/jais_guard/finetuning/evaluation.py \
  scripts/finetune.py scripts/prepare_test.py apps/annotate.py tests/test_evaluation_phase.py
```

Lexical/source checks cannot prove semantic independence or eliminate source/style
shortcuts. Review the tables and provenance manually. Near-duplicate blocking is
conservative. The final test remains single-reviewer unless another reviewer is
explicitly added. Public source licensing and generated-content publication rights
remain subject to the existing policy. Absolute checkpoint paths make the protocol
local to this workspace; relocation requires an explicitly reviewed migration.
The independent-test evaluator and mixed-script detector are not implemented here.

## Partial collection drafts

A partial draft may now be audited without relaxing import or freeze requirements:

```bash
uv run --no-sync python scripts/prepare_test.py check data/test400_v1/draft.jsonl --allow-partial
```

This accepts 1–399 records for read-only schema/overlap checks, reports coverage and
confounding gaps, and sets `ready_for_import=false`. The default full check/import
still require 400. See [the 29 October collection checkpoint](test400-collection-20261029.md)
for the current subset and local provenance report. No test candidates have been imported.
