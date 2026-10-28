# JaisGuard project context

JaisGuard investigates whether the Arabic-centric `IFM/Jais-3-0.9B` model can
be adapted into an accurate, efficient Arabic gibberish classifier. Preserve
meaningful Arabic-script inputs, including Modern Standard Arabic, regional
dialects, spelling mistakes, missing spaces, short phrases, and incomplete
sentences. Arabizi and mixed-script inputs are out of scope for v1. Unsupported
scripts must not be assigned GIBBERISH by default; report them as unsupported
or exclude them explicitly from an in-scope experiment.

## Authoritative task definition

For the general-purpose data collection, VALID means reasonably recoverable meaning
or communicative intent; GIBBERISH means no reasonably recoverable meaning without
inventing missing meaning. Accept dialects, typos, incomplete requests, greetings,
understandable metaphors, unusual coherent expressions and factual inaccuracies.
Relevance, safety and answerability do not determine the label. Prefer VALID when
a reasonable interpretation exists; flag uncertainty for review. There is no
third annotation label. Flagged or unlabeled records cannot be exported.

See `docs/annotation.md` for the current policy, source rights, and workflow. The
historical Arabic definition and generation prompts remain unchanged as benchmark
artifacts; they are not silently rewritten for the new collection.

## Development strategy

Use Hugging Face Transformers and PyTorch in the main `.venv` environment for:

- Model and tokenizer inspection.
- Baseline evaluation of the original model.
- Later fine-tuning experiments.
- Later custom classification heads and direct-logit inference.

Reserve the separate `.venv-vllm` environment for later inference optimization,
throughput benchmarking, and production serving experiments. Preserve the
existing working vLLM scripts and environment. Their presence does not make vLLM
the backend for the current baseline work. Use the main environment for the current smoke test; existing standalone vLLM
scripts remain available for later experiments.

Use the official model card, configuration, and modeling implementation as the
source of truth for Jais's custom K2 Horizon architecture. Do not assume standard
Llama or Qwen internals. Inspect installed dependencies and existing code before
changing them. Account for the local GPU's 6 GB VRAM limit.

## Learning progress and current milestone

The introductory study of tokenization, chat templates, prefill, decoding, and
KV caching is complete.

The checkpoint contains 808 stored candidates and 700 active training candidates.
27 transformed originals and 81 user-approved unrealistic, unreviewed semantic
constructions are retired without deleting records or audit history. The original
300 human reviews (currently 161 VALID / 139 GIBBERISH after user corrections) and
76 completed expansion reviews remain unchanged. The approved realism replacement
added exactly 81 fresh candidates: 30 keyboard failures, 20 character/long-press
failures, 11 invented-word proposals and 20 corruption transformations, including
6 recoverable intended VALID examples. These are proposals, not human labels.
The live collection has 376 human-reviewed records, 191 remaining reviews
(94 audit samples and 97 manual-only cases), and 133 unsampled provisional records.
Two new v2 audit groups require all 19 members to be sampled under the existing
minimums; the other 62 replacements remain manual-only. All old v1/v2 populations,
samples, failures and quarantine decisions remain frozen. Retired samples do not
silently pass their former groups. Stop for review; do not collect more.
See `docs/realism-replacement-checkpoint.md` for the receipt and limitations.
Earlier counts in `docs/training-700-checkpoint.md` describe the historical import.
The 400 now have explicit active-v2 pointers. All 22 v1 groups and 199 original
sample rows remain immutable history; the original 300 annotations and all human
revision history are unchanged. A checked SQLite backup and guarded pointer rollback
are available. See `docs/audit-policy-v2.md` for the migration receipt and procedure.
Keep only transformed
versions in review queues and future exports; retain originals solely for provenance
and annotation history. Historical collection reports remain in `data/milestone300_v1`.
That historical training checkpoint did not include validation/test collection or
model work. The separately approved validation milestone is recorded below.

### Validation collection checkpoint

`data/validation200_v1/` is a local, Git-ignored collection of 200 validation
candidates imported into the existing SQLite application: 100 intended VALID and
100 intended GIBBERISH, with every final label requiring blind human review.
No provisional labels or audits were assigned to validation. Public inputs use 30
fresh Aya original annotations from contributor groups unused in training and 30
fresh MASSIVE dev records; original IDs group related source translations. Codex
authored 40 meaningful inputs and 30 invented/garbled proposals; deterministic Python
generated 70 failure simulations using fresh validation-only runs and seeds.
Proposed meaningful dialect coverage is 50 MSA, 20 Egyptian, 17 Gulf, 10 Levantine
and 3 Tunisian. All text, family, source-ID and parent checks passed against all 808
stored training records and the religious benchmark. String checks cannot prove
semantic independence; the public corpora and synthetic author are shared across
partitions, while source records, contributor groups and generation seed families
are reserved separately.

At import, training remained 700 active records: 567 human-reviewed and 133
unsampled provisional. All 925 existing revisions, all 46 v1/v2 audit groups,
annotations, exports and results were preserved unchanged. Validation starts at
0/200 reviewed; choose partition `validation` and **Review Queue** in Streamlit.
See `docs/validation-200-checkpoint.md` for provenance, checks and the backup receipt.
Stop for human review. No general-purpose test collection or model work is authorized.

New training batches may explicitly opt into versioned provisional-label rules and
seeded, frozen audits. Keep automatic proposals separate from human labels. Under v2,
audit 20% (minimum 10 PER GROUP) for approved public/keyboard/repetition inputs and
50% (minimum 20 PER GROUP) for all Codex or other higher-risk groups, capped at group
size. Represent dialects where feasible within this fixed budget; never multiply
the minimum by dialect count. Preserve source/subtype/recipe/generation-family and
rule-version boundaries; declare genuine acquisition/generation runs before import.
Always project the workload before importing, and never append to a frozen run. One
confirmed error quarantines the affected group and disables that rule version.
Human labels override proposals; flags require explicit resolution. Unsampled
records remain provisional after a group passes. Human-only exports remain the
default, with audited provisional labels allowed only through an explicit training
export option. Validation/test exports must remain fully human-reviewed. Preserve
all existing records/history and exports; never promote historical intentions.

Prioritize realistic accidental input failures over artificial semantic puzzles.
The earlier 45/40/15 difficulty allocation led to excessive contrived word mixtures
and is superseded by the approved realism replacement. Retain already reviewed
semantic examples for diversity; do not generate more hard semantic gibberish in
this checkpoint. Recoverable corrupted inputs remain VALID proposals. Track difficulty
independently from subtype as proposed metadata. Historical pilot difficulty was
not recorded; do not invent or backfill it automatically. Prioritize natural VALID
inputs, dialects, typos and recoverable incomplete requests; keep meaningful metaphor
examples limited. Avoid excessive surreal poetry and repeated artificial templates.

The user is the sole reviewer. Preserve existing annotations and correction history.
Do not create more validation or any general-purpose test data, or implement
training, heads, LoRA, model evaluation or inference optimization in this checkpoint.
Future targets remain 1,000 training and eventually 400 isolated general-purpose
test examples; the 200 validation candidates now await human review.

Use Codex directly for semantic gibberish, invented words, and hard meaningful
counterexamples; save instructions, outputs, available metadata, and family IDs.
Use admitted public sources for human inputs and deterministic Python recipes for
noise and meaning-preserving perturbations. No paid API dependency is authorized.
Prefer permissive sources; allow CC BY-SA 4.0 for linguistic diversity after checking
upstream rights and publication compatibility. Preserve attribution/license metadata;
NC/ND or unclear rights require explicit review, not automatic admission.

Preserve `data/gibberish.jsonl` and its 40 religious dev / 160 religious test records,
labels and all historical results. Use this benchmark only for read-only contamination
checks in the data workflow. New train/validation/test partitions and generation
families remain separate. See `docs/annotation.md` for the application and exports.

Keep model inference, prompting, and evaluation metrics separate so later
experiments can reuse the evaluation framework. Preserve configurations, raw
model outputs, and relevant intermediate representations or measurements to make
results inspectable and reproducible. Do not claim accuracy, latency, or cost
improvements without supporting measurements.

## Learning-oriented workflow

This project is also a hands-on LLM engineering learning exercise.

- Make small, focused changes, one understandable step at a time.
- Start with the smallest useful implementation. Avoid extra scripts, elaborate
  command-line interfaces, and abstractions until a concrete need arises.
- Explain non-obvious implementation decisions and their trade-offs.
- Keep intermediate outputs inspectable so model behavior can be studied.
- Reuse existing code where appropriate; avoid unnecessary abstractions and
  dependencies.
- Prioritize correctness and reproducibility before performance optimization.
- When preparing commits, provide separate logical commits as needed, using only
  `git add` and `git commit -m` commands. Use subtle engineering humour in commit
  messages, and leave committing to the user unless explicitly asked to do it.
- Do not implement future milestones prematurely. Fine-tuning, custom heads,
  optimization, serving, and deployment require subsequent scoped work.

Implement only the work requested for the current step; this context does not
authorize future milestones on its own.
