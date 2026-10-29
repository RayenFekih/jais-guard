# test-400-v1 candidate collection, request-corruption-v2

400 **provisional** candidates have been imported into a new, isolated review DB.
No labels have been approved, no export frozen, and no Jais model used to generate,
select, judge or evaluate these inputs. The final benchmark does not exist until
human annotation, uncertainty resolution and a later explicit freeze.

## Start blind review

Run in the existing WSL uv environment from the repository root:

```bash
cd /home/rayen/projects/fine-tuning
JAISGUARD_ANNOTATION_DB="$PWD/data/test400_collection_v2/annotations.sqlite3" uv run --no-sync streamlit run apps/annotate.py --server.address 127.0.0.1 --server.port 8501
```

Open http://localhost:8501. The active partition is **test**. Choose the unreviewed
queue. All 400 require human review; proposals are never automatically accepted.
Use VALID for any reasonably recoverable meaning, including incomplete requests,
dialect, spelling mistakes and short topics. Use GIBBERISH only when meaning cannot
reasonably be recovered without inventing essential content. Flag uncertainty or
unfamiliar dialect; a saved label does not itself resolve a flag. Unresolved cases
must not enter the final scored export. Review the text alone: do not open the raw
candidate JSONL, this administrative document's statistics or the private originals
while making judgments. A second dialect-competent blind review is recommended for
disagreements; the reused UI is a single-reviewer application, not a multi-rater system.

Source reveal is disabled throughout this collection's review, including after a
label is saved. Originals and edit traces are in a separate private sidecar and
are not returned by the source-reveal API or displayed in the UI. This is workflow
separation, not filesystem access control: the repository owner can read the files.

## Files and preservation

- `data/test400_collection_v2/draft.jsonl`: completed 400-row construction snapshot.
- `data/test400_collection_v2/candidates.jsonl`: identical immutable imported snapshot.
- `data/test400_collection_v2/annotations.sqlite3`: new human-review database.
- `data/test400_collection_v2/quality_report.json`: counts and overlap results.
- `data/test400_collection_v2/import_receipt.json`: import/integrity receipt.
- `data/test400_collection_v2/adoption_receipt.json`: explicit collection-protocol adoption.
- `data/test400_collection_v2/private/parents.jsonl`: 340 authored request parents.
- `data/test400_collection_v2/private/corruptions.jsonl`: original text, parent hash,
  seed, actual step-by-step edits and parent dialect hints, keyed by candidate ID.
- `data/test400_collection_v2/private/historical_references.jsonl`: exclusion snapshot.
- `data/test400_collection_v2/private/reference_inventory.json`: historical file hashes.

The historical `data/test400_v1` directory, its 76-candidate draft, database,
plan and protocol lock remain unchanged. The new workspace retains the logical
`test-400-v1` name and inherits all six model/threshold selections and original
prompt-file hashes. Its new lock references the original lock hash and separately
pins the adopted plan, construction code, provenance and reference snapshot.
This is an append-only adoption, not an update to an existing locked protocol.
No trained checkpoint, training/validation export or MLflow experiment is changed.
No diagnostic candidates have been imported; all 400 families here are reserved to
the general test and must not later supply diagnostics, training or validation.

## Provisional composition

| Proposal category | Count |
|---|---:|
| Meaningful request | 80 |
| Recoverable corruption | 120 |
| Severe corruption | 90 |
| Partially recognizable, proposed unrecoverable | 50 |
| Random Arabic characters/words | 40 |
| Keyboard mashing/repetition | 20 |

Proposed VALID: MSA 80, Tunisian 30, Egyptian 30, Levantine 30, Gulf 30.
The 200 proposed GIBBERISH inputs carry `not-assigned` dialect; their original
request variety, where applicable, is retained only as a provenance hint.
Proposed VALID includes 32 uncorrupted MSA requests plus 12 per dialect; the
remaining 48 MSA and 18 per dialect are corrupted requests. Dialect hints are
assistant proposals requiring knowledgeable human confirmation, not verified
speaker demographics.

### Honest provenance

All 340 request parents are newly **assistant-authored scenarios**, not observed
user traffic, dictionary quotations, independently sampled authors or public corpus
records. There are 80 identity-transcribed requests, 260 scripted request
corruptions and 60 seeded sequence constructions. Python assembles every record,
so `source_method=python` appears in both proposed classes. Parent origin is
separately recorded; this does not imply the parents were produced by Python.
All 400 belong to the same disclosed authoring run but have distinct parent/seed
families. Authored and revised pre-collection parent snapshots are retained.
Public redistribution permissions have not been asserted; this is local review
material. There are **no verified naturally observed inputs** in this collection.

Each request contributes one scored candidate, with no clean counterpart or sibling
variant included. The 60 sequence families each have a distinct deterministic seed.
There are 400 unique family IDs and 340 unique original-request hashes. An entire
parent lineage remains reserved, even if a candidate is later rejected.

### Corruption

Both proposed classes include deletion, physical-keyboard-neighbor substitutions,
transposition, repetition, joined words and split words. The more severe proposals
also use dropped spans. Actual mutations, including their intermediate text, are
recorded. Word boundaries, keyboard rows and occasional compounded mistakes produce
diverse failure modes; these are simulations, not measured typing-error frequencies.
Random sequences alternate unrelated real Arabic words and Arabic characters;
keyboard sequences use varied contiguous key runs and repetitions.

The construction categories and strength parameters are **not labels**. Some
severe proposals may remain understandable; some mild proposals may be unclear.
Humans must relabel or flag them accordingly. Partial-versus-severe subtype
boundaries are provisional and can overlap semantically; final review must never
force a class to satisfy these quotas.

## Completed checks

- Read-only exclusion scan: 112 local historical JSON/JSONL files, 1,719 distinct
  stored text records, plus current annotation DB and religious exclusion records.
  This includes retired records, generation histories, earlier drafts and exports.
- Exact and normalized comparison (NFKC, diacritics/tatweel, whitespace), trigram
  Jaccard and sequence similarity checks: **zero candidate overlap flags**.
- The 340 original requests were independently checked against those references
  and one another: **zero parent overlap flags**.
- Source IDs, family IDs, parent references, recipe/seed and batch/run reservations
  were checked; **zero reserved-lineage conflicts**. No source corpus records were
  copied. Unselected raw source corpora are not an exhaustive comparison universe.
- Exactly 400 unique candidate IDs and family IDs; Arabic-script schema validation;
  requested provisional category counts and VALID variety targets pass.
- Human annotations at import: **0**; no automatic approvals.
- CPU suite: **125 passed**, including strict source hiding after labeling,
  deterministic construction/edit traces, unique-family enforcement, adoption
  preservation, existing UI behavior and frozen-export safeguards.
- Ruff passes on all changed Python files with Python 3.11 as the target.
  Repository-wide Ruff has two pre-existing E402 findings in `scripts/test_jais.py`
  (lines 13–14). `git diff --check` passes.

For a read-only repeat of the standard candidate checks:

```bash
uv run --no-sync python scripts/prepare_test.py --directory data/test400_collection_v2 check data/test400_collection_v2/candidates.jsonl
```

`scripts/build_test400.py --import-for-review` created this collection. Do **not**
rerun it over the completed workspace: it intentionally refuses overwrites.
No collection command performs classifier inference. CPU tests use isolated tiny
fixtures and temporary tracking stores, never these test candidates or the six
completed experiments.

## Remaining limitations for human review

There is no numerical candidate gap: all six provisional targets are filled.
There is no claim that 200/200 will survive human judgment. Rejected or unresolved
items may require append-only replacements before a final 400-example freeze.

This is a controlled constructed benchmark with one assistant authoring source;
it cannot establish production prevalence, independent human-author diversity or
real-user generalization. Native dialect quality and semantic recoverability still
need review. There is no empirical calibration of the typing-error simulator.

Length remains a potential shortcut: medians are 38 characters for proposed VALID
and 24 for proposed GIBBERISH. There are 1 versus 34 inputs of at most 15 characters,
141 versus 142 of 16–40, 56 versus 24 of 41–100, and 2 versus 0 above 100. The existing
coarse automated warning rules do not flag this imbalance; their empty warning
list is not evidence that shortcuts are absent. Very short corrupted fragments,
recognizable partial requests and accidentally meaningful random/repetitive inputs
need particular attention. Do not pad text or change human labels to balance bins.

Semantic paraphrase overlap cannot be ruled out by lexical checks; previously
unrecorded authoring material and model pretraining overlap are not detectable.
No model prediction was consulted. Keep the collection held out from all training,
checkpoint selection and threshold tuning. Do not evaluate any model before final
human review and an explicitly authorized freeze.

Additional checks completed: `cross_parent_check.json` reports zero overlaps
between candidates and other request parents (their own parent matches are
expected and excluded). The real 400-row review UI was smoke-tested against a
temporary database copy: test partition active, no exceptions, no source reveal.
`preservation_receipt.json` records SHA-256 verification of 987 pre-existing data,
configuration and fine-tuning artifact files, with zero changes.
