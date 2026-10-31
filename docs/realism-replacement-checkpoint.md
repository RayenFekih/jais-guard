# Realism replacement checkpoint — 28 October 2026

> Historical checkpoint: preserved as evidence, not current operating instructions.
> See [current public workflows](public-workflows.md).

The user approved retirement of exactly 81 unreviewed semantic-construction
candidates from the 400-record expansion: 61 `incoherent_real_words` and 20
`semantic_incoherence`. Their records remain stored; their labels were not changed.
All 300 original human reviews and the 76 completed expansion reviews are preserved,
as are all 731 annotation revisions at application time. Counts below are a snapshot;
subsequent human review will change progress.

| Replacement mechanism | Candidates | Intended VALID | Intended GIBBERISH |
|---|---:|---:|---:|
| Accidental keyboard gestures, adjacent walks, two-thumb bursts | 30 | 0 | 30 |
| Long presses, interrupted presses, duplicated touches | 20 | 0 | 20 |
| Short invented-word inputs, authored by Codex | 11 | 0 | 11 |
| Sustained key drift, intermittent capture, recoverable touch/spacing mistakes | 20 | 6 | 14 |
| Total | 81 | 6 | 75 |

Python generated 70 transformations/mechanical candidates using seed `70081028`.
Codex authored 11 invented-word proposals in saved 6/5-output batches and 20 fresh
meaningful transformation seeds. Seeds are provenance only, never active records.
Six corrupted requests retain recoverable meaning and are intended VALID. The
others are negative proposals, not automatic semantic judgments. The six meaningful
inputs cover MSA, Egyptian, Gulf and Levantine; language-neutral mechanical noise
and invented words carry dialect `none`. No hard semantic gibberish was generated.

These are mechanism-based simulations, **not observed user traffic**. Invented
words may resemble unfamiliar dialectal vocabulary. Severe drift or capture loss
can still leave interpretable fragments. Human review decides; source and intended
labels remain hidden in Review Queue. This small batch does not establish real
failure frequencies or broad dialect representativeness.

## Quality and workload

Lexical checks used all 727 previously stored candidates, including archived records
and the 81 retirements, plus the untouched 200-record religious benchmark. Fresh
transformation seeds were checked separately against those references. Final checks
found zero exact/normalized duplicates, near-duplicate matches, or cross-partition
family conflicts. Draft QC detected 11 near-duplicate pairs; the saved recipe records
parameter revisions for 10 affected draft records before import. Drafts and the QC
report remain provenance artifacts, not additional imported candidates. No source
seed is reused from the existing collection. Semantic independence remains an
editorial check, not a guarantee supplied by string similarity.

The frozen pre-import projection required **81 reviews**:

| Admission under unchanged provisional-v2 | Population | Audit samples | Manual-only |
|---|---:|---:|---:|
| Codex invented-word proposals, higher risk | 11 | 11 | 0 |
| Verified two-consonant long-press recipe, standard risk | 8 | 8 | 0 |
| Other mechanical/corruption recipes, not admitted | 62 | 0 | 62 |

Minimums are capped at group size, so both new audit groups require full review.
We did not constrain every recipe to the narrow admitted patterns or relax rules to
reduce workload. The old queue lost 81 retired cases and gained 81 replacement cases.
There are now **700 active training candidates**, 808 stored records and 108 retired
records (including the prior 27 transformed originals). Active status at import:

- 376 human-reviewed: 198 VALID / 178 GIBBERISH.
- 191 remaining reviews: 94 pending audit samples and 97 manual-only records.
- 133 unsampled provisional records; none is promoted by this replacement.

The 22 historical v1 groups remain intact. There are 24 groups under active v2
policy pointers: 22 pending and 2 failed, including the 2 new pending groups.
Twelve older v2 groups contain retired members. Their original populations, frozen
samples and statuses remain intact, even when no unresolved active member remains.
They cannot pass by silently removing retired samples. Failed semantic rules remain
disabled. Already reviewed semantic examples remain for diversity.

## Local artifacts and recovery

All operation artifacts are local and Git-ignored under:
`data/milestone700_v1/realism_replacements_v1/`.

- `approved_selection.json`: original approved 81-ID selection and payload hashes.
- `retirements.json`: one-to-one old/new ID mapping and retirement reason.
- `generation/authoring.json`: instructions, authored outputs/seeds and available
  model metadata; unavailable version/session/sampling settings are explicitly null.
- `generation/recipes.py`: deterministic Python recipe and QC parameter revisions.
- `candidates.jsonl`, `quality_report.json`, `workload_preview.json`: accepted inputs,
  final contamination checks and frozen workload projection.
- `dry_run_receipt.json`, `application_receipt.json`: preservation checks, table
  fingerprints, group IDs and protected benchmark/export/result file hashes.

SQLite backup, verified with `integrity_check` before import:
`data/annotation/backups/before-realism-replacements-2026-10-28T19-12-35.253378+00-00.sqlite3`.

The import and retirements were one transaction. A failure before commit would have
rolled both back; this operation committed successfully. For recovery, stop Streamlit
and all SQLite writers first. Compare the live database with the receipt to identify
any later human reviews. Restore the checked backup through SQLite's backup API only
if no later work would be lost, retaining a separate copy of the current database.
If later reviews exist, preserve them and plan an explicit forward correction instead
of overwriting the database. Do not rerun or roll back the old v1-to-v2 migration.

Reload Streamlit and keep **Review Queue** selected to review the 191 required
records. Classification and flag resolution remain human actions. No further
candidates, validation/test collections, exports, model training or evaluations
were created in this operation.

Verification: `pytest tests -q` passed all 90 tests, including transactional rollback,
review protection, immutable failed-audit preservation, retired export exclusion,
and Streamlit review navigation. Ruff and `git diff --check` passed. Bare repository
`pytest` also collects legacy GPU smoke scripts outside `tests/`; collection fails
on their existing `config`/CUDA-runtime imports. Those scripts and environments were
left unchanged.
