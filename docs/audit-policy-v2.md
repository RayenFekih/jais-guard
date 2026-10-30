# Audit policy v2 and the approved 400-record migration

`provisional-v2` is the default for new provisional training imports and is now active
for the approved existing 400 records. The migration has been applied transactionally:
**255 required reviews, 22 active v2 groups, 145 unsampled provisional records**.
The 300 existing human reviews, all 616 current human-history revisions, candidate
payloads and previous exports remain unchanged. Original v1 audits are immutable history.

## Admission and review policy

- Aya admission requires the existing pinned revision, `original-annotations`, the
  `human_instruction` subtype and a recognized standalone question/request form.
  Context-dependent/completion/summarization inputs, unsupported forms and bare topics
  remain manual. Comparison removes diacritics/tatweel without changing stored text.
  A rejected admission is not a GIBBERISH judgment. Human-authored does not mean error-free.
- The known `two_repeated_character_blocks` recipe may propose GIBBERISH only for
  two distinct consonants from `ثظضصقغخجطز`, repeated 5–10 times each in two blocks,
  with a deterministic seed, known project source and no parent transformation.
  Laughter, hesitation, ordinary words, mixed meaningful text, short repetitions,
  other recipes and other characters are excluded.
- The existing pinned TyDi/MASSIVE and strict keyboard-walk rules continue. All
  Codex proposals remain higher risk, including invented words and meaningful examples.
  They require generation metadata and cannot enter automatic exports before an audit passes.
- Meaning-preserving perturbations and unrestricted scrambling remain manual.
  One confirmed error quarantines its group and disables the labeling rule. A policy
  version change alone does not readmit a failed source/subtype rule.
- Labels remain strictly VALID/GIBBERISH. Human labels override proposals. Flags
  need explicit resolution. Validation/test remain fully human-reviewed; training
  exports remain human-only unless audited provisional inclusion is selected.

Review reason codes are `rule_not_admitted`, `content_uncertain`, `audit_sample`,
`failed_audit`, and `quality_issue`. Old v1 message strings are mapped for display
only; stored reasons and frozen sampling rows are not rewritten.

## Grouping and fixed-budget dialect coverage

| Risk | Total sample budget per group |
| --- | --- |
| Standard | min(population, max(10, ceil(20% × population))) |
| Higher | min(population, max(20, ceil(50% × population))) |

There is one minimum **per group**, never one per dialect. Reserve one sample per
dialect when the budget permits; distribute remaining slots proportionally to the
remaining populations with largest-remainder allocation and seeded tie-breaking.
Select randomly within each dialect. If dialects outnumber slots, randomly select
the represented dialects and report the coverage gap; do not inflate the budget.
These are coverage-oriented samples, not proof of a population error rate.

An audit group preserves source/revision, method, subtype, rule version and genuine
process-family boundaries. Collect all chunks of a declared run **before** freezing:

- Public records: `provenance.collection_run`, optionally `provenance.input_family`.
- Generated records: `generation.run_id` and a nonempty, versioned
  `generation.audit_family`, backed by the same instruction/recipe family. Different
  Python recipes remain separate even if the caller reuses a family name.
- Keep each original authoring `generation.batch_id` and candidate `family_id`.
  These record provenance/leakage relationships and must not be rewritten to shrink
  audit costs. An audit family is a process definition, not a label or quota.
- Without explicit run metadata, generation batches remain separate. Changing only
  a collection/upload ID cannot merge them. Frozen runs reject later additions.

Thus 5–10-record authoring chunks can form a larger, genuinely homogeneous run for
future collection, while each source/recipe/family remains independently auditable.
Do not combine unrelated prompts or retrospectively create families after seeing errors.

## Pre-import preview

Streamlit shows the projected audit samples, manual-only cases and total workload
before enabling provisional import. The backend compares the preview fingerprint
with the plan immediately before insertion; stale input/policy/eligibility requires
a fresh preview. The CLI expansion importer prints a source/subtype/risk breakdown
and imports the full declared run together rather than freezing every authoring chunk.

```python
plan = preview_import(rows, "new-training-run", seed=42, path=db_path)
print(plan["required_reviews"], plan["audit_samples"], plan["manual_only"])
import_candidates(
    rows,
    path=db_path,
    provisional_batch="new-training-run",
    seed=42,
    expected_projection=plan["fingerprint"],
)
```

`audit_policy.py` contains pure admission/sampling/projection functions; SQLite
persistence remains in `annotations.py`. New v2 and old v1 groups coexist without
changing existing decisions. Exports retain the policy version and v2 process boundary.

## Applied replacement for the existing 400

The read-only proposal is scoped to the imported `milestone700_v1` collection.
It verified the saved candidates match SQLite and rejects a new proposal if
annotation activity, resolved/failed audits, another v2 policy, or partially shared
groups or an already recorded replacement are detected. These checks passed before
application. The approved artifact remains unchanged; this collection can no longer
be silently proposed/applied again.

Only documented homogeneous runs are regrouped: the 75 MASSIVE acquisition records,
the eligible Aya acquisition records, the 40 keyboard records and the 16 repetition
records. Each source/subtype/recipe stays separate. All 18 Codex batch boundaries
are retained because larger shared generation families were not declared originally.

| Source/method and subtype | Candidates | Risk | Audit samples | Manual-only | Required |
| --- | ---: | --- | ---: | ---: | ---: |
| MASSIVE public requests | 75 | Standard | 15 | 0 | 15 |
| Aya original instructions | 75 | Standard / manual | 13 | 13 | 26 |
| Python keyboard noise | 40 | Standard | 10 | 0 | 10 |
| Python repeated-character blocks | 16 | Standard | 10 | 0 | 10 |
| Python meaning-preserving perturbations (Aya-derived) | 15 | Manual | 0 | 15 | 15 |
| Python scrambling (MASSIVE-derived) | 20 | Manual | 0 | 20 | 20 |
| Codex real-word incoherence | 85 | Higher | 85 | 0 | 85 |
| Codex hard semantic incoherence | 32 | Higher | 32 | 0 | 32 |
| Codex invented-word noise | 20 | Higher | 20 | 0 | 20 |
| Codex meaningful counterexamples | 15 | Higher | 15 | 0 | 15 |
| Codex dialectal requests | 7 | Higher | 7 | 0 | 7 |
| **Total** | **400** | | **207** | **48** | **255** |

The proposed total is **63.75%**, down from **81.25%**: 70 fewer reviews. The Aya
rule admits 62 inputs; 13 remain manual due to context/completion or conservative
form checks. Those checks do not assert that the remaining inputs are meaningless.
Audit failures can increase the eventual review count.

The proposed 22 groups contain 207 samples: 171 also belong to the original sample,
36 are newly selected and 28 old sample IDs would no longer be active under v2.
All 199 original sample rows, group definitions and audit events remain stored
unchanged as v1 history; no old candidate was deleted. The approved proposal is saved
locally at `data/milestone700_v1/policy_v2_proposal.json`, including lineage, seeds,
selected IDs and fingerprints. The applied replacement has explicit active pointers;
the frozen proposal itself is not rewritten.

Inspect the same calculation without writing to the database or existing artifacts
(the proposal now reports that a versioned migration is already recorded):

```bash
.venv/bin/python scripts/prepare_candidates.py --collection milestone700_v1 --project-v2
```

### Transaction, verification and rollback

Migration ID: `v2-c1beb1213406a918a176`. The implementation is in
`src/jais_guard/audit_migrations.py`. It takes `BEGIN IMMEDIATE`, rechecks the approved
fingerprints and untouched-review preconditions, creates an integrity-checked SQLite
backup through a separate reader, then applies schema/records/pointers in one transaction.
Any failure rolls back the DDL and data. There is no `executescript` inside this transaction.

The original tables are retained. Replacement proposals, samples and manual requests
live in separate versioned tables. `audit_policy_active` maps each of the 400 IDs to
the applied migration; `audit_policy_candidates` records payload hashes and the group
ledger links predecessor/replacement roles. Views select only the active version for
review, annotation updates and exports. `audit_report(include_history=True)` includes
the inactive original groups; the UI counts only active groups and displays retained
history separately. Database triggers reject changes to historical audit rows/events.

Record-level verification confirmed all 400 payloads unchanged; all 62 eligible Aya
inputs have v2 proposals and all 13 excluded Aya inputs remain manual-only. None of
the 400 is export-eligible while audits are pending. Rehearsal on a full database copy
verified the UI, pending exports (300 original human records only), and rollback.

Local, Git-ignored evidence files:

- `data/milestone700_v1/policy_v2_migration_mapping.jsonl`: all 400 IDs, immutable
  payload hashes, original intended labels, v1/v2 group/sample/reason mapping.
- `data/milestone700_v1/policy_v2_migration_receipt.json`: counts, fingerprint,
  preservation checks and exact backup path.
- `data/annotation/backups/before-approved-v2-*.sqlite3`: integrity-checked pre-migration backup.

Before any of the new 400 is reviewed/exported, this guarded operation can restore
v1 pointers transactionally while retaining all v2 rows as inactive history:

```python
from jais_guard.annotations import DEFAULT_DB
from jais_guard.audit_migrations import rollback_audit_migration

rollback_audit_migration("v2-c1beb1213406a918a176", path=DEFAULT_DB)
```

It refuses rollback after scoped annotations, audit decisions or exports change,
or if active pointers no longer match. A rollback after review begins needs a
separate reconciliation plan; restoring a whole backup must not discard later
human work. Do not rerun the migration artifact to obtain another sample. Existing
300 annotations/exports are never migration or rollback targets.

Lower sampling provides less individual human coverage; passing a group audit still
does not certify every unsampled label. No new examples, training code or model
evaluation were added; only the explicitly approved audit replacement and its
evidence files were produced.
