# Dataset pilot and local annotation

**Audit policy update:** the existing 400 and future imports use [provisional-v2](audit-policy-v2.md), with
narrow Aya/repetition admission, group-level sample minimums, explicit run/family
boundaries and pre-import workload previews. The approved replacement is active:
**255 reviews** (207 audit samples + 48 manual-only cases), with 145 unsampled
provisional records awaiting audits. All 22 v1 groups and 199 samples remain immutable
history. Restart/refresh Streamlit and choose **train → manual-review** to continue.

**Current checkpoint:** 700 active training candidates. Of the 727 stored records,
27 originals with transformed copies are archived and excluded from queues and future
exports. Original text, human labels, and revision history remain in SQLite for audit;
the source JSONL files and previous exports are historical snapshots, not active lists.
The original 300 examples remain human-reviewed and unflagged: **163 VALID / 137
GIBBERISH**, with all revision history preserved. The new 400 await the required audits
and manual cases. See the [700-record checkpoint](training-700-checkpoint.md).
The [replacement report](../data/replacements_v1/README.md)
and milestone reports are historical snapshots, not the current review counts.
Future imports apply the same modified-only policy automatically. Existing review flags
remain for explicit human resolution; exclusion does not silently clear flags.
Use **Queue → manual-review** for required samples and flagged cases. See the [milestone report](../data/milestone300_v1/README.md)
for the live audit, new difficulty mix, and quality flags. The original 100-candidate
pilot documentation below remains as provenance; do not re-import it into this database.

## Provisional training labels and audits

The classification labels remain **VALID / GIBBERISH**. Review status is separate:
**human-reviewed**, **provisional**, or **flagged**. Incoming candidates without a
label are pending. Flags take precedence; a human label overrides an automatic
proposal, but saving a label does not resolve a flag. Existing 300 reviews are not
retroactively assigned automatic labels or audits.

In **Import candidates**, select **Provisional training batch**, supply a new batch
ID and seed, then upload a new candidate JSONL. This is training-only and freezes
group membership and sampling in the same transaction as import. Reusing a batch
ID is rejected. The default import remains manual. An equivalent Python call is:

```python
plan = preview_import(rows, "train-next-v2", seed=42, path=db_path)
print(plan["required_reviews"], plan["audit_samples"], plan["manual_only"])
import_candidates(
    rows,
    path=db_path,
    provisional_batch="train-next-v2",
    seed=42,
    expected_projection=plan["fingerprint"],
)
```

The following admission/sampling details describe **historical v1 audits** from the
original import. For current review and new imports, [the v2 policy](audit-policy-v2.md) takes precedence.
Existing v1 reasons display clearer codes without rewriting stored rows or flags.
No separate operation assigns automatic labels to records already in SQLite.
The existing duplicate/script/license/family checks and modified-only policy still
apply before admission. Rule eligibility is conservative, not proof of meaning:

- Pinned TyDi question and MASSIVE localized-request subsets may receive provisional
  VALID after basic standalone-text checks. Source revisions are listed in
  `PUBLIC_RULES`; short or context-dependent inputs stay manual. Aya's varied
  instruction templates remain manual until a narrower subset is approved.
- Python may provisionally label GIBBERISH only for the `keyboard_noise` recipe
  with 10–30 characters drawn exclusively from the consonant-heavy first Arabic
  keyboard row and containing an eight-character contiguous forward/reverse walk.
  Repetition, short strings, other rows, shuffling and meaning-preserving typos stay
  manual. These restrictions reduce recognizable-word/laughter risks; they do not
  replace the random audit.
- Codex proposals require generation instructions (or their file path), available
  model metadata (unknown values are allowed), batch ID and family provenance.
  All Codex proposals receive the higher-risk audit rate, regardless of difficulty.
- Other candidates go to manual review with an uncertainty reason. Licensing
  admission alone never supplies a VALID label. Unsupported scripts are rejected,
  never automatically labeled GIBBERISH.

### Random audit policy

Each group fixes the collection batch, source/revision, method, subtype and rule
version. Dialect strata include explicit `unknown`/`none` values. Sampling is seeded,
without replacement, sorted independently of import order, and persisted once:

| Risk | Sample per dialect stratum |
| --- | --- |
| Standard public or mechanical-noise rules | 20%, minimum 10 |
| Higher risk: all Codex and any hard-difficulty group | 50%, minimum 20 |

Round upward, cap at stratum size. Small strata are fully reviewed. A hard member
raises the whole group's rate; subtype and difficulty remain separate metadata.
The frozen event stores membership, selected IDs and dialect denominators.

Use **Queue → manual-review** for failed-audit and uncertain cases first, then quality
flags and higher-risk audit samples. Proposed labels, source names and subtypes
remain hidden before initial human labeling. Enter your binary decision, then
explicitly resolve the flag. **Audit progress** shows opaque group IDs, status,
sample progress, errors and dialect-stratum coverage without revealing answers.

A group passes only after all sampled records are resolved with zero errors.
One confirmed error—including an error discovered in voluntary review—fails the
group, quarantines remaining provisional records and disables that rule version
for later imports. Resolved human reviews remain usable. Failure is retained even
if a subsequent correction agrees with the original proposal; fixing the rule
requires an explicit code/policy version revision and fresh audits for new batches.
An unresolved correction of a sampled label returns a passed audit to pending.
Sample outcomes store human decisions, notes and revision IDs; append-only audit
events preserve earlier decisions. Existing exports are never rewritten; the audit
report identifies exports affected by later audit invalidation.

For full administrative coverage by source, method, subtype and dialect, inspect:

```python
from jais_guard.annotations import audit_report

reports = audit_report(db_path)
```

This administrative report contains source metadata and is intentionally separate
from blind labeling. It includes sampled/resolved counts, current sample errors,
confirmed historical errors, frozen population sizes and affected export paths.
Voluntary reviews do not increase the random sample's denominator. A passing audit
does not establish that every unsampled record is correct or give a precise error rate.

### Export provenance and compatibility

**Human-reviewed only** remains the default export mode. **Include audited provisional
training labels** additionally permits unflagged provisional records from passed
groups. Validation and test reject this option. Archived, flagged, pending and
unaudited/failed-group automatic labels are never exported.

Schema-version 2 JSONL adds `review_status`, `label_origin`,
`original_provisional_label` and `audit` metadata. `annotation.label` remains the
actual human label (null for unsampled automatic records); top-level `label` and
`is_gibberish` carry the effective classification label. The audit metadata records
group/batch/rule IDs, policy version, sampled status, group outcome, sample size and
error count. A passed group never changes an unsampled record to human-reviewed.

Manifests separate human/provisional and origin counts, include audit coverage and
distinguish export completeness from human-annotation completeness. Existing version
folders are never overwritten. `read_export` still reads legacy human-only exports;
reading mixed exports requires `allow_provisional=True` in addition to the existing
partial-milestone opt-in where applicable. This permission never permits automatic
validation/test labels. Source licensing/publication review remains separate.

SQLite adds provisional decisions, review requests, frozen audit groups/samples,
audit events, disabled rules and export references. The additive upgrade is
transactional and creates a local `backups/before-provisional-*.sqlite3` backup for
an existing populated database. Human annotation rows/history are not migrated or
rewritten. No dependencies, new candidates, models or training jobs are added.


The first milestone is **100 candidates for your blind review**, not a labeled
training release. No model training or evaluation is included. The existing 40/160
religious benchmark, classification prompts, and all previous results are preserved.

## Start reviewing

```bash
.venv/bin/python -m streamlit run apps/annotate.py
```

Open **http://localhost:8501** in your Windows browser. The app runs in WSL,
binds to localhost, uses local assets, and has telemetry disabled. No account,
API key, model download, or Internet connection is needed for annotation.

The pilot is imported into the local database during setup. On another checkout,
use **Import 100-example pilot** in the sidebar once, or:

```bash
.venv/bin/python scripts/prepare_candidates.py --import
```

Repeated imports reject duplicate IDs and leave annotations intact. The database
is `data/annotation/annotations.sqlite3`, outside Git. To use another database,
set `JAISGUARD_ANNOTATION_DB` when starting the app. Torch, Transformers and vLLM retain their existing versions. Streamlit is an optional
annotation extra in the lockfile; its Pillow constraint changes Pillow from 13.0.0
to 12.3.0 in the main environment.
For a fresh environment, install the project's `annotation` extra.

## Annotation policy and workflow

**VALID (0):** reasonably recoverable meaning or communicative intent.
**GIBBERISH (1):** no reasonably recoverable meaning without inventing missing meaning.

Accept meaningful dialects, typos, informal writing, short topics, incomplete
requests, greetings, understandable metaphors, unusual coherent expressions,
and factual inaccuracies. Relevance, safety, and answerability do not decide the
label. Prefer VALID when a reasonable interpretation exists. Unsupported scripts
are excluded instead of labeled gibberish.

1. Read the RTL input without consulting candidate JSONL or generation files.
2. Choose VALID (`1`) or GIBBERISH (`2`). Nothing is preselected. Labels commit
   immediately and **do not automatically advance**.
3. Use Flag for Review (`F`) for uncertainty, unfamiliar dialect, or a quality
   concern. You may flag without labeling. Labeling does not clear an existing flag.
4. Add optional notes. They save when focus leaves the field and before navigation.
5. Use Previous/Next (`P`/`N`). Letter/number shortcuts are inactive in the notes field.
6. Revisit the flagged queue. Choose a binary label, then explicitly resolve its flag.

The source's intended label, subtype, rationale, and provenance are hidden during
initial review. Source details can be revealed only after a human label exists.
IDs are opaque and queue order is stable but shuffled independently of source or
label. Progress and your position survive application restarts. A failed save
shows an error and prevents navigation.

The historical pilot included five meaning-preserving near-duplicate pairs.
Their originals are now archived under the modified-only policy. New lexical
quality flags still require explicit resolution; blind review hides intended labels.
No candidate overlaps the religious benchmark at the configured lexical thresholds.
Lexical checks cannot establish absence of semantic overlap; your review remains necessary.

## Pilot composition and provenance

| Source method | Intended VALID | Intended GIBBERISH |
| --- | ---: | ---: |
| Public datasets | 30 | 0 |
| Codex | 15 | 40 |
| Deterministic Python | 5 | 10 |

Intended labels are **not ground truth**. Source acquisition does not automatically
prove an input is meaningful. There are no human labels before your review.

The 30 public inputs are selected verbatim: 15 TyDi Arabic questions, 10 Aya
original human instructions, and 5 MASSIVE localized requests. Source snapshots,
revisions, attribution and license evidence are documented in
`data/pilot_v1/sources/manifest.json` and its accompanying publisher cards.

- TyDi QA: Arabic questions only, Apache-2.0; original Wikipedia passages/answers
  are omitted. These are elicited questions, not production logs.
- Aya Dataset: original human annotations only, Apache-2.0. We found publisher
  dialect tags are not always consistent with the wording; proposed dialects
  remain unconfirmed. South Levantine entries use upstream code `ajp` in this snapshot.
- MASSIVE 1.1: CC BY 4.0; localized from English. `ar-SA` is not automatically
  treated as Gulf dialect. Its original license is retained.
- Naous dialogue data and ArBanking77 were not imported into this pilot. Their
  upstream/compatibility reviews remain pending for possible later collection.

Codex authored 55 new candidates in eleven batches of five. Each generation file
retains the instructions, output, intended labels, distinct construction-family
IDs, timestamp, and available model metadata. Exact runtime model/session IDs and
sampling parameters are unavailable and recorded as null. No local Jais generation
pipeline, paid API, or automatic label judge exists. Generated texts are not
claimed to be deterministic or already licensed for public redistribution.

Python recipes use seed 42: five meaning-preserving perturbations, four keyboard
runs, three repeated-character sequences, and three character-shuffled inputs.
Derived examples retain parent IDs, source rights, and family IDs. Rebuilding
preserves these texts, but timestamps naturally change.

MSA is most common; Egyptian, Gulf and Levantine examples are present, with limited
Tunisian/Moroccan coverage. These dialects are proposed metadata, not validated
coverage claims. The target for later meaningful inputs is approximately 55/20/12/10/3
percent MSA/Egyptian/Gulf/Levantine/Maghrebi. The pilot need not hit each quota exactly.

## Validation and reproducibility

Inspect distribution and lexical quality without running a model:

```bash
.venv/bin/python scripts/prepare_candidates.py
```

Raw public snapshots are cached under ignored `data/raw/pilot_v1/`. They total
about 176 MB because Aya is a single Parquet shard and MASSIVE an official release
archive; only 30 selected inputs are retained in the candidate corpus.
A new checkout can reacquire them with `--fetch`. It uses pinned publisher revisions
and official URLs, never executes downloaded dataset scripts, and rejects a changed
TyDi viewer revision. `--build` reconstructs the pilot from these caches and saved
Codex batches, but refuses to overwrite an existing candidate file. Candidate IDs,
selected source rows, and Python recipes are reproducible; Codex outputs are the
saved original artifact, not regenerated automatically.

Imports validate required fields, binary intended labels, Arabic script, provenance,
and source rights. They reject duplicate IDs/texts, cross-partition family matches,
and cross-partition near duplicates. Comparisons use NFKC, removed diacritics/tatweel,
and collapsed whitespace; stored inputs remain unchanged. Near matches use character
trigram Jaccard >=0.8 or character similarity >=0.9. Within-partition near matches
are flagged for review. Future collection still needs human semantic/family checks.

## Export and partition boundaries

Use the sidebar to export a partition with a unique version. Snapshots go under
`data/exports/<partition>-<version>/` and are never overwritten.

By default, only human-labeled, unflagged records are exported. The explicit
training-only provisional option is documented above. Every record preserves its
original intended label and provenance separately from the human label and derived
`is_gibberish`. The manifest records counts, exclusions, subtype breakdowns, policy,
source snapshot hash, data hash, and annotation completeness. `complete` means all
active records in that snapshot are exportable under the selected mode;
`annotation_complete` separately requires human review of every active record.
`target_reached` separately reports
whether the 1,000/200/400 collection target has been met. Partial snapshots remain
explicitly marked and must not masquerade as completed collections.

The `read_export` helper requires an explicit expected partition, checks hashes and
review metadata, and rejects incomplete milestones unless explicitly allowed.
No training consumer or model code is implemented. Do not combine partitions using
a directory glob.

General test imports and their UI remain locked. There is no need to unlock them
for the pilot. At the separately approved final-test milestone, record the project
freeze decision using the storage layer's `test_unlocked` state before importing
fresh test families. This is a trusted local workflow guard, not authentication.
Religious evaluation data is read only for contamination checking, never imported
into the annotation database or modified.

## Licensing and publication

Prefer permissive sources; CC BY-SA 4.0 is eligible when it adds linguistic diversity.
Preserve upstream provenance, license versions, attribution and modifications.
NC/ND sources require explicit rights review and are not admitted by the default
public-source validator. Do not infer redistribution rights from a code license.

Exports are **local review snapshots, not automatically approved public releases**.
Publication requires checking upstream rights and whether combined/derived data
triggers share-alike or database obligations. Record incompatibilities before
combining material, retain required notices, and keep unresolved material separate.
The repository's software license does not relicense upstream or generated data.

## Milestones and limits

The 300-candidate human review milestone is complete. The collection now has 700
active candidates; stop for audits/manual review of the new 400 before collecting more.
Future approved collection can grow toward 1,000 training examples
plus 200 separately collected validation examples, and eventually a separately
approved independent 400-example test collection. Source families and generated
paraphrase families must remain within one partition. No held-out inputs may seed
new generation prompts.

This is a single-reviewer pilot, not independent annotation or a representative
production corpus. Disclose synthetic/source proportions, dialect uncertainty,
annotation disagreements, and the limits of lexical leakage checks.

### Approved realism retirements

A replacement import can supply an explicit one-to-one retirement manifest and a
new backup path. It verifies the saved candidate hashes and rejects human-reviewed
or already retired records. New candidates, new v2 audits and retirement exclusions
commit together; a failed write rolls everything back. The checked backup is taken
before writes while holding the import's writer lock.

Retirement removes a record from active queues and future exports, without deleting
its candidate, annotation, proposal, audit sample or revision. Exclusion metadata
records the replacement ID and approval fingerprint. All frozen audit populations,
sample outcomes and failed-rule decisions remain unchanged. The audit panel reports
retired members separately: their pending historical samples do **not** become
resolved, and retirement never passes a group. A retained, resolved human review
remains usable independently of its group's provisional eligibility.

See [the realism replacement checkpoint](realism-replacement-checkpoint.md) for the
81-record operation. All candidate artifacts and the SQLite backup remain local.
