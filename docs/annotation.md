# Dataset pilot and local annotation

**Current checkpoint:** 300 active training candidates. Of the 327 stored records,
27 originals with transformed copies are archived and excluded from queues and future
exports. Original text, human labels, and revision history remain in SQLite for audit;
the source JSONL files and previous exports are historical snapshots, not active lists.
The [27 replacement inputs](../data/replacements_v1/README.md) are imported and await
blind review alongside the remaining expansion candidates (205 currently unreviewed).
Future imports apply the same modified-only policy automatically. Existing review flags
remain for explicit human resolution; exclusion does not silently clear flags.
Use **Queue → unreviewed** to resume. See the [milestone report](../data/milestone300_v1/README.md)
for the live audit, new difficulty mix, and quality flags. The original 100-candidate
pilot documentation below remains as provenance; do not re-import it into this database.


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

Five within-training near-duplicate pairs are intentional meaning-preserving
Python variants of public inputs. Their derivative records start flagged, and
must be resolved before export. The blind UI does not expose their intended labels.
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

Only human-labeled, unflagged records are exported. Every record preserves its
original intended label and provenance separately from the human label and derived
`is_gibberish`. The manifest records counts, exclusions, subtype breakdowns, policy,
source snapshot hash, data hash, and annotation completeness. `complete` means all
imported records in that snapshot are resolved; `target_reached` separately reports
whether the 1,000/200/400 collection target has been met. Partial snapshots remain
explicitly marked and must not masquerade as completed collections.

The `read_export` helper requires an explicit expected partition, checks hashes and
human annotations, and rejects incomplete milestones unless explicitly allowed.
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

The 100-candidate review milestone is complete; stop next after reviewing the new 200. After your feedback: refine policy
and collection quality, then grow to 300–500 reviewed training examples, 1,000 train
plus 200 separately collected validation examples, and eventually a separately
approved independent 400-example test collection. Source families and generated
paraphrase families must remain within one partition. No held-out inputs may seed
new generation prompts.

This is a single-reviewer pilot, not independent annotation or a representative
production corpus. Disclose synthetic/source proportions, dialect uncertainty,
annotation disagreements, and the limits of lexical leakage checks.
