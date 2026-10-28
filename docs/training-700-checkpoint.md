# 700-candidate training checkpoint

**Historical import report:** the approved [v2 replacement](audit-policy-v2.md) is
now active with 255 required reviews. Counts below retain the original v1 import
snapshot (325 reviews); the original audit records are preserved as history.

The local SQLite collection contains **700 active training candidates** and 27
archived originals. The existing 300 human reviews (163 VALID / 137 GIBBERISH),
615 revisions, candidate payloads, partitions and prior exports are unchanged.
No validation/test records, model runs, or training jobs were added.

## New 400 candidates

| Method | Intended VALID | Intended GIBBERISH | Total |
| --- | ---: | ---: | ---: |
| Public sources | 150 | 0 | 150 |
| Codex | 22 | 137 | 159 |
| Deterministic Python | 15 | 76 | 91 |
| Total | 187 | 213 | 400 |

Together with the existing human labels this targets 350/350, **if human review
agrees**. No labels are changed to meet a quota. Public inputs are 75 Aya original
annotations and 75 MASSIVE localized requests, selected from the existing admitted,
pinned training snapshots. No new source downloads or dependencies were needed.
Public inputs comprise 80% of the new intended VALID examples.

| New intended gibberish difficulty | Count | Share |
| --- | ---: | ---: |
| Easy | 96 | 45.1% |
| Medium | 85 | 39.9% |
| Hard | 32 | 15.0% |

Difficulty remains separate from subtype and human labels. New subtypes: 75 public
instructions, 75 localized requests, 85 incoherent-real-word fragments, 32 semantic
incoherence candidates, 20 invented-word strings, 40 keyboard-noise strings,
16 repeated-character examples, 20 scrambled inputs, 15 meaning-preserving Python
perturbations, 15 meaningful counterexamples and 7 dialectal requests. Meaningful
counterexamples include practical incomplete requests, typos and two metaphors.

Proposed dialect hints across the new collection: MSA 161, Egyptian 39, Gulf 14,
Levantine 12, Tunisian 1, Moroccan 2, unconfirmed ar-SA locale 75, and none 96.
Random noise is not assigned a dialect. Tags remain unconfirmed; publisher language
codes and locale are retained separately. MSA is prioritized, with limited Maghrebi
coverage. An ar-SA locale is not evidence of Gulf dialect.

## Audit and review checkpoint

Imports use **31 separate collection batches**. The **22 frozen audit groups**
keep source/revision, generation batch, subtype and rule version separate.
The 18 Codex generation batches contain 5–10 records each, with exact instructions,
outputs, family IDs and available model metadata saved. Unavailable runtime model,
session and sampling information remains explicitly unknown. No local generation
model or paid API was used.

| Active workflow status after import | Count |
| --- | ---: |
| Existing human-reviewed | 300 |
| Flagged: random audit samples | 199 |
| Flagged: other manual cases | 126 |
| Unsampled provisional, awaiting group audit | 75 |

The 199 samples comprise all 159 Codex candidates, 20 MASSIVE records and 20 Python
keyboard-noise records. The other 126 cases are 75 Aya inputs and 51 Python
transformations/repetitions outside the approved automatic rules. No new human
labels or annotation revisions were assigned.

**325 records currently require human review.** Small Codex batches fall below the
higher-risk minimum of 20, so they are fully sampled. This deliberately preserves
the approved audit policy rather than increasing automatic coverage by merging
batches. One confirmed error quarantines the affected group; additional records
may then require manual review. Unsampled records remain provisional after passing.

Refresh Streamlit, select **train → manual-review**, clear search, and open a matching
record. Label it, then explicitly resolve its flag. Labels stay hidden until your
initial decision. Saving stays on the same example; use Next or click the next
matching result. Use **Audit progress** to monitor groups. Avoid using the broader
unreviewed queue to determine which records are required samples.

## Quality, provenance and limitations

- Final checks found no duplicate IDs, normalized duplicate texts, near-duplicate
  pairs, cross-partition family conflicts or lexical matches against the existing
  religious benchmark. Draft repetition duplicates were corrected before import.
- The 35 transformation-source originals are retained as provenance only, never
  imported as active candidates. Stored texts and parent/family relationships are
  reproducible from saved source snapshots and Python seed 700428.
- Related source-author groups remain entirely in training. Lexical checks cannot
  establish semantic independence; no claim of independent semantic validation is made.
- Public inputs are elicited/localized rather than production traffic. Two sources
  dominate the public portion. Some publisher dialect tags are inconsistent with
  wording; geographic/dialect coverage must not be overstated.
- Hard semantic candidates may have recoverable interpretations. Human reviewers
  must retain VALID for those interpretations; oddity, metaphor and factual error
  are not grounds for GIBBERISH.
- Provenance, pinned revisions, attribution, rights evidence, cache hashes, seeds,
  model metadata and generation-family IDs remain in local artifacts. Existing
  source-license/publication restrictions still apply; this is not a public release.

Local artifacts are under `data/milestone700_v1/` (ignored by Git):
`README.md`, `checkpoint.json`, `audit.json`, `source_selection.json`,
`selected_sources.jsonl`, `candidates.jsonl`, and `generation/`. Earlier drafts are
retained in `drafts/`. SQLite was backed up before import under
`data/annotation/backups/before-milestone700-*.sqlite3`.

Inspect the saved collection without importing again:

```bash
.venv/bin/python scripts/prepare_candidates.py --collection milestone700_v1
```

The existing `--build` refuses to overwrite accepted candidates; `--import` rejects
existing IDs/batches. Tests use artificial fixtures and an optional reconstruction
check when these local files are available. Stop at this review checkpoint.
