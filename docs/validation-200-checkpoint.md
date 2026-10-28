# Validation collection checkpoint — 28 October 2026

Exactly 200 candidates were imported into partition `validation`, with **100 intended
VALID and 100 intended GIBBERISH**. Every candidate is pending blind manual review;
there are no provisional labels, automatic admissions or validation audit samples.
Human labels determine the final distribution. No validation export was created.

| Source / method | Intended VALID | Intended GIBBERISH |
|---|---:|---:|
| Aya original annotations | 30 | 0 |
| MASSIVE localized requests | 30 | 0 |
| Codex-authored fresh inputs | 40 | 30 |
| Deterministic Python failure simulations | 0 | 70 |
| Total | 100 | 100 |

Meaningful inputs cover knowledge questions, requests, ordinary reports, short topics,
incomplete requests, informal spelling and two understandable metaphors. Proposed
VALID dialect counts are **50 MSA, 20 Egyptian, 17 Gulf, 10 Levantine, 3 Tunisian**.
These are text-based collection hints, not human-confirmed dialect annotations. In
particular, MASSIVE's `ar-SA` locale alone does not establish Gulf dialect; neutral
standard requests remain MSA. Aya publisher codes are retained even where a record
tagged `ary` reads as MSA. Random noise and invented words carry dialect `none`.

The 100 negative proposals contain 30 keyboard-contact paths, 20 contact-bounce or
pressure failures, 20 severe capture/extraction failures, 20 short invented-word
inputs and 10 garbled short entries. The severe extraction cases include Unicode
replacement glyphs, preserved exactly. Corruption is not itself proof of gibberish;
recoverable intent should receive VALID during review. No surreal semantic puzzles
or additional hard semantic word-salad examples were added.

## Source rights and provenance

Public records reuse already admitted, locally cached, pinned sources:

- **Aya Dataset**, revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, Apache-2.0.
  Only `original-annotations` inputs are selected. Preserve license/notices and
  source attribution. Contributor IDs identify source groups, not human reviewers.
- **Amazon MASSIVE**, revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`,
  release 1.1, CC BY 4.0. The selected `ar-SA` records come from its upstream `dev`
  split. Preserve attribution, license links and modification notices. These are
  human-localized requests, not organic Arabic conversation logs.

Per-record provenance includes source URL, pinned revision, original split/ID,
license, attribution, rights evidence and origin. Full source cards/license evidence
and cache hashes are saved locally under `sources/`. The cached source hashes were
checked against the admitted training-source snapshot; no new download was required.
Original public text, including spelling and line breaks, is unchanged.

Synthetic content retains publication-review-pending metadata. Saved Codex batches
contain exact instructions and outputs, timestamps, family IDs and available model
metadata; unavailable model/session/sampling details are explicitly null. Python
uses seed `20020261028` with a saved recipe. Its 20 fresh meaningful source texts are
provenance-only validation seeds, never active examples in any partition. No public
release license is inferred for the whole collection and no dataset was published.

## Partition isolation and checks

The collection uses source records and generation families reserved for validation:

- No source ID or family ID is shared with training. Aya contributor groups are
  disjoint from those used in training; MASSIVE original-utterance IDs also keep
  related source translations together. Multiple examples within validation can
  share a reserved contributor or construction family.
- No training record is a transformation parent. Fresh validation seeds were checked
  separately; only their final corrupted versions enter the active collection.
- Exact/normalized comparisons use the existing NFKC, whitespace, diacritic and
  tatweel comparison view, preserving original stored text.
- Existing character-trigram Jaccard ≥0.8 and character similarity ≥0.9 checks found
  no near duplicates. Family intersections and parent-ID intersections are empty.
- References included **all 808 stored training records**, including 108 retired
  records, plus the untouched **200-record religious benchmark**. Within-validation
  pairs were checked too. All final lexical checks passed without overlaps.

The religious benchmark has no source-family IDs. Its contamination checks therefore
use text comparisons and editorial no-seed-reuse screening; source-family independence
against that benchmark cannot be mechanically established from unavailable metadata.

Editorial screening rejected no meaning solely for unusual wording or factual
accuracy. Lexical checks and source-group reservations cannot guarantee semantic
independence. Shared corpora and the same Codex author remain common influences;
new family IDs alone do not establish independent model generation. Generic Arabic
question forms and broad intents naturally recur across partitions.

Limitations: all negative candidates are synthetic mechanism simulations rather than
observed traffic. Public positives include elicited trivia and localized assistant
requests, with 20 Aya inputs from one newly reserved contributor. Dialect authenticity
needs the sole reviewer's judgment; Maghrebi coverage is limited to three synthetic
Tunisian positives. The balanced collection cannot establish production prevalence,
complete subtype coverage, or independent annotator agreement. This is internal
validation for later development, not the eventual independent general-purpose test.

## Import preservation and review

A checked SQLite backup and dry-run import preceded the live transaction. Only 200
candidate rows were appended. Every other database table remained identical,
including annotations, revision history, exclusions, source decisions, active-v2
pointers, frozen audits, failed rules and export ledgers. Training remains **700
active examples: 567 human-reviewed and 133 unsampled provisional** at import.
All **925 existing revisions and 46 historical/current audit groups** survived
unchanged. Hashes verified existing benchmark, export and result files were untouched.
No general-purpose test collection, model evaluation or fine-tuning was performed.

Artifacts are local and Git-ignored in `data/validation200_v1/`:
`candidates.jsonl`, `selected_sources.jsonl`, `source_selection.json`,
`quality_report.json`, `sources/manifest.json`, saved `generation/` instructions,
outputs and recipes, `dry_run_receipt.json`, and `import_receipt.json`.

Backup: `/home/rayen/projects/fine-tuning/data/annotation/backups/before-validation200-2026-10-28T20-11-47.377201+00-00.sqlite3`.

To review, reload the current Streamlit app, select **Partition → validation**, and
keep **Review Queue** selected. It starts with **200 remaining reviews**. Choose
VALID/GIBBERISH to save and advance; flag uncertain cases for later resolution.
Proposed labels and provenance stay hidden until initial labeling. Validation
exports remain human-only and exclude unresolved or unlabeled examples.

Verification: all **91 tests** passed, including the new artificial-fixture UI test
for fully manual validation review and preservation of training history. Ruff and
`git diff --check` passed. No new runtime dependency or annotation service was added.
