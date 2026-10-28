# Training milestone: 300 candidates

**Modified-only correction:** 27 source originals (5 pilot, 22 expansion) are now
archived, leaving **273 active training candidates**. Their transformed copies remain.
Original text, annotations and revision history are preserved; originals are excluded
from review queues and future exports. No replacements were generated. The counts
below describe the historical 300-record import before this exclusion. Candidate JSONL
and generation files remain unchanged as provenance, not an export for training.

The first 100 were audited from the **live SQLite database**, not exported JSONL. The next 200 are imported and await blind human review. No training or evaluation ran.

## Reviewed pilot audit

Human labels: **54 VALID / 46 GIBBERISH**. Unresolved flags: **0**. Revisions preserved: **285**.
Four original intentions differ from current human labels, all in the real-word incoherence category. Those corrections were preserved. Difficulty was never recorded for the pilot and remains unknown.

Source counts: public 30, Codex 55, Python 15.

### Pilot subtype metadata

| Category | Count |
| --- | ---: |
| arabic_like_nonsense | 10 |
| contiguous_keyboard_run | 4 |
| human_instruction | 10 |
| incoherent_real_words | 30 |
| localized_request | 5 |
| meaning_preserving_noise | 5 |
| meaningful_counterexample | 15 |
| question | 15 |
| repeated_arabic_character | 3 |
| seeded_word_character_shuffle | 3 |

### Pilot dialect hints

| Category | Count |
| --- | ---: |
| Arabic-ar-SA-unverified | 5 |
| Egyptian | 9 |
| Gulf | 7 |
| Levantine | 6 |
| MSA | 54 |
| Moroccan | 1 |
| Tunisian | 3 |
| none | 15 |

Pilot gaps: difficulty was not tracked, so historical difficulty proportions cannot be established; clear request prefixes with noisy tails could remain meaningful; source dialect tags were inconsistent; spontaneous conversation and short natural requests were sparse. No historical labels or metadata were changed.

## New 200-candidate batch

Intended labels: **96 VALID / 104 GIBBERISH**. Based on the reviewed pilot, the planning balance is 150/150 across 300 records. This is not 300 human labels: **100 reviewed + 200 unreviewed**.

Original candidate intentions across all 300 remain 146 VALID / 154 GIBBERISH, because the pilot intentions were not rewritten to match human corrections.

### New source methods

| Category | Count |
| --- | ---: |
| codex | 83 |
| public | 70 |
| python | 47 |

### New public sources

| Category | Count |
| --- | ---: |
| AmazonScience/massive | 20 |
| CohereLabs/aya_dataset | 35 |
| google-research-datasets/tydiqa | 15 |

### New subtypes

| Category | Count |
| --- | ---: |
| dialectal_question | 1 |
| greeting_request | 1 |
| human_instruction | 35 |
| incoherent_real_words | 42 |
| incomplete_request | 2 |
| informal_message | 1 |
| informal_spelling | 1 |
| invented_word_noise | 10 |
| keyboard_noise | 15 |
| localized_request | 20 |
| meaning_preserving_noise | 10 |
| meaningful_metaphor | 2 |
| missing_spaces | 2 |
| question | 17 |
| repeated_characters | 10 |
| request | 3 |
| semantic_incoherence | 15 |
| short_topic | 1 |
| unrecoverable_scrambling | 12 |

### New dialect hints (not human-confirmed)

| Category | Count |
| --- | ---: |
| Arabic-ar-SA-unverified | 20 |
| Egyptian | 17 |
| Gulf | 10 |
| Levantine | 9 |
| MSA | 95 |
| Moroccan | 1 |
| Tunisian | 1 |
| none | 47 |

### New difficulty, both labels

| Category | Count |
| --- | ---: |
| easy | 125 |
| hard | 17 |
| medium | 58 |

### New GIBBERISH difficulty

| Category | Count |
| --- | ---: |
| easy | 47 |
| hard | 15 |
| medium | 42 |

Gibberish targets: easy **47/104 (45.2%)**, medium **42/104 (40.4%)**, hard **15/104 (14.4%)**. Difficulty is a separate proposed field, not inferred from subtype and not measured model performance. VALID includes 70 public inputs, 16 Codex drafts, and 10 meaning-preserving Python variants; only two new VALID inputs are metaphor counterexamples.

## Quality checks and preservation

- No accepted duplicate IDs, raw texts, or normalized texts.
- No religious-benchmark near matches at trigram Jaccard >=0.8 or character similarity >=0.9.
- No cross-partition family conflicts; all 300 records are training candidates.
- Fourteen near-match pairs within training produced 14 new review flags. They include intentional typo variants and similar keyboard patterns. Original flags remain resolved.
- Five broad source-author groups recur within training. These are tracked grouping keys, not permission to distribute related examples across future partitions.
- One duplicate keyboard draft was regenerated before the accepted file was written.
- Existing candidate payloads, human annotations, and all 285 revisions were checked before and after import. A SQLite backup is under ignored `data/annotation/backups/`.
- Religious benchmark, classification prompts and prior results remain unchanged.

## Resume blind annotation

Refresh the existing Streamlit app. Keep **Partition → train**, clear the search field, and choose **Queue → unreviewed**. Select the first matching example, or press Next to open the first pending example if the old reviewed record remains selected. All 200 new candidates require your review. Resolve flags explicitly after deciding the human label.

To recheck candidate metadata/overlap without importing or running a model:

```bash
.venv/bin/python scripts/prepare_candidates.py --collection milestone300_v1
```

The source selection, pinned source revisions, deterministic seed 30027, and generation instructions are preserved. Codex batches were authored directly in this session; exact runtime model IDs/sampling settings remain unavailable, not fabricated. The saved outputs are the generation artifact. No paid API or local LLM pipeline exists.

## Limitations and next checkpoint

Aya and TyDi are human-elicited inputs, while MASSIVE is localized. Twenty new ar-SA inputs retain an unverified dialect hint. Some Aya authors recur. Source-rights evidence is retained from the already admitted snapshots; Naous and ArBanking77 are not used in this batch. Publication compatibility review remains separate from local annotation.

This is a single-reviewer collection. Hard semantic candidates may reasonably receive VALID labels under the policy; do not force labels to maintain quotas. Lexical checks and manual author screening cannot certify semantic independence. Stop after review of these 200 before further collection. No validation or test datasets were created.
