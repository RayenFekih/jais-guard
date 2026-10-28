# Replacement candidates

27 new, untransformed public-source inputs replace the 27 archived originals.
The SQLite collection now contains **300 active records** plus 27 archived originals.
All earlier texts, labels, flags, exclusion decisions and 285 annotation revisions
were verified unchanged. A local SQLite backup was taken before import.

## Sources and proposed metadata

| Source | Count | Terms | Origin |
| --- | ---: | --- | --- |
| Aya Dataset | 15 | Apache-2.0 | Original human annotations only |
| MASSIVE ar-SA | 10 | CC BY 4.0 | Human-localized assistant requests |
| TyDi QA | 2 | Apache-2.0 | Human-elicited Arabic questions |

All 27 have **intended VALID** labels and proposed **easy** difficulty. These are
candidate metadata, never human ground truth. Subtypes: 15 human instructions,
10 localized requests and 2 questions. Dialect hints: 7 MSA, 5 Egyptian, 3 Gulf,
2 Levantine and 10 ar-SA locale records with unconfirmed dialect. Publisher language
codes are retained separately; a publisher tag does not establish the actual dialect.
No additional Maghrebi candidates were added in this replacement batch.

Inputs are verbatim, with no new transformations, synthetic generation, downloads
or dependencies. Source revisions, IDs, attribution, rights evidence and modification
history are retained per record. Selection indices and cached-source SHA-256 hashes
are saved in `source_selection.json`. Source notices and rights evidence remain in
`../pilot_v1/sources/`. Existing provenance/licensing limitations still apply:
these are elicited/localized inputs, not production logs, and public redistribution
requires the documented license compatibility and attribution review.

## Quality checks and current review state

Compared all 27 against each other, all 300 previously stored candidates (including
archived originals), and the 200 religious benchmark records. No duplicate IDs,
exact/normalized duplicate texts, or near-duplicate matches were found at the existing
thresholds (character-trigram Jaccard >= 0.8 or character similarity >= 0.9).
Seven broad Aya author groups also occur in existing training records; these remain
in training only. No cross-partition family conflicts were found. Lexical checks
and manual selection cannot prove semantic independence; your review remains necessary.

Active candidate intentions: **146 VALID / 154 GIBBERISH**. Current active human
labels: **49 VALID / 46 GIBBERISH**, with **205 unreviewed** and **7 flagged**.
Five reviewed originals remain archived with their labels/history intact; this
explains why the active reviewed count is 95 rather than 100. No human labels or flags
were assigned to the replacements. See `audit.json` for the import/preservation checks.

Refresh Streamlit, select **train → unreviewed**, clear search, and open a matching
example. Resolve flagged records explicitly. Stop after reviewing this collection;
no validation/test collection, training or model evaluation was performed.
