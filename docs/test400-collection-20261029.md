# Independent test collection checkpoint — 29 October 2026

The existing test protocol verified and was preserved. A **76-example provisional
draft (44 VALID / 32 GIBBERISH)** is at `data/test400_v1/draft.jsonl`. Nothing was
imported, frozen, trained or evaluated. All human annotation tables remain empty.

Read the full local [collection report](/home/rayen/projects/fine-tuning/data/test400_v1/collection_20261029/collection_report.md)
and [machine-readable report](/home/rayen/projects/fine-tuning/data/test400_v1/collection_20261029/collection_report.json).
These local artifacts include source evidence, class/method/dialect/difficulty and
length tables, exact duplication checks, proposed corrections and remaining gaps.
The source corpus is independently acquired Derja.Ninja text, via a pinned mirror,
not unused rows from the existing Aya/MASSIVE/TyDi sources. The mirror's license
placeholder is not treated as permission for upstream content; publisher text
CC-BY-SA-4.0 evidence and attribution are retained. No audio was collected.

The subset deliberately stops short of 400: rights-cleared natural Egyptian,
Levantine and Gulf inputs and observed public failure text remain gaps. Public
source_method is currently VALID-only; the report flags that confound explicitly.
All generated labels, dialect hints and difficulty hints are provisional. Four
recoverable corruption proposals were corrected to VALID; four uncertain cases
were excluded and original drafting history preserved.

```bash
uv run --no-sync python scripts/prepare_test.py check data/test400_v1/draft.jsonl --allow-partial
less data/test400_v1/collection_20261029/blind_preview.csv
```

`--allow-partial` applies only to read-only checks. Import/freeze still require all
400 and the existing readiness conditions. `scripts/collect_test_draft.py` rebuilds
from archived source/authoring snapshots to a **new** output path with `--output`;
it refuses overwrite and never imports or calls a model. IDs/text/recipes are
reproducible; construction timestamps can differ. Do not rerun the collection
initializer or replace any existing plan/lock/annotation database.
