# Reproducibility and preservation

## What can be reproduced publicly

The intended public derivative is currently hosted in a **private** HF repository.
Authenticated download at the recorded revision was verified; anonymous access
returned 401. Public availability remains an owner-controlled visibility decision,
and this cleanup does not change it. Offline preparation accepts the exact eight-file
release and verifies all bytes against the publication receipt.

The pinned public derivative supports new training and evaluation, not exact replay
of models trained on the original 1,500-row training export. Its 1,498 train rows,
250 validation rows and 400 test rows retain published labels and provenance.
Original experiment positions 277 and 486 were omitted for privacy uncertainty.
Do not reintroduce their text or publish private exports to close that gap.

The public benchmark evidence package supports hash verification of figures and
reports and numerical verification of all 16 vLLM HTTP aggregates from timing-only
observations. Aggregate quality is checked against cached historical predictions
locally. Public copies exclude those prediction records and private provenance.
Thus full independent quality recomputation requires original prediction artifacts;
public aggregate consistency is a narrower verification claim.

## Historical replay

Keep `ea1986d` as the original code reference. Historical manifests lock source
bytes, paths, exports, checkpoints and thresholds. Cleanup intentionally changes
some source bytes. A historical verifier rejecting the cleanup tree is expected;
never patch an old manifest to accept new source. Use the original revision and
original artifacts at their recorded paths, or create a new versioned experiment
with new locks and explicitly record the difference. No old results were regenerated.

The original v1 dataset has 700 train and 200 validation examples. Iteration two
uses 1,500 and 250. The loader now honors declared counts while retaining the exact
v1 fallback. This fixes a validation defect without changing training computation.
Frozen configs remain byte-identical. The public adapter generates a separate config.

`data/test400_collection_v2/private/parents.jsonl` and `corruptions.jsonl` are missing.
Eighty passthrough test originals are inferable; parents for 260 transformed
requests cannot be reconstructed here. Restore only authentic saved evidence.
The default construction test now uses synthetic parents; it is not evidence of
historical reconstruction. The original-sidecar integration test reports a skip.

The test benchmark was previously examined and informed iteration two. Test[319]
is contained in test[204]. Preserve both and disclose dependence. Seed standard
deviations describe repeated models on the same benchmark, not independent test
samples. Jev latency includes remote networking and cannot establish a controlled
hardware speed comparison with local Jais.

## Integrity records

Before implementation, all 3,130 files under `data/`, `results/`, `artifacts/` and
`configs/` were hashed into `/tmp/jaisguard-cleanup/immutable-before.json`. This
private inventory intentionally stays untracked because it names local evidence.
The after-check and difference receipt are stored alongside it, and all three
receipts are preserved under ignored `artifacts/repository-cleanup/`. The only planned
removal in these trees is the empty `data/.gitkeep` placeholder; every substantive
historical artifact must match. Keep those local receipts private; they name original experimental artifacts. Public evidence source and destination hashes are in
[the allowlist](benchmarks/allowlist.json). A narrow public reference for the already
tracked frozen configs and v4 reports is [immutable-reference.json](immutable-reference.json).

No history was rewritten. Exact omitted-text searches and targeted credential
patterns found no matches in locally reachable historical blobs during the audit;
this is not a full privacy certification. Raw caches, annotation SQLite databases,
original experimental exports, internal review notes and publication approval
backups remain excluded. Commit author metadata is already in Git history and is
not removed by this cleanup.

Code remains under the repository LICENSE. Dataset components retain their scoped
CC BY 4.0 / Apache 2.0 terms, source revisions, attribution and modification notices.
Public preparation copies the published notice package. Publication code and
historical authorization configurations are preserved; no publication is performed.
