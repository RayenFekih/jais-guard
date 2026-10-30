# Cleanup implementation and verification

Branch: `cleanup/public-workflows`, based on `ea1986d`. All work is committed locally for review. No push, merge, dataset publication, model retraining,
paid API invocation or original scientific-artifact edit was performed.

## Meaningful changes

- Strict configured v2 class counts, with exact original v1 defaults and regression coverage.
- Installed `jais_guard` package, console/module entry points and canonical publication imports.
- Python 3.11 Ruff target and corrected package metadata; uv remains the dependency manager.
- Portable CPU test boundary; private publication/export/cache checks are explicit integrations.
- Pinned eight-file derivative preparation with checksums, scoped notices, preserved label origins and separate fresh-training configuration.
- Explicit cache preparation, fresh training, checkpoint prediction/evaluation and saved-threshold HF serving commands.
- Existing Predictor, FP32 head, token policy, labels, GPU serialization and API response fields preserved.
- Single-worker HF container command and restricted build context.
- Separate observed vLLM environment receipt; no change to its installed packages.
- Seventeen-file reviewed public benchmark allowlist, timing projections, parity summaries and reproducible copy utility.
- Current workflow/reproduction/catalog documentation, historical status labels and a 179-file audit register.
- Empty `main.py`, `.env.example` and redundant data/docs/tests placeholders removed; final README rewrite deferred.

## Checks

| Check | Before | After |
| --- | --- | --- |
| CPU tests | 238 passed, 2 failed, 3 errors in combined baseline | 232 passed in a checkout snapshot without ignored/private artifacts; 31 integration checks deselected |
| Integration tests | Not separated | 30 passed, 1 skipped (missing original test parents) |
| New public workflow tests | Absent | 19 passed as part of the final CPU suite |
| Ruff lint | 2 E402 errors | Passed |
| Ruff formatting | 6 files differed | Passed after Python 3.11 formatting |
| uv lock | Original lock consistent but misleading legacy extra | Updated lock consistent; 183 packages resolved offline |
| Installation | No declared package build | Editable installation and wheel build passed; installed help/inspect/evidence commands passed outside checkout |
| Published release | Local verified receipt | Authenticated download of pinned revision; all eight files verified |
| Public adapter | Absent | 1,498 train / 250 validation / 400 test; train/validation integrity passed |
| Public evidence | Only v4 aggregate report tracked | All 17 allowlisted files verified; 16 workloads / 6,400 durations; both FP16 gate summaries passed |
| Historical quality | Cached reports | 13 local models / nine quality groups recomputed; Jev cache and input/label identity verified, without API calls |
| Immutable artifacts | 3,130 files hashed before edits | 3,129 substantive/existing files unchanged; only empty `data/.gitkeep` removed |

The private before/after inventories and comparison receipt are preserved locally
under ignored `artifacts/repository-cleanup/`, with working copies in
`/tmp/jaisguard-cleanup/`.
The original `results/`, `artifacts/` and `configs/` bytes all matched. Public copies
were also reproduced into a separate temporary directory, yielding the exact same
allowlist and destination hashes.

## Verified command sheet

See [public workflows](public-workflows.md) for prerequisites and complete examples.

```bash
uv sync --frozen --extra finetune --extra annotation
jaisguard --help
python -m jais_guard --help
jaisguard prepare-data --output /absolute/path/new-public-v1
jaisguard inspect --config /absolute/path/new-public-v1/training.json
jaisguard verify-evidence --directory /absolute/path/jais-guard/docs/benchmarks
python scripts/check_docs.py
```

Installation and all command parsers were verified. Data preparation was run both
against the exact staged release and against the authenticated remote revision.
The normal sync recipe is validated by the lock/build checks; it was not used to
reinstall or upgrade the user's existing environments.

The following interfaces are covered with CPU fixtures/fake models; their real
model execution requires the pinned cache and GPU and was not run during cleanup:

```bash
jaisguard cache-model
jaisguard train --config /absolute/path/new-public-v1/training.json --approach frozen --seed 42
jaisguard predict --checkpoint /absolute/path/run/best --text 'كيف حالك'
jaisguard evaluate --checkpoint /absolute/path/run/best --data /absolute/path/new-public-v1/test/test.jsonl --output /absolute/path/new-evaluation
jaisguard serve --checkpoint /absolute/path/run/best
```

`cache-model` has parser coverage; its actual backbone download was not invoked.
Training uses existing training code; fixture checks confirm CUDA refusal before
creating outputs. Synthetic API tests use inline handlers, matching existing tests,
because this sandbox cannot reliably wake cross-thread HTTP test transports.

## Remaining limitations

- HF dataset metadata reports **private=true**. Anonymous access returned 401; authorized login succeeded. No visibility change was made.
- Original private 1,500-row training data differs from the public 1,498-row derivative. Historical model results are not exactly reproducible from that derivative.
- Original test parents/corruptions are absent; full replay of 260 transformed request parents remains impossible here.
- Historical manifests lock original source hashes and absolute paths. Cleanup source must not be silently substituted; use the original revision and authentic artifacts.
- Checkpoints are available locally, not promised as a public download. New public-data training is a separate GPU experiment.
- Generic vLLM export of fresh checkpoints is not implemented; the verified historical export/parity protocol remains the supported vLLM path.
- Public aggregate quality does not include private row predictions. Local recomputation passed, but independent public row-level verification requires those excluded artifacts.
- GPU parity, real checkpoint readiness, CUDA container runtime and clean vLLM installation were not run.
- Existing MLflow/SQLAlchemy/Pandas deprecation warnings remain; they do not indicate failed checks.

## Final verification

The final CPU run used a copied checkout containing tracked/public files and the
cleanup additions, with ignored private artifacts excluded: **232 passed, 31
integration checks deselected**. The available integrations separately passed 30
checks and skipped the missing original-sidecar check. A clean wheel was installed
without dependencies into an isolated temporary environment and its package/CLI
imported with Python isolated mode outside the checkout. The editable installation
also passed all command help, public-data inspection and evidence verification
outside the checkout.

Ruff lint, format check, uv offline lock check and `git diff --check` passed.
The documentation checker verified local link targets without requesting external
URLs. Dataset references were additionally validated by authenticated remote access.

The complete Dockerfile build passed with 142 locked runtime packages. A final
source-only layer refresh supplies fixes made while dependency downloads ran; the
image is tagged `jaisguard-cleanup:verified`. Container checks passed for installed command help, the exact single-worker
Uvicorn factory command, the actionable missing-artifact startup failure, and the
read-only evidence verifier with networking disabled. Actual GPU model readiness
was not exercised. Final local documentation verification covered 76 targets.
A final prospective-public-tree scan found zero excluded-text or targeted credential
pattern matches; private integrity receipts remain ignored.
