# Repository cleanup audit and implementation plan

Audited base: `ea1986db76bc180783c0afefc8cffa319da9473b`. Local branch:
`cleanup/public-workflows`. This document records the approved audit and practical
implementation choices; see [verification results](cleanup-verification.md).
No remote modifications, retraining, paid calls or dataset publication are included.

## 1. Current repository audit

The original tree contained 179 tracked files. Inspection covered tracked Python
imports, command parsers, configuration and dataset structures, documentation,
local frozen exports, checkpoints, benchmark reports and publication receipts.
A reachable-history scan examined 498 Git objects. The appendix identifies every
original tracked filename and its cleanup action.

The initial README described an empty template despite implemented training,
evaluation and inference. `main.py` was empty. Docker called undeclared Gunicorn
against nonexistent `main:app` with four workers. Ruff targeted Python 3.14 despite
Python 3.11 in `.python-version`, project metadata and actual environments.
`pyproject.toml` had placeholder description text and a legacy vLLM 0.2.5 extra
unrelated to the actual tested 0.32.0 environment. The package lacked installation
configuration, and publication used a mixture of `src.jais_guard` and `jais_guard`.

`src/config.py` and `src/logger.py` support only `scripts/test_jais.py`, a generation
demo with a broken import root. `scripts/test_jais_vllm.py` is also a generation
demo, not classification serving. They remain labeled historical, avoiding
unnecessary relocation. Most meaningful implementation modules remain in place.

## 2. Supported workflows and dependencies

| Workflow | Entry points | Inputs and implementation dependencies |
| --- | --- | --- |
| Legacy data/baseline | `scripts/smoke_baseline.py`, `jais_guard.data` | `data/gibberish.jsonl`, `data/eval/smoke.jsonl`, baseline config, prompts, inference, metrics/report |
| Annotation | `apps/annotate.py`, `scripts/prepare_candidates.py` | Annotation DB, generation batches, source manifests, `annotations`, `audit_policy`, `audit_migrations` |
| Test preparation | `scripts/collect_test_draft.py`, `scripts/build_test400.py`, `scripts/prepare_test.py` | Test plan, candidate workspace, parent sidecars, `test_collection`, `test_workflow` |
| Historical training | `scripts/finetune.py` | Frozen configs and exports, fine-tuning model/data/evaluation/training/tracking, PEFT/MLflow |
| Historical evaluation | `scripts/evaluate_test.py`, `scripts/compare_versions.py` | Saved validation selections, thresholds, test freeze, checkpoints and cached predictions |
| Jev | `scripts/benchmark_jev.py` | Jev config, guarded paid-execution flag, response cache, `jev`/`jev_reporting` |
| HF benchmark/service | `scripts/benchmark_latency.py`, `scripts/run_http_latency.py`, `scripts/serve_jaisguard.py` | Locked manifest, 13 models, quality snapshot, `serving`, historical selection helpers |
| Native vLLM | `scripts/vllm_classifier.py`, `scripts/serve_vllm_classifier.py` | Export receipts, head/bias, static merge, FP16 adapter, parity, separate GPU environment |
| Publication | `public_release.py`, `publication.py`, `scripts/investigate_publication.py` | Public-v1 policy, private original exports, license/attribution package, approval evidence |
| Current public workflow | `jaisguard`, `python -m jais_guard` | Pinned derivative, adapter manifests, explicit checkpoint and saved threshold; implementation in `public_workflow` |

`public_release.py` imports `publication.py`; the original publisher cannot be
removed as dead code. Annotation policy/migrations and source-selection evidence
are also active dependencies. Several tests dynamically import scripts or helpers
from other test modules. The exported `modeling_jaisguard.py` intentionally gains
its upstream model implementation during export; its relative import is not an
unused or accidentally missing repository module.

Source hashes are dependencies too: benchmark manifests pin selected Python files.
Editing the loader or service invalidates old source locks. Preserve old files and
locks; replay with the original revision or create a new explicitly versioned plan.

The [dependency register](repository-dependencies.json) records AST import edges and
literal tracked-path references at the audited revision. Computed paths and source
locks are covered by the workflow table and artifact catalog; literal scanning is
not a claim of exhaustive runtime dependency discovery.

## 3. Problems by priority

- P0: Privacy-uncertain originals and internal provenance must not enter public evidence.
- P0: Ignored local artifacts make original training/checkpoint reproduction unavailable to a fresh public checkout.
- P1: V2 loader applies v1 class counts; fixed by explicit configured counts and unchanged v1 defaults.
- P1: Missing test parents prevent complete construction replay; synthetic unit coverage and explicit integration boundaries are required.
- P1: Broken Docker, misleading vLLM extra and stale workflow status obstruct practical use.
- P1: HF derivative is uploaded but currently private; authorized access is required.
- P2: Import/package setup and absolute historical paths limit portability.
- P2: Only the v4 aggregate report was publicly tracked; quality and supporting numerical evidence needed review.
- P3: Empty placeholders and historical navigation obscure supported workflows.

## 4. Final organization

```text
README.md                         # limited corrections; final rewrite deferred
pyproject.toml / uv.lock           # Python 3.11 HF package, uv retained
requirements-vllm-tested.txt       # observed separate environment receipt
src/jais_guard/
  cli.py / __main__.py             # small public workflow interface
  public_workflow.py               # pinned data adapter and checkpoint evaluation
  evidence.py                     # read-only public evidence verification
  [existing modules]
  finetuning/ / vllm_support/      # existing functioning organization retained
scripts/                          # compatible historical experiment entry points
apps/annotate.py
configs/                          # frozen bytes and paths retained
data/                             # reviewed tracked provenance retained; private data ignored
results/vllm_serving/jaisguard-v4/report/ # existing tracked report unchanged
docs/
  public-workflows.md
  reproducibility.md
  artifact-catalog.md
  repository-cleanup-plan.md
  cleanup-verification.md
  benchmarks/allowlist.json        # explicit reviewed public files only
  [original protocols/checkpoints, clearly labeled historical]
```

## 5. Actions and safe-change criteria

Keep core modules, data-generation evidence, source cards, licenses, frozen configs
and all historical numerical artifacts at their existing paths. Do not introduce
a general CLI dispatch framework or move files purely for appearance.

Remove only empty `main.py`, `.env.example`, `data/.gitkeep`, `docs/.gitkeep` and
`tests/.gitkeep`; replace their README/Docker references first. Keep `logs/.gitkeep`.
Correct the generation demo import path and label both demos as historical.
Consolidate current navigation through public-workflows, reproducibility and the
artifact catalog. Preserve historical document bodies with explicit status banners.

Canonicalize publication imports, install the package from `src/`, add tested thin
public commands and keep old scripts compatible. Test package import and CLI help
outside the checkout. Frozen path configurations remain untouched; new public
configuration paths resolve independently of the working directory.

Prepare public copies only from an explicit reviewed allowlist. Record byte-copy
or projection transformations and source hashes. Never sanitize originals in place.
Public evidence excludes texts, private source IDs, original training data,
annotation notes and weights. Keep original metric JSON and benchmark CSV unchanged.

## 6. Required path/import/configuration updates

- Install `jais_guard` with setuptools using the existing uv workflow.
- Replace `src.jais_guard` imports in publication tests/investigation with `jais_guard`.
- Set Ruff to `py311`; declare plotting as runtime because evaluation imports it.
- Remove the misleading vLLM extra and its conflicts; record the separate installed stack without changing it.
- Default CPU pytest excludes explicitly marked integration checks; cached model architecture is an integration prerequisite.
- Public data preparation creates a separate config, exports and notice copies; no historical config is edited.
- HF FastAPI accepts an explicit checkpoint, keeping existing Predictor semantics, GPU serialization and response schema.
- Docker uses the installed app factory, explicit artifact mounts and one worker.

## 7. Reproducibility, privacy and publication

Original experiments: 1,500 train + 250 validation + 400 test = 2,150. Public
release: 1,498 + 250 + 400 = 2,148. Zero-based original training positions 277/486
are privacy exclusions, both human-reviewed VALID Aya inputs. Retained rows keep
texts, labels and order. Public train has 708 VALID/790 GIBBERISH and 1,365
human-reviewed/133 audited-provisional labels. Original metrics do not describe
training on this derivative.

The verified local publication receipt and authenticated remote download identify
HF commit `c3e739a396dcf4ae982633f19dc24555dab837d4`. All eight release hashes match.
Remote metadata says private=true, gated=false; anonymous access returned 401.
No repository visibility change or re-publication is part of cleanup.

Exact omitted texts and targeted credential patterns were absent from locally
reachable historical blobs. This is a scoped scan, not broad privacy clearance.
Keep original exports, raw Aya metadata, annotation databases, reviewer notes and
internal evidence private. Preserve MASSIVE CC BY 4.0 and Aya/TyDi Apache 2.0
attribution, revision and modification notices alongside the project contribution
license scope. Code licensing does not supersede dataset component terms.

Missing test parents/corruptions prevent replay for 260 transformed requests;
80 passthrough originals are inferable. Test[319] is contained in test[204]. The
benchmark was examined and informed iteration two. No claim of full parent-level
independence or untouched final evaluation is justified.

## 8. Testing and verification

Before: 238 passed, 2 failed, 3 errors; Ruff had two E402 errors; six formatting
differences. Failures came from missing private parents and hard-coded v1 counts.

Verify corrected v1/v2 counts, malformed counts, checksums and label errors; use
synthetic construction parents and mocked prediction/model transports. Run default
CPU tests, available explicit integrations, lint, formatting, installation, command
help outside the checkout, documentation links, exact release preparation and
immutable before/after comparison. Public evidence must reproduce HTTP aggregates
from timing observations. Recompute cached quality locally without rerunning models.

GPU-only: real checkpoint inference, model readiness, trained head/bias integrity,
LoRA static merge, FP16 parity and concurrent response parity. These are not inferred
from CPU tests. No retraining, paid Jev API requests or new datasets are authorized.

## 9. Implementation order

1. Create branch and private immutable inventories.
2. Fix count validation and establish portable test boundaries.
3. Install the package and add public-data/checkpoint commands.
4. Repair serving/Docker without changing prediction semantics.
5. Review public evidence and preserve its source hashes.
6. Correct current navigation and label historical protocols.
7. Run verification, compare immutable artifacts and document limitations.

The implementation report records outcomes and any deviations. No decision is
required for local completion; public HF visibility remains a separate owner action.

## Appendix: original tracked-file action and dependency register

The following register covers every filename at the audited base. “Retain” means
preserve its original role and path; historical numerical/data/configuration bytes
are immutable. Inbound Python references are recorded where applicable. For retained
data, path references are checked together with the consuming scripts/tests.

| Original file | Action / verification dependency |
| --- | --- |
| `.dockerignore` | Replace broken deployment configuration; build and validate one-worker startup. |
| `.env.example` | Remove empty template placeholder; references replaced; no scientific content. |
| `.gitignore` | Retain; verify existing role and references. |
| `.python-version` | Retain; verify existing role and references. |
| `.streamlit/config.toml` | Retain; verify existing role and references. |
| `Dockerfile` | Replace broken deployment configuration; build and validate one-worker startup. |
| `LICENSE` | Retain; verify existing role and references. |
| `PROJECT_CONTEXT.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `README.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `apps/annotate.py` | Retain; verify existing role and references. |
| `configs/baseline.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/finetuning-iteration2.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/finetuning.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/jev_benchmark.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/prompt_baseline.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/publication-public-v1.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/publication.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/publication.proposed.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `configs/test400_plan.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/.gitkeep` | Remove empty template placeholder; references replaced; no scientific content. |
| `data/eval/smoke.jsonl` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/gibberish.jsonl` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/README.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/audit.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/candidates.jsonl` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/checkpoint.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/easy-invented-1.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/easy-invented-2.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/hard-1.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/hard-2.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/hard-3.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-1.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-2.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-3.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-4.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-5.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/medium-6.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/valid-1.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/generation/valid-2.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/modified_only_audit.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/pilot_audit.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/milestone300_v1/source_selection.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/audit.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/candidates.jsonl` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/invented-a.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/invented-b.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-everyday.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-home.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-learning.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-shopping.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-tech.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/semantic-work.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/valid-dialects-a.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/valid-dialects-b.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/generation/valid-msa.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/sources/aya_README.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/sources/manifest.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/sources/massive_LICENSE.txt` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/sources/massive_README.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/pilot_v1/sources/tydi_README.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/replacements_v1/README.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/replacements_v1/audit.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/replacements_v1/candidates.jsonl` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `data/replacements_v1/source_selection.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `docs/.gitkeep` | Remove empty template placeholder; references replaced; no scientific content. |
| `docs/annotation.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/audit-policy-v2.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/baseline.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/dataset.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/evaluation-phase.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/evaluation-source-feasibility-20261029.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/final-test-evaluation.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/finetuning.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/huggingface-public-release-v1.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/huggingface-publication-review.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/huggingface-publication.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/jev-benchmark.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/latency-serving.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/realism-replacement-checkpoint.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/test400-candidate-review-v2.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/test400-collection-20261029.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/test400-qualitative-audit-20261029.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/training-700-checkpoint.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/validation-200-checkpoint.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/version-comparison.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/vllm-fp16-benchmark.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `docs/vllm-serving.md` | Retain evidence; correct navigation/status; check links and preserve numerical content. |
| `logs/.gitkeep` | Retain; verify existing role and references. |
| `main.py` | Remove empty template placeholder; references replaced; no scientific content. |
| `pyproject.toml` | Repair packaging/environment metadata; lock check, installation, Python 3.11 lint. |
| `results/vllm_serving/jaisguard-v4/report/comparison.csv` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/comparison.md` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/latency-throughput.png` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/latency-throughput.svg` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/metrics.json` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/startup-memory.png` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `results/vllm_serving/jaisguard-v4/report/startup-memory.svg` | Retain bytes/path; verify immutable hashes and consuming workflow references. |
| `ruff.toml` | Repair packaging/environment metadata; lock check, installation, Python 3.11 lint. |
| `scripts/benchmark_jev.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/benchmark_latency.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/build_test400.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/collect_test_draft.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/compare_versions.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/evaluate_test.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/finetune.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/investigate_publication.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/prepare_candidates.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/prepare_test.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/prepare_vllm_fp16_experiment.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/repair_vllm_serialization.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/run_http_latency.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/serve_jaisguard.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/serve_vllm_classifier.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/smoke_baseline.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/test_jais.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/test_jais_vllm.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `scripts/vllm_classifier.py` | Retain compatible entry point; mark demos historical; check dynamic imports, help and docs. |
| `src/config.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/test_jais.py`. |
| `src/jais_guard/__init__.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/annotations.py` | Retain path; regression tests and source-lock disclosure. Imports: `apps/annotate.py`, `scripts/benchmark_jev.py`, `scripts/build_test400.py`, `scripts/collect_test_draft.py`, `scripts/prepare_candidates.py`, `scripts/prepare_test.py`, `src/jais_guard/audit_migrations.py`, `src/jais_guard/finetuning/test_baseline.py`, `src/jais_guard/finetuning/version_comparison.py`, `tests/test_annotation_app.py`, `tests/test_annotations.py`, `tests/test_candidate_expansion.py`, `tests/test_evaluation_phase.py`, `tests/test_milestone300.py`, `tests/test_test_collection.py`. |
| `src/jais_guard/audit_migrations.py` | Retain path; regression tests and source-lock disclosure. Imports: `src/jais_guard/annotations.py`. |
| `src/jais_guard/audit_policy.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/prepare_candidates.py`, `src/jais_guard/annotations.py`, `src/jais_guard/audit_migrations.py`, `tests/test_audit_migrations.py`, `tests/test_audit_policy_v2.py`. |
| `src/jais_guard/data.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/smoke_baseline.py`, `tests/test_data.py`. |
| `src/jais_guard/finetuning/__init__.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/finetuning/data.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`, `src/jais_guard/vllm_support/parity.py`, `src/jais_guard/vllm_support/server.py`, `tests/test_final_test.py`, `tests/test_finetuning.py`. |
| `src/jais_guard/finetuning/evaluation.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/compare_versions.py`, `scripts/evaluate_test.py`, `scripts/finetune.py`, `src/jais_guard/latency_benchmark.py`, `tests/test_final_baseline.py`, `tests/test_final_test.py`, `tests/test_finetuning.py`, `tests/test_version_comparison.py`. |
| `src/jais_guard/finetuning/final_test.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/compare_versions.py`, `scripts/evaluate_test.py`, `src/jais_guard/latency_benchmark.py`. |
| `src/jais_guard/finetuning/model.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`, `src/jais_guard/finetuning/tracking.py`, `src/jais_guard/serving.py`, `src/jais_guard/vllm_support/parity.py`, `tests/test_final_test.py`, `tests/test_finetuning.py`. |
| `src/jais_guard/finetuning/smoke.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`. |
| `src/jais_guard/finetuning/test_baseline.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/evaluate_test.py`, `src/jais_guard/serving.py`. |
| `src/jais_guard/finetuning/tracking.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/compare_versions.py`, `scripts/evaluate_test.py`, `scripts/finetune.py`, `tests/test_final_test.py`, `tests/test_finetuning.py`. |
| `src/jais_guard/finetuning/training.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`, `tests/test_finetuning.py`. |
| `src/jais_guard/finetuning/version_comparison.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/compare_versions.py`, `src/jais_guard/latency_benchmark.py`, `src/jais_guard/serving.py`. |
| `src/jais_guard/inference.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`, `scripts/smoke_baseline.py`, `src/jais_guard/finetuning/test_baseline.py`, `src/jais_guard/serving.py`, `tests/test_evaluation_phase.py`. |
| `src/jais_guard/jev.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/jev_reporting.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/benchmark_jev.py`. |
| `src/jais_guard/latency_benchmark.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/benchmark_latency.py`, `src/jais_guard/vllm_support/artifacts.py`, `src/jais_guard/vllm_support/benchmark.py`, `src/jais_guard/vllm_support/experiment_upgrade.py`, `src/jais_guard/vllm_support/parity.py`, `src/jais_guard/vllm_support/serialization_repair.py`. |
| `src/jais_guard/metrics.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/smoke_baseline.py`, `tests/test_baseline.py`. |
| `src/jais_guard/prompts.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/finetune.py`, `src/jais_guard/finetuning/test_baseline.py`, `src/jais_guard/finetuning/version_comparison.py`, `tests/test_baseline.py`, `tests/test_evaluation_phase.py`, `tests/test_final_baseline.py`. |
| `src/jais_guard/public_release.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/publication.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/report.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/smoke_baseline.py`, `tests/test_report.py`. |
| `src/jais_guard/serving.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/benchmark_latency.py`, `scripts/run_http_latency.py`, `scripts/serve_jaisguard.py`, `src/jais_guard/latency_benchmark.py`, `src/jais_guard/vllm_support/artifacts.py`, `src/jais_guard/vllm_support/benchmark.py`, `src/jais_guard/vllm_support/experiment_upgrade.py`, `src/jais_guard/vllm_support/fp16.py`, `src/jais_guard/vllm_support/parity.py`, `src/jais_guard/vllm_support/runtime.py`, `src/jais_guard/vllm_support/serialization_repair.py`, `src/jais_guard/vllm_support/server.py`, `tests/test_vllm_serving.py`. |
| `src/jais_guard/test_collection.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/build_test400.py`, `tests/test_test_collection.py`. |
| `src/jais_guard/test_workflow.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/build_test400.py`, `scripts/collect_test_draft.py`, `scripts/evaluate_test.py`, `scripts/prepare_test.py`, `src/jais_guard/finetuning/final_test.py`, `src/jais_guard/finetuning/test_baseline.py`, `src/jais_guard/finetuning/version_comparison.py`, `src/jais_guard/latency_benchmark.py`, `src/jais_guard/serving.py`, `src/jais_guard/vllm_support/parity.py`, `tests/test_final_baseline.py`, `tests/test_final_test.py`, `tests/test_latency_serving.py`, `tests/test_version_comparison.py`. |
| `src/jais_guard/vllm_support/__init__.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/vllm_support/artifacts.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/serve_vllm_classifier.py`, `scripts/vllm_classifier.py`. |
| `src/jais_guard/vllm_support/benchmark.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/vllm_classifier.py`. |
| `src/jais_guard/vllm_support/experiment_upgrade.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/prepare_vllm_fp16_experiment.py`. |
| `src/jais_guard/vllm_support/fp16.py` | Retain path; regression tests and source-lock disclosure. Imports: `tests/test_vllm_serving.py`. |
| `src/jais_guard/vllm_support/modeling_jaisguard.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/vllm_support/parity.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/vllm_classifier.py`. |
| `src/jais_guard/vllm_support/runtime.py` | Retain path; regression tests and source-lock disclosure. |
| `src/jais_guard/vllm_support/serialization_repair.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/repair_vllm_serialization.py`. |
| `src/jais_guard/vllm_support/server.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/serve_vllm_classifier.py`. |
| `src/logger.py` | Retain path; regression tests and source-lock disclosure. Imports: `scripts/test_jais.py`. |
| `tests/.gitkeep` | Remove empty template placeholder; references replaced; no scientific content. |
| `tests/test_annotation_app.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_annotations.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_audit_migrations.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_audit_policy_v2.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_baseline.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_candidate_expansion.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_data.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_evaluation_phase.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_final_baseline.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_final_test.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_finetuning.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_jev.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_latency_serving.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_milestone300.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_provisional_annotations.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_public_release.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_publication.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_report.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_test_collection.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_version_comparison.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `tests/test_vllm_serving.py` | Retain coverage; canonical imports/integration markers as required; run CPU suite. |
| `uv.lock` | Repair packaging/environment metadata; lock check, installation, Python 3.11 lint. |
