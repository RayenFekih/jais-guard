# Package layout

The package root contains only the public command entry points. Implementation is
organized by workflow, without changing model behavior or introducing a plugin framework.

```text
src/jais_guard/
  __init__.py
  __main__.py
  cli.py
  data/             # annotation, audit policy, candidate/test construction, legacy data
  baselines/        # untuned Jais inference, prompts, baseline metrics/reports
  benchmarks/       # Jev, local latency and reviewed evidence verification
  publication/      # frozen publisher and privacy-filtered release
  serving/          # HF Predictor and FastAPI factory
  workflows/        # public data preparation, checkpoint evaluation and orchestration
  finetuning/       # existing training/model/evaluation implementation
  vllm_support/     # existing native vLLM export, parity and serving implementation
```

`jaisguard` commands and existing `scripts/` entry points are unchanged. Docker's
`jais_guard.serving:create_app` factory is retained through the serving package.
The legacy `python -m jais_guard.data` command still inspects its original dataset.
The guarded frozen publisher remains `python -m jais_guard.publication`; the
public derivative publisher is now `python -m jais_guard.publication.public_release`.
No publication is performed by this reorganization.

## Import migration

These implementation imports moved. Callers inside the project, including test
monkeypatch targets, use the new modules. Old root-level implementation imports are
not retained as a second set of forwarding files.

| Previous module | Current module |
| --- | --- |
| `jais_guard.annotations` | `jais_guard.data.annotations` |
| `jais_guard.audit_policy` | `jais_guard.data.audit_policy` |
| `jais_guard.audit_migrations` | `jais_guard.data.audit_migrations` |
| `jais_guard.test_collection` | `jais_guard.data.test_collection` |
| `jais_guard.test_workflow` | `jais_guard.data.test_workflow` |
| `jais_guard.data` | `jais_guard.data.legacy` |
| `jais_guard.inference` | `jais_guard.baselines.inference` |
| `jais_guard.prompts` | `jais_guard.baselines.prompts` |
| `jais_guard.metrics` | `jais_guard.baselines.metrics` |
| `jais_guard.report` | `jais_guard.baselines.report` |
| `jais_guard.jev` | `jais_guard.benchmarks.jev` |
| `jais_guard.jev_reporting` | `jais_guard.benchmarks.jev_reporting` |
| `jais_guard.latency_benchmark` | `jais_guard.benchmarks.latency` |
| `jais_guard.evidence` | `jais_guard.benchmarks.evidence` |
| `jais_guard.publication` | `jais_guard.publication.frozen` |
| `jais_guard.public_release` | `jais_guard.publication.public_release` |
| `jais_guard.serving` | `jais_guard.serving.hf` |
| `jais_guard.public_workflow` | `jais_guard.workflows.public` |

## Path and integrity implications

Repository-relative path calculations in annotation, legacy data and publication
account for their additional package depth. New benchmark preparation records the
new prompt/service source paths. Existing frozen plans and recorded hashes are
never edited to accommodate moves. Exact historical replay still requires the
original revision and artifacts, as described in [reproducibility](reproducibility.md).
The original audit and dependency register remain records of their audited revision.

Training/evaluation algorithms, thresholds, class IDs, public dataset bytes and
original benchmark evidence are unchanged. Package discovery includes the new
subpackages. CPU tests, integrations, installed commands and wheel contents are
checked after the moves. GPU retraining and parity collection are not part of this
source organization change.
