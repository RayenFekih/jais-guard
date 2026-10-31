# Current public workflows

These commands use Python 3.11 and uv. They do not require private experimental
exports. Run installation in the checkout; installed `jaisguard` commands can run
from any directory. All output paths below are examples, and output directories
must be new. Training and real model inference require appropriate hardware and
cached model weights; the cleanup verification did not execute these GPU stages.

## Install and verify

```bash
uv sync --frozen --extra finetune --extra annotation
uv run --no-sync jaisguard --help
uv run --no-sync pytest -q
uv run --no-sync pytest -m integration -q -rs
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The default suite excludes `integration`. The explicit integration command runs
available private-artifact and cached-architecture checks and reports unavailable
inputs as skips. No tests download model weights or reconstruct private evidence.
The distribution retains its existing `jais-fine-tune` name; the installed Python
package is `jais_guard`. See [package layout](package-layout.md) for the grouped implementation. The `finetune` extra supplies MLflow and PEFT. Matplotlib
is a runtime dependency because evaluation and historical serving import plotting
helpers. Keep the existing vLLM environment separate.

## Load the published dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "RayenFekih/arabic-gibberish-detection",
    revision="c3e739a396dcf4ae982633f19dc24555dab837d4",
)
assert {split: len(rows) for split, rows in dataset.items()} == {
    "train": 1498,
    "validation": 250,
    "test": 400,
}
```

The derivative is intended for public use, but the HF repository was verified as
**private** during cleanup. Anonymous access returned HTTP 401. An authorized HF
login (`hf auth login`) or `HF_TOKEN` is currently required; the example above uses
the existing login automatically. No visibility change is included in cleanup.

This public derivative has 2,148 examples. Two privacy-uncertain VALID training
records were omitted from the frozen 2,150-example experiment collection. Existing
metrics describe those original experiments, not training on this derivative.
Labels are VALID=0 and GIBBERISH=1. Train contains 1,365 human-reviewed and 133
audited-provisional labels. Validation and test are human-reviewed.

Prepare checksum-verified exports and a separate fresh-training configuration:

```bash
jaisguard prepare-data --output /absolute/path/public-v1
# Offline alternative: exact eight-file public release already downloaded.
jaisguard prepare-data --source /absolute/path/published-release --output /absolute/path/public-offline
jaisguard inspect --config /absolute/path/public-v1/training.json
```

Preparation pins and verifies all eight published files, including licenses and
attribution. It preserves public texts, ordering and label origins, creates public
row identifiers, and copies the published notice package into `publication/`.
Those row identifiers do not establish independence of underlying source families.
Generated paths are absolute: regenerate a new adapter directory after relocation.
The public export and run directories must stay separate from historical exports.
No source reconstruction or human annotation is invented.

## Cache the backbone and train a fresh model

```bash
jaisguard cache-model
jaisguard train --config /absolute/path/public-v1/training.json --approach frozen --seed 42
jaisguard train --config /absolute/path/public-v1/training.json --approach lora --seed 42
```

`cache-model` explicitly downloads the pinned `IFM/Jais-3-0.9B` revision
`e51941b37091287fee4934e3f8a799e0d3fef37f`, including its remote model implementation.
Review that trusted code before execution. Training loaders then use local files
only. The two training examples above are alternatives; run sequentially. The
configuration retains the existing optimizer, pooling, precision, LoRA and
validation-selection policies. Seeds 42, 43 and 44 are supported. MLflow uses a
new local SQLite database, and every run has a separate output directory.

Training is an explicit expensive command. It was not run during cleanup. The
public historical test was examined and informed iteration two; evaluating a new
model on it does not provide a new independent generalization estimate.

## Predict, evaluate and serve an available checkpoint

A checkpoint must contain `classifier.json`, `head.safetensors`, `tokenizer/`, and
`adapter/` for LoRA. Its pinned backbone must be cached. Existing historical
checkpoints remain private/local; this repository does not promise downloadable
trained weights. New training writes a `best/` checkpoint and sibling
`threshold.json`.

```bash
jaisguard predict --checkpoint /absolute/path/run/best --text 'كيف يمكنني تعلم العربية؟'
jaisguard evaluate --checkpoint /absolute/path/run/best --data /absolute/path/public-v1/test/test.jsonl --output /absolute/path/new-evaluation
jaisguard serve --checkpoint /absolute/path/run/best --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"كيف يمكنني تعلم العربية؟"}'
```

The default operating point loads the saved validation-selected threshold; it does
not search evaluation data. Add `--operating-point default` for the explicit 0.5
operating point. Missing saved thresholds fail rather than silently substituting a
value. Decisions use `p_gibberish >= threshold`; empty, unsupported-script and
over-budget inputs remain errors. Prediction/evaluation also accept `--device cpu`
for debugging, which can be slow. Serving retains the verified CUDA predictor,
FP32 classification head, GPU lock and one resident model worker.

## Container

```bash
docker build -t jaisguard:local .
docker run --rm --gpus all -p 8000:8000 -v /absolute/path/run:/model:ro -v /absolute/path/huggingface-cache:/cache:ro -e HF_HOME=/cache -e JAISGUARD_CHECKPOINT=/model/best jaisguard:local
```

The cache mount must include the pinned backbone and its trusted-code cache, with
compatible read permissions. If dynamic-module caching needs writes, mount a
separate writable `HF_MODULES_CACHE` directory. The image installs the locked HF
stack and the fine-tuning extra, starts the existing FastAPI app through a Uvicorn
factory, binds port 8000, and uses one worker. Missing artifacts produce an explicit
startup error. Historical manifests can instead be selected with
`JAISGUARD_DIRECTORY`; their original source hashes must still match.

## Native vLLM backend

The tested classification stack is Python 3.11, vLLM 0.32.0, Torch
2.13.0+cu132 and Transformers 5.19.0. The HF environment uses Torch 2.14.1 and
Transformers 5.20.2. Never `uv sync` the HF project into `.venv-vllm`.
The complete observed package inventory is
[requirements-vllm-tested.txt](../requirements-vllm-tested.txt). It is an environment
receipt, not a claim that a clean GPU installation was revalidated. To provision a
separate candidate environment using the same dependency manager:

```bash
uv venv --python 3.11 .venv-vllm-reproduction
uv pip install --python .venv-vllm-reproduction/bin/python -r requirements-vllm-tested.txt
```

This requires compatible CUDA drivers and available wheel distributions. Run the
original export/parity checks before using it for performance claims. The existing
`.venv-vllm` was neither reinstalled nor upgraded. The removed
`vllm-legacy==0.2.5` extra did not represent this tested environment.

For an existing export with matching source locks, from the checkout:

```bash
.venv-vllm/bin/python scripts/serve_vllm_classifier.py --root results/vllm_serving/jaisguard-v4 --model v2-lora-44 --backend vllm
```

The v4 historical root is tied to the original source revision. Cleanup changes
source hashes, so use an isolated checkout of `ea1986d` and authentic artifact
paths to replay it; do not bypass gates or edit its plan. See
[vLLM protocol](vllm-serving.md) and [FP16 protocol](vllm-fp16-benchmark.md).
Fresh public-data checkpoints use the HF workflow above. A general vLLM export
workflow for fresh checkpoints is not provided: the verified exporter is tied to
validation-selected historical manifests. Extending it requires new export
receipts and the same GPU parity gates; no unverified generic conversion is claimed.

## Inspect existing evidence

```bash
jaisguard verify-evidence --directory /absolute/path/jais-guard/docs/benchmarks
```

See the [artifact catalog](artifact-catalog.md), [reproduction limits](reproducibility.md)
and [cleanup audit](repository-cleanup-plan.md). Historical scripts retain their
interfaces; no large generic command-dispatch framework was introduced.
