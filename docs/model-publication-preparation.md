# JaisGuard model publication preparation

Prepared locally on `cleanup/public-workflows`. Nothing was uploaded, no model was
retrained, no paid API was called, and no historical artifacts were modified.

## Staged repositories and exact source checkpoints

| Staged directory | Historical model | Source checkpoint under `results/fine_tuning_iteration2/` | Epoch | Saved threshold |
| --- | --- | --- | ---: | ---: |
| `artifacts/huggingface-models/JaisGuard-LoRA-0.9B` | v2 LoRA seed 44 | `0d227aa2e3954a27837687aef10ddd31/lora-44/best` | 4 | 0.942931056022644 |
| `artifacts/huggingface-models/JaisGuard-Frozen-0.9B` | v2 frozen seed 43 | `51ce252a2cbb4e6e98d487bf2b9f0847/frozen-43/best` | 5 | 0.9563547372817993 |

These are the selected checkpoints in the historical v4 FP16 parity and HTTP
serving plan, not newly selected models. Validation losses are respectively
0.17111791670322418 and 0.22949336469173431; complete exact metadata is in the
packages. Both use the first minimum validation-loss epoch. Threshold selection
was previously locked on validation; the separate default threshold remains 0.5.

Artifact lineage: `results/vllm_serving/jaisguard-v4/plan.json`, its per-model
`exports/*/export.json`, per-model `parity/*/gate.json`, and
`results/version_comparison/jaisguard-v1-v2-v1/benchmark/metrics.json`.
Their sealed identities were checked. Checkpoint metadata, history, dataset hashes,
head weight/bias, adapters, tokenizer and threshold bytes matched the locked entries.
The actual training configuration is recorded in each checkpoint's `classifier.json`;
`configs/finetuning-iteration2.json` remains frozen and unchanged.

Training records name git revision `fab35805150d3808f1dea686d809222aa0119ca2`
**with uncommitted changes**. That commit is not a complete clean source snapshot.
`ea1986db76bc180783c0afefc8cffa319da9473b` is the preserved historical repository
reference, not a claim that all training occurred from that clean commit. A separate
read-only source snapshot is available locally at
`/tmp/jaisguard-model-publication/original-source`. No historical locks were edited.

Some original absolute source paths were relocated during cleanup. Their historical
source-lock validators are not relabeled as current-code passes. Instead, original
non-code artifact bytes were checked independently and real inference reproduced
all matching saved outputs exactly. The refactored checkpoint loader itself loaded
both models successfully, so no fallback was needed to obtain model predictions.

## Architecture and training data

Pinned base: `IFM/Jais-3-0.9B` revision
`e51941b37091287fee4934e3f8a799e0d3fef37f`. The official Hub API now redirects this
identifier to `inception42/Jais-3-0.9B` at the same revision. The original identifier
is retained; the redirect is disclosed, not silently substituted with latest weights.
Cached base weights/config match historical export hashes; upstream modeling and
configuration files match the copied historical export code.

The architecture uses a 1536-dimensional final-nonpadding hidden representation,
right padding, no chat prompt and no silent truncation beyond the 64-token limit.
The FP32 linear head has 3,074 parameters, including the original trained bias.
Labels remain VALID=0 and GIBBERISH=1; classification uses `p_gibberish >= threshold`.
LoRA uses rank 8, alpha 16, dropout 0.05 and q_proj/v_proj across 28 layers. Its
unmerged adapter is preserved; the frozen model includes only the trained head.

Original training: **1,500 rows**, 710 VALID / 790 GIBBERISH, with 1,367 human-reviewed
and 133 provisional labels admitted through the audited policy. Validation: 250
human-reviewed rows. The public derivative is **1,498 train + 250 validation + 400
test = 2,148**, and was **not** used for these checkpoints. The two removed VALID
Aya training narratives had privacy uncertainty. Their original use in training is
explicitly disclosed; no memorization or extraction clearance is claimed.
The dataset was private when last checked; authorized access is required.

## Metrics: individual checkpoints versus experiments

All rows below use the previously examined constructed test benchmark of 400 inputs
(188 VALID / 212 GIBBERISH). Original quality evaluation used **BF16, batch size 2**.
Numbers are not recomputed from FP16 or serving observations.

| Individual checkpoint | Operating point | Accuracy | Macro F1 | GIBBERISH recall | VALID rejection |
| --- | --- | ---: | ---: | ---: | ---: |
| LoRA seed 44 | default 0.5 | 0.912500 | 0.912038 | 0.929245 | 0.106383 |
| LoRA seed 44 | validation locked | 0.922500 | 0.922441 | 0.896226 | 0.047872 |
| Frozen seed 43 | default 0.5 | 0.880000 | 0.879020 | 0.915094 | 0.159574 |
| Frozen seed 43 | validation locked | 0.815000 | 0.812297 | 0.655660 | 0.005319 |

Three-seed experiment means ± sample SD, seeds 42/43/44, same test inputs:

| Experiment | Operating point | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| LoRA v2 | default | 0.896667 ± 0.017736 | 0.895453 ± 0.018730 |
| LoRA v2 | validation locked | 0.910000 ± 0.010897 | 0.909960 ± 0.010877 |
| Frozen v2 | default | 0.879167 ± 0.003819 | 0.878114 ± 0.003553 |
| Frozen v2 | validation locked | 0.827500 ± 0.019526 | 0.825631 ± 0.020333 |

The packages' `metrics.json` contains full precision source values, both scopes and
confusion counts. Individual cached predictions were independently rescored and
matched archived metrics. Neither model was selected using its test score.

## Real inference verification

Hardware: RTX 4050 Laptop GPU (6 GB), Torch 2.14.1+cu130, Transformers 5.20.2,
PEFT 0.21.2, Python 3.11. GPU access required execution outside the restricted sandbox.
Only one model was resident per process. BF16/FP16 backbones retained FP32 heads.

- Refactored code: both models × BF16/FP16, 658 inputs per run.
- Standalone packages copied to a fresh directory: both models × BF16/FP16, 658 inputs per run.
- Inputs: validation250 + historical test400 + eight saved probes, with identical token IDs/order.
- Every same-precision comparison had **maximum and mean probability difference 0.0** and zero default/locked classification mismatches.
- Both standalone models additionally reproduced all **400 BF16 batch-two quality probabilities exactly**, including both threshold decisions.
- Isolated Python execution confirmed no `jais_guard` training-repository module was imported by the standalone API.
- The exact documented example commands ran successfully on real GPU hardware.

Reduced-precision batching matters. Batch-one serving observations on the same 400
inputs differ slightly from the historical batch-two quality table:

| Standalone runtime | Default accuracy | Locked accuracy |
| --- | ---: | ---: |
| LoRA BF16 batch 1 | 0.912500 | 0.925000 |
| LoRA FP16 batch 1 | 0.915000 | 0.925000 |
| Frozen BF16 batch 1 | 0.885000 | 0.815000 |
| Frozen FP16 batch 1 | 0.880000 | 0.815000 |

These differences are disclosed, not treated as failures of the matching reference
conditions. Historical BF16 batch-two and serving batch-one outputs each reproduced
exactly. Existing metrics and thresholds remain unchanged. The single-input API
makes no claim to return the same rounded probability as padded batched inference.

Against matching **historical** vLLM FP16 observations, the new package outputs had:

| Model | Max absolute probability difference | Mean difference | Default/locked mismatches |
| --- | ---: | ---: | ---: |
| Frozen seed 43 | 0.00748598575592041 | 0.0004821549827004682 | 0 / 0 |
| LoRA seed 44 | 0.012874335050582886 | 0.00037334483295069546 | 0 / 0 |

Both satisfy the original max/mean 0.02/0.005 tolerances. No vLLM engine or HTTP
benchmark was rerun. These compact packages are **not native vLLM exports** and do
not claim AutoModelForSequenceClassification, pipeline or Inference Provider support.
The old native exports remain separate full-backbone artifacts.

Upstream code emits `cache_position` docstring diagnostics and reports the unused
language-model head when loading the backbone. Those upstream files were not edited;
loading checks accept only the expected `lm_head.*` extra and inference passed.

## Package contents and supported commands

Each package includes a standalone `jaisguard_model.py`, `example.py`, FP32
`head.safetensors`, copied tokenizer files, portable `config.json`, hyperparameters,
metrics, provenance, validation receipts, pinned requirements, MIT/Apache notices,
comprehensive model card and SHA-256 manifest. LoRA additionally includes its exact
adapter config and safetensors. Optimizer state, training data, reviewer notes,
MLflow databases, private paths and base weights are excluded.

Approximate sizes: LoRA 10 MB; frozen 5 MB. These are ignored local staging trees,
not Git-tracked model binaries. Private validation inputs/receipts are under
`artifacts/model-publication-review/`. Public receipts contain aggregate validation,
not input texts, private candidate IDs or per-record predictions.

From the project root, using its tested environment and existing pinned base cache:

```bash
.venv/bin/python artifacts/huggingface-models/JaisGuard-LoRA-0.9B/example.py --local-files-only --text 'كيف يمكنني تعلم اللغة العربية؟'
.venv/bin/python artifacts/huggingface-models/JaisGuard-Frozen-0.9B/example.py --local-files-only --precision fp16 --text 'ضصثقفغعهخحجشس'
```

From either downloaded model directory, independent of the project checkout:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python example.py --text 'كيف يمكنني تعلم اللغة العربية؟'
```

Use `--operating-point default` for 0.5; otherwise the saved locked threshold is
used. `--precision historical` selects the historical hardware policy; explicit
`bf16` and `fp16` are validated CUDA modes. `--device cpu` is an unbenchmarked FP32
debugging path. Requirements were checked against the executing environment; a
second installation of all GPU dependencies was not performed.

Reproduction helpers (new output paths required):

```bash
.venv/bin/python scripts/prepare_model_publication.py --output /absolute/path/new-staging --base-license-review artifacts/model-publication-review/base-license-review.json
.venv/bin/python scripts/validate_model_publication.py --model v2-lora-44 --precision bf16 --package /absolute/path/new-staging/JaisGuard-LoRA-0.9B --output /absolute/path/new-lora-bf16.json
.venv/bin/python scripts/validate_model_publication.py --model v2-frozen-43 --precision bf16 --quality-batch2 --package /absolute/path/new-staging/JaisGuard-Frozen-0.9B --output /absolute/path/new-frozen-batch2.json
```

Preparation initially marks validation pending. The helper's optional
`--validation-reviews` and `--validated-packages` arguments attach completed receipts
only if the inference-critical bytes match the GPU-tested copies. It does not
perform publication or grant approval. The `--quality-batch2` path exercises the
underlying package model with the historical batch collation; the public `predict`
API intentionally remains single-input.

## Integrity and licensing review

The pinned publisher metadata declares **Apache-2.0**. Its inventory contains no
separate LICENSE or NOTICE file. Full Apache license text and upstream credits are
retained for copied tokenizer assets and the separately downloaded base. The source
headers credit the Qwen team, Alibaba Group and Hugging Face. No upstream backbone
or modeling code is repackaged, avoiding unnecessary duplication.

Project loader/classifier component licensing follows the repository's existing
MIT license. The model cards clearly scope this separately from Apache-licensed
upstream components and the MASSIVE/Aya/TyDi training-source terms. Dataset licensing
does not automatically relicense model components. Final publication approval also
covers the staged component licensing and disclosed training-privacy limitation;
no upload approval is inferred from successful numerical checks.

Primary evidence: the [pinned publisher card](https://huggingface.co/IFM/Jais-3-0.9B/blob/e51941b37091287fee4934e3f8a799e0d3fef37f/README.md),
[redirected publisher repository](https://huggingface.co/inception42/Jais-3-0.9B),
and [Apache redistribution terms](https://www.apache.org/licenses/LICENSE-2.0).
Review used the actual pinned Hub API metadata and cached/file hashes, not a current
model-card assumption. The MIT text is copied unchanged from the project LICENSE.

All 44 checkpoint/reference files captured before preparation matched afterward;
base weights/config and copied upstream code matched the historical exports.
Each staged manifest checks every intended package file. Card YAML, dependency
pins, local links and scans for private paths/credential patterns passed. New
regression tests cover pooling/bias equivalence, thresholds, integrity failures,
label incompatibility and absence of training-repository imports.

## Private publication commands — only after owner approval

These commands have **not** been executed. They create new **private model**
repositories and refuse preexisting targets rather than overwriting anything.
Authenticate separately with an authorized account, inspect both cards and manifests,
and run only after actual approval:

```bash
hf auth login
.venv/bin/python - <<'PY'
import importlib.util
import json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
for name in ("JaisGuard-LoRA-0.9B", "JaisGuard-Frozen-0.9B"):
    folder = Path("artifacts/huggingface-models") / name
    spec = importlib.util.spec_from_file_location("jaisguard_model", folder / "jaisguard_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify_package(folder)
    validation = json.loads((folder / "validation.json").read_text())
    assert validation["status"] == "passed real GPU inference"
    repo_id = f"RayenFekih/{name}"
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
    info = api.model_info(repo_id)
    assert info.private
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=[*manifest["files"], "manifest.json"],
        parent_commit=info.sha,
        commit_message="Publish validated compact JaisGuard classifier",
    )
PY
```

The exact allowlist excludes caches or any files added after sealing. Repositories
remain private. After upload, record the returned commit and independently verify
remote file hashes at that commit before making public-access claims. Visibility
changes and any future revisions require a separate explicit instruction.

## Verification summary

- CPU suite: **239 passed**, 31 integrations deselected.
- Available integrations: **30 passed, 1 skipped** (original missing test parents).
- New standalone-runtime regressions: **7 passed**.
- Ruff lint/formatting, documentation links, command help and package card parsing passed.
- GPU: eight 658-input same-precision checks plus two 400-input historical batch-two checks passed with exact probabilities.
- No repository dependency was loaded in isolated standalone GPU smoke checks.
- Original checkpoint/reference hashes and additional data/optimizer/quality artifacts matched their prior inventories.
- No model upload, target repository creation, dataset publication or visibility change was performed.

The remaining release decision is owner approval of the actual staged packages,
component licensing and disclosed privacy/reproduction limitations. There is no
unresolved numerical mismatch under the matched reference conditions. A new vLLM
engine run and a fresh installation of the complete GPU dependency stack were not
performed and are not claimed.
