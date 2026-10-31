# JaisGuard public model publication

Published on 2026-10-31 following explicit owner authorization for **public** publication of the exact reviewed packages and acceptance of the documented residual uncertainty. Numerical validation is not described as privacy clearance.

## Published repositories

| Model | Public URL | Final commit |
| --- | --- | --- |
| JaisGuard-LoRA-0.9B | [RayenFekih/JaisGuard-LoRA-0.9B](https://huggingface.co/RayenFekih/JaisGuard-LoRA-0.9B) | [`74977da942572cf15e0cbf6c815a343b1bd61af1`](https://huggingface.co/RayenFekih/JaisGuard-LoRA-0.9B/commit/74977da942572cf15e0cbf6c815a343b1bd61af1) |
| JaisGuard-Frozen-0.9B | [RayenFekih/JaisGuard-Frozen-0.9B](https://huggingface.co/RayenFekih/JaisGuard-Frozen-0.9B) | [`169fe462f83667ddc9af547ebeea538d0a0b9602`](https://huggingface.co/RayenFekih/JaisGuard-Frozen-0.9B/commit/169fe462f83667ddc9af547ebeea538d0a0b9602) |

Both names were checked with the authenticated owner before creation. Both repositories were created using `private=False` and `exist_ok=False`; no preexisting repository was overwritten. Uploads used the reviewed exact file allowlists and the freshly created parent commits.

## Exact-byte and anonymous verification

| Model | Reviewed manifest SHA-256 | Matching payload files | Total remote files |
| --- | --- | ---: | ---: |
| JaisGuard-LoRA-0.9B | `3a55facc788c9f3ca5aa75e361716eb376f52acce5f62590ad59b2918212cdc4` | 19 | 20 |
| JaisGuard-Frozen-0.9B | `fe0add7a8dc00bc3986ea4f8b021921bac4b36a0e1a6b3d35dbec9ab1415f3d5` | 17 | 18 |

All reviewed payload files, including cards, licensing, attribution, limitations, code, tensors and manifest, downloaded byte-for-byte at their published commits. No reviewed package file was edited. The sole extra remote file in each repository is Hub-generated `.gitattributes`, whose standard filter rules were inspected; its SHA-256 is `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361`.

Every remote file was downloaded and hashed: **38 files total**, comprising **36 reviewed payload files** and two platform-generated attributes files. Anonymous Hub API requests and snapshot downloads explicitly used `token=False`, with implicit authentication disabled. Both returned public visibility. A final anonymous default-branch check matched each commit above.

## Fresh-download real inference

Fresh anonymous download directories:

- `/tmp/jaisguard-public-release-anonymous/JaisGuard-LoRA-0.9B`
- `/tmp/jaisguard-public-release-anonymous/JaisGuard-Frozen-0.9B`

The downloaded bundled runtimes were loaded on the RTX 4050, sequentially in separate processes, using the existing pinned base-model cache and tested Python environment. This verifies a fresh download, not a newly provisioned GPU dependency stack.

| Model | Precision | Reference inputs | Max / mean probability difference | Default / locked mismatches |
| --- | --- | ---: | ---: | ---: |
| v2-lora-44 | BF16 | 658 | 0.0 / 0.0 | 0 / 0 |
| v2-lora-44 | FP16 | 658 | 0.0 / 0.0 | 0 / 0 |
| v2-frozen-43 | BF16 | 658 | 0.0 / 0.0 | 0 / 0 |
| v2-frozen-43 | FP16 | 658 | 0.0 / 0.0 | 0 / 0 |

All four runs passed against their matching historical HF references: 250 validation inputs, 400 historical test inputs and eight probes each. FP32 trained heads/biases and saved thresholds were retained. Historical FP16 vLLM comparisons remained within the previously approved tolerance; no new vLLM server or HTTP benchmark was run.

Known upstream docstring diagnostics and the expected unused `lm_head.weight` loading notice appeared, as in preparation. No loading, prediction, transfer or verification failure occurred. No warning was fixed by changing the reviewed packages.

## Pinned anonymous download and inference example

Run from the project checkout with its existing tested environment:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

for repo, revision, folder in [
    ("RayenFekih/JaisGuard-LoRA-0.9B", "74977da942572cf15e0cbf6c815a343b1bd61af1", "downloads/JaisGuard-LoRA-0.9B"),
    ("RayenFekih/JaisGuard-Frozen-0.9B", "169fe462f83667ddc9af547ebeea538d0a0b9602", "downloads/JaisGuard-Frozen-0.9B"),
]:
    snapshot_download(repo, revision=revision, token=False, local_dir=folder)
PY
.venv/bin/python downloads/JaisGuard-LoRA-0.9B/example.py --local-files-only --text 'كيف يمكنني تعلم اللغة العربية؟'
.venv/bin/python downloads/JaisGuard-Frozen-0.9B/example.py --local-files-only --precision fp16 --text 'ضصثقفغعهخحجشس'
```

Use the bundled requirements and custom `JaisGuardClassifier` API as documented in each published card. No native AutoModel sequence-classification or compact-package vLLM support is claimed. BF16/FP16 and batching scopes remain separate.

## Preservation and receipts

All 44 original checkpoint/reference files captured before preparation matched again after publication. No model was retrained and no historical checkpoint, dataset, threshold, evaluation or benchmark result was edited. The approved model cards and license/attribution files are unchanged.

Local detailed receipts are preserved under ignored `artifacts/model-publication-review/public-release/`: authorization/preflight, creation/upload progress, final commit receipts, all remote SHA-256 values, anonymous visibility checks and four downloaded-package GPU results. The completion receipt records zero failures.

The previous [release review](model-release-review.md) and [preparation report](model-publication-preparation.md) remain historical records. Their earlier pending/private-upload instructions were superseded by the explicit public-publication authorization. Residual uncertainty remains disclosed in the unchanged public model cards.

**Result: both public publications completed and independently verified. No unresolved publication or numerical verification failures.**
