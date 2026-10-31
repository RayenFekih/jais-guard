# Final JaisGuard model release review

> Subsequent explicit authorization and completed public publication are recorded in
> [the publication report](model-publication-release.md). The pending/private-upload
> statements below document the earlier review state; model package bytes are unchanged.

Review date: 2026-10-31, Africa/Tunis. Branch: `cleanup/public-workflows`.
No upload, repository creation, visibility change, checkpoint change or historical
result change was performed. The two staged packages were reviewed without
modifying their bytes or invalidating their previous GPU receipts.

## Decision and unresolved blockers

**Technical/package review passes. Privacy clearance is unresolved. Upload is not authorized.**

The evidence supports a conditional owner decision about uploading these exact
packages to new, owner-controlled **private** repositories. It does not support an
unqualified privacy-safe/public-release statement. Before any upload, record a
specific disposition of the residual training-data risk and explicitly authorize
the private destinations. Owner risk acceptance is a workflow decision; it is not
proof of consent, lawful disclosure, or absence of memorization.

There is no new numerical, integrity or loader-compatibility blocker. The remaining
items are:

1. **Privacy disposition:** fictional/authorized status of the two original
   personal-sounding Aya records remains unestablished. No membership-inference,
   white-box inversion or generative extraction evaluation has been performed.
   Resolve with actual source/permission evidence where available, or document the
   basis and limits of a restricted private-upload decision. A future public
   release remains a separate decision; accepting a private upload does not clear it.
2. **Owner approval:** accept the reviewed licensing scope and both exact package
   manifests, identify authorized private-repository collaborators, and explicitly
   authorize upload. No such authorization is inferred from the previous request
   to prepare or review these files.

No new bespoke permission from every licensed source publisher is presumed necessary.
Conversely, an open-source license is not evidence resolving personal-information
uncertainty. These findings do not establish that the records describe real people
or that the models actually reveal their content.

## Exact reviewed packages

| Target repository | Files including manifest | Bytes | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| `RayenFekih/JaisGuard-LoRA-0.9B` | 19 | 10133414 | `3a55facc788c9f3ca5aa75e361716eb376f52acce5f62590ad59b2918212cdc4` |
| `RayenFekih/JaisGuard-Frozen-0.9B` | 17 | 5071278 | `fe0add7a8dc00bc3986ea4f8b021921bac4b36a0e1a6b3d35dbec9ab1415f3d5` |

Staging root: `artifacts/huggingface-models/`. The exact inventory equals each
manifest's files plus `manifest.json`; no unlisted upload files were present.
All per-file hashes matched. These manifest hashes also match the preparation
release index. Any package change requires resealing and review of the new hash.

Publicly intended files are classifier components, tokenizer assets, the standalone
loader/example, configuration, aggregate metrics, provenance, validation receipts,
requirements, model card and scoped licenses. LoRA includes its adapter config and
safetensors; frozen includes no trained backbone updates. Neither bundles the base
weights, optimizer states, training exports, annotation databases or review notes.

## Targeted privacy-risk assessment

### Established facts

- Both checkpoints were trained on the original 1,500-row training set. The two
  excluded records are human-reviewed VALID Aya examples, original zero-based
  positions 277 and 486, already identified in the publication investigation.
- The recorded concerns are first-person family/biography-style information.
  Their narratives, names and identifying details are deliberately not reproduced
  here or sent to external search/review services.
- Their checkpoint-tokenizer lengths are 30 and 29 tokens with special tokens,
  below the 64-token policy. The exclusion therefore cannot be explained as a
  training-token-budget rejection.
- The 1,498-row public derivative omits them. These historical weights were not
  retrained or unlearned after that omission. Numerical parity preserves the
  historical function; it does not remove or measure privacy risk.
- Frozen-backbone training changed a 3,074-parameter classification head, including
  bias. LoRA additionally changed low-rank attention projections. The classifier
  objective was binary cross-entropy, not next-token generation.
- The documented API returns labels and two probabilities, not generated text.
  Its small output interface limits direct text emission through that API.
- Model repositories distribute tensors. A recipient can inspect or adapt them
  outside the documented API. A private Hub upload is still a transfer of those
  tensors to the hosting provider and authorized collaborators.

### What these facts do not establish

The restricted API and absence of literal text are **not** proof that weights
cannot expose information. Classifier prediction distributions can support
membership inference in some settings; no attack success rate can be inferred for
JaisGuard from that general result. See the primary
[membership-inference study](https://arxiv.org/abs/1610.05820).

For frozen-head, the unchanged public backbone cannot have acquired new training
information through this fine-tuning step. The learned head can still encode
information about training representations/labels. Its small size is not a formal
privacy bound. No differential-privacy training/accounting was recorded.

For LoRA, learned backbone updates provide an additional white-box exposure surface.
An attacker need not keep the supplied two-output wrapper and may attempt to combine
updates with a generative backbone/head. This is a structural threat assessment,
not evidence that this adapter can reconstruct either record. Published
[generative extraction research](https://arxiv.org/abs/2012.07805) establishes why
training-data extraction cannot be dismissed generally; its GPT-2 results are not a
measured vulnerability or quantitative risk estimate for this binary classifier.

A high confidence score on either known example would not demonstrate memorization:
there is no matched nonmember distribution, pre-training membership evidence,
shadow-model attack calibration or held-out extraction criterion. Therefore no
single-example confidence probe is being presented as privacy clearance. A finite
unsuccessful attack test would also not prove that all attacks fail.

The unresolved facts are fictional/real status, any relevant individual authority
to disclose, and the actual exploitable influence of these two examples. Public
availability of an upstream example and its dataset license do not establish those
privacy facts. Do not claim the weights are privacy-cleared, free of personal data,
or unlearned. The current model cards already disclose the original training and
absence of memorization/extraction clearance.

### Direct-content and artifact screening

Both complete excluded narratives were absent from all textual package payloads
and as literal UTF-8 byte sequences in safetensors. Complete original training
texts were absent from non-tokenizer package text. Complete train/validation record
values were absent from serialized JSON values. The tokenizer is a byte-verified
upstream asset, not a newly built vocabulary from these training records.

No email/targeted credential-pattern/private-path hits were found. Safetensors
metadata had no internal paths. The reviewed inventory contains no training records,
source caches, annotator hashes, reviewer notes, SQLite state or private-path
configuration. Intentional generic Arabic examples, public attribution names and
upstream tokenizer vocabulary are not private dataset exports.

These scans check **direct carriage** and known patterns. They do not detect
latent memorization in numerical tensors, establish identities, or certify every
possible secret format. Findings are preserved privately in
`artifacts/model-publication-review/final-review-evidence.json` without narrative
text. Do not upload that private review directory.

## Metrics and locked-threshold trade-offs

The model cards and `metrics.json` match the selected individual checkpoint entries
in the archived v2 comparison. Three-seed means and sample SD are explicitly in a
separate section. The table below uses the historical BF16 **batch-two** quality
run, 400 inputs: 188 VALID and 212 GIBBERISH.

| Checkpoint | Operating point | Accuracy | Macro F1 | VALID rejected / 188 | GIBBERISH caught / 212 |
| --- | --- | ---: | ---: | ---: | ---: |
| LoRA v2 seed 44 | default 0.5 | 91.25% | 0.912038 | 20 | 197 |
| LoRA v2 seed 44 | locked 0.942931056022644 | 92.25% | 0.922441 | 9 | 190 |
| Frozen v2 seed 43 | default 0.5 | 88.00% | 0.879020 | 30 | 194 |
| Frozen v2 seed 43 | locked 0.9563547372817993 | 81.50% | 0.812297 | 1 | 139 |

Raising LoRA's threshold accepts 11 more meaningful inputs but misses seven more
gibberish inputs: VALID rejection falls from 10.64% to 4.79%, while gibberish recall
falls from 92.92% to 89.62%. Accuracy increases by one percentage point on this
particular examined benchmark; it is not a general guarantee.

Frozen-head makes a much stronger trade-off: 29 fewer meaningful inputs are
rejected, but 55 additional gibberish inputs are missed. VALID rejection falls from
15.96% to 0.53%, recall falls from 91.51% to 65.57%, and accuracy falls by 6.5 points.
The locked setting is consequently a low-false-rejection operating point, not an
unconditional best-quality recommendation. Neither threshold is changed here.

Selections were validation-locked, but the test benchmark was previously examined
and informed iteration two. Containment and missing-parent limitations are disclosed.
Batch-one serving observations and BF16/FP16 differences are separately labeled;
the original quality figures are not promises of identical rounded scores under
other batching/precision conditions. Full preparation evidence and aggregates are
in [the preparation report](model-publication-preparation.md).

## Licensing, attribution and loader review

The project's MIT license is copied byte-for-byte. Cards scope it to the project
loader/classifier components, with explicit Apache-2.0 scope for copied tokenizer
assets and the separately downloaded pinned base. The full Apache text matches the
reviewed standard copy. Source attribution preserves IFM/Inception42, Qwen,
Alibaba/Hugging Face credits and training-source references. Upstream modeling code
and base weights are not redistributed in these compact packages.

The pinned publisher review found no separate LICENSE/NOTICE file. Apache allows
redistribution subject to its applicable license, attribution and modification-notice
conditions; those packaging checks do not clear unrelated privacy questions.
See [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The owner must accept the staged trained-component licensing at release approval;
this review does not infer authority over third-party personal information from MIT.

Both cards require `jaisguard_model.JaisGuardClassifier`, explain pinned trusted base
code and dependencies, and explicitly disclaim AutoModelForSequenceClassification,
Transformers pipelines, hosted Inference Providers and direct native-vLLM loading.
`config.json` has a custom checkpoint model type and no misleading `auto_map` or
standard sequence-classification architecture registration. Historical vLLM claims
refer only to separate full-backbone exports with matching FP16 conditions, not to
these compact repositories. No new compatibility claim is needed or supported.

## Final owner approval checklist

Technical items were verified; the unchecked decisions remain genuinely pending.

- [x] Exact file inventory, manifests and unchanged inference-critical hashes reviewed.
- [x] Cards, scoped license text, attribution and pinned base dependency reviewed.
- [x] Original training versus public derivative, dirty training-source revision and test limitations disclosed.
- [x] Individual/aggregate metrics and threshold/batching/precision trade-offs reviewed.
- [x] Custom-loader requirement and unsupported compatibility exclusions confirmed.
- [x] Direct private-artifact/text/path/credential screening completed within its stated limits.
- [ ] Record an evidence-based privacy disposition or the basis for accepting residual risk for restricted private upload; do not label this as proof of privacy clearance.
- [ ] Accept the exact reviewed component licensing, including project trained tensors.
- [ ] Confirm the two private destinations and intended collaborators/access scope.
- [ ] Explicitly authorize upload of the two manifest hashes listed above.

Do not change approval state merely because this checklist was prepared. If either
package changes, this checklist's exact-byte review no longer applies. No approval
for a future public visibility change is included.

## Private-upload commands, conditional on explicit authorization

**Not executed. Do not run while the decisions above are pending.** These commands
pin the reviewed manifests, hash every file before importing any model code, reject
symlinks/extra files, preflight both target names and create only new private model
repositories. They refuse to overwrite preexisting targets and do not change visibility.

```bash
hf auth login
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

approved = {
    "JaisGuard-LoRA-0.9B": "3a55facc788c9f3ca5aa75e361716eb376f52acce5f62590ad59b2918212cdc4",
    "JaisGuard-Frozen-0.9B": "fe0add7a8dc00bc3986ea4f8b021921bac4b36a0e1a6b3d35dbec9ab1415f3d5",
}

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

prepared = []
for name, expected in approved.items():
    folder = (Path("artifacts/huggingface-models") / name).resolve()
    manifest_path = folder / "manifest.json"
    if digest(manifest_path) != expected:
        raise RuntimeError(f"Reviewed manifest changed: {name}")
    manifest = json.loads(manifest_path.read_text())
    allowlist = set(manifest["files"]) | {"manifest.json"}
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    if actual != allowlist:
        raise RuntimeError(f"Unexpected or missing package files: {name}")
    if any(p.is_symlink() for p in folder.rglob("*")):
        raise RuntimeError(f"Symlink in package: {name}")
    for filename, expected_file in manifest["files"].items():
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder) or digest(path) != expected_file:
            raise RuntimeError(f"Package integrity failure: {name}/{filename}")
    validation = json.loads((folder / "validation.json").read_text())
    if validation["status"] != "passed real GPU inference":
        raise RuntimeError(f"Inference validation not passed: {name}")
    prepared.append((f"RayenFekih/{name}", folder, sorted(allowlist)))

# Run only after explicit owner approval and privacy disposition, recorded separately.
api = HfApi()
for repo_id, _, _ in prepared:
    try:
        api.model_info(repo_id, timeout=15)
    except RepositoryNotFoundError:
        continue
    raise RuntimeError(f"Target already exists; stop for review: {repo_id}")

receipts = []
for repo_id, folder, allowlist in prepared:
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=False)
    info = api.model_info(repo_id, timeout=15)
    if not info.private:
        raise RuntimeError("Repository is not private; refuse upload")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=allowlist, parent_commit=info.sha,
        commit_message="Upload reviewed compact JaisGuard classifier privately",
    )
    receipts.append({"repo_id": repo_id, "commit": commit.oid})
print(json.dumps(receipts, indent=2))
PY
```

These are prepared instructions, not an authorization record. If an upload fails,
retain any private partial repository and inspect it; do not silently retry against
changed remote state. After an authorized upload, save commit receipts, independently
verify downloaded bytes at those commits, and reconfirm private visibility. No
public-access or privacy-cleared claim follows from upload success.

## Review preservation and verification

No staged package file changed during this review. The preparation release-index
manifest hashes still match, so the previous inference-critical-byte receipts remain
applicable. Original checkpoint/reference files captured before preparation were
rehash-checked, and no changed historical file was found. This review did not
perform new fitting, GPU collection, membership inference or extraction experiments.
The private-upload snippet was parsed and its **local preflight only** executed;
no Hub client was instantiated by that check and no upload branch was executed.
Local documentation links and diff whitespace checks passed. The evidence branch
referenced by the cards was found on GitHub at the preparation commit via read-only
metadata; no remote branch or repository was modified.

A live check of the two proposed HF target names did not complete and was stopped.
No claim is made that those names are currently absent or available. The prepared
commands perform a fresh bounded metadata check for both names before creating
either repository; existing targets require separate review rather than overwrite.
