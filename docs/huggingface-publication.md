# Publishing the JaisGuard dataset

For the owner-authorized 2,148-row privacy-filtered release, use the
[public-v1 runbook](huggingface-public-release-v1.md) and
`src.jais_guard.public_release`. Commands below describe the preserved original
2,150-row candidate.

Run these commands in WSL with the existing environment. No GPU or package installation
is required. No remote operations were executed during implementation.

## Build and validate locally

```bash
cd /home/rayen/projects/fine-tuning
.venv/bin/python -m src.jais_guard.publication --build
.venv/bin/python -m src.jais_guard.publication --dry-run
.venv/bin/python -m pytest tests/test_publication.py -q
```

Build writes exactly `README.md`, `metadata.json` and `data/{train,validation,test}.parquet`
to `artifacts/huggingface/arabic-gibberish-detection/`. Both build and validation
re-read pinned original exports, verify checksums/class counts, and compare every
public column against its source projection. No original dataset or experiment is
written. v1 is retained through its frozen version, source and manifest hashes and
an unchanged-subset check, rather than an additional overlapping evaluation split.
`metadata.json` inventories recorded source claims without publishing record IDs,
local paths or reviewer notes. Data and local review artifacts are Git-ignored.

Build exits zero when preparation succeeds, even if release is blocked. Dry-run
makes no network calls and exits **2** when local checks pass but publication is
blocked, **1** on validation errors, **0** only when a supplied review clears all gates.

Review these local-only files in `artifacts/huggingface-review/`:

- `validation.json`: local result, readiness and relationship counts.
- `audit.json`: source/license/attribution groups, record references as zero-based
  `[split, row_index]`, regex findings and exact/normalized/family/near-duplicate pairs.
- `review-template.json`: pending review tied to the complete metadata SHA-256.
- `publication-receipt.json`: created only after an actual upload; records commit,
  uploaded file hashes, review hash, remote inventory and verification outcome.

The audit examines all records. Personal-info, secret, copyright and unsafe-content
regexes are triage only: they cannot identify every Arabic name, protected passage,
sensitive request or rights restriction. Document the scope of automated and targeted text review, resolve concrete findings,
and verify source/derivative rights. The five category approvals apply to the entire
dataset, including records with no regex hits. The evidence investigation and the
remaining focused decisions are in [huggingface-publication-review.md](huggingface-publication-review.md). Review any relationship findings
without changing or dropping source records. Document intentional shared structure
and its evaluation consequences; stop if any critical issue cannot be resolved.
`source_family` checks recorded family/parent/source IDs across splits, not shared
recipe names. Near-duplicate detection uses documented character similarity and
cannot certify semantic independence or ancestry absent from the records.

## Preview

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset
p = 'artifacts/huggingface/arabic-gibberish-detection'
ds = load_dataset(p, cache_dir='/tmp/jaisguard-publication-preview')
print(ds)
for split in ds:
    print(split, ds[split].features)
    print(ds[split].select(range(3)).to_pandas().to_string(index=False))
PY
```

Read the staged README for methodology, class and dialect distributions and historical
benchmark disclosure. The 400-record test informed iteration 2 and is not independent.

## Resolve publication blockers

No license has been inferred from the project code license, generated content or
upstream source claims. In particular, local-review-only/publication-review-pending
records remain blocked. Check recorded source revisions and license/attribution
requirements, including derivatives, and obtain any necessary permissions.

After rights and content review, set `license` in `configs/publication.json` to the
approved Hugging Face license identifier, and set `licensing_statement` to the actual
license grant, redistribution scope, required source attributions/notices, permission
evidence references and content-review outcome. Do not merely replace pending with
approved. If a composite license cannot describe the collection, document the exact
per-source terms with a suitable Hugging Face metadata identifier and licensing
statement. Modify the card implementation if additional license files are required;
the publisher currently accepts only the five expected files.

Rebuild; this adds the selected license to YAML and replaces the pending licensing
section with the reviewed statement. Copy the newly generated review template to
`artifacts/huggingface-review/approved-review.json`. Record the approved `license`,
`license_evidence`, and `status: "approved"` with substantive `evidence` for every
category and the relationship review. Keep the new `metadata_sha256` unchanged.
Evidence should reference the actual completed review/permissions, not assert that
regex screening proves clearance. Review evidence stays local and is hashed in the
receipt. Any rebuilt content or stale approval fails closed.

```bash
.venv/bin/python -m src.jais_guard.publication --dry-run \
  --review artifacts/huggingface-review/approved-review.json
```

## Authenticate and publish privately (only after clearance)

```bash
.venv/bin/hf auth login
.venv/bin/python -m src.jais_guard.publication \
  --repo-id RayenFekih/arabic-gibberish-detection \
  --review artifacts/huggingface-review/approved-review.json \
  --publish
```

Use the login prompt; never place a token in source, commands, or staging. The
publisher creates a private dataset repo if absent, refuses an existing public repo,
checks preexisting file collisions, freezes the validated bytes and uploads only the
five allowed files. It uses the observed parent commit to reject concurrent changes,
does not delete files, and preserves unrelated remote files. A preexisting managed
filename with different content is refused, including README.md: inspect and resolve
ownership manually or use a fresh repo rather than overwrite unrelated content.
Authentication/network/permission failures may require retry; if upload succeeded
but verification failed, inspect the unverified receipt and remote commit first.
Identical existing files permit safe verification/retry.

## Verify the remote commit

The publisher checks inventory and downloads each published file at the returned
commit for byte-level verification. Independently repeat this check and load the
commit-pinned dataset:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset
r = json.loads(Path('artifacts/huggingface-review/publication-receipt.json').read_text())
api = HfApi()
files = api.list_repo_files(r['repo_id'], repo_type='dataset', revision=r['commit'])
assert set(r['files']) <= set(files)
for name, expected in r['files'].items():
    p = hf_hub_download(r['repo_id'], name, repo_type='dataset', revision=r['commit'])
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
s = load_dataset(r['repo_id'], revision=r['commit'], token=True)
assert {k: len(v) for k, v in s.items()} == {'train': 1500, 'validation': 250, 'test': 400}
print(r['commit'], s)
PY
```

## Make public after separate manual approval

Inspect the private dataset card, viewer, attribution terms and verified receipt first.
The publisher never changes visibility. Once public release is explicitly approved,
run this separate command:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
HfApi().update_repo_settings(
    repo_id='RayenFekih/arabic-gibberish-detection',
    repo_type='dataset', private=False,
)
PY
```

API/file-layout references: [Hub upload guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload),
[dataset repository configuration](https://huggingface.co/docs/datasets/repository_structure),
[dataset loading](https://huggingface.co/docs/datasets/loading).
