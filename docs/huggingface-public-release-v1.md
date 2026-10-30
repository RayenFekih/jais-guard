# JaisGuard public-v1 publication candidate

The owner authorized a separate derivative, excluding original experiment training
positions 277 and 486 for privacy uncertainty. The owner states this is a personal
learning project independent of the employer and authorizes preparing project-created
contributions for CC BY 4.0 release, subject to source terms and final review.
This authorization is recorded in `configs/publication-public-v1.json` and the new
approval file. It is not a claim that the excluded narratives are fictional or cleared,
and it is not the final content/license release sign-off.

## Which data this version contains

| Dataset | Train | Validation | Historical test | Total |
| --- | ---: | ---: | ---: | ---: |
| Original experiments (unchanged) | 1,500 | 250 | 400 | 2,150 |
| Public-v1 derivative | 1,498 | 250 | 400 | 2,148 |

| Public split | VALID (0) | GIBBERISH (1) |
| --- | ---: | ---: |
| Train | 708 | 790 |
| Validation | 120 | 130 |
| Historical test | 188 | 212 |

Public training retains 1,365 human-reviewed and 133 audited-provisional records.
The two exclusions were human-reviewed VALID Aya examples. All retained texts, labels
and selected metadata match the original source projection exactly, in original order.
Validation and test remain identical. The original 700-record historical training set
remains a frozen source reference; 698 of its rows occur in public-v1. Existing model
results describe the original experiments, not training on this public derivative.

The original staging directory and its approval/evidence files remain preserved.
The new, separate staging directory is:

```text
artifacts/huggingface/arabic-gibberish-detection-public-v1/
  README.md
  metadata.json
  data/train.parquet
  data/validation.parquet
  data/test.parquet
  LICENSES/Apache-2.0.txt
  LICENSES/CC-BY-4.0.txt
  LICENSES/ATTRIBUTION.md
```

README keeps the scoped licensing summary and source attribution table. The LICENSES/
files contain the complete licenses and record-level attribution/modification notices.
The strict eight-file upload allowlist and manifest cover all of them. No weights, checkpoints, results or local review files
are staged. The public attribution table maps public split positions to public upstream
record IDs/revisions and changes, with the two omitted source IDs excluded. No internal
candidate IDs, annotator hashes or reviewer notes are exposed.

## Licensing package and disclosures

- CC BY 4.0 applies to authorized project-controlled contributions/compilation and
  MASSIVE components; original copyrightability or uniqueness of AI outputs is not asserted.
- Aya and TyDi components retain Apache 2.0; the YAML license list describes component
  scope, not a choice of licenses for every record.
- The README contains contributor attributions and links to LICENSES/, which preserves
  pinned upstream revision links, per-row historical modification notices and both
  complete licenses. Retained
  external origins total 313: MASSIVE 130, Aya 148 and TyDi 35. There are 248 verbatim
  upstream inputs and 65 historical adaptations. The rest are project-generated text/seeds.
- Test[319] is contained in test[204]; both remain. The benchmark is previously examined
  and informed iteration 2. Zero detected cross-split overlaps does not make it independent.
- Test parent sidecars are absent in this checkout. Eighty passthrough originals are
  inferable; 260 corruption parents cannot be reconstructed. No replacement evidence is made up.

The previous source investigation remains in
[huggingface-publication-review.md](huggingface-publication-review.md). Its two privacy
findings are handled here by authorized omission, not by inventing consent or clearance.

## Build, inspect and validate in WSL

```bash
cd /home/rayen/projects/fine-tuning
.venv/bin/python -m src.jais_guard.public_release --build
.venv/bin/python -m pytest tests/test_publication.py tests/test_public_release.py -q
.venv/bin/python -m src.jais_guard.public_release --dry-run \
  --repo-id RayenFekih/arabic-gibberish-detection \
  --review artifacts/huggingface-review/public-release-v1/approved-review.json
```

Build writes a fresh review template but does not overwrite `approved-review.json`.
A rebuild changing any staged/policy bytes invalidates an older approval hash. The
original `src.jais_guard.publication` command still describes the 2,150-row candidate;
use **public_release** for this derivative.

Dry-run performs no network calls. It returns 2 when technical validation succeeds
but final approval is pending; it returns 1 for validation/input errors. It returns
0 only when evidence reviews and final content/license approvals are complete.
The known within-test containment case is disclosed and expected; an unexpected
relationship or screening finding blocks publication even with an approval file.

Preview the public derivative without using the read-only home cache:

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset
p = 'artifacts/huggingface/arabic-gibberish-detection-public-v1'
ds = load_dataset(p, cache_dir='/tmp/jaisguard-public-v1-preview')
print(ds)
print(ds['train'].features)
print(ds['train'].select(range(3)).to_pandas())
PY
```

Local evidence (not uploaded) lives in `artifacts/huggingface-review/public-release-v1/`:
`audit.json`, `validation.json`, `retained-source-positions.json`, `review-template.json`,
`approved-review.json` and the preparation preservation receipt. The source-position
map documents exactly how the public train row numbers shift after omissions.

## Final review and upload

Inspect the staged README, all three LICENSES/ files and `validation.json`.
The README uses approval-neutral language; pending/final approval status stays in the
local approval file, so actual approval does not require changing the reviewed payload. The approval file
already records the owner's preparation authorization and verified review findings.
Only its `final_review.content` and `final_review.license` are deliberately pending.
Record actual final approval and its evidence in those entries; do not fabricate
approval, change the metadata hash to bypass a mismatch, or use the old 2,150-row review.
The final license review should confirm the scoped notice package and any account terms
applicable to the project authoring, without claiming unverified permission facts.

After final sign-off, repeat the dry-run above. Only when it returns zero:

```bash
.venv/bin/hf auth login
.venv/bin/python -m src.jais_guard.public_release \
  --repo-id RayenFekih/arabic-gibberish-detection \
  --review artifacts/huggingface-review/public-release-v1/approved-review.json \
  --publish
```

**This command has not been executed.** The publisher requires a private dataset
repository (creates one privately if absent), refuses differing preexisting remote
files, guards against concurrent remote changes and uploads only the eight staged
files. It never changes visibility to public. Pending final approvals prevent even
repository creation or other Hub calls. An upload receipt includes commit, file/review
hashes and remote inventory, with downloaded-byte verification at that commit.

To independently verify an actual future upload:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
r = json.loads(Path('artifacts/huggingface-review/public-release-v1/publication-receipt.json').read_text())
files = HfApi().list_repo_files(r['repo_id'], repo_type='dataset', revision=r['commit'])
assert set(r['files']) <= set(files)
for name, expected in r['files'].items():
    f = hf_hub_download(r['repo_id'], name, repo_type='dataset', revision=r['commit'])
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == expected
s = load_dataset(r['repo_id'], revision=r['commit'], token=True,
                 cache_dir='/tmp/jaisguard-public-v1-remote-check')
assert {k: len(v) for k, v in s.items()} == {'train':1498, 'validation':250, 'test':400}
print(r['commit'], s)
PY
```
