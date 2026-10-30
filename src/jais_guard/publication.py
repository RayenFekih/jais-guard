"""Frozen-data export and fail-closed Hugging Face publication (CPU only)."""
# Long lines below preserve readable dataset-card prose and audit descriptions.
# ruff: noqa: E501

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import datasets
import yaml
from datasets import ClassLabel, Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "artifacts/huggingface/arabic-gibberish-detection"
LOCAL = ROOT / "artifacts/huggingface-review"
LABELS = {"VALID": 0, "GIBBERISH": 1}
SOURCES = {
    "train": (
        "train-train-1500-v3-combined-v1",
        "51c9e4778f4b587b2ca6c824188e5ecb236b05275f1e7a72ac1cd65c498e0f33",
        1500,
        {"VALID": 710, "GIBBERISH": 790},
    ),
    "validation": (
        "validation-validation-250-v2",
        "0db272b4fc8c5f02cd039f8f05b1204e5f56aa21acd26987bcede869b9ed2bda",
        250,
        {"VALID": 120, "GIBBERISH": 130},
    ),
    "test": (
        "test-test-400-v1",
        "5912582d055c8079e773b4b27a205966033ae49aa034e7bdd979eab2732c57da",
        400,
        {"VALID": 188, "GIBBERISH": 212},
    ),
}
HISTORICAL = "99106292f85bea8b9a4be18b2019c6cccb541aa4cb05857528e4655786345962"
COLUMNS = [
    "text",
    "label",
    "label_name",
    "dialect",
    "dialect_status",
    "subtype",
    "label_origin",
    "source_method",
    "original_label_origin",
]
FEATURES = Features(
    {
        k: ClassLabel(names=["VALID", "GIBBERISH"]) if k == "label" else Value("string")
        for k in COLUMNS
    }
)
FILES = {"README.md", "metadata.json", *(f"data/{s}.parquet" for s in SOURCES)}
CATEGORIES = [
    "personal_information",
    "secrets",
    "copyrighted_third_party_text",
    "unsafe_content",
    "redistribution_rights",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_data(root=ROOT):
    rows, provenance = {}, {}
    for split, (version, expected, count, classes) in SOURCES.items():
        path = root / "data/exports" / version / f"{split}.jsonl"
        manifest = path.with_name("manifest.json")
        m = json.loads(manifest.read_text())
        if (
            sha(path) != expected
            or m["sha256"] != expected
            or not m["complete"]
            or m["partition"] != split
        ):
            raise ValueError(f"Frozen source mismatch: {split}")
        rs = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rs) != count or Counter(r["label"] for r in rs) != classes:
            raise ValueError(f"Source count/class mismatch: {split}")
        if len({r["id"] for r in rs}) != count:
            raise ValueError("Duplicate source identifiers")
        for r in rs:
            if (
                r["partition"] != split
                or not isinstance(r["input_text"], str)
                or not r["input_text"].strip()
            ):
                raise ValueError("Invalid source record")
            if r["is_gibberish"] != (r["label"] == "GIBBERISH") or r["annotation"]["flagged"]:
                raise ValueError("Inconsistent or flagged source label")
            if r["review_status"] == "human-reviewed":
                if r["annotation"]["label"] != r["label"]:
                    raise ValueError("Human annotation mismatch")
            elif (
                split != "train"
                or r["review_status"] != "provisional"
                or (r.get("audit") or {}).get("status") != "passed"
            ):
                raise ValueError("Unapproved provisional label")
        rows[split] = rs
        provenance[split] = {
            "version": version,
            "sha256": expected,
            "manifest_sha256": sha(manifest),
        }
    old = root / "data/exports/train-train-700-v2/train.jsonl"
    if sha(old) != HISTORICAL:
        raise ValueError("Historical source mismatch")
    v1 = [json.loads(x) for x in old.read_text().splitlines()]
    if len(v1) != 700 or rows["train"][:700] != v1:
        # Ordering need not be a prefix, but the complete original records must survive.
        if len(v1) != 700 or any(r not in rows["train"] for r in v1):
            raise ValueError("v1 is not an unchanged subset of v2")
    provenance["historical_v1"] = {
        "version": "train-700-v2",
        "sha256": HISTORICAL,
        "rows": 700,
        "relationship": "unchanged subset of train; 800 additional records",
        "manifest_sha256": sha(old.with_name("manifest.json")),
    }
    return rows, provenance


def project(r):
    values = {
        "text": r["input_text"],
        "label": LABELS[r["label"]],
        "label_name": r["label"],
        **{k: r.get(k) for k in ["dialect", "dialect_status", "subtype", "source_method"]},
        "label_origin": "human-reviewed"
        if r["review_status"] == "human-reviewed"
        else "audited-provisional",
        "original_label_origin": r.get("label_origin"),
    }
    return {k: values[k] for k in COLUMNS}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        c for c in text if not c.isspace() and c != "ـ" and unicodedata.category(c) != "Mn"
    )


def relations(r):
    keys = {("id", r["id"])}
    if r.get("family_id"):
        keys.add(("family", r["family_id"]))
    g, p = r.get("generation") or {}, r.get("provenance") or {}
    for key in ["parent_id", "source_id"]:
        value = g.get(key) if key == "parent_id" else p.get(key)
        if value:
            keys.add(("id", value))
    return keys


def audit(rows):
    """Heuristic screening is triage, never a certification of safety or rights."""
    findings, groups = [], defaultdict(list)
    patterns = {
        "email": r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}",
        "phone_or_identifier": r"(?:\+?\d[\d ()-]{6,}\d)",
        "secret": r"(?i)(?:hf_[a-z0-9]{15,}|sk-[a-z0-9_-]{15,}|AKIA[A-Z0-9]{16}|BEGIN .*PRIVATE KEY|password\s*[:=])",
        "personal_information": r"رقم هاتفي|عنواني|بطاقتي|رقم حسابي|كلمة السر",
        "unsafe_content": r"انتحار|اقتل|متفجر|اغتصاب|إرهاب|ارهاب",
        "copyright_candidate": r"حقوق النشر|جميع الحقوق|محفوظة|قال الشاعر",
    }
    flat = []
    for split, rs in rows.items():
        for i, r in enumerate(rs):
            ref = [split, i]
            for name, pattern in patterns.items():
                if re.search(pattern, r["input_text"]):
                    findings.append({"record": ref, "kind": name})
            p = r.get("provenance") or {}
            key = (
                p.get("source"),
                p.get("revision"),
                p.get("license"),
                p.get("rights_status"),
                p.get("attribution"),
            )
            groups[key].append(ref)
            flat.append((ref, r, normalize(r["input_text"])))
    overlap = {k: [] for k in ["exact", "normalized", "source_family", "near_duplicate"]}
    # Full pairwise character similarity after cheap length/character-set bounds.
    for (a, ra, na), (b, rb, nb) in combinations(flat, 2):
        if ra["input_text"] == rb["input_text"]:
            overlap["exact"].append([a, b])
        elif na == nb:
            overlap["normalized"].append([a, b])
        if a[0] != b[0] and relations(ra) & relations(rb):
            overlap["source_family"].append([a, b])
        if a[0] != b[0] and na != nb and min(len(na), len(nb)) >= 8:
            if (
                min(len(na), len(nb)) / max(len(na), len(nb)) >= 0.9
                and len(set(na) & set(nb)) / max(len(set(na)), len(set(nb)), 1) >= 0.8
            ):
                sm = SequenceMatcher(None, na, nb, autojunk=False)
                if sm.quick_ratio() >= 0.9 and sm.ratio() >= 0.9:
                    overlap["near_duplicate"].append([a, b])
    return {
        "screening_findings": findings,
        "relationships": overlap,
        "method": "NFKC, remove whitespace/tatweel/combining marks; cross-split SequenceMatcher >=0.90, length >=8, length ratio >=0.9, character-set overlap >=0.8; family_id and parent/source identifiers. Same-run/recipe alone is not a shared example family. Heuristics miss paraphrases, names and unrecorded ancestry.",
        "rights_groups": [
            dict(
                zip(
                    ["source", "revision", "license", "rights_status", "attribution"],
                    k,
                    strict=True,
                ),
                count=len(v),
                records=v,
            )
            for k, v in sorted(groups.items(), key=lambda x: str(x[0]))
        ],
    }


def file_hashes(stage, files=None):
    selected = FILES if files is None else set(files)
    return {p: sha(stage / p) for p in sorted(selected - {"metadata.json"})}


def inventory(stage, files=None):
    selected = FILES if files is None else set(files)
    if stage.is_symlink() or any(p.is_symlink() for p in stage.rglob("*")):
        raise ValueError("Staging symlinks are forbidden")
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if actual != selected:
        raise ValueError(f"Unexpected staging inventory: {actual ^ selected}")


def build(stage=STAGE, local=LOCAL, root=ROOT):
    if stage.exists():
        if stage.is_symlink() or any(p.is_symlink() for p in stage.rglob("*")):
            raise ValueError("Staging symlinks are forbidden")
        existing = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
        if not existing <= FILES:
            raise ValueError("Refusing to build over unrelated files")
    rows, provenance = source_data(root)
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "data").mkdir(exist_ok=True)
    for split, rs in rows.items():
        Dataset.from_list([project(r) for r in rs], features=FEATURES).to_parquet(
            stage / f"data/{split}.parquet"
        )
    report = audit(rows)
    write_json(local / "audit.json", report)
    (stage / "README.md").write_text(card(rows), encoding="utf-8")
    metadata = {
        "format_version": 1,
        "label_mapping": LABELS,
        "sources": provenance,
        "splits": {
            s: {
                "rows": len(rs),
                "classes": dict(Counter(r["label"] for r in rs)),
                "label_origin": dict(Counter(project(r)["label_origin"] for r in rs)),
                "dialect": dict(Counter(r.get("dialect") for r in rs)),
            }
            for s, rs in rows.items()
        },
        "files": file_hashes(stage),
        "audit_sha256": sha(local / "audit.json"),
        "recorded_sources": [
            {k: v for k, v in group.items() if k != "records"} for group in report["rights_groups"]
        ],
        "publication_status": "blocked_pending_content_and_redistribution_review",
        "software": {"datasets": datasets.__version__},
    }
    write_json(stage / "metadata.json", metadata)
    write_json(
        local / "review-template.json",
        {
            "metadata_sha256": sha(stage / "metadata.json"),
            "license": None,
            "license_evidence": None,
            "categories": {k: {"status": "pending", "evidence": None} for k in CATEGORIES},
            "relationships": {"status": "pending", "evidence": None},
        },
    )
    return validate(stage, local, root)


def validate(stage=STAGE, local=LOCAL, root=ROOT, review=None):
    inventory(stage)
    rows, provenance = source_data(root)
    m = json.loads((stage / "metadata.json").read_text())
    if (
        m["sources"] != provenance
        or m["label_mapping"] != LABELS
        or m["files"] != file_hashes(stage)
    ):
        raise ValueError("Publication manifest/hash mismatch")
    with tempfile.TemporaryDirectory(prefix="jaisguard-hf-") as cache:
        ds = load_dataset(
            "parquet",
            data_files={s: str(stage / f"data/{s}.parquet") for s in SOURCES},
            cache_dir=cache,
        )
        for s, rs in rows.items():
            if ds[s].features != FEATURES or ds[s].to_list() != [project(r) for r in rs]:
                raise ValueError(f"Export differs from exact source projection: {s}")
            expected = {
                "rows": len(rs),
                "classes": dict(Counter(r["label"] for r in rs)),
                "label_origin": dict(Counter(project(r)["label_origin"] for r in rs)),
                "dialect": dict(Counter(r.get("dialect") for r in rs)),
            }
            if m["splits"][s] != expected:
                raise ValueError("Split metadata mismatch")
    if (stage / "README.md").read_text() != card(rows):
        raise ValueError("Dataset card differs from reproducible template")
    report = audit(rows)
    if m["recorded_sources"] != [
        {k: v for k, v in group.items() if k != "records"} for group in report["rights_groups"]
    ]:
        raise ValueError("Source attribution inventory mismatch")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != m["audit_sha256"]:
        raise ValueError("Audit mismatch")
    write_json(local / "audit.json", report)
    blockers = [
        "Content, privacy, copyright and redistribution review pending; no dataset license established",
        "Duplicate/family/similarity findings require documented review",
    ]
    if review:
        approval = json.loads(Path(review).read_text())
        if approval.get("metadata_sha256") != sha(stage / "metadata.json"):
            raise ValueError("Review applies to a different publication")
        if not approval.get("license") or not approval.get("license_evidence"):
            raise ValueError("License and redistribution evidence required")
        for k in CATEGORIES:
            item = approval.get("categories", {}).get(k, {})
            if item.get("status") != "approved" or not item.get("evidence"):
                raise ValueError(f"Unresolved review: {k}")
        item = approval.get("relationships", {})
        if item.get("status") != "approved" or not item.get("evidence"):
            raise ValueError("Unresolved relationship review")
        card_meta = yaml.safe_load((stage / "README.md").read_text().split("---", 2)[1])
        if card_meta.get("license") != approval["license"]:
            raise ValueError(
                "Approved license must also appear in the dataset card; update licensing policy and rebuild"
            )
        blockers = []
    result = {
        "local_validation": "passed",
        "publication_ready": not blockers,
        "blockers": blockers,
        "metadata_sha256": sha(stage / "metadata.json"),
        "relationship_counts": {k: len(v) for k, v in report["relationships"].items()},
        "screening_findings": len(report["screening_findings"]),
    }
    write_json(local / "validation.json", result)
    return result


def publish(
    repo_id, stage=STAGE, local=LOCAL, root=ROOT, review=None, api=None, validator=None, files=None
):
    selected = FILES if files is None else set(files)
    result = (validator or validate)(stage, local, root, review)
    if not result["publication_ready"]:
        raise ValueError("Publication blocked: " + "; ".join(result["blockers"]))
    inventory(stage, files=selected)
    api = api or HfApi()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=False)
        info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise ValueError("Publishing requires a private repository")
    before = set(api.list_repo_files(repo_id, repo_type="dataset", revision=info.sha))
    # Refuse different colliding files, including README, rather than taking ownership.
    for name in before & selected:
        path = hf_hub_download(repo_id, name, repo_type="dataset", revision=info.sha)
        if sha(path) != sha(stage / name):
            raise ValueError(f"Refusing to overwrite existing remote file: {name}")
    # Freeze the validated bytes to prevent a local edit racing with upload.
    with tempfile.TemporaryDirectory(prefix="jaisguard-upload-") as tmp:
        snapshot = Path(tmp)
        import shutil

        for name in selected:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage / name, target)
        if (
            sha(snapshot / "metadata.json") != result["metadata_sha256"]
            or file_hashes(snapshot, files=selected)
            != json.loads((snapshot / "metadata.json").read_text())["files"]
        ):
            raise ValueError("Staging changed during publication")
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=snapshot,
            allow_patterns=sorted(selected),
            parent_commit=info.sha,
            commit_message="Publish frozen JaisGuard dataset",
        )
        remote = set(api.list_repo_files(repo_id, repo_type="dataset", revision=commit.oid))
        receipt = {
            "repo_id": repo_id,
            "commit": commit.oid,
            "metadata_sha256": result["metadata_sha256"],
            "review_sha256": sha(review),
            "files": {n: sha(snapshot / n) for n in sorted(selected)},
            "remote_files": sorted(remote),
            "verified": False,
        }
        write_json(local / "publication-receipt.json", receipt)
        if not (before | selected) <= remote:
            raise ValueError("Remote inventory verification failed; inspect receipt")
        for name in selected:
            if (
                sha(hf_hub_download(repo_id, name, repo_type="dataset", revision=commit.oid))
                != receipt["files"][name]
            ):
                raise ValueError(f"Remote content verification failed: {name}")
        receipt["verified"] = True
        write_json(local / "publication-receipt.json", receipt)
    return receipt


def card(rows, policy=None):
    header = {
        "language": ["ar"],
        "task_categories": ["text-classification"],
        "size_categories": ["1K<n<10K"],
        "tags": ["arabic", "gibberish-detection", "jaisguard"],
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": s, "path": f"data/{s}.parquet"} for s in SOURCES],
            }
        ],
    }
    if policy is None:
        policy = json.loads((ROOT / "configs/publication.json").read_text())
    if policy.get("license"):
        if not policy.get("licensing_statement"):
            raise ValueError("Approved licensing statement and attribution terms required")
        header["license"] = policy["license"]
    table = "\n".join(
        f"| {s} | {len(rs)} | {sum(r['label'] == 'VALID' for r in rs)} | {sum(r['label'] == 'GIBBERISH' for r in rs)} |"
        for s, rs in rows.items()
    )
    examples = []
    for split, predicate, title in [
        ("train", lambda r: r["label"] == "VALID" and r["dialect"] == "MSA", "Meaningful Arabic"),
        (
            "train",
            lambda r: r["label"] == "VALID" and r["dialect"] == "Tunisian",
            "Dialectal Arabic",
        ),
        (
            "validation",
            lambda r: r["label"] == "VALID" and r["subtype"] == "recoverable_corruption",
            "Recoverable corruption",
        ),
        ("test", lambda r: r["label"] == "GIBBERISH", "Gibberish"),
    ]:
        r = next(r for r in rows[split] if predicate(r))
        examples.append(f"- {title} ({r['label']}): {r['input_text']}")
    dialects = "\n".join(
        f"| {s} | {k} | {n} |"
        for s, rs in rows.items()
        for k, n in sorted(Counter(r["dialect"] for r in rs).items())
    )
    rendered = (
        "---\n"
        + yaml.safe_dump(header, sort_keys=False)
        + "---\n"
        + """# JaisGuard Arabic gibberish detection

## Dataset description
A frozen 2,150-example research collection for distinguishing unintelligible Arabic-script input from recoverable meaning. Original text and final labels are preserved exactly. This is a local publication candidate; redistribution review is pending.

## Intended use and label definitions
Train and evaluate Arabic input quality classifiers. VALID (0) means a meaningful question, topic, keyword or request can be recovered. Dialects, spelling errors, missing spaces, incomplete phrases, and limited corruption remain VALID when meaning survives. GIBBERISH (1) means no coherent question, topic or request can be recovered: random characters, keyboard noise, unrecoverable scrambling, invented word noise or incoherent real-word sequences. Unusual topics and out-of-domain requests alone are not gibberish. This is not a toxicity or factuality classifier. Arabizi and mixed-script generalization are not established.

## Examples (verbatim source records)
"""
        + "\n".join(examples)
        + f"""

## Construction methodology
The training export combines the original 700 records with 800 new records. The first collection includes selected public requests, assistant-authored examples and deterministic Python transformations. Recorded public sources include Amazon MASSIVE, Aya and TyDi; exact recorded source revisions, claimed licenses and attributions are inventoried in the local review audit. These claims require compatibility review before redistribution; no external permissions are inferred. Later training, validation and test collections use constructed scenarios and Python corruption recipes. Source metadata is incomplete in places. Nothing was generated, normalized, rewritten or relabeled during publication export.

## Human review and annotation
The recorded standard is single-reviewer annotation; no inter-annotator agreement statistic is available. Training has 1,367 human-reviewed and 133 audited-provisional labels (98 public-source-rule, 35 python-rule). Passing group audits does not mean every provisional example was individually reviewed. Validation (250) and test (400) are fully human-reviewed. Dialect metadata can be proposed and unconfirmed; preserve `dialect_status` where recorded.

## Splits and class distributions
| Split | Total | VALID (0) | GIBBERISH (1) |
| --- | ---: | ---: | ---: |
{table}

## Schema
`text` is original Unicode input; `label` is a Hugging Face ClassLabel integer (0 VALID, 1 GIBBERISH); `label_name` is its string name. `dialect`, `dialect_status`, `subtype`, and `source_method` preserve available source values (null if absent). `label_origin` distinguishes human-reviewed and audited-provisional; `original_label_origin` preserves the recorded human/public-source-rule/python-rule origin. Internal IDs, reviewer notes, paths and generation payloads are excluded.

## Dialect coverage
| Split | Recorded dialect | Count |
| --- | --- | ---: |
{dialects}

`none`, `not-assigned` and `Arabic-ar-SA-unverified` are original annotations, not verified dialect coverage. Dialect counts do not establish representative regional performance.

## Limitations and biases
Small, constructed, recipe-heavy samples do not represent natural production traffic. Shared corruption processes and unrecorded ancestry can produce correlated errors even without exact overlap. Near-duplicate screening is heuristic, not proof of independence. Single-reviewer judgments and provisional rules can be wrong; dialect expertise and recovery judgments vary. Content, privacy, copyright and redistribution review remain pending. Do not interpret the absence of regex findings as clearance. No records were silently removed to resolve concerns.

## Evaluation methodology
Report gibberish precision/recall/F1, confusion matrices and VALID false rejection rate, including per-subtype and per-dialect results where sample sizes permit. Choose thresholds on validation and disclose reuse of validation. **The 400-example test set is a previously examined, constructed and human-reviewed historical benchmark. Its results informed development of training iteration 2. It is not an untouched independent final test set.** Use a newly collected independent benchmark for final generalization claims. The local audit identifies exact, normalized, source-family and heuristic near-duplicate relationships for review.

## Version history
- Training iteration 1: 700 records, source version `train-700-v2`.
- Training iteration 2: 1,500 records, `train-1500-v3-combined-v1`; retains all original 700 unchanged plus 800.
- Validation: `validation-250-v2`, 250 human-reviewed records.
- Historical benchmark: `test-400-v1`, 400 human-reviewed records.
The historical 700 are not exposed as an evaluation split. `metadata.json` records frozen source and source-manifest hashes for reproducibility; original manifests remain in the project. Publication version 1 exports these three splits without changing experiments.

## Licensing and redistribution status
**BLOCKED pending documented rights and content review. No dataset license is granted by this candidate card.** Project code licensing does not establish rights to dataset text. Generated-content local-review-only/publication-review-pending markers and third-party source terms require explicit review, including required attribution and notices. A recorded upstream license is not by itself clearance for this combined release. Before upload, the publishing command requires a hash-bound review with licensing evidence and all audit categories approved. That review must specify the approved dataset license and any attribution obligations; update the publication implementation/card with those terms and rebuild before approving the new hash.

## Loading
After authorized publication (authenticate for private access):
```python
from datasets import load_dataset

ds = load_dataset("RayenFekih/arabic-gibberish-detection")
assert ds["train"].features["label"].names == ["VALID", "GIBBERISH"]
```
For local preview:
```python
from datasets import load_dataset
from pathlib import Path
root = Path("artifacts/huggingface/arabic-gibberish-detection")
ds = load_dataset("parquet", data_files={{s: str(root / "data" / f"{{s}}.parquet") for s in ("train", "validation", "test")}})
print(ds)
```

## Citation and project links
No paper, DOI or public project repository has been confirmed. Cite the dataset repository and the exact commit used after release: https://huggingface.co/datasets/RayenFekih/arabic-gibberish-detection . No formal bibliographic attribution is invented.
"""
    )
    if policy.get("license"):
        start = rendered.index("**BLOCKED pending documented rights")
        end = rendered.index("\n\n## Loading", start)
        rendered = rendered[:start] + policy["licensing_statement"] + rendered[end:]
    return rendered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="RayenFekih/arabic-gibberish-detection")
    p.add_argument("--stage", type=Path, default=STAGE)
    p.add_argument("--review", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--publish", action="store_true")
    args = p.parse_args()
    try:
        if args.build:
            result = build(args.stage)
        elif args.publish:
            result = publish(args.repo_id, args.stage, review=args.review)
        else:
            result = validate(args.stage, review=args.review)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.build and not args.publish and not result["publication_ready"]:
            return 2
        return 0
    except (ValueError, OSError, KeyError) as e:
        p.exit(1, f"Publication failed: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
