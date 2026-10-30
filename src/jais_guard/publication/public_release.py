"""Prepare the privacy-filtered public derivative; never alter experiment exports."""

# Long strings are dataset-card prose and audit descriptions.
# ruff: noqa: E501
import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

from datasets import Dataset, load_dataset

from jais_guard.publication import frozen

ROOT = frozen.ROOT
STAGE = ROOT / "artifacts/huggingface/arabic-gibberish-detection-public-v1"
LOCAL = ROOT / "artifacts/huggingface-review/public-release-v1"
POLICY = ROOT / "configs/publication-public-v1.json"
EXCLUSIONS = (277, 486)
COUNTS = {"train": 1498, "validation": 250, "test": 400}
CLASSES = {
    "train": {"VALID": 708, "GIBBERISH": 790},
    "validation": {"VALID": 120, "GIBBERISH": 130},
    "test": {"VALID": 188, "GIBBERISH": 212},
}
EXTERNAL = {
    "AmazonScience/massive": ("cc-by-4.0", "Amazon MASSIVE contributors"),
    "CohereLabs/aya_dataset": ("apache-2.0", "Aya Open Science contributors / Cohere Labs"),
    "google-research-datasets/tydiqa": ("apache-2.0", "TyDi QA contributors / Google Research"),
}


LICENSE_FILES = {"LICENSES/Apache-2.0.txt", "LICENSES/CC-BY-4.0.txt", "LICENSES/ATTRIBUTION.md"}
FILES = frozen.FILES | LICENSE_FILES
VALID_EXAMPLE = "كيف يصنع حديد الزهر ؟"


def policy():
    p = json.loads(POLICY.read_text())
    if (
        p["excluded_original_train_positions"] != list(EXCLUSIONS)
        or p["release_version"] != "public-v1"
    ):
        raise ValueError("Unexpected public-v1 exclusion/version policy")
    if (
        p["license"] != ["cc-by-4.0", "apache-2.0"]
        or p["owner_authorization"]["status"] != "authorized_for_preparation"
    ):
        raise ValueError("Unexpected licensing/preparation policy")
    return p


def release_data(root=ROOT):
    original, sources = frozen.source_data(root)
    # These positions refer to the checksum-pinned 1,500-row training export.
    for i in EXCLUSIONS:
        r = original["train"][i]
        if (
            r["provenance"]["source"] != "CohereLabs/aya_dataset"
            or r["label"] != "VALID"
            or r["review_status"] != "human-reviewed"
        ):
            raise ValueError("Privacy exclusion does not identify expected frozen record")
    indices = {
        s: [i for i in range(len(rs)) if s != "train" or i not in EXCLUSIONS]
        for s, rs in original.items()
    }
    rows = {s: [original[s][i] for i in ix] for s, ix in indices.items()}
    for s, rs in rows.items():
        if len(rs) != COUNTS[s] or Counter(r["label"] for r in rs) != CLASSES[s]:
            raise ValueError(f"Unexpected derivative counts: {s}")
    # A text must not sneak back into another row or any public-facing card example.
    omitted = {original["train"][i]["input_text"] for i in EXCLUSIONS}
    if any(r["input_text"] in omitted for rs in rows.values() for r in rs):
        raise ValueError("Excluded text reappeared in derivative")
    return rows, sources, indices


def stats(rows):
    return {
        s: {
            "rows": len(rs),
            "classes": dict(Counter(r["label"] for r in rs)),
            "label_origin": dict(Counter(frozen.project(r)["label_origin"] for r in rs)),
            "dialect": dict(Counter(r.get("dialect") for r in rs)),
        }
        for s, rs in rows.items()
    }


def audit(rows):
    report = frozen.audit(rows)
    flat = [
        (s, i, frozen.normalize(r["input_text"]))
        for s, rs in rows.items()
        for i, r in enumerate(rs)
    ]
    report["relationships"]["containment"] = [
        [[a, i], [b, j]]
        for (a, i, x), (b, j, y) in combinations(flat, 2)
        if x != y and min(len(x), len(y)) >= 20 and (x in y or y in x)
    ]
    patterns = {
        "biography_age": r"عمري\s*[0-9٠-٩۰-۹]+|أنا فارس",
        "named_family_narrative": r"مرتي\s+حنان|بنت زي القمر",
        "harmful_instruction_candidate": r"كيف.*(?:اقتل|أقتل|اصنع قنبلة|أصنع قنبلة|انتحر|أنتحر)|اغتصاب|محتوى جنسي",
    }
    report["expanded_screening"] = {
        name: [
            [s, i]
            for s, rs in rows.items()
            for i, r in enumerate(rs)
            if re.search(pattern, r["input_text"])
        ]
        for name, pattern in patterns.items()
    }
    known = {s: [] for s in rows}
    for split, directory in [
        ("train", "train800_iteration2_v1"),
        ("validation", "validation250_iteration2_v1"),
    ]:
        path = ROOT / "data" / directory / "private/corruptions.jsonl"
        sidecars = {
            r["id"]: r for r in [json.loads(line) for line in path.read_text().splitlines()]
        }
        for r in rows[split]:
            if r["id"] in sidecars and sidecars[r["id"]].get("original_text"):
                known[split].append(frozen.normalize(sidecars[r["id"]]["original_text"]))
    known["train"] += [
        frozen.normalize(r["generation"]["parent"]["text"])
        for r in rows["train"]
        if r.get("generation", {}).get("parent")
    ]
    known["test"] = [
        frozen.normalize(r["input_text"])
        for r in rows["test"]
        if r["subtype"] == "meaningful_request"
    ]
    report["known_parent_counts"] = {s: len(values) for s, values in known.items()}
    report["known_parent_cross_split_overlap_counts"] = {
        f"{a}/{b}": len(set(known[a]) & set(known[b])) for a, b in combinations(known, 2)
    }
    report["scope"] = (
        "Public derivative only. Automated triage plus prior AI-assisted targeted content inspection; no fictional-status assertion or comprehensive plagiarism certification."
    )
    return report


def legal_assets(rows):
    lines = [
        "## Scoped licenses, attribution and changes",
        "",
        "Prepared under the owner's authorization for project-created contributions; final content/license sign-off is a separate local publication gate. CC BY 4.0 applies to project-controlled original contributions, annotations and compilation rights to the extent those rights exist, and to the MASSIVE components. Aya/TyDi components retain Apache 2.0. The YAML list records applicable component licenses, not a choice of either license for every record. No third-party rights are waived, and AI-output uniqueness or copyrightability is not guaranteed.",
        "",
        "| Source | Attribution | Retained examples | Applicable terms |",
        "| --- | --- | ---: | --- |",
    ]
    for source, (license, credit) in EXTERNAL.items():
        n = sum(r["provenance"]["source"] == source for rs in rows.values() for r in rs)
        lines.append(
            f"| [{source}](https://huggingface.co/datasets/{source}) | {credit} | {n} | {license} |"
        )
    lines += [
        "",
        "The owner identifies this as a personal learning project independent of the employer. This is an authorization to prepare the release, not a claim of employer endorsement or bespoke permission from upstream publishers. No endorsement by Amazon, Cohere Labs, Google Research or OpenAI is implied.",
        "",
        "Selection, JaisGuard task annotation, metadata minimization and Parquet conversion are project changes relative to upstream datasets. The historical adaptations below modified upstream input text with the named recipes. This public export does not modify any retained frozen text or final label. Two privacy-uncertain original training records were omitted solely from this derivative; their original text and source identifiers are not reproduced here.",
        "",
        "MASSIVE: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Aya/TyDi: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0). Preserve these component attributions, source/revision links, modification notices and applicable license conditions on redistribution. The cached pinned repository listings had no separate Aya/TyDi NOTICE file; no additional upstream notice is invented. The pinned source cards supply their contributor/citation context.",
        "",
        "| Public split row (zero-based) | Upstream input source | Revision | Public upstream record | Historical input change |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s, rs in rows.items():
        for i, r in enumerate(rs):
            p, g = r["provenance"], r.get("generation") or {}
            if p["source"] in EXTERNAL:
                repo = p["source"]
                revision = p["revision"]
                lines.append(
                    f"| {s}[{i}] | [{repo}](https://huggingface.co/datasets/{repo}/blob/{revision}/README.md) | {revision} | {p['source_id']} | {g.get('recipe') or 'Input verbatim; other upstream fields omitted'} |"
                )
    # Public license texts are read from existing, locally inspected evidence.
    apache_path = next(
        (ROOT / ".venv/lib/python3.11/site-packages").glob(
            "huggingface_hub-*.dist-info/licenses/LICENSE"
        )
    )
    apache = apache_path.read_text()
    if "Apache License" not in apache or "Version 2.0, January 2004" not in apache:
        raise ValueError("Expected Apache 2.0 license text unavailable")
    cc = (ROOT / "data/pilot_v1/sources/massive_LICENSE.txt").read_text()
    if "Attribution 4.0 International" not in cc:
        raise ValueError("Expected CC BY 4.0 license text unavailable")
    return {
        "LICENSES/Apache-2.0.txt": apache,
        "LICENSES/CC-BY-4.0.txt": cc,
        "LICENSES/ATTRIBUTION.md": "\n".join(lines) + "\n",
    }


def notices(rows):
    summary = (
        legal_assets(rows)["LICENSES/ATTRIBUTION.md"].split("| Public split row", 1)[0].rstrip()
    )
    summary = summary.replace(
        "The historical adaptations below",
        "The historical adaptations in the detailed attribution table",
    )
    return (
        summary
        + "\n\n- [Full Apache 2.0 license](LICENSES/Apache-2.0.txt) — Aya/TyDi components.\n- [Full CC BY 4.0 license](LICENSES/CC-BY-4.0.txt) — MASSIVE and authorized project contributions.\n- [Complete record-level attributions and modification notices](LICENSES/ATTRIBUTION.md).\n"
    )


def card(rows):
    text = frozen.card(
        rows, policy={"license": policy()["license"], "licensing_statement": notices(rows)}
    )
    text = text.replace(
        "A frozen 2,150-example research collection",
        "A 2,148-example public derivative of the frozen 2,150-example experiment collection",
    )
    text = text.replace(
        "This is a local publication candidate; redistribution review is pending.",
        "This public-v1 candidate excludes two privacy-uncertain training examples by owner instruction. All retained inputs and labels are unchanged; this derivative was not the dataset consumed by completed experiments.",
    )
    text = text.replace(
        "The training export combines the original 700 records with 800 new records.",
        "The experiment training export combined 700 historical records with 800 new records (1,500). Public-v1 retains 698 of those historical records plus all 800 new records (1,498). Original experiment train[277] and train[486] (zero-based frozen positions) are excluded for unresolved privacy uncertainty, without rewriting or clearing them. Validation and historical test remain unchanged.",
    )
    text = text.replace(
        "Training has 1,367 human-reviewed and 133 audited-provisional labels",
        "The original experiment training has 1,367 human-reviewed labels; public-v1 retains 1,365 human-reviewed and 133 audited-provisional labels",
    )
    text = text.replace(
        "These claims require compatibility review before redistribution; no external permissions are inferred.",
        "The scoped license/notice appendix preserves source terms; no bespoke external permissions are inferred. Final review remains a separate publication gate.",
    )
    text = text.replace(
        "No records were silently removed to resolve concerns.",
        "Exactly two privacy exclusions are documented in this derivative; no retained text or label was rewritten. Original experiment datasets and results remain unchanged.",
    )
    text = text.replace(
        "Publication version 1 exports these three splits without changing experiments.",
        "Public-v1 exports 1,498/250/400 (2,148 total) from original experiment splits of 1,500/250/400 (2,150 total). The original v1 700-row training set remains a historical source reference, not a public evaluation split; only 698 of its records are public. Published counts and source/derivative hashes are in metadata.json. Existing experiment metrics describe their original training data; do not claim those classifiers were trained on the 1,498-row derivative.",
    )
    text = text.replace(
        'root = Path("artifacts/huggingface/arabic-gibberish-detection")',
        'root = Path("artifacts/huggingface/arabic-gibberish-detection-public-v1")',
    )
    disclosure = """\n\n### Known historical test dependencies and missing evidence
Test[319] is contained verbatim in test[204] after comparison normalization; the longer cooked-lentil storage question adds details. Both VALID records and their historical labels are preserved. This within-test dependency is not detected cross-split leakage; do not claim 400 fully independent scenarios. Exact/normalized equality, recorded cross-split family and heuristic cross-split near-duplicate checks found no overlaps; that does not prove semantic independence.

Private parent/corruption sidecars for the test construction are absent from this checkout. Of 340 scenario-based test rows, 80 passthrough originals are inferable from their meaningful_request recipe; the 260 corrupted-request parents cannot be replayed here. Record-level run/seed provenance and frozen texts survive. No replacement parent evidence was generated. Shared recipe bias and the prior use of benchmark results remain limitations.
"""
    text = text.replace("\n## Version history", disclosure + "\n## Version history")
    text = text.replace(
        "Content, privacy, copyright and redistribution review remain pending.",
        "Privacy exclusions, source rights and scoped licensing are documented; automated screening does not constitute comprehensive content clearance.",
    )
    if not any(
        r["input_text"] == VALID_EXAMPLE
        and r["label"] == "VALID"
        and r["review_status"] == "human-reviewed"
        for r in rows["train"]
    ):
        raise ValueError("README VALID example is not a retained human-reviewed VALID input")
    text, changed = re.subn(
        r"(?m)^- Meaningful Arabic \(VALID\): .*$",
        "- Meaningful Arabic (VALID): " + VALID_EXAMPLE,
        text,
    )
    if changed != 1:
        raise ValueError("Expected exactly one Meaningful Arabic example in the README")
    original, _ = frozen.source_data()
    if any(original["train"][i]["input_text"] in text for i in EXCLUSIONS):
        raise ValueError("Excluded text leaked into dataset card")
    return text


def metadata(rows, sources, stage, report):
    return {
        "format_version": 3,
        "release_version": "public-v1",
        "experiment_total": 2150,
        "release_total": 2148,
        "sources": sources,
        "derivation": {
            "excluded_split": "train",
            "excluded_original_positions": list(EXCLUSIONS),
            "excluded_count": 2,
            "reason": policy()["exclusion_reason"],
            "retained_order": "original source order",
            "text_and_label_changes": 0,
        },
        "label_mapping": frozen.LABELS,
        "splits": stats(rows),
        "license": policy()["license"],
        "license_scope": policy()["license_scope"],
        "policy_sha256": frozen.sha(POLICY),
        "files": frozen.file_hashes(stage, files=FILES),
        "audit_sha256": hashlib.sha256(
            (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "recorded_sources": [
            {k: v for k, v in g.items() if k != "records"} for g in report["rights_groups"]
        ],
        "publication_status": "candidate_requires_hash_bound_final_content_and_license_approval",
    }


def build(stage=STAGE, local=LOCAL, root=ROOT):
    if (
        stage.resolve().is_relative_to(frozen.STAGE.resolve())
        or local.resolve() == frozen.LOCAL.resolve()
    ):
        raise ValueError(
            "Public derivative must not overwrite original publication candidate/review"
        )
    if stage.exists():
        if stage.is_symlink() or any(p.is_symlink() for p in stage.rglob("*")):
            raise ValueError("Staging symlinks are forbidden")
        if any(
            p.relative_to(stage).as_posix() not in FILES for p in stage.rglob("*") if p.is_file()
        ):
            raise ValueError("Refusing to build over unrelated files")
    rows, sources, indices = release_data(root)
    (stage / "data").mkdir(parents=True, exist_ok=True)
    for s, rs in rows.items():
        Dataset.from_list([frozen.project(r) for r in rs], features=frozen.FEATURES).to_parquet(
            stage / f"data/{s}.parquet"
        )
    for name, content in legal_assets(rows).items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    report = audit(rows)
    frozen.write_json(local / "audit.json", report)
    (stage / "README.md").write_text(card(rows), encoding="utf-8")
    frozen.write_json(stage / "metadata.json", metadata(rows, sources, stage, report))
    frozen.write_json(local / "retained-source-positions.json", indices)
    frozen.write_json(local / "review-template.json", review_template(stage))
    return validate(stage, local, root)


def review_template(stage):
    return {
        "metadata_sha256": frozen.sha(stage / "metadata.json"),
        "license": policy()["license"],
        "license_evidence": "Owner preparation authorization in configs/publication-public-v1.json; retained upstream source terms and scoped summary in staged README.md and full license texts/row-specific notices in LICENSES/. No final release approval is implied.",
        "owner_authorization": policy()["owner_authorization"],
        "categories": {k: {"status": "pending", "evidence": None} for k in frozen.CATEGORIES},
        "relationships": {"status": "pending", "evidence": None},
        "final_review": {
            k: {"status": "pending", "evidence": None} for k in ["content", "license"]
        },
    }


def validate(stage=STAGE, local=LOCAL, root=ROOT, review=None):
    frozen.inventory(stage, files=FILES)
    rows, sources, indices = release_data(root)
    report = audit(rows)
    m = json.loads((stage / "metadata.json").read_text())
    if m != metadata(rows, sources, stage, report):
        raise ValueError("Public derivative manifest/hash/policy mismatch")
    with tempfile.TemporaryDirectory(prefix="jaisguard-public-validation-") as cache:
        # Exercise native directory/card loading, not just individual Parquet reads.
        ds = load_dataset(str(stage.resolve()), cache_dir=cache)
        for s, rs in rows.items():
            if ds[s].features != frozen.FEATURES or ds[s].to_list() != [
                frozen.project(r) for r in rs
            ]:
                raise ValueError(
                    f"Public derivative differs from exact retained source projection: {s}"
                )
    if (stage / "README.md").read_text() != card(rows):
        raise ValueError("Public dataset card/notice mismatch")
    for name, content in legal_assets(rows).items():
        if (stage / name).read_text() != content:
            raise ValueError(f"Public license/attribution content mismatch: {name}")
    frozen.write_json(local / "audit.json", report)
    # Findings stop publication unless the one documented within-test case is the only relationship.
    critical = [
        k
        for k in ["exact", "normalized", "source_family", "near_duplicate"]
        if report["relationships"][k]
    ]
    if report["relationships"]["containment"] != [[["test", 204], ["test", 319]]]:
        critical.append("unexpected containment relationships")
    if any(report["known_parent_cross_split_overlap_counts"].values()):
        critical.append("known-parent cross-split overlap")
    public_meta = (stage / "metadata.json").read_text()
    if re.search(
        r'/home/|/mnt/[a-z]/|"(?:author_group|reviewer|user_id|notes|api_key|token|password)"\s*:',
        public_meta,
    ):
        critical.append("private staged metadata")
    if report["screening_findings"] or any(report["expanded_screening"].values()):
        critical.append("unresolved content screening findings")
    blockers = [f"Unresolved critical finding: {x}" for x in critical]
    approval = json.loads(Path(review).read_text()) if review else None
    if approval:
        if approval.get("metadata_sha256") != frozen.sha(stage / "metadata.json"):
            raise ValueError("Review applies to a different public derivative")
        if approval.get("license") != policy()["license"] or not approval.get("license_evidence"):
            raise ValueError("Scoped license and redistribution evidence required")
        if approval.get("owner_authorization") != policy()["owner_authorization"]:
            raise ValueError("Owner preparation authorization mismatch")
        for key in [*frozen.CATEGORIES, "relationships"]:
            item = (
                approval.get("relationships", {})
                if key == "relationships"
                else approval.get("categories", {}).get(key, {})
            )
            if item.get("status") not in ["reviewed", "approved"] or not item.get("evidence"):
                blockers.append(f"Unresolved evidence review: {key}")
        for key in ["content", "license"]:
            item = approval.get("final_review", {}).get(key, {})
            if item.get("status") != "approved" or not item.get("evidence"):
                blockers.append(f"Final {key} review approval pending")
    else:
        blockers += [
            "Final content review approval pending",
            "Final license review approval pending",
        ]
    result = {
        "local_validation": "passed",
        "publication_ready": not blockers,
        "blockers": blockers,
        "release_version": "public-v1",
        "split_counts": {s: len(rs) for s, rs in rows.items()},
        "release_total": sum(map(len, rows.values())),
        "exclusions_verified": list(EXCLUSIONS),
        "retained_records_exactly_preserved": sum(map(len, rows.values())),
        "metadata_sha256": frozen.sha(stage / "metadata.json"),
        "relationship_counts": {k: len(v) for k, v in report["relationships"].items()},
        "screening_findings": len(report["screening_findings"])
        + sum(map(len, report["expanded_screening"].values())),
    }
    frozen.write_json(local / "validation.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="RayenFekih/arabic-gibberish-detection")
    p.add_argument("--stage", type=Path, default=STAGE)
    p.add_argument("--review", type=Path, default=LOCAL / "approved-review.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--publish", action="store_true")
    args = p.parse_args()
    try:
        if args.build:
            result = build(args.stage)
        elif args.publish:
            result = frozen.publish(
                args.repo_id, args.stage, LOCAL, review=args.review, validator=validate, files=FILES
            )
        else:
            result = validate(args.stage, review=args.review)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if args.dry_run and not result["publication_ready"] else 0
    except (ValueError, OSError, KeyError) as e:
        p.exit(1, f"Public release failed: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
