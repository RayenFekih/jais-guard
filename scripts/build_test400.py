# ruff: noqa: E402
"""Build approved, provisional test candidates. No model imports or human labels."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jais_guard.data.annotations import benchmark_references, quality_issues
from jais_guard.data.test_collection import CATEGORIES, RUN, adopt, construct
from jais_guard.data.test_workflow import (
    check_candidates,
    import_collection,
    read_jsonl,
    sha,
    verify_lock,
    write_new,
)


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def historical_references(exclude):
    """All locally stored candidate/generation histories, not unselected raw corpora."""
    found, inventory = {}, {}

    def visit(value, path):
        if isinstance(value, dict):
            text = value.get("input_text")
            # Historical selected parent material uses the same text key in some receipts.
            if not text and value.get("parent_id"):
                text = value.get("text") or value.get("parent_text")
            if isinstance(text, str) and text.strip():
                row = {
                    **value,
                    "id": value.get("id", "archive:" + str(path) + str(len(found))),
                    "input_text": text,
                    "partition": value.get("partition", "historical-reserved"),
                }
                found[(row["id"], text)] = row
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item, path)
        elif isinstance(value, list):
            for item in value:
                visit(item, path)

    for path in sorted((ROOT / "data").rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".json", ".jsonl"}
            or exclude in path.parents
            or ROOT / "data/raw" in path.parents
        ):
            continue
        try:
            values = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        visit(values, path)
        inventory[str(path.relative_to(ROOT))] = sha(path)
    return list(found.values()), inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/test400_collection_v2")
    parser.add_argument("--import-for-review", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    # Fail before any write if a completed construction already exists.
    if (directory / "draft.jsonl").exists():
        raise ValueError("Immutable draft already exists; do not regenerate over candidate history")
    parents = read_jsonl(directory / "private/parents.jsonl")
    rows, sidecar = construct(parents)
    references, inventory = historical_references(directory)
    plan = {
        **json.loads((ROOT / "configs/test400_plan.json").read_text()),
        "collection_protocol": "request-corruption-v2",
        "proposed_categories": CATEGORIES,
        "origin_disclosure": (
            "Assistant-authored request scenarios and simulated corruption; "
            "not observed production traffic"
        ),
        "one_scored_example_per_family": True,
        "review_policy": "Input-only judgments; uncertainty flagged; original never exposed in UI",
    }
    adopt(directory, ROOT / "data/test400_v1", plan)
    write_rows(directory / "draft.jsonl", rows)
    write_rows(directory / "private/corruptions.jsonl", sidecar)
    write_rows(directory / "private/historical_references.jsonl", references)
    write_new(directory / "private/reference_inventory.json", inventory)
    lockpath = directory / "selection_lock.json"
    lock = json.loads(lockpath.read_text())
    lock["additional_reference_files"] = {
        str(directory / "private/historical_references.jsonl"): sha(
            directory / "private/historical_references.jsonl"
        )
    }
    lock["collection_files"] = {
        str(p): sha(p)
        for p in (
            directory / "private/parents.jsonl",
            directory / "private/corruptions.jsonl",
            directory / "private/reference_inventory.json",
            ROOT / "src/jais_guard/data/test_collection.py",
        )
    }
    # This is initial construction of a NEW lock; never changes the old protocol.
    lockpath.write_text(json.dumps(lock, indent=2) + "\n")
    _, report = check_candidates(directory, directory / "draft.jsonl")
    parent_rows = [
        {
            "id": s["family_id"],
            "input_text": s["original_text"],
            "partition": "test",
            "family_id": s["family_id"],
        }
        for s in sidecar
        if s["original_text"]
    ]
    # quality_issues validates only incoming rows, so wrap parents using their candidate schema.
    by_family = {r["family_id"]: r for r in rows}
    parent_rows = [{**by_family[p["family_id"]], **p} for p in parent_rows]
    _, all_refs = verify_lock(directory)
    parent_issues = quality_issues(parent_rows, all_refs + benchmark_references())
    report.update(
        {
            "run_id": RUN,
            "category_counts": dict(Counter(r["subtype"] for r in rows)),
            "parent_overlap_issues": parent_issues,
            "history_files_scanned": len(inventory),
            "historical_text_records": len(references),
            "unique_candidate_families": len({r["family_id"] for r in rows}),
            "request_parent_count": len(parent_rows),
            "human_labels": 0,
            "origin": "340 assistant-authored request families; 60 simulated sequence families",
            "mechanisms_by_proposal": {
                label: dict(
                    Counter(
                        m for s in sidecar if s["proposal_label"] == label for m in s["mechanisms"]
                    )
                )
                for label in ("VALID", "GIBBERISH")
            },
        }
    )
    write_new(directory / "quality_report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"issues", "parent_overlap_issues"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "Candidate overlap flags:",
        len(report["issues"]),
        "Parent overlap flags:",
        len(parent_issues),
    )
    if args.import_for_review:
        if parent_issues:
            raise ValueError("Parent overlap must be resolved before importing")
        print(
            "Imported provisional candidates:",
            import_collection(directory, directory / "draft.jsonl"),
        )


if __name__ == "__main__":
    main()
