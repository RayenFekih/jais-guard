"""Independent-test preparation. No model imports, predictions, or evaluation."""

import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from .annotations import (
    ROOT,
    benchmark_references,
    comparison_text,
    connection,
    export_dataset,
    import_candidates,
    now,
    quality_issues,
    read_export,
    set_state,
    validate_candidate,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def readonly_candidates(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Missing reference DB: {path}")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        return [json.loads(row[0]) for row in db.execute("SELECT payload FROM candidates")]


def selection_inventory(results):
    """Lock all six existing checkpoints/thresholds; never choose using test labels."""
    inventory = []
    for approach in ("frozen", "lora"):
        for seed in (42, 43, 44):
            candidates = sorted(Path(results).glob(f"*/{approach}-{seed}/best/classifier.json"))
            complete = [
                p for p in candidates if (p.parent.parent / "evaluation/metrics.json").exists()
            ]
            if len(complete) != 1:
                raise ValueError(f"Expected one completed {approach}-{seed}; found {len(complete)}")
            metadata = complete[0]
            directory = metadata.parent.parent
            paths = [metadata, metadata.parent / "head.safetensors", directory / "threshold.json"]
            paths += sorted((metadata.parent / "tokenizer").glob("*"))
            paths = [p for p in paths if p.is_file() or p.suffix in {".safetensors", ".json"}]
            if approach == "lora":
                paths += [
                    metadata.parent / "adapter/adapter_model.safetensors",
                    metadata.parent / "adapter/adapter_config.json",
                ]
            saved = json.loads(metadata.read_text())
            if saved["approach"] != approach or saved["seed"] != seed:
                raise ValueError("Checkpoint metadata mismatch")
            inventory.append(
                {
                    "approach": approach,
                    "seed": seed,
                    "model": saved["config"]["model"],
                    "revision": saved["config"]["revision"],
                    "train_sha256": saved["config"]["train_sha256"],
                    "validation_sha256": saved["config"]["validation_sha256"],
                    "files": {str(p.resolve()): sha(p) for p in paths},
                }
            )
    if (
        len(
            {
                (r["model"], r["revision"], r["train_sha256"], r["validation_sha256"])
                for r in inventory
            }
        )
        != 1
    ):
        raise ValueError("Experiments do not share the same pinned model and datasets")
    return inventory


def initialize(directory, results, reference_db, plan_path):
    directory = Path(directory)
    if directory.exists():
        raise ValueError(
            "Collection directory already exists; do not overwrite an existing protocol"
        )
    inventory = selection_inventory(results)
    references = readonly_candidates(reference_db)
    if any(r["partition"] == "test" for r in references):
        raise ValueError(
            "Reference DB already contains test records; review existing collection first"
        )
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("version") != "test-400-v1" or plan.get("target") != 400:
        raise ValueError("Expected the test-400-v1 400-example plan")
    directory.mkdir(parents=True)
    write_new(directory / "plan.json", plan)
    write_new(
        directory / "selection_lock.json",
        {
            "created_at": now(),
            "plan_sha256": sha(directory / "plan.json"),
            "evaluation_policy": "All six selected checkpoints; default 0.5 and "
            "their already-selected validation thresholds. No test-based selection or tuning.",
            "checkpoints": inventory,
            "prompt_files": {
                str(p.resolve()): sha(p)
                for p in (ROOT / "configs/prompt_baseline.json", ROOT / "src/jais_guard/prompts.py")
            },
            "reference_db": str(Path(reference_db).resolve()),
            "reference_candidates_sha256": hashlib.sha256(
                json.dumps(
                    sorted(references, key=lambda r: r["id"]), sort_keys=True, ensure_ascii=False
                ).encode()
            ).hexdigest(),
        },
    )
    # Separate DB: importing/reviewing test candidates cannot affect training annotations.
    with connection(directory / "annotations.sqlite3"):
        pass
    set_state(
        "test_unlocked",
        "test-400-v1 protocol locked; collection/review only",
        directory / "annotations.sqlite3",
    )
    set_state("active_partition", "test", directory / "annotations.sqlite3")
    set_state("independent_test_workflow", "test-400-v1", directory / "annotations.sqlite3")
    return directory


def verify_lock(directory):
    directory = Path(directory)
    lock = json.loads((directory / "selection_lock.json").read_text())
    if sha(directory / "plan.json") != lock["plan_sha256"]:
        raise ValueError("Collection plan changed after protocol lock")
    for record in lock["checkpoints"]:
        for path, expected in record["files"].items():
            if sha(path) != expected:
                raise ValueError(f"Locked checkpoint/threshold changed: {path}")
    for path, expected in lock["prompt_files"].items():
        if sha(path) != expected:
            raise ValueError(f"Locked prompt changed: {path}")
    references = readonly_candidates(lock["reference_db"])
    actual = hashlib.sha256(
        json.dumps(
            sorted(references, key=lambda r: r["id"]), sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    if actual != lock["reference_candidates_sha256"]:
        raise ValueError("Reference collection changed; review independence against a new protocol")
    if lock.get("parent_protocol"):
        parent_lock = Path(lock["parent_protocol"]) / "selection_lock.json"
        if sha(parent_lock) != lock["parent_lock_sha256"]:
            raise ValueError("Parent protocol lock changed")
    for path, expected in lock.get("collection_files", {}).items():
        if sha(path) != expected:
            raise ValueError(f"Collection provenance changed: {path}")
    for path, expected in lock.get("additional_reference_files", {}).items():
        if sha(path) != expected:
            raise ValueError(f"Historical reference snapshot changed: {path}")
        references.extend(read_jsonl(path))
    return lock, references


def distribution(rows, label_field):
    by_method = defaultdict(Counter)
    lengths = defaultdict(list)
    bins = defaultdict(Counter)
    for row in rows:
        label = row[label_field]
        by_method[row["source_method"]][label] += 1
        length = len(comparison_text(row["input_text"]))
        lengths[label].append(length)
        bucket = (
            "1-15"
            if length <= 15
            else "16-40"
            if length <= 40
            else ("41-100" if length <= 100 else "101+")
        )
        bins[bucket][label] += 1
    warnings = []
    for method, counts in by_method.items():
        if len(counts) < 2:
            warnings.append(f"source_method={method} has only one {label_field}: {dict(counts)}")
    for bucket, counts in bins.items():
        if sum(counts.values()) >= 10 and len(counts) < 2:
            warnings.append(f"length bucket {bucket} has one class only")
    summary = {
        label: {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "median": sorted(values)[len(values) // 2],
        }
        for label, values in lengths.items()
    }
    if len(summary) == 2:
        medians = [r["median"] for r in summary.values()]
        if max(medians) > 2 * max(1, min(medians)):
            warnings.append("Class median character lengths differ by more than 2x")
    return {
        "class_counts": dict(Counter(r[label_field] for r in rows)),
        "source_method_by_label": {k: dict(v) for k, v in by_method.items()},
        "length_bin_by_label": {k: dict(v) for k, v in bins.items()},
        "lengths": summary,
        "confounding_warnings": warnings,
        "dialect_by_label": dict(Counter(f"{r['dialect']}:{r[label_field]}" for r in rows)),
    }


def reference_overlap(rows, references):
    """Beyond lexical matches: never reuse source records, contributors, or parents."""
    issues = quality_issues(rows, references)
    for row in rows:
        provenance = row["provenance"]
        for other in references:
            previous = other.get("provenance", {})
            if (provenance.get("source"), provenance.get("source_id")) == (
                previous.get("source"),
                previous.get("source_id"),
            ):
                issues.append(
                    {
                        "id": row["id"],
                        "other_id": other["id"],
                        "kind": "reserved_source_record",
                        "cross_partition": True,
                    }
                )
            contributor = provenance.get("contributor_group") or provenance.get("author_group")
            if (
                contributor
                and contributor
                == (previous.get("contributor_group") or previous.get("author_group"))
                and (provenance.get("source") == previous.get("source"))
            ):
                issues.append(
                    {
                        "id": row["id"],
                        "other_id": other["id"],
                        "kind": "reserved_contributor_group",
                        "cross_partition": True,
                    }
                )
            generation = row.get("generation", {})
            older_generation = other.get("generation", {})
            for field in ("run_id", "batch_id"):
                if generation.get(field) and generation[field] == older_generation.get(field):
                    issues.append(
                        {
                            "id": row["id"],
                            "other_id": other["id"],
                            "kind": f"reserved_{field}",
                            "cross_partition": True,
                        }
                    )
            if (
                generation.get("recipe")
                and generation.get("seed") is not None
                and (generation["recipe"], generation["seed"])
                == (older_generation.get("recipe"), older_generation.get("seed"))
            ):
                issues.append(
                    {
                        "id": row["id"],
                        "other_id": other["id"],
                        "kind": "reserved_recipe_seed",
                        "cross_partition": True,
                    }
                )
            parent = row.get("parent_id") or row.get("generation", {}).get("parent_id")
            if parent and parent == other["id"]:
                issues.append(
                    {
                        "id": row["id"],
                        "other_id": parent,
                        "kind": "reserved_parent",
                        "cross_partition": True,
                    }
                )
    return issues


def check_candidates(directory, candidate_path, *, allow_partial=False):
    lock, references = verify_lock(directory)
    del lock
    rows = read_jsonl(candidate_path)
    if not rows or len(rows) > 400 or (len(rows) != 400 and not allow_partial):
        raise ValueError(
            "Exactly 400 candidates are required; use read-only allow_partial for a draft subset"
        )
    for row in rows:
        validate_candidate(row)
        if row["partition"] != "test":
            raise ValueError("Only test candidates can enter this workflow")
        if row.get("review_status") != "pending-human-review":
            raise ValueError(
                "Candidates must declare pending-human-review; intentions are provisional"
            )
        if row.get("intended_label_status") != "provisional":
            raise ValueError("Candidate intentions must explicitly remain provisional")
        if any(
            "prediction" in key or key in {"score", "p_gibberish", "classifier_label"}
            for key in row
        ):
            raise ValueError("Classifier-derived candidate selection/labels are forbidden")
        generation = row.get("generation", {})
        if row["source_method"] == "python" and not generation.get("recipe"):
            raise ValueError("Python records need an explicit recipe as well as a fresh seed/run")
        if row["source_method"] in {"codex", "python"} and not generation.get("run_id"):
            raise ValueError("Generated records need reserved test-only run_id metadata")
    ids = {r["id"] for r in rows}
    if len({r["family_id"] for r in rows}) != len(rows):
        raise ValueError("Only one scored candidate per parent family is permitted")
    if any((r.get("parent_id") or r.get("generation", {}).get("parent_id")) in ids for r in rows):
        raise ValueError(
            "Do not include a transformation and its parent in the same 400-row collection"
        )
    issues = reference_overlap(rows, references + benchmark_references())
    summary = distribution(rows, "intended_label")
    plan = json.loads((Path(directory) / "plan.json").read_text())
    meaningful_dialects = Counter(r["dialect"] for r in rows if r["intended_label"] == "VALID")
    summary["valid_dialect_target_shortfall"] = {
        dialect: target - meaningful_dialects[dialect]
        for dialect, target in plan["proposed_valid_dialects"].items()
        if meaningful_dialects[dialect] < target
    }
    summary["proposed_class_target_met"] = summary["class_counts"] == plan["proposed_classes"]
    return rows, {
        "candidate_sha256": sha(candidate_path),
        "count": len(rows),
        "count_complete": len(rows) == 400,
        "issues": issues,
        "distribution": summary,
        "ready_for_import": len(rows) == 400
        and not any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in issues),
    }


def import_collection(directory, candidate_path):
    directory = Path(directory)
    if (directory / "candidates.jsonl").exists():
        raise ValueError("Candidate collection is immutable once imported; use a new version")
    rows, report = check_candidates(directory, candidate_path)
    if not report["ready_for_import"]:
        raise ValueError("Overlap blocks import; inspect check report and replace candidates")
    if (
        report["distribution"]["confounding_warnings"]
        or report["distribution"]["valid_dialect_target_shortfall"]
        or not report["distribution"]["proposed_class_target_met"]
    ):
        raise ValueError("Resolve proposed coverage/source/length confounding before import")
    # Importer flags within-test near-duplicates for explicit blind human resolution.
    count = import_candidates(rows, path=directory / "annotations.sqlite3")
    shutil.copyfile(candidate_path, directory / "candidates.jsonl")
    write_new(
        directory / "import_receipt.json",
        {
            **report,
            "imported": count,
            "created_at": now(),
            "selection_lock_sha256": sha(directory / "selection_lock.json"),
        },
    )
    return count


def freeze(directory, destination, confounding_note=None):
    directory = Path(directory)
    lock, references = verify_lock(directory)
    candidate_path = directory / "candidates.jsonl"
    receipt = json.loads((directory / "import_receipt.json").read_text())
    if sha(directory / "selection_lock.json") != receipt["selection_lock_sha256"]:
        raise ValueError("Selection protocol changed after import")
    if sha(candidate_path) != receipt["candidate_sha256"]:
        raise ValueError("Imported candidate snapshot changed")
    # Read the DB without upgrading it or mutating annotation/history tables.
    with sqlite3.connect(
        f"file:{(directory / 'annotations.sqlite3').resolve()}?mode=ro", uri=True
    ) as db:
        stored = db.execute(
            "SELECT c.payload,a.human_label,a.flagged,a.notes FROM candidates c "
            "LEFT JOIN annotations a ON a.id=c.id WHERE c.partition='test'"
        ).fetchall()
        excluded = db.execute("SELECT COUNT(*) FROM candidate_exclusions").fetchone()[0]
    if excluded:
        raise ValueError("Freeze requires exactly 400 active candidates with no excluded originals")
    if len(stored) != 400 or any(
        label not in {"VALID", "GIBBERISH"} or flagged for _, label, flagged, _ in stored
    ):
        raise ValueError("Freeze requires all 400 human labels and no unresolved flags")
    rows = []
    for payload, label, _, _ in stored:
        rows.append({**json.loads(payload), "final_label": label})
    original = {r["id"]: r for r in read_jsonl(candidate_path)}
    if {r["id"] for r in rows} != set(original) or any(
        {k: v for k, v in r.items() if k != "final_label"} != original[r["id"]] for r in rows
    ):
        raise ValueError("DB candidates differ from immutable imported snapshot")
    candidates = [{k: v for k, v in r.items() if k != "final_label"} for r in rows]
    issues = reference_overlap(candidates, references + benchmark_references())
    if any(i["kind"] != "near_duplicate" or i["cross_partition"] for i in issues):
        raise ValueError("Independent-test contamination checks failed")
    final_distribution = distribution(rows, "final_label")
    if len(final_distribution["class_counts"]) != 2:
        raise ValueError("Both human-labeled classes must be present")
    if final_distribution["confounding_warnings"] and not (confounding_note or "").strip():
        raise ValueError(
            "Final-label confounding needs an explicit review note, never relabel to quotas"
        )
    # Existing exporter creates a new directory and refuses overwrite.
    folder = export_dataset(
        "test", "test-400-v1", Path(destination), path=directory / "annotations.sqlite3"
    )
    if len(read_export(folder, "test")) != 400:
        raise ValueError("Expected exactly 400 exported final human labels")
    write_new(
        folder / "freeze.json",
        {
            "version": "test-400-v1",
            "frozen_at": now(),
            "human_review_complete": True,
            "test_sha256": sha(folder / "test.jsonl"),
            "manifest_sha256": sha(folder / "manifest.json"),
            "candidate_sha256": receipt["candidate_sha256"],
            "selection_lock": lock,
            "selection_lock_sha256": sha(directory / "selection_lock.json"),
            "distribution": final_distribution,
            "confounding_review_note": confounding_note,
            "near_duplicate_pairs": [i for i in issues if i["kind"] == "near_duplicate"],
            "evaluation_allowed_only_with_locked_protocol": True,
        },
    )
    return folder


def read_frozen_test(folder):
    """Future test evaluators must call this gate, not load candidate JSONL directly."""
    folder = Path(folder)
    frozen = json.loads((folder / "freeze.json").read_text())
    if (
        not frozen["human_review_complete"]
        or sha(folder / "test.jsonl") != frozen["test_sha256"]
        or sha(folder / "manifest.json") != frozen["manifest_sha256"]
    ):
        raise ValueError("Frozen export changed or review incomplete")
    for record in frozen["selection_lock"]["checkpoints"]:
        for path, expected in record["files"].items():
            if sha(path) != expected:
                raise ValueError("Selected model or threshold changed after protocol lock")
    for path, expected in frozen["selection_lock"]["prompt_files"].items():
        if sha(path) != expected:
            raise ValueError("Prompt changed after protocol lock")
    rows = read_export(folder, "test")
    if len(rows) != 400 or any(r["review_status"] != "human-reviewed" for r in rows):
        raise ValueError("Expected 400 final human labels")
    return rows
