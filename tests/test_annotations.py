import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import (
    connection,
    exclude_transformed_originals,
    export_dataset,
    get_state,
    import_candidates,
    progress,
    provenance_after_review,
    quality_issues,
    queue,
    read_export,
    save_annotation,
    set_state,
    validate_candidate,
)


@pytest.fixture
def candidate():
    return {
        "id": "opaque-1",
        "input_text": "كيف أختار دراجة مناسبة للتنقل اليومي",
        "partition": "train",
        "source_method": "codex",
        "intended_label": "GIBBERISH",
        "subtype": "hidden",
        "dialect": "MSA",
        "domain": "travel",
        "family_id": "family-1",
        "provenance": {
            "source": "fixture",
            "source_id": "1",
            "license": "project-original",
            "attribution": "fixture",
            "rights_evidence": "fixture",
        },
        "generation": {"batch_id": "fixture"},
    }


def test_blind_storage_and_human_override_survive_restart(tmp_path, candidate):
    db = tmp_path / "annotations.sqlite3"
    import_candidates([candidate], db)
    row = queue("train", path=db)[0]
    assert "intended_label" not in row and "provenance" not in row and "subtype" not in row
    assert row["human_label"] is None
    with pytest.raises(ValueError, match="hidden"):
        provenance_after_review(candidate["id"], db)
    save_annotation(candidate["id"], label="VALID", flagged=False, notes="معنى واضح", path=db)
    set_state("cursor:train:all", candidate["id"], db)
    assert queue("train", path=db)[0]["human_label"] == "VALID"
    assert get_state("cursor:train:all", db) == candidate["id"]
    assert provenance_after_review(candidate["id"], db)["intended_label"] == "GIBBERISH"
    folder = export_dataset("train", "v1", tmp_path / "exports", path=db)
    rows = read_export(folder, "train", allow_partial=True)
    assert rows[0]["label"] == "VALID" and rows[0]["is_gibberish"] is False
    assert rows[0]["intended_label"] == "GIBBERISH"
    with pytest.raises(ValueError, match="Incomplete"):
        read_export(folder, "train")
    with pytest.raises(ValueError, match="partition"):
        read_export(folder, "validation", allow_partial=True)
    with pytest.raises(FileExistsError):
        export_dataset("train", "v1", tmp_path / "exports", path=db)
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


def test_flags_exclude_until_explicit_resolution(tmp_path, candidate):
    db = tmp_path / "annotations.sqlite3"
    import_candidates([candidate], db)
    save_annotation(candidate["id"], label=None, flagged=True, notes="uncertain", path=db)
    assert progress(db)["train"]["flagged"] == 1
    assert len(queue("train", "flagged", db)) == 1
    with pytest.raises(ValueError, match="No resolved"):
        export_dataset("train", "v1", tmp_path, path=db)
    save_annotation(candidate["id"], label="VALID", flagged=True, notes="still reviewing", path=db)
    assert progress(db)["train"]["exportable"] == 0
    save_annotation(candidate["id"], label="VALID", flagged=False, notes="resolved", path=db)
    assert progress(db)["train"]["exportable"] == 1
    assert queue("train", "flagged", db) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("intended_label", "AMBIGUOUS"),
        ("partition", "dev"),
        ("input_text", "hello"),
        ("input_text", "   "),
        ("input_text", "مرحبا hello"),
    ],
)
def test_invalid_candidate(field, value, candidate):
    candidate[field] = value
    with pytest.raises(ValueError):
        validate_candidate(candidate)


def test_labels_are_binary_and_foreign_keys_fail_atomically(tmp_path, candidate):
    db = tmp_path / "a.sqlite3"
    import_candidates([candidate], db)
    with pytest.raises(ValueError, match="Only VALID"):
        save_annotation(candidate["id"], label="AMBIGUOUS", flagged=False, notes="", path=db)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        save_annotation("missing", label="VALID", flagged=False, notes="", path=db)
    assert progress(db)["train"]["reviewed"] == 0


def test_exact_near_and_family_overlap(candidate, tmp_path):
    other = copy.deepcopy(candidate)
    other["id"] = "opaque-2"
    assert quality_issues([other], [candidate])[0]["kind"] == "duplicate_text"
    other["input_text"] += "؟"
    assert quality_issues([other], [candidate])[0]["kind"] == "near_duplicate"
    db = tmp_path / "a.sqlite3"
    import_candidates([candidate, other], db)
    assert progress(db)["train"]["flagged"] == 1
    other["id"] = "opaque-3"
    other["partition"] = "validation"
    other["input_text"] = "هل تتوفر مواعيد لزيارة المتحف خلال عطلة نهاية الأسبوع"
    assert quality_issues([other], [candidate])[0]["kind"] == "cross_partition_family"
    with pytest.raises(ValueError, match="overlap"):
        import_candidates([other], db)


def test_test_partition_locked(tmp_path, candidate):
    candidate["partition"] = "test"
    with pytest.raises(ValueError, match="locked"):
        import_candidates([candidate], tmp_path / "a.sqlite3")


def test_pilot_distribution_and_provenance():
    path = Path(__file__).resolve().parents[1] / "data/pilot_v1/candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    from collections import Counter

    assert Counter((r["source_method"], r["intended_label"]) for r in rows) == {
        ("public", "VALID"): 30,
        ("codex", "VALID"): 15,
        ("codex", "GIBBERISH"): 40,
        ("python", "VALID"): 5,
        ("python", "GIBBERISH"): 10,
    }
    for row in rows:
        validate_candidate(row)
        assert row["partition"] == "train"
        assert "human_label" not in row


def test_export_detects_tampering(tmp_path, candidate):
    db = tmp_path / "a.sqlite3"
    import_candidates([candidate], db)
    save_annotation(candidate["id"], label="VALID", flagged=False, notes="", path=db)
    folder = export_dataset("train", "v1", tmp_path / "exports", path=db)
    with (folder / "train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash"):
        read_export(folder, "train", allow_partial=True)


def test_modified_only_policy_preserves_reviews_and_excludes_originals(tmp_path, candidate):
    path = tmp_path / "annotations.sqlite3"
    original = candidate
    modified = copy.deepcopy(original)
    modified.update(id="modified", input_text=original["input_text"] + "؟")
    modified["generation"]["parent_id"] = original["id"]
    import_candidates([original], path)
    save_annotation(original["id"], label="VALID", flagged=False, notes="Reviewed", path=path)
    import_candidates([modified], path)
    assert [r["id"] for r in queue("train", path=path)] == ["modified"]
    save_annotation("modified", label="VALID", flagged=False, notes="Resolved", path=path)
    with connection(path) as db:
        before = [tuple(r) for r in db.execute("SELECT * FROM annotations ORDER BY id")]
        history = [tuple(r) for r in db.execute("SELECT * FROM revisions ORDER BY revision")]
        db.execute("DELETE FROM candidate_exclusions")  # Simulate the pre-policy database.
    assert exclude_transformed_originals(path) == [original["id"]]
    assert exclude_transformed_originals(path) == [original["id"]]
    with connection(path) as db:
        assert [tuple(r) for r in db.execute("SELECT * FROM annotations ORDER BY id")] == before
        assert [
            tuple(r) for r in db.execute("SELECT * FROM revisions ORDER BY revision")
        ] == history
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM candidate_exclusions").fetchone()[0] == 1
    folder = export_dataset("train", "modified-only", tmp_path / "exports", path=path)
    assert [r["id"] for r in read_export(folder, "train", allow_partial=True)] == ["modified"]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["archived_originals"][0]["id"] == original["id"]
    assert progress(path)["train"]["imported"] == 1
