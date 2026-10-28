import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import (
    connection,
    import_candidates,
    queue,
    save_annotation,
    validate_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data/milestone300_v1"


def candidates(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_new_batch_counts_and_independent_difficulty():
    rows = candidates(BATCH / "candidates.jsonl")
    assert len(rows) == 200
    assert Counter(r["intended_label"] for r in rows) == {"VALID": 96, "GIBBERISH": 104}
    assert Counter(r["difficulty"] for r in rows if r["intended_label"] == "GIBBERISH") == {
        "easy": 47,
        "medium": 42,
        "hard": 15,
    }
    assert all(r["partition"] == "train" for r in rows)
    assert all("human_label" not in r and "annotation" not in r for r in rows)
    for row in rows:
        validate_candidate(row)
    old = candidates(ROOT / "data/pilot_v1/candidates.jsonl")
    assert all("difficulty" not in r for r in old)
    invalid = {**rows[0], "difficulty": "guess"}
    with pytest.raises(ValueError, match="Difficulty"):
        validate_candidate(invalid)


def test_import_preserves_prior_reviews_and_history(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    old = candidates(ROOT / "data/pilot_v1/candidates.jsonl")
    new = candidates(BATCH / "candidates.jsonl")
    import_candidates(old, path)
    for row in old[:3]:
        save_annotation(row["id"], label="VALID", flagged=False, notes="Original review", path=path)
    save_annotation(old[0]["id"], label="GIBBERISH", flagged=False, notes="Correction", path=path)
    with connection(path) as db:
        before = {r[0]: tuple(r) for r in db.execute("SELECT * FROM annotations")}
        revisions = [tuple(r) for r in db.execute("SELECT * FROM revisions ORDER BY revision")]
    assert import_candidates(new, path) == 200
    assert len(queue("train", path=path)) == 273
    with connection(path) as db:
        after = {r[0]: tuple(r) for r in db.execute("SELECT * FROM annotations")}
        assert all(after[key] == value for key, value in before.items())
        assert [
            tuple(r) for r in db.execute("SELECT * FROM revisions ORDER BY revision")
        ] == revisions
    old_ids = {r["id"] for r in old}
    assert all(
        r["human_label"] is None for r in queue("train", path=path) if r["id"] not in old_ids
    )
    with pytest.raises(ValueError, match="Duplicate ID"):
        import_candidates(new, path)
    assert len(queue("train", path=path)) == 273


def test_cached_reconstruction_is_deterministic():
    if not (ROOT / "data/raw/pilot_v1/aya_train.parquet").exists():
        pytest.skip("Pinned raw source cache is not available; never download during tests")
    spec = importlib.util.spec_from_file_location(
        "candidate_recipes", ROOT / "scripts/prepare_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = module.milestone_candidates(BATCH)
    saved = candidates(BATCH / "candidates.jsonl")
    fields = ("id", "input_text", "intended_label", "difficulty", "family_id")
    assert {tuple(r[k] for k in fields) for r in rebuilt} == {
        tuple(r[k] for k in fields) for r in saved
    }
