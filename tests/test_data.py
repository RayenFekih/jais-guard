import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.data import DATA_PATH, load_dataset


def test_existing_dataset_preserves_records_and_splits():
    original = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]
    loaded = load_dataset()
    assert len(load_dataset(split="dev")) == 40
    assert len(load_dataset(split="test")) == 160
    for source, row in zip(original, loaded, strict=True):
        assert all(row[key] == value for key, value in source.items())
        assert row["text"] == source["input_text"]
        assert row["label"] == ("GIBBERISH" if source["is_gibberish"] else "VALID")
    assert any(row["review_reason"] for row in loaded)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "input_text"),
        ("boolean", "JSON boolean"),
        ("duplicate_id", "duplicate ID"),
        ("duplicate_text", "duplicate text"),
        ("split", "unknown split"),
        ("counts", "Expected dev=40"),
    ],
)
def test_invalid_dataset(tmp_path, problem, match):
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]
    if problem == "missing":
        del rows[0]["input_text"]
    elif problem == "boolean":
        rows[0]["is_gibberish"] = "false"
    elif problem == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "duplicate_text":
        rows[1]["input_text"] = " " + rows[0]["input_text"] + " "
    elif problem == "split":
        rows[0]["split"] = "train"
    else:
        rows.pop()
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match=match):
        load_dataset(path)
