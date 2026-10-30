"""CPU-only candidate construction and strict source blindness regressions."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard import test_workflow as workflow
from jais_guard.annotations import (
    import_candidates,
    provenance_after_review,
    save_annotation,
    set_state,
)
from jais_guard.test_collection import CATEGORIES, adopt, construct, corrupt

ROOT = Path(__file__).resolve().parents[1]


def test_construction_preserves_provisional_labels_and_families():
    # Synthetic parents exercise construction without distributing private sidecars.
    dialects = (
        ["MSA"] * 80
        + sum(([d] * 30 for d in ("Tunisian", "Egyptian", "Levantine", "Gulf")), [])
        + ["MSA"] * 140
    )
    parents = [
        {
            "text": "كيف أنقل صوري من الهاتف إلى الحاسوب من غير فقدان الملفات "
            + "".join("ابتثجحخدذر"[int(c)] for c in str(i)),
            "dialect": dialect,
        }
        for i, dialect in enumerate(dialects)
    ]
    rows, private = construct(parents)
    assert (rows, private) == construct(parents)
    assert Counter(r["subtype"] for r in rows) == CATEGORIES
    assert Counter(r["intended_label"] for r in rows) == {"VALID": 200, "GIBBERISH": 200}
    assert len({r["id"] for r in rows}) == len({r["family_id"] for r in rows}) == 400
    assert len({s["parent_sha256"] for s in private if s["original_text"]}) == 340
    assert all(r["review_status"] == "pending-human-review" for r in rows)
    assert all("original_text" not in r and "operations" not in r for r in rows)
    mechanisms = {}
    for label in ("VALID", "GIBBERISH"):
        mechanisms[label] = {
            m for s in private if s["proposal_label"] == label for m in s["mechanisms"]
        }
    assert {"delete", "adjacent_key", "transpose", "repeat", "join", "split"} <= (
        mechanisms["VALID"] & mechanisms["GIBBERISH"]
    )


def test_corruption_trace_is_actual_and_replayable():
    original = "كيف أنقل صوري من الهاتف إلى الحاسوب من غير فقدان الملفات"
    output, trace = corrupt(original, 710043, "recoverable_corruption")
    current = original
    for step in trace:
        assert step["before"] == current
        assert step["before"] != step["after"]
        current = step["after"]
    assert output == " ".join(current.split())


def test_strict_blindness_survives_human_label(tmp_path):
    row = workflow.read_jsonl(ROOT / "data/pilot_v1/candidates.jsonl")[0]
    db = tmp_path / "review.sqlite3"
    import_candidates([row], db)
    save_annotation(row["id"], label="VALID", flagged=False, notes="", path=db)
    assert provenance_after_review(row["id"], db)["id"] == row["id"]
    set_state("blind_sources", "1", db)
    with pytest.raises(ValueError, match="remains hidden"):
        provenance_after_review(row["id"], db)


def test_duplicate_parent_family_rejected_before_import(tmp_path, monkeypatch):
    rows = workflow.read_jsonl(ROOT / "data/pilot_v1/candidates.jsonl")[:2]
    for row in rows:
        row.update(
            partition="test",
            review_status="pending-human-review",
            intended_label_status="provisional",
            source_method="python",
            family_id="shared-parent",
            generation={"seed": 1, "recipe": "test", "run_id": "test"},
        )
    path = tmp_path / "draft.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(workflow, "verify_lock", lambda _: ({}, []))
    with pytest.raises(ValueError, match="one scored candidate"):
        workflow.check_candidates(tmp_path, path, allow_partial=True)


def test_adoption_preserves_parent_and_refuses_overwrite(tmp_path, monkeypatch):
    previous = tmp_path / "old"
    previous.mkdir()
    old = previous / "selection_lock.json"
    old.write_text('{"immutable":"old"}')
    original = old.read_bytes()
    import jais_guard.test_collection as collection

    monkeypatch.setattr(collection, "verify_lock", lambda _: ({"checkpoints": ["locked"]}, []))
    new = tmp_path / "new"
    adopt(new, previous, {"version": "test-400-v1", "target": 400})
    assert old.read_bytes() == original
    assert json.loads((new / "selection_lock.json").read_text())["checkpoints"] == ["locked"]
    with pytest.raises(ValueError, match="never overwrite"):
        adopt(new, previous, {})


@pytest.mark.integration
def test_original_construction_sidecars():
    path = ROOT / "data/test400_collection_v2/private/parents.jsonl"
    if not path.exists():
        pytest.skip("Original test parents unavailable; historical reconstruction is limited")
    rows, sidecar = construct(workflow.read_jsonl(path))
    assert len(rows) == 400 and len(sidecar) == 400
