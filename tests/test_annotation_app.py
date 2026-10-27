import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import import_candidates, queue

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def launch(db):
    app = AppTest.from_file(str(ROOT / "apps/annotate.py"))
    app.session_state["db_path"] = db
    return app.run()


def test_blind_review_flag_notes_and_restart(tmp_path):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:2]
    db = tmp_path / "app.sqlite3"
    import_candidates(candidates, db)
    app = launch(db)
    assert not app.exception
    assert not app.json
    assert not app.checkbox
    current = queue("train", path=db)[0]
    button(app, "Flag for Review").click().run()
    assert queue("train", path=db)[0]["flagged"]
    button(app, "VALID").click().run()
    assert queue("train", path=db)[0]["flagged"]
    assert queue("train", path=db)[0]["human_label"] == "VALID"
    app.text_area[0].set_value("مراجعة مكتملة").run()
    assert queue("train", path=db)[0]["notes"] == "مراجعة مكتملة"
    button(app, "Resolve review flag").click().run()
    assert not queue("train", path=db)[0]["flagged"]
    button(app, "Next").click().run()
    resumed = launch(db)
    assert not resumed.exception
    assert any("2 of 2" in c.value for c in resumed.caption)
    assert current["id"] != queue("train", path=db)[1]["id"]


def test_save_failure_does_not_navigate(tmp_path, monkeypatch):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:2]
    db = tmp_path / "app.sqlite3"
    import_candidates(candidates, db)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("jais_guard.annotations.save_annotation", fail)
    app = launch(db)
    button(app, "Next").click().run()
    assert not app.exception
    assert any("1 of 2" in c.value for c in app.caption)
    assert any("disk full" in e.value for e in app.error)
