import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.annotations import connection, import_candidates, queue, save_annotation

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
    app.text_area[0].set_value("Unsaved note")
    button(app, "Next").click().run()
    assert not app.exception
    assert any("1 of 2" in c.value for c in app.caption)
    assert any("disk full" in e.value for e in app.error)


def search(app, value):
    return (
        next(w for w in app.text_input if w.label == "Search text or example ID")
        .set_value(value)
        .run()
    )


def matching_ids(app):
    return [w.key.removeprefix("open:") for w in app.button if (w.key or "").startswith("open:")]


@pytest.fixture
def reviewed_db(tmp_path):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:3]
    db = tmp_path / "reviewed.sqlite3"
    import_candidates(candidates, db)
    records = queue("train", path=db)
    for index, row in enumerate(records):
        save_annotation(
            row["id"],
            label="VALID" if index == 0 else "GIBBERISH",
            flagged=index == 2,
            notes=f"Original note {index}",
            path=db,
        )
    return db, records


def test_search_exact_partial_arabic_and_ids(reviewed_db):
    db, records = reviewed_db
    app = launch(db)
    target = records[1]
    assert matching_ids(search(app, target["input_text"])) == [target["id"]]
    assert target["id"] in matching_ids(search(app, target["input_text"][:8]))
    assert matching_ids(search(app, target["id"].upper())) == [target["id"]]
    assert target["id"] in matching_ids(search(app, target["id"][3:9]))
    assert not app.json
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 3


@pytest.mark.parametrize(
    "mode,indices", [("VALID", [0]), ("GIBBERISH", [1, 2]), ("flagged", [2]), ("all", [0, 1, 2])]
)
def test_human_label_and_flag_filters(reviewed_db, mode, indices):
    db, records = reviewed_db
    app = launch(db)
    next(w for w in app.selectbox if w.label == "Queue").select(mode).run()
    assert matching_ids(app) == [records[i]["id"] for i in indices]
    for widget in app.button:
        if (widget.key or "").startswith("open:"):
            assert "VALID" in widget.label or "GIBBERISH" in widget.label


def test_direct_selection_correction_preserves_history_and_stays_open(reviewed_db):
    db, records = reviewed_db
    app = launch(db)
    next(w for w in app.selectbox if w.label == "Queue").select("VALID").run()
    target = records[0]
    next(w for w in app.button if w.key == f"open:{target['id']}").click().run()
    button(app, "GIBBERISH").click().run()
    assert not app.exception
    assert matching_ids(app) == []
    assert any(target["id"] in c.value and "outside" in c.value for c in app.caption)
    assert app.text_area[0].value == "Original note 0"
    with connection(db) as conn:
        history = [
            json.loads(r[0])["human_label"]
            for r in conn.execute(
                "SELECT payload FROM revisions WHERE id=? ORDER BY revision", (target["id"],)
            )
        ]
    assert history == ["VALID", "GIBBERISH"]
    assert queue("train", path=db)[0]["human_label"] == "GIBBERISH"
    next(w for w in app.selectbox if w.label == "Queue").select("all").run()
    target = records[2]
    next(w for w in app.button if w.key == f"open:{target['id']}").click().run()
    assert app.text_area[0].value == "Original note 2"
    assert any(target["id"] in c.value and "3 of 3" in c.value for c in app.caption)


def test_empty_search_keeps_selection_and_annotation_unchanged(reviewed_db):
    db, records = reviewed_db
    app = launch(db)
    search(app, "لا يوجد نص مطابق لهذا البحث")
    assert not matching_ids(app)
    assert any("No matching examples" in message.value for message in app.info)
    assert any(records[0]["id"] in c.value for c in app.caption)
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 3
