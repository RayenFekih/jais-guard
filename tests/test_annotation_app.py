import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.data.annotations import (
    connection,
    import_candidates,
    queue,
    save_annotation,
    set_state,
)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


def launch(db, view="Browse All"):
    app = AppTest.from_file(str(ROOT / "apps/annotate.py"))
    app.session_state["db_path"] = db
    if view is not None:
        app.session_state["annotation_view"] = view
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

    monkeypatch.setattr("jais_guard.data.annotations.save_annotation", fail)
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


def test_end_of_queue_links_back_to_unlabeled_records_without_changing_labels(tmp_path):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:3]
    db = tmp_path / "app.sqlite3"
    import_candidates(candidates, db)
    records = queue("train", path=db)
    save_annotation(records[-1]["id"], label="VALID", flagged=False, notes="Reviewed", path=db)
    set_state("cursor:train:all", records[-1]["id"], db)
    app = launch(db)
    assert any("Position 3 of 3" in c.value for c in app.caption)
    assert any("2 examples" in message.value for message in app.warning)
    assert button(app, "Next").disabled
    button(app, "Open next unlabeled example").click().run()
    assert not app.exception
    assert any(records[0]["id"] in c.value and "Position 1 of 3" in c.value for c in app.caption)
    assert [r["human_label"] for r in queue("train", path=db)] == [None, None, "VALID"]
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1


def test_unlabeled_navigation_does_not_bypass_failed_notes_save(tmp_path, monkeypatch):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:2]
    db = tmp_path / "app.sqlite3"
    import_candidates(candidates, db)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("jais_guard.data.annotations.save_annotation", fail)
    app = launch(db)
    app.text_area[0].set_value("Unsaved note")
    button(app, "Open next unlabeled example").click().run()
    assert any("Position 1 of 2" in c.value for c in app.caption)
    assert any("disk full" in message.value for message in app.error)


def test_provisional_audit_is_blind_and_requires_explicit_resolution(tmp_path):
    from jais_guard.data.annotations import audit_report

    candidate = json.loads((ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()[0])
    candidate.update(id="audit-fixture", source_method="codex", intended_label="GIBBERISH")
    candidate["generation"] = {
        "batch_id": "audit-fixture",
        "instructions": "Artificial fixture",
        "model_metadata": {"model_id": None},
    }
    db = tmp_path / "audit.sqlite3"
    import_candidates([candidate], db, provisional_batch="audit-fixture")
    app = launch(db)
    assert not app.exception
    assert not app.json and not app.checkbox
    assert any("Your label: **Not labeled**" in w.value for w in app.markdown)
    assert any("pending" in w.value and "0/1" in w.value for w in app.markdown)
    assert button(app, "Resolve review flag").disabled
    next(w for w in app.selectbox if w.label == "Queue").select("manual-review").run()
    button(app, "GIBBERISH").click().run()
    assert audit_report(db)[0]["status"] == "pending"
    assert any("audit-fixture" in c.value for c in app.caption)
    button(app, "Resolve review flag").click().run()
    assert audit_report(db)[0]["status"] == "passed"
    assert not app.exception
    assert queue("train", path=db)[0]["status"] == "human-reviewed"


def test_failed_audit_write_does_not_complete_or_advance_ui(tmp_path, monkeypatch):
    from jais_guard.data.annotations import audit_report

    candidate = json.loads((ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()[0])
    candidate.update(id="audit-failure-fixture", source_method="codex")
    candidate["generation"] = {
        "batch_id": "audit-fixture",
        "instructions": "Artificial fixture",
        "model_metadata": {"model_id": None},
    }
    db = tmp_path / "audit.sqlite3"
    import_candidates([candidate], db, provisional_batch="audit-fixture")
    app = launch(db)

    def fail(*args):
        raise OSError("Audit write failed")

    monkeypatch.setattr("jais_guard.data.annotations._audit_event", fail)
    button(app, "VALID").click().run()
    assert any("Audit write failed" in w.value for w in app.error)
    assert queue("train", path=db)[0]["human_label"] is None
    assert audit_report(db)[0]["resolved"] == 0


def test_import_preflight_displays_v2_workload_before_any_candidate_write(tmp_path, monkeypatch):
    from test_provisional_annotations import artificial_rows

    from jais_guard.data.annotations import audit_report

    rows = artificial_rows(30, method="public", dialects=("MSA", "Egyptian", "Gulf"))

    class Upload:
        def getvalue(self):
            return "\n".join(json.dumps(r) for r in rows).encode()

    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: Upload())
    db = tmp_path / "preflight.sqlite3"
    app = launch(db)
    next(w for w in app.selectbox if w.label == "Import labeling").select(
        "Provisional training batch"
    ).run()
    next(w for w in app.text_input if w.label == "New batch ID").set_value("v2-source-run").run()
    assert not app.exception
    assert any("10/30" in m.value and "Projected" in m.value for m in app.info)
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    button(app, "Import uploaded file").click().run()
    assert not app.exception
    assert audit_report(db)[0]["policy_version"] == "provisional-v2"
    assert len(audit_report(db)[0]["samples"]) == 10


@pytest.fixture
def review_db(tmp_path):
    from test_provisional_annotations import artificial_rows

    db = tmp_path / "review-queue.sqlite3"
    public = artificial_rows(20, method="public", prefix="review-public")
    import_candidates(public, db, provisional_batch="review-public")
    manual = artificial_rows(3, prefix="manual")
    import_candidates(manual, db)
    save_annotation(manual[0]["id"], label="VALID", flagged=False, notes="Completed", path=db)
    save_annotation(manual[1]["id"], label="GIBBERISH", flagged=True, notes="Revisit", path=db)
    unsampled = next(r for r in queue("train", path=db) if r["status"] == "provisional")
    save_annotation(unsampled["id"], label=None, flagged=True, notes="Uncertain", path=db)
    from jais_guard.data.annotations import audit_report

    sample_id = audit_report(db)[0]["samples"][0]["id"]
    return db, sample_id, manual[0]["id"]


def remaining_count(app):
    return next(m.value for m in app.metric if m.label == "Remaining reviews")


def test_review_queue_is_default_and_ignores_old_browse_cursor(review_db):
    from jais_guard.data.annotations import get_state

    db, _, completed = review_db
    set_state("active_queue", "all", db)
    set_state("cursor:train:all", completed, db)
    set_state("cursor:train:review", completed, db)
    with connection(db) as conn:
        before = {
            t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")]
            for t in ("candidates", "annotations", "revisions", "audit_groups", "audit_samples")
        }
    app = launch(db, view=None)
    assert not app.exception
    assert app.radio[0].value == "Review Queue"
    assert remaining_count(app) == "13"
    current_id = get_state("cursor:train:review", db)
    current = next(r for r in queue("train", path=db) if r["id"] == current_id)
    assert current["flagged"] or current["status"] == "pending"
    assert current_id != completed
    assert not app.json and not app.checkbox
    assert not any(w.label == "Search text or example ID" for w in app.text_input)
    assert not any(
        w.label == "Source details (available after initial labeling)" for w in app.expander
    )
    with connection(db) as conn:
        assert all(
            [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")] == rows
            for t, rows in before.items()
        )


def test_review_label_resolves_once_saves_notes_and_advances(review_db):
    from jais_guard.data.annotations import get_state

    db, sample_id, _ = review_db
    set_state("cursor:train:review", sample_id, db)
    app = launch(db, view=None)
    app.text_area[0].set_value("معنى واضح").run()
    with connection(db) as conn:
        history = conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    button(app, "VALID").click().run()
    assert not app.exception
    assert remaining_count(app) == "12"
    saved = next(r for r in queue("train", path=db) if r["id"] == sample_id)
    assert saved["human_label"] == "VALID" and not saved["flagged"]
    assert saved["notes"] == "معنى واضح"
    assert get_state("cursor:train:review", db) != sample_id
    assert not any(sample_id in c.value for c in app.caption)
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == history + 1
        assert (
            conn.execute("SELECT outcome FROM audit_samples WHERE id=?", (sample_id,)).fetchone()[0]
            == "agreement"
        )


def test_uncertain_flag_stays_unresolved_and_can_be_found_later(review_db):
    from jais_guard.data.annotations import audit_report, get_state

    db, sample_id, _ = review_db
    set_state("cursor:train:review", sample_id, db)
    app = launch(db, view=None)
    button(app, "Flag as uncertain").click().run()
    assert not app.exception
    assert remaining_count(app) == "13"
    saved = next(r for r in queue("train", path=db) if r["id"] == sample_id)
    assert saved["human_label"] is None and saved["flagged"]
    assert get_state("cursor:train:review", db) != sample_id
    assert audit_report(db)[0]["status"] == "pending"
    resumed = launch(db, view=None)
    assert remaining_count(resumed) == "13"
    resumed.radio[0].set_value("Browse All").run()
    search(resumed, sample_id)
    assert matching_ids(resumed) == [sample_id]
    next(w for w in resumed.button if w.key == f"open:{sample_id}").click().run()
    assert any("Your label: **Not labeled**" in w.value for w in resumed.markdown)
    assert not resumed.json


def test_review_completion_does_not_fall_back_to_unsampled_provisional(tmp_path):
    from test_provisional_annotations import artificial_rows

    from jais_guard.data.annotations import audit_report

    db = tmp_path / "finish-review.sqlite3"
    import_candidates(artificial_rows(20, method="public"), db, provisional_batch="public")
    app = launch(db, view=None)
    assert remaining_count(app) == "10"
    for _ in range(10):
        button(app, "VALID").click().run()
        assert not app.exception
    assert remaining_count(app) == "0"
    assert any("No examples currently require manual review" in m.value for m in app.success)
    assert not any(w.label in {"VALID", "GIBBERISH"} for w in app.button)
    assert audit_report(db)[0]["status"] == "passed"
    assert sum(r["status"] == "provisional" for r in queue("train", path=db)) == 10
    assert remaining_count(launch(db, view=None)) == "0"


def test_failed_review_transaction_does_not_resolve_or_advance(review_db, monkeypatch):
    from jais_guard.data.annotations import audit_report, get_state

    db, sample_id, _ = review_db
    set_state("cursor:train:review", sample_id, db)
    app = launch(db, view=None)
    with connection(db) as conn:
        before = {
            t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")]
            for t in ("annotations", "revisions", "audit_groups", "audit_samples")
        }

    def fail(*args):
        raise OSError("Audit write failed")

    monkeypatch.setattr("jais_guard.data.annotations._audit_event", fail)
    button(app, "VALID").click().run()
    assert get_state("cursor:train:review", db) == sample_id
    assert remaining_count(app) == "13"
    assert any("Audit write failed" in m.value for m in app.error)
    assert audit_report(db)[0]["resolved"] == 0
    with connection(db) as conn:
        assert all(
            [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")] == rows
            for t, rows in before.items()
        )


def test_review_count_includes_new_cases_quarantined_by_audit_failure(tmp_path):
    from test_provisional_annotations import artificial_rows

    from jais_guard.data.annotations import get_state

    db = tmp_path / "quarantine-review.sqlite3"
    import_candidates(artificial_rows(20, method="public"), db, provisional_batch="public")
    app = launch(db, view=None)
    assert remaining_count(app) == "10"
    completed_id = get_state("cursor:train:review", db)
    button(app, "GIBBERISH").click().run()
    assert not app.exception
    assert remaining_count(app) == "19"
    assert get_state("cursor:train:review", db) != completed_id
    assert (
        next(r for r in queue("train", path=db) if r["id"] == completed_id)["status"]
        == "human-reviewed"
    )


def test_browse_all_has_its_own_filter_and_returns_to_required_selection(review_db):
    from jais_guard.data.annotations import get_state

    db, _, completed = review_db
    set_state("active_queue", "provisional", db)  # Old UI preference must not limit Browse All.
    set_state("cursor:train:all", completed, db)
    app = launch(db, view=None)
    review_id = get_state("cursor:train:review", db)
    app.radio[0].set_value("Browse All").run()
    assert not app.exception
    assert next(w for w in app.selectbox if w.label == "Queue").value == "all"
    assert len(matching_ids(app)) == len(queue("train", path=db))
    assert any(completed in c.value for c in app.caption)
    app.radio[0].set_value("Review Queue").run()
    assert not app.exception
    assert get_state("cursor:train:review", db) == review_id
    assert not any(completed in c.value for c in app.caption)
    assert remaining_count(app) == "13"


def test_validation_collection_requires_blind_manual_review_without_training_changes(tmp_path):
    from test_provisional_annotations import artificial_rows

    from jais_guard.data.annotations import audit_report, export_dataset

    db = tmp_path / "validation-review.sqlite3"
    training = artificial_rows(1, prefix="preserved-training")
    import_candidates(training, db)
    save_annotation(training[0]["id"], label="VALID", flagged=False, notes="Keep", path=db)
    with connection(db) as conn:
        before = {
            t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")]
            for t in ("annotations", "revisions", "audit_groups", "audit_samples")
        }
    validation = artificial_rows(2, prefix="held-out-validation")
    for row in validation:
        row["partition"] = "validation"
    import_candidates(validation, db)
    assert {r["status"] for r in queue("validation", path=db)} == {"pending"}
    assert audit_report(db) == []
    with pytest.raises(ValueError, match="human-reviewed only"):
        export_dataset("validation", "automatic", tmp_path, path=db, include_provisional=True)
    app = launch(db, view=None)
    next(w for w in app.selectbox if w.label == "Partition").select("validation").run()
    assert not app.exception and remaining_count(app) == "2"
    assert not app.json and not app.checkbox
    with connection(db) as conn:
        assert before == {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}")] for t in before}
    button(app, "VALID").click().run()
    assert remaining_count(app) == "1"
    assert sum(r["human_label"] is not None for r in queue("validation", path=db)) == 1
    assert queue("train", path=db)[0]["human_label"] == "VALID"
    with connection(db) as conn:
        assert [
            tuple(r)
            for r in conn.execute("SELECT * FROM annotations WHERE id=?", (training[0]["id"],))
        ] == before["annotations"]
        assert [
            tuple(r)
            for r in conn.execute("SELECT * FROM revisions WHERE id=?", (training[0]["id"],))
        ] == before["revisions"]
        assert conn.execute("SELECT COUNT(*) FROM provisional_labels").fetchone()[0] == 0


def test_strict_test_sources_hidden_after_label(tmp_path):
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/pilot_v1/candidates.jsonl").read_text().splitlines()
    ][:1]
    db = tmp_path / "strict-test.sqlite3"
    import_candidates(candidates, db)
    set_state("blind_sources", "1", db)
    app = launch(db)
    button(app, "VALID").click().run()
    assert not app.exception
    assert not app.json
    assert not any("Show source" in c.label for c in app.checkbox)
