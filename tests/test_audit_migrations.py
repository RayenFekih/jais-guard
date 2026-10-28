"""Transactional replacement, immutable history and rollback use isolated databases."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_provisional_annotations import artificial_rows

from jais_guard import annotations as a
from jais_guard import audit_migrations as migrations
from jais_guard.audit_policy import AYA_REVISION, digest


def snapshot(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {t: sorted(db.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in tables}


@pytest.fixture
def collection(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    human = artificial_rows(1, prefix="human-root")
    a.import_candidates(human, path)
    a.save_annotation(
        human[0]["id"], label="VALID", flagged=False, notes="Existing review", path=path
    )
    rows = artificial_rows(20, method="public", prefix="public", dialects=("MSA", "Egyptian"))
    aya = artificial_rows(15, method="public", prefix="aya")
    for r in aya:
        r.update(subtype="human_instruction")
        r["provenance"].update(
            source="CohereLabs/aya_dataset",
            revision=AYA_REVISION,
            annotation_type="original-annotations",
        )
    rows += aya + artificial_rows(15, prefix="codex")
    for r in rows:
        r["collection_name"] = "fixture"
    a.import_candidates(
        rows, path, provisional_batch="legacy-import", seed=42, policy_version="provisional-v1"
    )
    plan = a.migration_preview("fixture", seed=42, path=path)
    artifact = {"proposal": plan}
    artifact["fingerprint"] = digest(artifact)
    return path, rows, artifact


def apply(collection):
    path, _, artifact = collection
    return migrations.apply_audit_migration(
        artifact, "fixture", path=path, expected_fingerprint=artifact["fingerprint"]
    )


def test_apply_preserves_every_original_record_and_blocks_unaudited_exports(collection, tmp_path):
    path, rows, artifact = collection
    before = snapshot(path)
    result = apply(collection)
    assert result["records"] == 50 and result["required_reviews"] == 35
    assert result["audit_groups"] == 3 and result["audit_samples"] == 35
    after = snapshot(path)
    for table, records in before.items():
        if table in {"audit_groups", "audit_events", "collection_batches"}:
            assert set(records) <= set(after[table])
        else:
            assert records == after[table], table
    assert len(a.audit_report(path)) == 3
    all_reports = a.audit_report(path, include_history=True)
    assert len(all_reports) == 5
    assert all(
        not r["active"] and r["policy_version"] == "provisional-v1"
        for r in all_reports
        if r["id"] in artifact["proposal"]["original_group_ids"]
    )
    assert len(a.queue("train", "manual-review", path)) == 35
    assert a.progress(path)["train"]["export_eligible"] == 1
    folder = a.export_dataset(
        "train", "pending", tmp_path / "exports", path=path, include_provisional=True
    )
    assert len(a.read_export(folder, "train", allow_partial=True)) == 1
    with sqlite3.connect(result["backup_path"]) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 51
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='audit_policy_active'"
        ).fetchone()
    with a.connection(path) as db:
        assert db.execute("SELECT COUNT(*) FROM audit_policy_active").fetchone()[0] == 50
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
    with pytest.raises(ValueError, match="already recorded"):
        apply(collection)


def test_legacy_audits_are_immutable_and_new_reviews_update_only_v2(collection):
    path, _, artifact = collection
    apply(collection)
    with a.connection(path) as db:
        original_samples = [tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")]
        original_groups = [
            tuple(r)
            for r in db.execute(
                "SELECT * FROM audit_groups WHERE policy_version='provisional-v1' ORDER BY id"
            )
        ]
        sample_id = db.execute("SELECT id FROM audit_samples LIMIT 1").fetchone()[0]
    for sql, args in [
        ("UPDATE audit_samples SET outcome=? WHERE id=?", ("agreement", sample_id)),
        (
            "UPDATE audit_groups SET status=? WHERE id=?",
            ("passed", artifact["proposal"]["original_group_ids"][0]),
        ),
        ("DELETE FROM audit_samples WHERE id=?", (sample_id,)),
    ]:
        with pytest.raises(sqlite3.IntegrityError):
            with a.connection(path) as db:
                db.execute(sql, args)
    report = next(
        r
        for r in a.audit_report(path)
        if r["method"] == "public" and r["source"] == "google-research-datasets/tydiqa"
    )
    sample = report["samples"][0]["id"]
    a.save_annotation(
        sample, label="GIBBERISH", flagged=False, notes="Confirmed mismatch", path=path
    )
    assert next(r for r in a.audit_report(path) if r["id"] == report["id"])["status"] == "failed"
    with a.connection(path) as db:
        assert [
            tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")
        ] == original_samples
        assert [
            tuple(r)
            for r in db.execute(
                "SELECT * FROM audit_groups WHERE policy_version='provisional-v1' ORDER BY id"
            )
        ] == original_groups
        assert (
            db.execute(
                "SELECT outcome FROM versioned_audit_samples WHERE id=?", (sample,)
            ).fetchone()[0]
            == "error"
        )
    with pytest.raises(ValueError, match="Review has started"):
        migrations.rollback_audit_migration("v2-" + artifact["fingerprint"][:20], path=path)


def test_passed_v2_audit_exports_unsampled_records_without_old_flags(collection, tmp_path):
    path, _, _ = collection
    apply(collection)
    for report in a.audit_report(path):
        label = "GIBBERISH" if report["method"] == "codex" else "VALID"
        for sample in report["samples"]:
            a.save_annotation(sample["id"], label=label, flagged=False, notes="Reviewed", path=path)
    assert a.progress(path)["train"]["flagged"] == 0
    assert a.progress(path)["train"]["export_eligible"] == 51
    folder = a.export_dataset(
        "train", "resolved", tmp_path / "exports", path=path, include_provisional=True
    )
    exported = a.read_export(folder, "train", allow_partial=True, allow_provisional=True)
    assert len(exported) == 51
    automatic = [r for r in exported if r["review_status"] == "provisional"]
    assert len(automatic) == 15
    assert all(
        r["audit"]["policy_version"] == "provisional-v2" and r["audit"]["status"] == "passed"
        for r in automatic
    )
    with a.connection(path) as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM audit_groups "
                "WHERE policy_version='provisional-v1' AND status='pending'"
            ).fetchone()[0]
            == 2
        )


def test_failed_pointer_write_rolls_back_schema_and_all_data(collection, monkeypatch):
    path, _, _ = collection
    before = snapshot(path)

    def fail(db, migration_id, ids):
        db.execute("INSERT INTO audit_policy_active VALUES (?,?)", (sorted(ids)[0], migration_id))
        raise OSError("Pointer write failed")

    monkeypatch.setattr(migrations, "_activate", fail)
    with pytest.raises(OSError, match="Pointer write failed"):
        apply(collection)
    assert snapshot(path) == before
    assert len(list((path.parent / "backups").glob("before-approved-v2-*.sqlite3"))) == 1


def test_rollback_restores_v1_queue_and_retains_v2_history(collection):
    path, _, _ = collection
    before_queue = a.queue("train", path=path)
    result = apply(collection)
    migrations.rollback_audit_migration(result["migration_id"], path=path)
    assert a.queue("train", path=path) == before_queue
    assert len(a.audit_report(path)) == 2
    with a.connection(path) as db:
        assert db.execute("SELECT COUNT(*) FROM audit_policy_active").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM versioned_audit_samples").fetchone()[0] == 35
        assert (
            db.execute("SELECT status FROM audit_policy_migrations").fetchone()[0] == "rolled_back"
        )
    with pytest.raises(sqlite3.IntegrityError):
        with a.connection(path) as db:
            db.execute("UPDATE versioned_audit_samples SET outcome='agreement'")


def test_stale_or_tampered_proposal_cannot_apply(collection):
    path, rows, artifact = collection
    bad = json.loads(json.dumps(artifact))
    bad["proposal"]["required_reviews"] += 1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        migrations.apply_audit_migration(
            bad, "fixture", path=path, expected_fingerprint=bad["fingerprint"]
        )
    a.save_annotation(rows[0]["id"], label="VALID", flagged=True, notes="Review started", path=path)
    before = snapshot(path)
    with pytest.raises(ValueError, match="stale or review has started"):
        apply(collection)
    assert snapshot(path) == before


def test_ineligible_aya_remains_manual_only_under_active_replacement(collection):
    path, _, _ = collection
    row = artificial_rows(1, method="public", prefix="context-dependent")[0]
    row.update(
        collection_name="fixture",
        subtype="human_instruction",
        input_text="كمل الحكي: وصلت اليوم على المدينة",
    )
    row["provenance"].update(
        source="CohereLabs/aya_dataset",
        revision=AYA_REVISION,
        annotation_type="original-annotations",
    )
    a.import_candidates(
        [row], path, provisional_batch="extra-legacy", policy_version="provisional-v1"
    )
    artifact = {"proposal": a.migration_preview("fixture", seed=42, path=path)}
    artifact["fingerprint"] = digest(artifact)
    migrations.apply_audit_migration(
        artifact, "fixture", path=path, expected_fingerprint=artifact["fingerprint"]
    )
    current = next(r for r in a.queue("train", path=path) if r["id"] == row["id"])
    assert current["review_reasons"] == ["content_uncertain"]
    assert current["flagged"] and not current["exportable"] and current["human_label"] is None
    with a.connection(path) as db:
        assert (
            db.execute(
                "SELECT label FROM active_provisional_labels WHERE id=?", (row["id"],)
            ).fetchone()
            is None
        )
