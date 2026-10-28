"""Artificial records only; no source downloads, dataset generation, or live DB writes."""

import copy
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard import annotations as a


def artificial_rows(count=60, *, method="codex", prefix="fixture", dialects=("MSA",)):
    rng = random.Random(prefix)
    rows = []
    for i in range(count):
        # Varied artificial strings avoid testing import with near-identical numeric placeholders.
        words = ["".join(rng.choices("ابتجدرزسشصطظعغفقكلمنهوي", k=8)) for _ in range(5)]
        public = method == "public"
        source = "google-research-datasets/tydiqa" if public else "artificial-Codex"
        rows.append(
            {
                "id": f"{prefix}-{i}",
                "input_text": "كيف " + " ".join(words) + "؟",
                "partition": "train",
                "source_method": method,
                "intended_label": "VALID" if public else "GIBBERISH",
                "subtype": "question" if public else "incoherent_real_words",
                "dialect": dialects[i % len(dialects)],
                "domain": "fixture-only",
                "family_id": f"{prefix}-family-{i}",
                "difficulty": "easy" if public else "medium",
                "provenance": {
                    "source": source,
                    "source_id": f"{prefix}-{i}",
                    "license": "Apache-2.0",
                    "attribution": "artificial fixture",
                    "rights_evidence": "fixture-only",
                    "rights_status": "verified",
                    "revision": a.PUBLIC_RULES[source][0] if public else "unknown",
                },
                "generation": {
                    "batch_id": prefix,
                    "instructions": "Artificial test fixture only",
                    "model_metadata": {"model_id": None},
                },
            }
        )
    return rows


def seed_batch(path, *, method="codex", prefix="fixture", count=60, dialects=("MSA",)):
    rows = artificial_rows(count, method=method, prefix=prefix, dialects=dialects)
    a.import_candidates(rows, path, provisional_batch=prefix, seed=42)
    return rows, a.audit_report(path)[0]


def resolve_samples(path, report, label):
    for sample in report["samples"]:
        a.save_annotation(
            sample["id"], label=label, flagged=False, notes="Reviewed fixture", path=path
        )


def test_frozen_stratified_sampling_and_blind_queue(tmp_path):
    reports = []
    for name in ("first", "second"):
        path = tmp_path / f"{name}.sqlite3"
        rows, report = seed_batch(path, count=100, dialects=("MSA", "Egyptian"))
        reports.append(report)
        assert report["population"] == 100 and report["sampled"] == 50
        assert report["risk"] == "higher" and report["status"] == "pending"
        assert all(
            v == {"population": 50, "sampled": 25, "resolved": 0, "errors": 0}
            for v in report["coverage"].values()
        )
        assert len(a.queue("train", "manual-review", path)) == 50
        for row in a.queue("train", path=path):
            assert (
                not {"provisional_label", "intended_label", "origin", "subtype", "candidate"}
                & row.keys()
            )
            assert row["human_label"] is None
        with a.connection(path) as db:
            assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0
            before = [tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")]
        a.audit_report(path)
        with a.connection(path) as db:
            assert [
                tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")
            ] == before
    assert [s["id"] for s in reports[0]["samples"]] == [s["id"] for s in reports[1]["samples"]]
    with pytest.raises(ValueError, match="Duplicate ID"):
        a.import_candidates(rows, path, provisional_batch="another-seed", seed=99)


def test_public_rule_and_standard_sampling_are_narrow(tmp_path):
    path = tmp_path / "public.sqlite3"
    _, report = seed_batch(path, method="public", count=100, dialects=("MSA", "Egyptian"))
    assert report["sampled"] == 20 and report["risk"] == "standard"
    row = artificial_rows(1, method="public")[0]
    assert a._proposal(row)[0:2] == ("VALID", "public-source-rule")
    row["input_text"] = "أكمل الفقرة التالية؟"
    assert a._proposal(row) is None

    row["input_text"] = "كيف نستخدم هذه الأداة؟"
    row["provenance"]["revision"] = "unapproved-revision"
    assert a._proposal(row) is None
    row = artificial_rows(1)[0]
    row.update(source_method="python", subtype="keyboard_noise", intended_label="GIBBERISH")
    row["generation"] = {"seed": 42, "recipe": "keyboard_noise"}
    row["input_text"] = "ضصثقفغعهخحج"
    assert a._proposal(row)[0:2] == ("GIBBERISH", "python-rule")
    for text in ("هههههههههههه", "اه اه اه اه اه", "نعم نعم نعم", "ضصث", "مرحبا كيف حالك"):
        row["input_text"] = text
        assert a._proposal(row) is None
    row["input_text"] = "ضصثقفغعهخحج"
    row["generation"]["recipe"] = "whole_input_character_shuffle"
    assert a._proposal(row) is None


def test_batch_id_is_immutable_and_failed_reuse_rolls_back(tmp_path):
    path = tmp_path / "a.sqlite3"
    seed_batch(path, count=1)
    with a.connection(path) as db:
        before = [tuple(r) for r in db.execute("SELECT * FROM audit_samples")]
    fresh = artificial_rows(1, prefix="fresh")
    with pytest.raises(ValueError, match="already frozen"):
        a.import_candidates(fresh, path, provisional_batch="fixture", seed=999)
    with a.connection(path) as db:
        assert [tuple(r) for r in db.execute("SELECT * FROM audit_samples")] == before
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1
    # Explicit manual imports never promote their intended labels.
    a.import_candidates(fresh, path)
    assert (
        next(r for r in a.queue("train", path=path) if r["id"] == fresh[0]["id"])["status"]
        == "pending"
    )


def test_passed_audit_allows_explicit_mixed_training_export(tmp_path):
    path = tmp_path / "a.sqlite3"
    rows, report = seed_batch(path)
    with pytest.raises(ValueError, match="No resolved"):
        a.export_dataset(
            "train", "pending", tmp_path / "exports", path=path, include_provisional=True
        )
    first = report["samples"][0]["id"]
    a.save_annotation(first, label="GIBBERISH", flagged=True, notes="Still checking", path=path)
    assert a.audit_report(path)[0]["status"] == "pending"
    assert a.audit_report(path)[0]["resolved"] == 0
    resolve_samples(path, report, "GIBBERISH")
    assert a.audit_report(path)[0]["status"] == "passed"
    assert a.progress(path)["train"]["human_reviewed"] == 30
    assert a.progress(path)["train"]["provisional"] == 30
    folder = a.export_dataset("train", "human", tmp_path / "exports", path=path)
    assert len(a.read_export(folder, "train", allow_partial=True)) == 30
    folder = a.export_dataset(
        "train", "mixed", tmp_path / "exports", path=path, include_provisional=True
    )
    with pytest.raises(ValueError, match="provisional"):
        a.read_export(folder, "train", allow_partial=True)
    exported = a.read_export(folder, "train", allow_partial=True, allow_provisional=True)
    assert len(exported) == len(rows)
    automatic = [r for r in exported if r["review_status"] == "provisional"]
    assert len(automatic) == 30
    assert all(r["annotation"]["label"] is None and not r["audit"]["sampled"] for r in automatic)
    assert all(r["label_origin"] == "codex-proposal" for r in automatic)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["review_status_counts"] == {"human-reviewed": 30, "provisional": 30}
    assert manifest["complete"] and not manifest["annotation_complete"]
    # Reader also rejects forged metadata even when the payload hash is recomputed.
    exported[0]["review_status"] = "human-reviewed"
    exported[0]["label_origin"] = "codex-proposal"
    content = "".join(json.dumps(r) + "\n" for r in exported)
    (folder / "train.jsonl").write_text(content)
    manifest["sha256"] = hashlib.sha256(content.encode()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="review metadata"):
        a.read_export(folder, "train", allow_partial=True, allow_provisional=True)


def test_one_error_quarantines_only_affected_group_and_disables_rule(tmp_path):
    path = tmp_path / "a.sqlite3"
    _, report = seed_batch(path)
    resolve_samples(path, report, "GIBBERISH")
    folder = a.export_dataset(
        "train", "before-error", tmp_path / "exports", path=path, include_provisional=True
    )
    export_bytes = (folder / "train.jsonl").read_bytes()
    sample = report["samples"][0]["id"]
    a.save_annotation(sample, label="VALID", flagged=True, notes="Uncertain correction", path=path)
    assert a.audit_report(path)[0]["status"] == "pending"
    a.save_annotation(sample, label="VALID", flagged=False, notes="Confirmed error", path=path)
    failed = a.audit_report(path)[0]
    assert failed["status"] == "failed" and failed["errors"] == 1
    assert str(folder.resolve()) in failed["affected_exports"]
    assert (folder / "train.jsonl").read_bytes() == export_bytes
    assert len(a.queue("train", "manual-review", path)) == 30
    assert all(
        "failed audit" in r["review_reasons"] for r in a.queue("train", "manual-review", path)
    )
    # Human-reviewed samples remain usable despite the failed group.
    folder2 = a.export_dataset(
        "train", "after-error", tmp_path / "exports", path=path, include_provisional=True
    )
    assert len(a.read_export(folder2, "train", allow_partial=True)) == 30
    next_rows = artificial_rows(2, prefix="later")
    a.import_candidates(next_rows, path, provisional_batch="later")
    assert all(
        "failed audit" in " ".join(r["review_reasons"])
        for r in a.queue("train", path=path)
        if r["id"].startswith("later-")
    )
    unrelated = artificial_rows(1, prefix="unrelated", method="public")
    a.import_candidates(unrelated, path, provisional_batch="unrelated")
    assert a.audit_report(path)[-1]["status"] == "pending"
    # Errors remain in history even if a subsequent correction agrees with the proposal.
    a.save_annotation(
        sample, label="GIBBERISH", flagged=False, notes="Another correction", path=path
    )
    assert a.audit_report(path)[0]["status"] == "failed"
    assert a.audit_report(path)[0]["errors"] == 1


def test_voluntary_review_error_also_quarantines_group(tmp_path):
    path = tmp_path / "a.sqlite3"
    rows, report = seed_batch(path)
    selected = {s["id"] for s in report["samples"]}
    unsampled = next(r["id"] for r in rows if r["id"] not in selected)
    a.save_annotation(unsampled, label="VALID", flagged=False, notes="Found an error", path=path)
    assert a.audit_report(path)[0]["status"] == "failed"
    assert a.audit_report(path)[0]["errors"] == 1
    assert len(a.audit_report(path)[0]["samples"]) == 30


def test_pending_audit_blocks_unsampled_exports_and_failure_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "a.sqlite3"
    _, report = seed_batch(path)
    sample = report["samples"][0]["id"]
    with a.connection(path) as db:
        history = db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]

    def fail(*args):
        raise OSError("Audit write failed")

    monkeypatch.setattr(a, "_audit_event", fail)
    with pytest.raises(OSError, match="Audit write"):
        a.save_annotation(sample, label="GIBBERISH", flagged=False, notes="", path=path)
    assert a.audit_report(path)[0]["status"] == "pending"
    assert a.audit_report(path)[0]["resolved"] == 0
    with a.connection(path) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == history
        assert db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 0


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_evaluation_partitions_reject_provisional_import_and_export(tmp_path, partition):
    path = tmp_path / "a.sqlite3"
    rows = artificial_rows(1)
    rows[0]["partition"] = partition
    with pytest.raises(ValueError, match="training-only"):
        a.import_candidates(rows, path, provisional_batch="not-allowed")
    a.set_state("test_unlocked", "approved fixture", path)
    a.import_candidates(rows, path)
    a.save_annotation(rows[0]["id"], label="VALID", flagged=False, notes="Reviewed", path=path)
    with pytest.raises(ValueError, match="human-reviewed only"):
        a.export_dataset(
            partition, "mixed", tmp_path / "exports", path=path, include_provisional=True
        )
    folder = a.export_dataset(partition, "human", tmp_path / "exports", path=path)
    assert len(a.read_export(folder, partition, allow_partial=True)) == 1


def test_legacy_export_and_additive_upgrade_preserve_existing_rows(tmp_path):
    path = tmp_path / "a.sqlite3"
    row = artificial_rows(1)[0]
    a.import_candidates([row], path)
    a.save_annotation(row["id"], label="VALID", flagged=False, notes="Existing", path=path)
    with a.connection(path) as db:
        before = {
            t: [tuple(r) for r in db.execute(f"SELECT * FROM {t}")]
            for t in ("candidates", "annotations", "revisions")
        }
        for table in (
            "audited_exports",
            "disabled_rules",
            "audit_events",
            "audit_samples",
            "review_requests",
            "provisional_labels",
            "audit_groups",
            "collection_batches",
        ):
            db.execute(f"DROP TABLE {table}")
    with a.connection(path) as db:
        assert all(
            [tuple(r) for r in db.execute(f"SELECT * FROM {t}")] == values
            for t, values in before.items()
        )
        assert db.execute("SELECT COUNT(*) FROM provisional_labels").fetchone()[0] == 0
    backups = list((tmp_path / "backups").glob("before-provisional-*.sqlite3"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
    folder = a.export_dataset("train", "legacy-fixture", tmp_path / "exports", path=path)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest.pop("schema_version")
    (folder / "manifest.json").write_text(json.dumps(manifest))
    assert a.read_export(folder, "train", allow_partial=True)[0]["label"] == "VALID"
    # Existing IDs cannot be submitted again to acquire provisional labels.
    with pytest.raises(ValueError, match="Duplicate ID"):
        a.import_candidates([copy.deepcopy(row)], path, provisional_batch="retroactive")
