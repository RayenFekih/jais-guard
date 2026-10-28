"""V2 workflow tests use artificial fixtures; migration preview must remain read-only."""

import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_provisional_annotations import artificial_rows

from jais_guard import annotations as a
from jais_guard.audit_policy import AYA_REVISION, admission, digest, project_workload


def aya(text, **updates):
    row = artificial_rows(1, method="public")[0]
    row.update(subtype="human_instruction", input_text=text)
    row["provenance"].update(
        source="CohereLabs/aya_dataset",
        revision=AYA_REVISION,
        annotation_type="original-annotations",
    )
    row.update(updates)
    return row


def test_aya_subset_is_pinned_original_and_standalone():
    for text in ("كيف أصلح مقبض الباب؟", "قارن بين طريقتين لتنظيم الوقت", "القمر اتكون ازاى؟"):
        proposal, reason = admission(aya(text))
        assert proposal["label"] == "VALID" and proposal["risk"] == "standard" and reason is None
    for text in (
        "كمل الحكي: اليوم وصلنا على البيت",
        "لخص الفقرة التالية",
        "ما رأيك في هذا النص؟",
        "منظمة الصحة العالمية",
        "أَكْمِلْ الفقرة التالية",
    ):
        assert admission(aya(text))[1] == "content_uncertain"
    row = aya("كيف أصلح مقبض الباب؟")
    row["provenance"]["annotation_type"] = "human-revisions"
    assert admission(row)[1] == "rule_not_admitted"
    row["provenance"].update(annotation_type="original-annotations", revision="another-revision")
    assert admission(row)[1] == "rule_not_admitted"


def test_repetition_rule_does_not_admit_meaningful_or_unverified_forms():
    row = artificial_rows(1)[0]
    row.update(source_method="python", subtype="repeated_characters", intended_label="GIBBERISH")
    row["provenance"]["source"] = "JaisGuard deterministic Python recipes"
    row["generation"] = {"seed": 42, "recipe": "two_repeated_character_blocks"}
    row["input_text"] = "ثثثثثث ظظظظظ"
    assert admission(row)[0]["label"] == "GIBBERISH"
    for text in (
        "هههههههه مممممم",
        "اه اه اه اه",
        "لا لا لا",
        "ثث ظظ",
        "ثثثثثث ثثثثثث",
        "ثثثثثث طيب",
    ):
        row["input_text"] = text
        assert admission(row)[1] == "content_uncertain"
    row["generation"]["recipe"] = "unknown-repetition"
    assert admission(row)[1] == "rule_not_admitted"


def test_minimum_applies_to_group_and_dialects_remain_represented():
    rows = artificial_rows(30, method="public", dialects=("MSA", "Egyptian", "Gulf"))
    plan = project_workload(rows, "source-run", seed=42)
    assert len(plan["groups"]) == 1 and plan["audit_samples"] == 10
    group = plan["groups"][0]
    assert all(s["sampled"] >= 1 for s in group["strata"].values())
    assert sum(s["sampled"] for s in group["strata"].values()) == 10
    assert project_workload(list(reversed(rows)), "source-run", seed=42) == plan
    more_dialects = artificial_rows(
        30, method="public", dialects=tuple(f"dialect-{i}" for i in range(15))
    )
    plan = project_workload(more_dialects, "many-dialects")
    assert plan["audit_samples"] == 10
    assert sum(s["sampled"] > 0 for s in plan["groups"][0]["strata"].values()) == 10


def test_generation_runs_and_families_control_grouping_not_authoring_chunk_size():
    rows = artificial_rows(60)
    for i, row in enumerate(rows):
        row["generation"].update(
            batch_id=f"chunk-{i // 10}",
            run_id="authoring-run",
            audit_family="semantic-fragments-v1",
        )
    plan = project_workload(rows, "collection")
    assert len(plan["groups"]) == 1 and plan["audit_samples"] == 30
    assert plan["groups"][0]["risk"] == "higher"
    for row in rows[30:]:
        row["generation"]["audit_family"] = "another-construction-v1"
    plan = project_workload(rows, "collection")
    assert len(plan["groups"]) == 2 and plan["audit_samples"] == 40
    rows[0]["generation"].pop("audit_family")
    with pytest.raises(ValueError, match="requires an explicit audit_family"):
        project_workload(rows, "collection")


def test_v2_source_subtype_and_recipe_boundaries_are_preserved():
    first = artificial_rows(25, method="public", prefix="source-a")
    second = artificial_rows(25, method="public", prefix="source-b")
    for row in second:
        row["provenance"]["source"] = "AmazonScience/massive"
        row["provenance"]["revision"] = a.PUBLIC_RULES["AmazonScience/massive"][0]
        row["subtype"] = "localized_request"
    assert len(project_workload(first + second, "one-upload")["groups"]) == 2
    python = artificial_rows(2, prefix="python")
    for i, row in enumerate(python):
        row.update(
            source_method="python",
            subtype="repeated_characters",
            intended_label="GIBBERISH",
            input_text="ثثثثث ظظظظظ" if i == 0 else "صصصصص غغغغغ",
        )
        row["generation"] = {
            "seed": 42,
            "recipe": "two_repeated_character_blocks",
            "run_id": "recipe-run",
            "audit_family": "family-a" if i == 0 else "family-b",
        }
        row["provenance"]["source"] = "JaisGuard deterministic Python recipes"
    assert len(project_workload(python, "one-upload")["groups"]) == 2


def test_preview_matches_import_and_rejects_stale_approval(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    with a.connection(path):
        pass
    rows = artificial_rows(30, method="public", dialects=("MSA", "Egyptian", "Gulf"))
    preview = a.preview_import(rows, "public-run", 42, path)
    with a.connection(path) as db:
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    changed = copy.deepcopy(rows)
    changed[0]["difficulty"] = "hard"
    with pytest.raises(ValueError, match="Projected workload changed"):
        a.import_candidates(
            changed,
            path,
            provisional_batch="public-run",
            seed=42,
            expected_projection=preview["fingerprint"],
        )
    a.import_candidates(
        rows,
        path,
        provisional_batch="public-run",
        seed=42,
        expected_projection=preview["fingerprint"],
    )
    report = a.audit_report(path)[0]
    assert report["policy_version"] == "provisional-v2"
    assert {s["id"] for s in report["samples"]} == set(preview["groups"][0]["sampled_ids"])
    assert len(a.queue("train", "manual-review", path)) == 10


def test_frozen_run_cannot_be_extended_by_a_new_import(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    for prefix in ("first", "second"):
        rows = artificial_rows(3, prefix=prefix)
        for row in rows:
            row["generation"].update(run_id="fixed-run", audit_family="fixed-family-v1")
        if prefix == "first":
            a.import_candidates(rows, path, provisional_batch=prefix)
        else:
            with pytest.raises(ValueError, match="already frozen"):
                a.preview_import(rows, prefix, path=path)
    with a.connection(path) as db:
        assert db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 3


def test_v1_and_v2_audits_coexist_and_errors_still_quarantine(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    old = artificial_rows(2, prefix="legacy")
    a.import_candidates(old, path, provisional_batch="old", policy_version="provisional-v1")
    with a.connection(path) as db:
        before = [tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")]
    new = artificial_rows(60, prefix="future")
    a.import_candidates(new, path, provisional_batch="future")
    reports = a.audit_report(path)
    assert {r["policy_version"] for r in reports} == {"provisional-v1", "provisional-v2"}
    current = next(r for r in reports if r["policy_version"] == "provisional-v2")
    sample = current["samples"][0]["id"]
    a.save_annotation(sample, label="VALID", flagged=False, notes="Confirmed mismatch", path=path)
    assert next(r for r in a.audit_report(path) if r["id"] == current["id"])["status"] == "failed"
    followup = artificial_rows(1, prefix="followup")
    a.import_candidates(followup, path, provisional_batch="followup")
    assert next(r for r in a.queue("train", path=path) if r["id"] == followup[0]["id"])[
        "review_reasons"
    ] == ["failed_audit"]
    with a.connection(path) as db:
        assert [
            tuple(r)
            for r in db.execute(
                "SELECT * FROM audit_samples WHERE group_id=? ORDER BY id", (reports[0]["id"],)
            )
        ] == before


def test_migration_proposal_is_read_only_and_refuses_started_reviews(tmp_path):
    path = tmp_path / "annotations.sqlite3"
    rows = artificial_rows(6)
    for row in rows:
        row["collection_name"] = "existing-fixture"
    a.import_candidates(rows, path, provisional_batch="legacy", policy_version="provisional-v1")

    def snapshot():
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return {t: db.execute(f"SELECT * FROM {t}").fetchall() for t in tables}

    before = snapshot()
    plan = a.migration_preview("existing-fixture", path=path)
    assert plan["safe_to_propose"] and plan["approval_required"] and plan["proposal_only"]
    assert before == snapshot()
    a.save_annotation(rows[0]["id"], label="GIBBERISH", flagged=True, notes="Started", path=path)
    assert not a.migration_preview("existing-fixture", path=path)["safe_to_propose"]


def test_local_400_proposal_preserves_live_database_and_frozen_samples():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/milestone700_v1"
    if not (folder / "candidates.jsonl").exists() or not a.DEFAULT_DB.exists():
        pytest.skip("Local unreviewed artifacts unavailable; never download or generate in tests")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "candidate_preparation", root / "scripts/prepare_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with sqlite3.connect(f"file:{a.DEFAULT_DB}?mode=ro", uri=True) as db:
        before = [tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")]
    plan, summary = module.propose_collection_v2(folder)
    assert plan["approval_required"] and plan["proposal_only"]
    # Counts describe the pristine pre-migration snapshot, not a live collection
    # after human corrections, disabled rules or approved retirements.
    if plan["safe_to_propose"]:
        assert summary["required_reviews"] == 255 and summary["audit_samples"] == 207
        assert summary["manual_only"] == 48
        assert len([g for g in plan["groups"] if g["method"] == "codex"]) == 18
    else:
        assert plan["blockers"]
    artifact_path = folder / "policy_v2_proposal.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        assert artifact["fingerprint"] == digest(
            {k: v for k, v in artifact.items() if k != "fingerprint"}
        )
        if plan["safe_to_propose"]:
            assert artifact["proposal"]["fingerprint"] == plan["fingerprint"]
        else:
            assert any("versioned migration" in reason for reason in plan["blockers"])
    with sqlite3.connect(f"file:{a.DEFAULT_DB}?mode=ro", uri=True) as db:
        assert [tuple(r) for r in db.execute("SELECT * FROM audit_samples ORDER BY id")] == before
