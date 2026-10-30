"""Public derivative regression and explicit final-approval gates, CPU only."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from datasets import Dataset, load_dataset

from src.jais_guard import public_release as release
from src.jais_guard import publication as original


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    folder = tmp_path_factory.mktemp("public-derivative")
    stage, local = folder / "stage", folder / "local"
    old_hash = original.sha(original.STAGE / "metadata.json")
    result = release.build(stage, local)
    assert original.sha(original.STAGE / "metadata.json") == old_hash
    return stage, local, result


@pytest.fixture
def fast_audit(prepared, monkeypatch):
    report = json.loads((prepared[1] / "audit.json").read_text())
    monkeypatch.setattr(release, "audit", lambda rows: report)


def reviewed_draft(stage):
    review = release.review_template(stage)
    for item in [*review["categories"].values(), review["relationships"]]:
        item.update(status="reviewed", evidence="Simulated test review evidence")
    return review


def test_exact_exclusions_and_preservation(prepared, tmp_path):
    stage, _, result = prepared
    source, _ = original.source_data()
    ds = load_dataset(str(stage), cache_dir=str(tmp_path / "cache"))
    expected_train = [
        original.project(r) for i, r in enumerate(source["train"]) if i not in (277, 486)
    ]
    assert ds["train"].to_list() == expected_train
    assert ds["train"][277] == original.project(source["train"][278])
    assert ds["train"][485] == original.project(source["train"][487])
    for split in ["validation", "test"]:
        assert ds[split].to_list() == [original.project(r) for r in source[split]]
    assert result["split_counts"] == {"train": 1498, "validation": 250, "test": 400}
    assert result["release_total"] == 2148
    assert ds["train"].features == original.FEATURES
    assert sum(r["label_origin"] == "audited-provisional" for r in ds["train"]) == 133
    assert sum(r["label"] == 0 for r in ds["train"]) == 708
    card = (stage / "README.md").read_text()
    assert all(source["train"][i]["input_text"] not in card for i in (277, 486))
    card += (stage / "LICENSES/ATTRIBUTION.md").read_text()
    # Excluded upstream identifiers must not appear in the public attribution appendix.
    assert all(source["train"][i]["provenance"]["source_id"] not in card for i in (277, 486))


def test_scoped_card_and_disclosures(prepared):
    text = (prepared[0] / "README.md").read_text()
    header = yaml.safe_load(text.split("---", 2)[1])
    assert header["license"] == ["cc-by-4.0", "apache-2.0"]
    assert "2,148-example public derivative" in text
    assert "1,365 human-reviewed and 133 audited-provisional" in text
    assert "Test[319]" in text and "test[204]" in text
    assert "260 corrupted-request parents cannot be replayed" in text
    assert "not an untouched independent final test set" in text
    assert "Version 2.0, January 2004" not in text
    assert "Attribution 4.0 International Public License" not in text
    assert "| Public split row" not in text
    assert "pending" not in text.lower()
    for name in release.LICENSE_FILES:
        assert f"]({name})" in text
        assert (prepared[0] / name).is_file()
    assert "Version 2.0, January 2004" in (prepared[0] / "LICENSES/Apache-2.0.txt").read_text()
    assert (
        "Attribution 4.0 International Public License"
        in (prepared[0] / "LICENSES/CC-BY-4.0.txt").read_text()
    )
    attribution = (prepared[0] / "LICENSES/ATTRIBUTION.md").read_text()
    assert sum(line.startswith("| train[") for line in attribution.splitlines()) == 313
    assert "- Meaningful Arabic (VALID): كيف يصنع حديد الزهر ؟" in text
    assert "| 148 | apache-2.0 |" in text
    assert "| 130 | cc-by-4.0 |" in text
    assert "| 35 | apache-2.0 |" in text


def test_pending_final_review_never_contacts_hub(prepared, tmp_path, fast_audit):
    stage, local, _ = prepared
    review = reviewed_draft(stage)
    path = tmp_path / "pending.json"
    original.write_json(path, review)
    result = release.validate(stage, local, review=path)
    assert result["blockers"] == [
        "Final content review approval pending",
        "Final license review approval pending",
    ]
    api = Mock()
    with pytest.raises(ValueError, match="Publication blocked"):
        original.publish(
            "example/data", stage, local, review=path, validator=release.validate, api=api
        )
    assert not api.mock_calls


def test_final_review_must_bind_exact_release(prepared, tmp_path, fast_audit):
    stage, local, _ = prepared
    review = reviewed_draft(stage)
    for item in review["final_review"].values():
        item.update(status="approved", evidence="Simulated owner approval for test only")
    path = tmp_path / "approved.json"
    original.write_json(path, review)
    before_card = original.sha(stage / "README.md")
    assert release.validate(stage, local, review=path)["publication_ready"]
    assert original.sha(stage / "README.md") == before_card
    assert "pending" not in (stage / "README.md").read_text().lower()
    review["metadata_sha256"] = "stale"
    original.write_json(path, review)
    with pytest.raises(ValueError, match="different public derivative"):
        release.validate(stage, local, review=path)


def test_rehashed_mutation_still_fails_source_comparison(prepared, tmp_path, fast_audit):
    import shutil

    stage = tmp_path / "mutated"
    shutil.copytree(prepared[0], stage)
    rows, _, _ = release.release_data()
    projected = [original.project(r) for r in rows["train"]]
    projected[0]["text"] = "تغيير غير مسموح"
    Dataset.from_list(projected, features=original.FEATURES).to_parquet(
        stage / "data/train.parquet"
    )
    m = json.loads((stage / "metadata.json").read_text())
    m["files"] = original.file_hashes(stage, files=release.FILES)
    original.write_json(stage / "metadata.json", m)
    with pytest.raises(ValueError, match="retained source projection"):
        release.validate(stage, prepared[1])


def test_old_candidate_cannot_be_build_target():
    with pytest.raises(ValueError, match="must not overwrite"):
        release.build(original.STAGE)


def test_license_files_rehashed_tamper_rejected(prepared, tmp_path, fast_audit):
    import shutil

    stage = tmp_path / "tampered-license"
    shutil.copytree(prepared[0], stage)
    (stage / "LICENSES/ATTRIBUTION.md").write_text("Removed required attributions")
    m = json.loads((stage / "metadata.json").read_text())
    m["files"] = original.file_hashes(stage, files=release.FILES)
    original.write_json(stage / "metadata.json", m)
    with pytest.raises(ValueError, match="license/attribution content mismatch"):
        release.validate(stage, prepared[1])


def test_missing_or_unexpected_license_files_rejected(prepared, tmp_path):
    import shutil

    stage = tmp_path / "incomplete"
    shutil.copytree(prepared[0], stage)
    (stage / "LICENSES/Apache-2.0.txt").unlink()
    with pytest.raises(ValueError, match="inventory"):
        original.inventory(stage, files=release.FILES)
    (stage / "LICENSES/Apache-2.0.txt").write_bytes(
        (prepared[0] / "LICENSES/Apache-2.0.txt").read_bytes()
    )
    (stage / "LICENSES/private-notes.txt").write_text("not for release")
    with pytest.raises(ValueError, match="inventory"):
        original.inventory(stage, files=release.FILES)


def test_publisher_includes_and_verifies_all_license_files(prepared, tmp_path, monkeypatch):
    stage, _, result = prepared
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(private=True, sha="base")
    api.list_repo_files.side_effect = [[], sorted(release.FILES)]
    api.upload_folder.return_value = SimpleNamespace(oid="simulated")
    download = Mock(side_effect=lambda repo, name, **kw: str(stage / name))
    monkeypatch.setattr(original, "hf_hub_download", download)
    review = tmp_path / "simulated-review.json"
    review.write_text("{}")

    def ready(*a, **kw):
        return {**result, "publication_ready": True}

    receipt = original.publish(
        "example/data",
        stage,
        tmp_path / "receipt",
        review=review,
        api=api,
        validator=ready,
        files=release.FILES,
    )
    assert receipt["verified"]
    assert set(receipt["files"]) == release.FILES
    assert set(api.upload_folder.call_args.kwargs["allow_patterns"]) == release.FILES
    assert {call.args[1] for call in download.call_args_list} == release.FILES
    # A forgotten extended allowlist must fail before contacting the Hub.
    untouched = Mock()
    with pytest.raises(ValueError, match="inventory"):
        original.publish("example/data", stage, review=review, api=untouched, validator=ready)
    assert not untouched.mock_calls
