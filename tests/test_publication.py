"""CPU coverage for frozen export and fail-closed publication."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pyarrow.parquet as pq
import pytest
import yaml

from jais_guard import publication as pub

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    path = tmp_path_factory.mktemp("publication")
    stage, local = path / "stage", path / "review"
    result = pub.build(stage, local)
    return stage, local, result


@pytest.fixture
def fast_audit(prepared, monkeypatch):
    report = json.loads((prepared[1] / "audit.json").read_text())
    monkeypatch.setattr(pub, "audit", lambda rows: report)


def test_export_schema_exact_and_history(prepared):
    stage, _, result = prepared
    rows, provenance = pub.source_data()
    assert result["local_validation"] == "passed"
    assert result["publication_ready"] is False
    assert provenance["historical_v1"]["rows"] == 700
    for split, rs in rows.items():
        table = pq.read_table(stage / f"data/{split}.parquet")
        assert table.to_pylist() == [pub.project(r) for r in rs]
        assert table.column_names == pub.COLUMNS
        assert set(table.column("label").to_pylist()) == {0, 1}
    assert (
        sum(pub.project(r)["label_origin"] == "audited-provisional" for r in rows["train"]) == 133
    )


def test_card_metadata(prepared):
    stage = prepared[0]
    text = (stage / "README.md").read_text()
    header = yaml.safe_load(text.split("---", 2)[1])
    assert "license" not in header
    assert header["configs"][0]["data_files"] == [
        {"split": s, "path": f"data/{s}.parquet"} for s in pub.SOURCES
    ]
    assert "not an untouched independent final test set" in text
    assert "1,367 human-reviewed and 133 audited-provisional" in text


def test_publish_without_review_never_connects(prepared, fast_audit):
    api = Mock()
    with pytest.raises(ValueError, match="Publication blocked"):
        pub.publish("example/data", prepared[0], prepared[1], api=api)
    assert not api.mock_calls


def test_extra_file_and_symlink_rejected(prepared, tmp_path):
    import shutil

    stage = tmp_path / "copy"
    shutil.copytree(prepared[0], stage)
    extra = stage / "checkpoint.bin"
    extra.write_bytes(b"private")
    with pytest.raises(ValueError, match="inventory"):
        pub.inventory(stage)
    extra.unlink()
    extra.symlink_to(stage / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        pub.inventory(stage)


def test_manifest_tamper_rejected(prepared, tmp_path, fast_audit):
    import shutil

    stage = tmp_path / "copy"
    shutil.copytree(prepared[0], stage)
    with (stage / "data/train.parquet").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest/hash"):
        pub.validate(stage, prepared[1])


def test_review_cannot_grant_unlisted_license(prepared, tmp_path, fast_audit):
    review = json.loads((prepared[1] / "review-template.json").read_text())
    review["license"] = "cc-by-4.0"
    review["license_evidence"] = "test evidence"
    for v in [*review["categories"].values(), review["relationships"]]:
        v.update(status="approved", evidence="test evidence")
    path = tmp_path / "approval.json"
    pub.write_json(path, review)
    with pytest.raises(ValueError, match="dataset card"):
        pub.validate(prepared[0], prepared[1], review=path)
    review["metadata_sha256"] = "stale"
    pub.write_json(path, review)
    with pytest.raises(ValueError, match="different publication"):
        pub.validate(prepared[0], prepared[1], review=path)


def test_relationship_and_content_screening():
    def row(text, id, family):
        return {"id": id, "family_id": family, "input_text": text, "provenance": {}}

    report = pub.audit(
        {
            "train": [row("مرحبا بالعالم الجميل", "one", "family")],
            "test": [
                row("مرحبا بالعالم الجميل", "two", "other"),
                row("مَرْحبا بالعالم الجميل", "three", "family"),
                row("مرحبا بالعالم الجميلة", "four", "four"),
                row("test@example.com hf_abcdefghijk123456789", "five", "five"),
            ],
        }
    )
    assert report["relationships"]["exact"]
    assert report["relationships"]["normalized"]
    assert report["relationships"]["near_duplicate"]
    assert report["relationships"]["source_family"]
    assert {f["kind"] for f in report["screening_findings"]} >= {"email", "secret"}


def test_remote_public_and_collision_rejected(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(pub, "validate", lambda *a, **kw: {"publication_ready": True})
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(private=False, sha="old")
    with pytest.raises(ValueError, match="private"):
        pub.publish("example/data", prepared[0], prepared[1], api=api)
    api.upload_folder.assert_not_called()
    api.repo_info.return_value.private = True
    api.list_repo_files.return_value = ["README.md", "unrelated.txt"]
    remote = tmp_path / "remote"
    remote.write_text("unrelated preexisting card")
    monkeypatch.setattr(pub, "hf_hub_download", lambda *a, **kw: str(remote))
    with pytest.raises(ValueError, match="overwrite"):
        pub.publish("example/data", prepared[0], prepared[1], api=api)
    api.upload_folder.assert_not_called()


def test_private_create_allowlist_and_receipt(prepared, monkeypatch, tmp_path):
    stage, local, result = prepared
    monkeypatch.setattr(pub, "validate", lambda *a, **kw: {**result, "publication_ready": True})
    api = Mock()
    api.repo_info.side_effect = [
        pub.RepositoryNotFoundError(
            "missing",
            response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")),
        ),
        SimpleNamespace(private=True, sha="base"),
    ]
    api.list_repo_files.side_effect = [
        [".gitattributes", "unrelated.txt"],
        [*pub.FILES, ".gitattributes", "unrelated.txt"],
    ]
    api.upload_folder.return_value = SimpleNamespace(oid="published")
    monkeypatch.setattr(pub, "hf_hub_download", lambda repo, name, **kw: str(stage / name))
    review = tmp_path / "review.json"
    review.write_text("{}")
    receipt = pub.publish("example/data", stage, local, review=review, api=api)
    assert receipt["verified"]
    api.create_repo.assert_called_once_with(
        "example/data", repo_type="dataset", private=True, exist_ok=False
    )
    kwargs = api.upload_folder.call_args.kwargs
    assert set(kwargs["allow_patterns"]) == pub.FILES
    assert kwargs["parent_commit"] == "base"
    assert "delete_patterns" not in kwargs


def test_native_folder_loading(prepared, tmp_path):
    ds = pub.load_dataset(str(prepared[0]), cache_dir=str(tmp_path / "cache"))
    assert {s: len(rs) for s, rs in ds.items()} == {"train": 1500, "validation": 250, "test": 400}
    assert ds["train"].features["label"].names == ["VALID", "GIBBERISH"]


def test_source_tamper_rejected(tmp_path):
    version, _, _, _ = pub.SOURCES["train"]
    path = tmp_path / "data/exports" / version
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}")
    (path / "train.jsonl").write_text('{"input_text": "changed"}\n')
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        pub.source_data(tmp_path)
