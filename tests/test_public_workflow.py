"""Portable public workflow tests; no private data, network, or pretrained weights."""

import copy
import json
from pathlib import Path

import httpx
import pytest

from jais_guard import public_workflow as workflow
from jais_guard.cli import main
from jais_guard.finetuning.data import expected_counts, load_datasets
from jais_guard.serving import create_app


@pytest.fixture
def prepared(tmp_path):
    dataset = {}
    for split in ("train", "validation", "test"):
        dataset[split] = [
            {
                "text": f"كيف أتعلم العربية {'الآن' if label == 0 else 'غدا'} {split}",
                "label": label,
                "label_name": ["VALID", "GIBBERISH"][label],
                "label_origin": "human-reviewed"
                if split != "train" or label == 0
                else "audited-provisional",
            }
            for label in (0, 1)
        ]
        # Arabic-script only and unique across partitions.
        for row in dataset[split]:
            row["text"] = row["text"].replace(
                split, {"train": "تدريب", "validation": "تحقق", "test": "اختبار"}[split]
            )
    config = workflow.export_public(
        dataset, tmp_path / "public", {split: {"VALID": 1, "GIBBERISH": 1} for split in dataset}
    )
    return dataset, config, tmp_path / "public"


def test_public_export_preserves_origin_and_is_exclusive(prepared):
    dataset, config, folder = prepared
    train, validation, _, _ = load_datasets(config)
    assert train[1]["label_origin"] == train[1]["review_status"] == "audited-provisional"
    assert "label" not in train[1]["annotation"]
    assert train[1]["family_id_scope"].startswith("public-row-only")
    assert all(row["review_status"] == "human-reviewed" for row in validation)
    assert config["historical_reproduction"] is False
    with pytest.raises(FileExistsError):
        workflow.export_public(dataset, folder, {s: {"VALID": 1, "GIBBERISH": 1} for s in dataset})


def test_v1_defaults_and_explicit_v2_counts(prepared):
    assert expected_counts({}, "train") == {"VALID": 359, "GIBBERISH": 341}
    assert expected_counts({}, "validation") == {"VALID": 99, "GIBBERISH": 101}
    config = prepared[1]
    assert len(load_datasets(config)[0]) == 2
    wrong = {**config, "train_counts": {"VALID": 2, "GIBBERISH": 1}}
    with pytest.raises(ValueError, match="class counts"):
        load_datasets(wrong)
    wrong = {k: v for k, v in config.items() if k != "train_counts"}
    with pytest.raises(ValueError, match="class counts"):
        load_datasets(wrong)


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"VALID": 1},
        {"VALID": True, "GIBBERISH": 1},
        {"VALID": 1, "GIBBERISH": -1},
        {"VALID": 1, "GIBBERISH": 1, "INVALID": 1},
    ],
)
def test_counts_cannot_weaken_validation(counts):
    with pytest.raises(ValueError, match="class counts"):
        expected_counts({"train_counts": counts}, "train")


def test_adapter_rejects_fabricated_review_and_mapping(prepared):
    row = copy.deepcopy(prepared[0]["train"][0])
    row["label"] = 1
    with pytest.raises(ValueError, match="mapping"):
        workflow.public_row(row, "train", 0)
    row["label"] = 0
    row["label_origin"] = "audited-provisional"
    with pytest.raises(ValueError, match="human-reviewed"):
        workflow.public_row(row, "test", 0)


def test_tampered_export_is_rejected(prepared):
    config = prepared[1]
    path = Path(config["train_path"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_datasets(config)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "run/best"
    path.mkdir(parents=True)
    (path / "classifier.json").write_text(
        json.dumps(
            {
                "label_mapping": {"VALID": 0, "GIBBERISH": 1},
                "pooling": "last_nonpadding_final_hidden",
                "approach": "frozen",
                "seed": 42,
                "config": {"max_length": 64},
            }
        )
    )
    (path / "head.safetensors").write_bytes(b"synthetic checkpoint identity")
    (path.parent / "threshold.json").write_text(
        json.dumps(
            {
                "exploratory": 0.9,
                "selection_data": "validation",
            }
        )
    )
    return path


def test_checkpoint_uses_saved_threshold_and_refuses_missing(checkpoint):
    assert workflow.checkpoint_manifest(checkpoint)["models"]["checkpoint"]["threshold"] == 0.9
    assert (
        workflow.checkpoint_manifest(checkpoint, "default")["models"]["checkpoint"]["threshold"]
        == 0.5
    )
    (checkpoint.parent / "threshold.json").unlink()
    with pytest.raises(ValueError, match="threshold missing"):
        workflow.checkpoint_manifest(checkpoint)


def test_evaluation_preserves_threshold_without_selection(prepared, checkpoint, monkeypatch):
    from jais_guard.finetuning import model

    monkeypatch.setattr(
        model, "load_checkpoint", lambda *a: (object(), object(), {"config": {"max_length": 64}})
    )
    monkeypatch.setattr(model, "predict_rows", lambda *a: ([0.9, 0.8], None))
    output = prepared[2] / "evaluation"
    workflow.evaluate(checkpoint, prepared[2] / "test/test.jsonl", output, device="cpu")
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [r["prediction"] for r in rows] == [1, 0]  # >= at saved boundary
    assert json.loads((output / "protocol.json").read_text())["threshold"] == 0.9
    with pytest.raises(FileExistsError):
        workflow.evaluate(checkpoint, prepared[2] / "test/test.jsonl", output, device="cpu")


def test_checkpoint_api_without_historical_manifest(checkpoint, monkeypatch):
    import asyncio

    class FakePredictor:
        def __init__(self, manifest, key):
            assert key == "checkpoint"
            assert manifest["models"][key]["threshold"] == 0.9

        def info(self):
            return {"model_key": "checkpoint"}

        def predict(self, text):
            if not text.strip():
                raise ValueError("Expected nonempty Arabic text")
            return {"class_id": 0}

    # Match existing API unit tests: avoid sandbox threadpool wakeup restrictions.
    import fastapi.routing

    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(fastapi.routing, "run_in_threadpool", inline)
    app = create_app(checkpoint=checkpoint, predictor_factory=FakePredictor)

    async def run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (await client.get("/health")).json()["status"] == "ready"
                assert (await client.post("/predict", json={"text": "كيف حالك"})).json() == {
                    "class_id": 0
                }
                assert (await client.post("/predict", json={"text": " "})).status_code == 422

    asyncio.run(run())


def test_inspect_command_is_portable(prepared, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    main(["inspect", "--config", str(prepared[2] / "training.json")])
    assert json.loads(capsys.readouterr().out)["train"] == 2


def test_pinned_release_rejects_wrong_bytes(tmp_path):
    (tmp_path / "README.md").write_text("wrong release")
    with pytest.raises(ValueError, match="checksum"):
        workflow.prepare(tmp_path / "output", tmp_path)
    assert not (tmp_path / "output").exists()


def test_predict_command_uses_locked_boundary(checkpoint, monkeypatch, capsys):
    from jais_guard.finetuning import model

    monkeypatch.setattr(
        model, "load_checkpoint", lambda *a: (object(), object(), {"config": {"max_length": 64}})
    )
    monkeypatch.setattr(model, "predict_rows", lambda *a: ([0.9], None))
    main(["predict", "--checkpoint", str(checkpoint), "--text", "كيف حالك", "--device", "cpu"])
    result = json.loads(capsys.readouterr().out)
    assert result["class_id"] == 1 and result["threshold"] == 0.9


def test_training_refuses_cpu_before_outputs(prepared, monkeypatch):
    from jais_guard import serving

    def no_cuda():
        raise RuntimeError("CUDA required")

    monkeypatch.setattr(serving, "cuda_required", no_cuda)
    with pytest.raises(RuntimeError, match="CUDA"):
        main(
            [
                "train",
                "--config",
                str(prepared[2] / "training.json"),
                "--approach",
                "frozen",
                "--seed",
                "42",
            ]
        )
    assert not Path(prepared[1]["output_dir"]).exists()


def test_evidence_verification_and_tampering(tmp_path):
    import shutil

    from jais_guard.evidence import verify

    source = Path(__file__).resolve().parents[1] / "docs/benchmarks"
    assert verify(source)["requests"] == 6400
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    (target / "vllm/metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        verify(target)


def test_missing_serving_artifacts_are_actionable(tmp_path):
    import asyncio

    app = create_app(directory=tmp_path / "missing")

    async def run():
        with pytest.raises(ValueError, match="JAISGUARD_CHECKPOINT"):
            async with app.router.lifespan_context(app):
                pass

    asyncio.run(run())
