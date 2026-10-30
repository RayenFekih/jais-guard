"""CPU-only baseline runner and independent-test workflow checks."""

import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.baselines.prompts import (
    RECOVERABLE_INSTRUCTION,
    RECOVERABLE_PROMPT_VERSION,
    build_prompt,
    parse_diagnostic,
)
from jais_guard.data import test_workflow as workflow
from jais_guard.data.annotations import (
    connection,
    quality_issues,
    queue,
    save_annotation,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "raw,stopped,status,prediction",
    [
        ("reason</ifm|think_faster>VALID", True, "success", "VALID"),
        ("reason</ifm|think_faster>GIBBERISH", True, "success", "GIBBERISH"),
        ("reason</ifm|think_faster>VALID GIBBERISH", True, "ambiguous", None),
        ("reason</ifm|think_faster>VALID VALID", True, "ambiguous", None),
        ("reason</ifm|think_faster>VALID because...", True, "malformed", None),
        ("reason</ifm|think_faster>", True, "missing_classification", None),
        ("VALID", True, "missing_reasoning_close", None),
        ("reason</ifm|think_faster>VALID", False, "truncated", None),
    ],
)
def test_conservative_parsing(raw, stopped, status, prediction):
    assert parse_diagnostic(raw, "low", stopped) == {
        "parse_status": status,
        "prediction": prediction,
    }


def test_new_prompt_contains_only_policy_and_unlabeled_input():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["reasoning_effort"] == "low"
            return messages

    messages = build_prompt(Tokenizer(), "طلب معنى", "low", False, "recoverable_v1")
    assert messages[0]["content"] == RECOVERABLE_INSTRUCTION
    assert "طلب معنى" in messages[1]["content"]
    assert "GIBBERISH" not in messages[1]["content"].split("<text>")[1]
    assert "recoverable" in RECOVERABLE_PROMPT_VERSION


def test_prompt_baseline_runner_with_fake_inference(tmp_path, monkeypatch):
    """Exercise real reports and SQLite logging; no model initialization or GPU."""
    import mlflow

    import jais_guard.baselines.inference

    spec = importlib.util.spec_from_file_location("finetune_cli", ROOT / "scripts/finetune.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    replies = iter(
        [
            {"prediction": "VALID", "parse_status": "success"},
            {"prediction": None, "parse_status": "ambiguous"},
            {"prediction": "GIBBERISH", "parse_status": "success"},
            {"prediction": None, "parse_status": "truncated"},
        ]
    )

    class FakeClassifier:
        def __init__(self, config):
            assert config.max_tokens == 64 and config.local_files_only
            self.model = SimpleNamespace(
                dtype=torch.float16, generation_config=SimpleNamespace(to_dict=lambda: {})
            )

        def classify(self, text, variant):
            assert variant == "recoverable_v1"
            return {
                **next(replies),
                "raw_output": "fixture",
                "prompt": text,
                "finish_reason": "stop",
                "input_tokens": 12,
                "generated_tokens": 4,
            }

    monkeypatch.setattr(jais_guard.baselines.inference, "BaselineClassifier", FakeClassifier)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = mlflow.create_experiment("prompt-test", artifact_location=str(tmp_path / "artifacts"))
    rows = [
        {"id": str(i), "input_text": "نص عربي", "label": label}
        for i, label in enumerate(["VALID", "VALID", "GIBBERISH", "GIBBERISH"])
    ]
    with mlflow.start_run(experiment_id=exp) as run:
        runner.baseline_run({"model": "fixture", "revision": "pinned"}, rows, tmp_path, mlflow)
        mlflow.log_artifacts(str(tmp_path / "evaluation"), "evaluation")
        recorded = mlflow.MlflowClient().get_run(run.info.run_id)
        assert recorded.data.params["prompt_version"] == RECOVERABLE_PROMPT_VERSION
        assert recorded.data.metrics["default/parsing_success_rate"] == 0.5
        assert recorded.data.metrics["default/accuracy"] == 0.5
    metrics = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert metrics["default"]["confusion"] == [[1, 0, 1], [0, 1, 1]]
    assert metrics["default"]["valid_false_rejection_rate"] == 0
    assert "INVALID" in metrics["unparseable_policy"]
    assert "Parsing success: 50.0%" in (tmp_path / "evaluation/report.md").read_text()
    assert "parse_status" in (tmp_path / "evaluation/predictions.csv").read_text()


def candidate(index, text=None, partition="test"):
    return {
        "id": f"test-{index}",
        "input_text": text or "رسالة مفهومة مختلفة",
        "partition": partition,
        "intended_label": "VALID",
        "source_method": "codex",
        "subtype": "fixture",
        "dialect": "MSA",
        "domain": "general",
        "family_id": f"test-family-{index}",
        "review_status": "pending-human-review",
        "intended_label_status": "provisional",
        "provenance": {
            "source": "fixture",
            "source_id": str(index),
            "license": "generated-content-publication-review-pending",
            "attribution": "fixture",
            "rights_evidence": "fixture",
        },
        "generation": {"batch_id": "test-only-batch", "run_id": "test-only-run"},
    }


def test_normalized_near_source_author_parent_and_run_overlap():
    reference = candidate("train", "كيف أختار دراجة مناسبة للتنقل اليومي", "train")
    item = candidate("new", "كَيف أختار دراجة مناسبة للتنقل اليومي")
    assert any(i["kind"] == "duplicate_text" for i in quality_issues([item], [reference]))
    item["input_text"] = "كيف أختار دراجة مناسبة للتنقل اليومي؟"
    assert any(i["kind"] == "near_duplicate" for i in quality_issues([item], [reference]))
    item["input_text"] = "موضوع مغاير بعيدا عن المرجع"
    item["provenance"]["source_id"] = "train"
    item["provenance"]["author_group"] = reference["provenance"]["author_group"] = "author-a"
    item["generation"]["parent_id"] = reference["id"]
    kinds = {i["kind"] for i in workflow.reference_overlap([item], [reference])}
    assert {
        "reserved_source_record",
        "reserved_contributor_group",
        "reserved_parent",
        "reserved_run_id",
        "reserved_batch_id",
    } <= kinds


def test_confounding_report():
    a, b = candidate(0, "نص"), candidate(1, "نص طويل جدا جدا جدا")
    b.update(intended_label="GIBBERISH", source_method="python")
    assert workflow.distribution([a, b], "intended_label")["confounding_warnings"]


@pytest.fixture
def collection(tmp_path):
    reference = tmp_path / "references.sqlite3"
    with connection(reference):
        pass
    results = tmp_path / "previous-results"
    for approach in ("frozen", "lora"):
        for seed in (42, 43, 44):
            directory = results / f"parent-{approach}-{seed}" / f"{approach}-{seed}"
            best = directory / "best"
            best.mkdir(parents=True)
            (directory / "evaluation").mkdir()
            (directory / "evaluation/metrics.json").write_text("{}")
            (best / "classifier.json").write_text(
                json.dumps(
                    {
                        "approach": approach,
                        "seed": seed,
                        "config": {
                            "model": "fixture",
                            "revision": "pinned",
                            "train_sha256": "train",
                            "validation_sha256": "validation",
                        },
                    }
                )
            )
            (best / "head.safetensors").write_bytes(b"fixture")
            (directory / "threshold.json").write_text('{"default":0.5,"exploratory":0.7}')
            if approach == "lora":
                (best / "adapter").mkdir()
                (best / "adapter/adapter_model.safetensors").write_bytes(b"fixture")
                (best / "adapter/adapter_config.json").write_text("{}")
    directory = tmp_path / "collection"
    workflow.initialize(directory, results, reference, ROOT / "configs/test400_plan.json")
    return directory, reference


def test_lock_detects_checkpoint_changes_and_refuses_overwrite(collection):
    directory, reference = collection
    lock, _ = workflow.verify_lock(directory)
    checkpoint = next(iter(lock["checkpoints"][0]["files"]))
    Path(checkpoint).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.verify_lock(directory)
    with pytest.raises(ValueError, match="already exists"):
        workflow.initialize(directory, "unused", reference, ROOT / "configs/test400_plan.json")


def make_fixture_rows():
    rng = random.Random(400)
    dialects = ["MSA"] * 80 + [
        d for d in ("Tunisian", "Egyptian", "Levantine", "Gulf") for _ in range(30)
    ]
    rows = []
    for i in range(400):
        row = candidate(i, "".join(rng.choices("ابتجحدرزسشصضطظعغفقكلمنهوي", k=45)))
        row["intended_label"] = "VALID" if i < 200 else "GIBBERISH"
        row["dialect"] = dialects[i] if i < 200 else "not_applicable"
        rows.append(row)
    return rows


def test_isolated_review_freeze_and_tamper_detection(collection, tmp_path, monkeypatch):
    directory, reference = collection
    # Unit tests for real deduplication are above. Avoid repeated O(n²) similarity
    # scans on 400 synthetic fixture strings in this state-machine integration test.
    monkeypatch.setattr(workflow, "quality_issues", lambda rows, references: [])
    import jais_guard.data.annotations

    monkeypatch.setattr(jais_guard.data.annotations, "quality_issues", lambda rows, references: [])
    candidates = tmp_path / "draft.jsonl"
    rows = make_fixture_rows()
    candidates.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert workflow.import_collection(directory, candidates) == 400
    assert workflow.readonly_candidates(reference) == []
    assert queue("test", path=directory / "annotations.sqlite3")[0]["human_label"] is None
    with pytest.raises(ValueError, match="human labels"):
        workflow.freeze(directory, tmp_path / "exports")
    for row in rows:
        save_annotation(
            row["id"],
            label=row["intended_label"],
            flagged=False,
            notes="fixture human judgment",
            path=directory / "annotations.sqlite3",
        )
    folder = workflow.freeze(directory, tmp_path / "exports")
    assert len(workflow.read_frozen_test(folder)) == 400
    with pytest.raises(ValueError, match="immutable"):
        workflow.import_collection(directory, candidates)
    with pytest.raises(FileExistsError):
        workflow.freeze(directory, tmp_path / "exports")
    (folder / "test.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.read_frozen_test(folder)


def test_candidates_cannot_be_predictions_or_mixed_script(collection, tmp_path, monkeypatch):
    directory, _ = collection
    monkeypatch.setattr(workflow, "quality_issues", lambda rows, references: [])
    rows = make_fixture_rows()
    path = tmp_path / "draft.jsonl"
    for change, message in [
        ({"prediction": 0}, "Classifier"),
        ({"input_text": "طلب hello"}, "Unsupported"),
        ({"intended_label_status": "human"}, "provisional"),
    ]:
        altered = copy.deepcopy(rows)
        altered[0].update(change)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in altered))
        with pytest.raises(ValueError, match=message):
            workflow.check_candidates(directory, path)
    with pytest.raises(FileNotFoundError):
        workflow.read_frozen_test(directory)


def test_partial_draft_audit_never_permits_import(collection, tmp_path):
    directory, _ = collection
    rows = make_fixture_rows()[:12]
    draft = tmp_path / "partial.jsonl"
    draft.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    checked, report = workflow.check_candidates(directory, draft, allow_partial=True)
    assert len(checked) == report["count"] == 12
    assert not report["count_complete"] and not report["ready_for_import"]
    assert not report["issues"]
    assert all(r["review_status"] == "pending-human-review" for r in checked)
    with pytest.raises(ValueError, match="Exactly 400"):
        workflow.check_candidates(directory, draft)
    with pytest.raises(ValueError, match="Exactly 400"):
        workflow.import_collection(directory, draft)
    assert workflow.readonly_candidates(directory / "annotations.sqlite3") == []


def test_collection_recipes_are_deterministic_and_do_not_label_ground_truth():
    spec = importlib.util.spec_from_file_location(
        "collect_test_draft", ROOT / "scripts/collect_test_draft.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = "رسالة عربية فيها أخطاء بسيطة"
    assert len(module.repeat_key(original)) == len(original) + 1
    assert module.scramble_words(original, 123) == module.scramble_words(original, 123)
    assert module.keyboard_displacement(original) != original
    record = module.record(
        original,
        "VALID",
        "codex",
        "MSA",
        "fixture",
        "general",
        "family",
        {},
        {},
        "medium",
        "fixture-time",
    )
    assert record["intended_label_status"] == "provisional"
    assert record["review_status"] == "pending-human-review"
    assert not {"label", "annotation", "prediction"} & record.keys()
