"""CPU fixtures only: historical settings, strict invalid policy and baseline-only execution."""

import hashlib
import inspect
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_final_test import command
from test_final_test import protocol as protocol

from jais_guard.baselines.prompts import (
    RECOVERABLE_INSTRUCTION,
    RECOVERABLE_PROMPT_VERSION,
    build_prompt,
)
from jais_guard.data.test_workflow import sha
from jais_guard.finetuning import final_test
from jais_guard.finetuning import test_baseline as baseline
from jais_guard.finetuning.evaluation import metrics, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archived(protocol, tmp_path):
    folder, rows, frozen = protocol
    config_path = ROOT / "configs/prompt_baseline.json"
    frozen["selection_lock"]["prompt_files"] = {str(config_path): sha(config_path)}
    write_json(folder / "freeze.json", frozen)
    cfg = json.loads(config_path.read_text())
    cfg["prompt_sha256"] = hashlib.sha256(
        (
            RECOVERABLE_PROMPT_VERSION
            + cfg["variant"]
            + RECOVERABLE_INSTRUCTION
            + inspect.getsource(build_prompt)
        ).encode()
    ).hexdigest()
    source = tmp_path / "history" / ("a" * 32)
    (source / "evaluation").mkdir(parents=True)
    write_json(source / "baseline_config.json", cfg)
    identity = {
        k: frozen["selection_lock"]["checkpoints"][0][k]
        for k in ("model", "revision", "train_sha256", "validation_sha256")
    }
    write_json(source / "config.json", identity)
    write_json(
        source / "evaluation/metrics.json",
        {"default": {"examples": 200}, "unparseable_policy": baseline.INVALID_POLICY},
    )
    artifact = source.parent / "artifacts/1" / source.name / "artifacts"
    artifact.mkdir(parents=True)
    write_json(artifact / "generation_config.json", {"eos_token_id": [1]})
    params = {
        k: str(v)
        for k, v in {
            **cfg,
            "approach": "untuned_prompt",
            "model": identity["model"],
            "revision": identity["revision"],
            "do_sample": False,
            "batch_size": 1,
            "dtype": "torch.float16",
            "seed": 42,
        }.items()
    }
    with sqlite3.connect(source.parent / "mlflow.db") as db:
        db.execute(
            "CREATE TABLE runs(run_uuid TEXT,status TEXT,artifact_uri TEXT,experiment_id INT)"
        )
        db.execute("CREATE TABLE params(run_uuid TEXT,key TEXT,value TEXT)")
        db.execute(
            "INSERT INTO runs VALUES (?,?,?,1)",
            (source.name, "FINISHED", f"mlflow-artifacts:/1/{source.name}/artifacts"),
        )
        db.executemany(
            "INSERT INTO params VALUES (?,?,?)", [(source.name, k, v) for k, v in params.items()]
        )
    _, _, registry = final_test.locked_registry(folder)
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    m = metrics([0, 1], [0, 1])
    results = [
        {
            "approach": r["approach"],
            "seed": r["seed"],
            "threshold_validation_locked": r["threshold"],
            "metrics": {
                "default": {**m, "threshold": 0.5},
                "validation_locked": {**m, "threshold": r["threshold"]},
            },
            "timing": {
                "model_load_seconds": 1,
                "steady_state_batch_ms_mean": 1,
                "steady_state_batch_ms_p95": 1,
                "amortized_ms_per_example": 1,
            },
        }
        for r in registry
    ]
    final_test.summarize(comparison, results, [])
    write_json(
        comparison / "completion.json",
        {"status": "complete", "models_evaluated": 6, "test_sha256": frozen["test_sha256"]},
    )
    write_json(comparison / "preflight.json", {"test_sha256": frozen["test_sha256"]})
    return folder, source, comparison, rows, frozen, registry


def arguments(archived, tmp_path):
    folder, source, comparison, *_ = archived
    return [
        "--baseline-only",
        "--test",
        str(folder),
        "--comparison",
        str(comparison),
        "--validation-baseline",
        str(source),
        "--output-root",
        str(tmp_path / "out"),
    ]


def test_archived_settings_exact_and_mismatches_rejected(archived):
    _, source, _, _, frozen, _ = archived
    settings = baseline.validation_settings(source, frozen)
    assert settings["baseline"]["max_tokens"] == 64
    assert settings["invalid_policy"] == baseline.INVALID_POLICY
    cfg = json.loads((source / "baseline_config.json").read_text())
    cfg["max_tokens"] = 65
    write_json(source / "baseline_config.json", cfg)
    with pytest.raises(ValueError, match="configuration changed"):
        baseline.validation_settings(source, frozen)


def test_invalid_policy_keeps_invalid_separate(tmp_path):
    records = [
        {
            "id": str(i),
            "input_text": "نص",
            "label": label,
            "prediction_id": prediction,
            "parse_status": status,
        }
        for i, (label, prediction, status) in enumerate(
            [
                ("VALID", 0, "success"),
                ("VALID", 1, "success"),
                ("VALID", -1, "ambiguous"),
                ("GIBBERISH", 1, "success"),
                ("GIBBERISH", -1, "truncated"),
            ]
        )
    ]
    report = baseline.baseline_report(tmp_path, records)
    m = report["default"]
    assert m["accuracy"] == 2 / 5
    assert m["parsing_success_rate"] == 3 / 5
    assert m["valid_false_rejection_rate"] == 1 / 3
    assert m["valid_acceptance_rate"] == 1 / 3
    assert m["invalid"] == 2 and m["valid_invalid_outputs"] == 1
    assert m["confusion"] == [[1, 1, 1], [0, 1, 1]]
    assert m["macro_f1"] == metrics([0, 0, 0, 1, 1], [0, 1, -1, 1, -1])["macro_f1"]


def test_generation_timer_does_not_change_arguments_or_recover_invalid(monkeypatch):
    from jais_guard.baselines.prompts import parse_diagnostic

    calls = []

    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return "</ifm|think_faster>VALID GIBBERISH"

    class Classifier:
        model = Model()

        def classify(self, text, variant):
            assert text == "طلب" and variant == "recoverable_v1"
            raw = self.model.generate(max_new_tokens=64, do_sample=False, pad_token_id=1)
            return {**parse_diagnostic(raw, "low", True), "raw_output": raw}

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    classifier = Classifier()
    original = classifier.model.generate
    record = baseline.classify_timed(classifier, "طلب", "recoverable_v1")
    assert calls == [{"max_new_tokens": 64, "do_sample": False, "pad_token_id": 1}]
    assert classifier.model.generate == original
    assert record["prediction_id"] == -1 and record["parse_status"] == "ambiguous"
    assert record["latency_ms"] >= record["generation_ms"] >= 0


def test_comparison_insertion_preserves_classifier_results(archived, tmp_path):
    _, _, directory, _, frozen, registry = archived
    snapshot = baseline.comparison_snapshot(directory, frozen, registry)
    original = (directory / "comparison.md").read_bytes()
    metric_bytes = (directory / "metrics.json").read_bytes()
    out = tmp_path / "baseline"
    out.mkdir()
    report = {
        "default": {
            **metrics([0, 1], [0, 1]),
            "parsing_success_rate": 1,
            "valid_invalid_outputs": 0,
        }
    }
    timing = {"latency_ms_mean": 4, "generation_ms_mean": 3, "model_load_seconds": 1}
    baseline.attach_comparison(snapshot, out, report, timing, "new-run")
    assert (directory / "metrics.json").read_bytes() == metric_bytes
    assert (out / "classifier_comparison.original.md").read_bytes() == original
    updated = (directory / "comparison.md").read_text()
    assert updated.count("| untuned_prompt |") == 1
    # Removing only the newly inserted row and new section recovers the exact old report.
    restored = "\n".join(
        line
        for line in updated.split("\n## Untuned zero-shot baseline")[0].split("\n")
        if not line.startswith("| untuned_prompt |")
    )
    assert restored.encode() == original
    combined = json.loads((out / "comparison_with_baseline.json").read_text())
    assert combined["runs"] == snapshot["metrics"]["runs"]
    with pytest.raises(ValueError, match="already attached"):
        baseline.comparison_snapshot(directory, frozen, registry)


def test_baseline_cli_check_and_cuda_guard_never_rerun_classifiers(archived, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Classifier execution is forbidden in baseline-only mode")

    monkeypatch.setattr(final_test, "timed_classifier", forbidden)
    monkeypatch.setattr(command, "source_runs", forbidden)
    monkeypatch.setattr(baseline, "prompt_preflight", lambda *a: {})
    command.main(arguments(archived, tmp_path) + ["--check-only"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU inference is disabled"):
        command.main(arguments(archived, tmp_path))
    assert not (tmp_path / "out").exists()


def test_baseline_argument_mistake_cannot_launch_six_classifiers(tmp_path):
    with pytest.raises(SystemExit):
        command.main(["--comparison", str(tmp_path)])
    with pytest.raises(SystemExit):
        command.main(["--baseline-only"])


def test_baseline_only_mlflow_integration_on_synthetic_rows(archived, tmp_path, monkeypatch):
    import mlflow

    from jais_guard.finetuning import tracking

    monkeypatch.chdir(tmp_path)
    _, source, comparison, rows, frozen, _ = archived
    original_metrics = (comparison / "metrics.json").read_bytes()
    settings = baseline.validation_settings(source, frozen)
    original_get_run = mlflow.tracking.MlflowClient.get_run

    def get_run(self, run_id):
        if run_id == source.name:
            return SimpleNamespace(
                info=SimpleNamespace(status="FINISHED", experiment_id="1"),
                data=SimpleNamespace(params=settings["validation_params"]),
            )
        return original_get_run(self, run_id)

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "get_run", get_run)
    monkeypatch.setattr(baseline, "prompt_preflight", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=6)
    )
    monkeypatch.setattr(tracking, "environment", lambda: {"hardware": "CPU fixture"})

    def infer(fixture_rows, _settings, _output):
        assert fixture_rows == rows
        records = [
            {**r, "prediction_id": int(r["label"] == "GIBBERISH"), "parse_status": "success"}
            for r in rows
        ]
        return records, {"latency_ms_mean": 4, "generation_ms_mean": 3, "model_load_seconds": 1}

    monkeypatch.setattr(baseline, "infer_baseline", infer)
    uri = "sqlite:///" + str(tmp_path / "target.db")
    command.main(arguments(archived, tmp_path) + ["--tracking-uri", uri])
    assert (comparison / "metrics.json").read_bytes() == original_metrics
    assert (comparison / "comparison.md").read_text().count("| untuned_prompt |") == 1
    outputs = list((tmp_path / "out").iterdir())
    assert len(outputs) == 1
    assert json.loads((outputs[0] / "completion.json").read_text())["classifiers_rerun"] == 0
    runs = mlflow.tracking.MlflowClient(tracking_uri=uri).search_runs(["1"])
    assert len(runs) == 1 and runs[0].info.status == "FINISHED"
    assert runs[0].data.tags["approach"] == "untuned_prompt"
    assert runs[0].data.tags["evaluation_stage"] == "independent_test"
