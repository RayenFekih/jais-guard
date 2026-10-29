"""CPU-only final-test gates and orchestration; fixture data only, no Jais inference."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.finetuning import final_test
from jais_guard.finetuning.evaluation import write_json
from jais_guard.test_workflow import sha

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_test_command", ROOT / "scripts/evaluate_test.py"
)
command = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command)


@pytest.fixture
def protocol(tmp_path):
    folder = tmp_path / "test"
    folder.mkdir()
    rows = [
        {
            "id": f"fixture-{i}",
            "family_id": f"family-{i}",
            "input_text": f"نص مثال {i}",
            "partition": "test",
            "label": "VALID" if i < 188 else "GIBBERISH",
            "annotation": {"label": "VALID" if i < 188 else "GIBBERISH", "flagged": False},
            "is_gibberish": i >= 188,
            "review_status": "human-reviewed",
            "label_origin": "human",
            "subtype": "fixture",
            "dialect": "unknown",
            "source_method": "python",
        }
        for i in range(400)
    ]
    (folder / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        folder / "manifest.json",
        {
            "schema_version": 2,
            "partition": "test",
            "complete": True,
            "target_reached": True,
            "exported": 400,
            "sha256": sha(folder / "test.jsonl"),
            "review_status_counts": {"human-reviewed": 400},
            "label_origin_counts": {"human": 400},
        },
    )
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text("fixture train")
    val.write_text("fixture validation")
    entries = []
    for approach in ("frozen", "lora"):
        for seed in (42, 43, 44):
            run = tmp_path / f"{approach}-{seed}" / "best"
            run.mkdir(parents=True)
            (run / "tokenizer").mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (run / "tokenizer" / name).write_text("{}")
            identity = {
                "model": "fixture-model",
                "revision": "fixture-revision",
                "train_sha256": sha(train),
                "validation_sha256": sha(val),
            }
            config = {
                **identity,
                "train_path": str(train),
                "validation_path": str(val),
                "max_length": 64,
            }
            write_json(
                run / "classifier.json",
                {
                    "approach": approach,
                    "seed": seed,
                    "config": config,
                    "label_mapping": {"VALID": 0, "GIBBERISH": 1},
                    "pooling": "last_nonpadding_final_hidden",
                },
            )
            (run / "head.safetensors").write_bytes(b"fixture, not model weights")
            if approach == "lora":
                (run / "adapter").mkdir()
                (run / "adapter/adapter_model.safetensors").write_bytes(b"fixture adapter")
                (run / "adapter/adapter_config.json").write_text("{}")
            threshold = run.parent / "threshold.json"
            write_json(
                threshold,
                {
                    "default": 0.5,
                    "exploratory": 0.8,
                    "selection_data": "validation",
                    "production_requirement": False,
                },
            )
            files = [p for p in run.rglob("*") if p.is_file()] + [threshold]
            entries.append(
                {
                    "approach": approach,
                    "seed": seed,
                    **identity,
                    "files": {str(p): sha(p) for p in files},
                }
            )
    frozen = {
        "version": "test-400-v1",
        "human_review_complete": True,
        "test_sha256": sha(folder / "test.jsonl"),
        "manifest_sha256": sha(folder / "manifest.json"),
        "evaluation_allowed_only_with_locked_protocol": True,
        "selection_lock": {"checkpoints": entries, "prompt_files": {}},
    }
    write_json(folder / "freeze.json", frozen)
    return folder, rows, frozen


def test_registry_exact_six_and_fixed_thresholds(protocol):
    folder, _, _ = protocol
    rows, _, registry = final_test.locked_registry(folder)
    assert len(rows) == 400
    assert [(r["approach"], r["seed"]) for r in registry] == [
        (a, s) for a in ("frozen", "lora") for s in (42, 43, 44)
    ]
    assert all(r["threshold"] == 0.8 for r in registry)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "threshold", "unpinned", "metadata"])
def test_registry_fails_closed(protocol, mutation):
    folder, _, frozen = protocol
    entries = frozen["selection_lock"]["checkpoints"]
    if mutation == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    elif mutation == "missing":
        entries.pop()
    else:
        entry = entries[0]
        metadata = next(Path(p) for p in entry["files"] if p.endswith("/classifier.json"))
        if mutation == "unpinned":
            (metadata.parent / "tokenizer/added_tokens.json").write_text("{}")
        elif mutation == "metadata":
            value = json.loads(metadata.read_text())
            value["seed"] = 44
            write_json(metadata, value)
            entry["files"][str(metadata)] = sha(metadata)
        else:
            threshold = next(Path(p) for p in entry["files"] if p.endswith("/threshold.json"))
            write_json(threshold, {"default": 0.5, "exploratory": 0.6, "selection_data": "test"})
            entry["files"][str(threshold)] = sha(threshold)
    write_json(folder / "freeze.json", frozen)
    with pytest.raises(ValueError):
        final_test.locked_registry(folder)


def test_changed_weights_and_test_export_rejected(protocol):
    folder, _, frozen = protocol
    file = next(
        Path(p)
        for p in frozen["selection_lock"]["checkpoints"][0]["files"]
        if p.endswith("head.safetensors")
    )
    file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model or threshold changed"):
        final_test.locked_registry(folder)
    (folder / "test.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        final_test.locked_registry(folder)


def test_metrics_do_not_select_threshold(tmp_path, monkeypatch):
    import jais_guard.finetuning.evaluation as evaluation

    def forbidden(*args, **kwargs):
        raise AssertionError("Test threshold selection is forbidden")

    monkeypatch.setattr(evaluation, "threshold_report", forbidden)
    rows = [
        {"id": "a", "input_text": "طلب مفهوم", "label": "VALID"},
        {"id": "b", "input_text": "نص معطوب", "label": "GIBBERISH"},
    ]
    report, records = final_test.fixed_report(tmp_path / "report", rows, [0.7, 0.9], 0.8)
    assert report["default"]["accuracy"] == 0.5
    assert report["validation_locked"]["accuracy"] == 1
    assert report["default"]["valid_false_rejection_rate"] == 1
    assert report["validation_locked"]["valid_acceptance_rate"] == 1
    assert report["validation_locked"]["threshold"] == 0.8
    assert len(records) == 2
    assert not (tmp_path / "report/thresholds.csv").exists()
    assert (tmp_path / "report/confusion_default.png").is_file()


def test_no_cuda_never_loads_model_or_starts_mlflow(protocol, tmp_path, monkeypatch):
    folder, _, _ = protocol
    monkeypatch.setattr(final_test, "token_preflight", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU inference is intentionally unsupported"):
        command.main(["--test", str(folder), "--output-root", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_source_runs_require_unique_finished_child():
    def run(identifier, status="FINISHED", params=None):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=identifier, status=status, experiment_id="1"),
            data=SimpleNamespace(params=params or {}),
        )

    client = SimpleNamespace(
        get_run=lambda _: run("parent"),
        search_runs=lambda *a, **k: [run("child", params={"approach": "frozen", "seed": "42"})],
    )
    registry = [{"source_parent_run_id": "a" * 32, "approach": "frozen", "seed": 42}]
    assert command.source_runs(client, registry, "1")["frozen-42"]["training_run_id"] == "child"
    client.search_runs = lambda *a, **k: []
    with pytest.raises(ValueError, match="uniquely resolve"):
        command.source_runs(client, registry, "1")


def test_full_orchestration_uses_new_local_mlflow_runs(protocol, tmp_path, monkeypatch):
    """Synthetic CPU scores exercise artifacts and MLflow, never the real test or Jais."""
    import jais_guard.finetuning.tracking as tracking

    folder, fixture_rows, _ = protocol
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(final_test, "token_preflight", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=100, major=8, minor=9),
    )
    monkeypatch.setattr(tracking, "environment", lambda: {"hardware": "CPU fixture"})
    monkeypatch.setattr(
        command,
        "source_runs",
        lambda client, registry, experiment_id: {
            f"{r['approach']}-{r['seed']}": {"training_run_id": "fixture"} for r in registry
        },
    )
    visits = []

    def fake_inference(item, rows, batch, warmup):
        assert rows == fixture_rows
        visits.append((item["approach"], item["seed"]))
        scores = [0.1 if r["label"] == "VALID" else 0.9 for r in rows]
        return (
            scores,
            rows,
            {
                "model_load_seconds": 1.0,
                "steady_state_batch_ms_mean": 2.0,
                "steady_state_batch_ms_p95": 3.0,
                "amortized_ms_per_example": 1.0,
            },
        )

    monkeypatch.setattr(final_test, "timed_classifier", fake_inference)
    uri = "sqlite:///" + str(tmp_path / "tracking.db")
    command.main(
        ["--test", str(folder), "--output-root", str(tmp_path / "out"), "--tracking-uri", uri]
    )
    outputs = list((tmp_path / "out").iterdir())
    assert len(outputs) == 1 and len(visits) == 6
    assert (outputs[0] / "comparison.md").is_file()
    assert json.loads((outputs[0] / "completion.json").read_text())["models_evaluated"] == 6
    metrics = json.loads((outputs[0] / "metrics.json").read_text())
    assert metrics["std_ddof"] == 1
    assert metrics["across_seeds"]["lora/default"]["accuracy"] == {"mean": 1, "std": 0}
    assert all(r["metrics"]["validation_locked"]["threshold"] == 0.8 for r in metrics["runs"])
    import mlflow

    runs = mlflow.tracking.MlflowClient(tracking_uri=uri).search_runs(["1"])
    assert len(runs) == 7
    assert all(r.info.status == "FINISHED" for r in runs)
    assert all(r.data.tags["evaluation_stage"] == "independent_test" for r in runs)


def test_cpu_timing_fixture_hides_labels_and_releases_model(tmp_path, monkeypatch):
    import weakref

    import jais_guard.finetuning.model as model_module

    references = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)

    def loader(*args):
        model = Model()
        references.append(weakref.ref(model))
        return model, object(), {"config": {"max_length": 64}}

    def predict(model, tokenizer, rows, *args):
        assert all(set(r) == {"id", "input_text"} for r in rows)
        return [0.2] * len(rows), torch.zeros((len(rows), 2))

    import jais_guard.finetuning.data as data

    monkeypatch.setattr(data, "encode_rows", lambda rows, *a: [{"input_ids": [1]} for r in rows])
    monkeypatch.setattr(model_module, "load_checkpoint", loader)
    monkeypatch.setattr(model_module, "predict_rows", predict)
    monkeypatch.setattr(model_module, "seed_everything", lambda _: None)
    for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    scores, enriched, timing = final_test.timed_classifier(
        {"seed": 42, "checkpoint": str(tmp_path)},
        [{"id": "a", "input_text": "طلب", "label": "VALID"}] * 3,
        2,
        1,
    )
    assert scores == [0.2, 0.2, 0.2]
    assert len(enriched) == 3 and timing["warmup_batches"] == 1
    assert references[0]() is None
    assert timing["steady_state_batch_ms_p95"] >= 0


@pytest.mark.parametrize("mode", ["environment", "active"])
def test_existing_mlflow_run_cannot_be_resumed(protocol, tmp_path, monkeypatch, mode):
    import mlflow

    folder, _, _ = protocol
    monkeypatch.setattr(final_test, "token_preflight", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    if mode == "environment":
        monkeypatch.setenv("MLFLOW_RUN_ID", "existing-run")
    else:
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
        monkeypatch.setattr(mlflow, "active_run", lambda: object())
    with pytest.raises(ValueError, match="not resume an existing"):
        command.main(["--test", str(folder), "--output-root", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
