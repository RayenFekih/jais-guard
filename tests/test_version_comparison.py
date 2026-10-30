"""CPU-only validation lock, cache compatibility, paired errors and fake GPU orchestration."""

import copy
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.data.test_workflow import sha
from jais_guard.finetuning import version_comparison as vc
from jais_guard.finetuning.evaluation import metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_versions_command", ROOT / "scripts/compare_versions.py"
)
command = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command)


def row(i, prefix="validation", valid=None):
    valid = i % 2 == 0 if valid is None else valid
    label = "VALID" if valid else "GIBBERISH"
    topic = {"train": "تدريب", "validation": "تحقق", "test": "اختبار"}[prefix]
    return {
        "id": f"{prefix}-{i}",
        "input_text": f"نص {i} مختلف للتجربة {topic}",
        "label": label,
        "partition": prefix if prefix != "test" else "test",
        "family_id": f"family-{prefix}-{i}",
        "subtype": "recoverable_corruption",
        "dialect": "Tunisian",
        "label_origin": "human",
        "source_method": "python",
        "review_status": "human-reviewed",
        "is_gibberish": not valid,
        "annotation": {"label": label, "flagged": False},
    }


def fixture_export(directory, partition, rows):
    directory.mkdir()
    path = directory / f"{partition}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_json(
        directory / "manifest.json",
        {
            "version": f"fixture-{partition}",
            "schema_version": 2,
            "partition": partition,
            "complete": True,
            "annotation_complete": True,
            "target_reached": True,
            "exported": len(rows),
            "sha256": sha(path),
            "review_status_counts": {"human-reviewed": len(rows)},
            "label_origin_counts": {"human": len(rows)},
        },
    )
    return path


@pytest.fixture
def registry_case(tmp_path):
    train = fixture_export(tmp_path / "train", "train", [row(i, "train") for i in range(1500)])
    val_rows = [row(i) for i in range(250)]
    validation = fixture_export(tmp_path / "validation", "validation", val_rows)
    cfg = {
        "iteration": 2,
        "validation_review_complete": True,
        "model": "fixture-model",
        "revision": "fixture-revision",
        "train_path": str(train),
        "train_sha256": sha(train),
        "validation_path": str(validation),
        "validation_sha256": sha(validation),
        "train_counts": {"VALID": 750, "GIBBERISH": 750},
        "validation_counts": {"VALID": 125, "GIBBERISH": 125},
        "max_length": 64,
        "experiment": "JaisGuard/general-arabic-v2",
        "output_dir": str(tmp_path / "models"),
    }
    config = tmp_path / "config.json"
    write_json(config, cfg)
    db = tmp_path / "tracking-source.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE experiments(experiment_id INT,name TEXT);"
            "CREATE TABLE runs(run_uuid TEXT,status TEXT,experiment_id INT);"
            "CREATE TABLE tags(run_uuid TEXT,key TEXT,value TEXT);"
            "CREATE TABLE params(run_uuid TEXT,key TEXT,value TEXT);"
        )
        conn.execute("INSERT INTO experiments VALUES (2,?)", (cfg["experiment"],))
        for index, (approach, seed) in enumerate(sorted(vc.EXPECTED)):
            parent, child = f"{index + 1:032x}", f"{index + 100:032x}"
            conn.executemany("INSERT INTO runs VALUES (?, 'FINISHED', 2)", [(parent,), (child,)])
            conn.execute("INSERT INTO tags VALUES (?, 'mlflow.parentRunId', ?)", (child, parent))
            conn.executemany(
                "INSERT INTO params VALUES (?,?,?)",
                [
                    (child, "approach", approach),
                    (child, "seed", str(seed)),
                    (child, "train_sha256", cfg["train_sha256"]),
                    (child, "validation_sha256", cfg["validation_sha256"]),
                ],
            )
            root = Path(cfg["output_dir"]) / parent / f"{approach}-{seed}"
            checkpoint = root / "best"
            (checkpoint / "tokenizer").mkdir(parents=True)
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (checkpoint / "tokenizer" / name).write_text("{}")
            (checkpoint / "head.safetensors").write_bytes(b"fixture-not-weights")
            if approach == "lora":
                (checkpoint / "adapter").mkdir()
                (checkpoint / "adapter/adapter_config.json").write_text("{}")
                (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"fixture")
            records = []
            for r in val_rows:
                score = 0.2 if r["label"] == "VALID" else 0.7
                records.append(
                    {
                        **r,
                        "p_gibberish": score,
                        "truth_id": int(r["label"] == "GIBBERISH"),
                        "prediction_default": int(score >= 0.5),
                        "prediction_exploratory": int(score >= 0.7),
                        "logit_valid": float(np.log(1 - score)),
                        "logit_gibberish": float(np.log(score)),
                    }
                )
            loss = float(
                np.mean([-np.log(0.8) if r["label"] == "VALID" else -np.log(0.7) for r in records])
            )
            history = [
                {
                    "epoch": i,
                    "seed": seed,
                    "global_step": i * 10,
                    "validation_loss": v,
                    "train_loss": 0.1,
                }
                for i, v in enumerate((0.5, loss, 0.4), 1)
            ]
            write_json(root / "history.json", history)
            write_json(
                checkpoint / "classifier.json",
                {
                    "approach": approach,
                    "seed": seed,
                    "config": cfg,
                    "metadata": history[1],
                    "label_mapping": {"VALID": 0, "GIBBERISH": 1},
                    "pooling": "last_nonpadding_final_hidden",
                },
            )
            write_json(
                root / "threshold.json",
                {"default": 0.5, "exploratory": 0.7, "selection_data": "validation"},
            )
            evaluation = root / "evaluation"
            evaluation.mkdir()
            (evaluation / "predictions.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            measured = metrics(
                [r["truth_id"] for r in records], [r["prediction_default"] for r in records]
            )
            write_json(
                evaluation / "metrics.json",
                {
                    "default": measured,
                    "exploratory": {"threshold": 0.7, **measured},
                    "selection_data": "validation; not independent test performance",
                },
            )
            (evaluation / "report.md").write_text("fixture validation")
            (evaluation / "thresholds.csv").write_text("threshold\n0.5\n0.7\n")
    return config, db


def test_inventory_checks_all_validation_epochs_and_saved_thresholds(registry_case):
    registry, files, _ = vc.v2_inventory(*registry_case)
    assert len(registry) == 6
    assert all(r["selected_epoch"] == 2 and r["threshold"] == 0.7 for r in registry)
    vc.verify_files(files)


@pytest.mark.parametrize("mutation", ["epoch", "threshold", "validation_text", "seed"])
def test_invalid_selection_provenance_stops(registry_case, mutation):
    config, db = registry_case
    cfg = json.loads(config.read_text())
    path = next(Path(cfg["output_dir"]).glob("*/*/best/classifier.json"))
    if mutation in ("epoch", "seed"):
        value = json.loads(path.read_text())
        if mutation == "epoch":
            value["metadata"]["epoch"] = 3
        else:
            value["seed"] = 99
        write_json(path, value)
    elif mutation == "threshold":
        value = path.parents[1] / "threshold.json"
        write_json(value, {"default": 0.5, "exploratory": 0.7, "selection_data": "test"})
    else:
        file = path.parents[1] / "evaluation/predictions.jsonl"
        values = [json.loads(s) for s in file.read_text().splitlines()]
        values[0]["input_text"] = "نص لم يرد في بيانات التحقق"
        file.write_text("".join(json.dumps(r) + "\n" for r in values))
    with pytest.raises(ValueError):
        vc.v2_inventory(config, db)


def test_alignment_requires_identical_raw_text_labels_and_ids():
    rows = [row(1)]
    for field in ("id", "label", "input_text"):
        altered = copy.deepcopy(rows)
        altered[0][field] = "changed"
        with pytest.raises(ValueError):
            vc.identity_rows(rows, altered)


def test_cached_prediction_metric_compatibility():
    rows = [row(0), row(1)]
    truth = [0, 1]
    pred = [
        {
            **r,
            "p_gibberish": s,
            "prediction_default": int(s >= 0.5),
            "prediction_validation_locked": int(s >= 0.7),
        }
        for r, s in zip(rows, [0.6, 0.8], strict=True)
    ]
    reports = {"default": metrics(truth, [1, 1]), "validation_locked": metrics(truth, [0, 1])}
    measured, _ = vc.cached_metrics(rows, pred, 0.7, reports)
    assert measured["default"]["valid_false_rejection_rate"] == 1
    assert measured["validation_locked"]["valid_acceptance_rate"] == 1
    reports["default"]["accuracy"] = 0.9
    with pytest.raises(ValueError, match="compatibility mismatch"):
        vc.cached_metrics(rows, pred, 0.7, reports)


def test_paired_transitions_and_recoverable_dialect_groups(tmp_path):
    rows = [row(i, "test", i < 2) for i in range(4)]
    results = []
    for version, preds in (("v1", [1, 0, 0, 1]), ("v2", [0, 1, 0, 1])):
        for approach, seed in sorted(vc.EXPECTED):
            results.append(
                {
                    "version": version,
                    "approach": approach,
                    "seed": seed,
                    "metrics": {m: {"threshold": 0.5} for m in vc.MODES},
                    "records": [
                        {
                            **r,
                            "prediction_default": p,
                            "prediction_validation_locked": p,
                            "p_gibberish": 0.9 if p else 0.1,
                        }
                        for r, p in zip(rows, preds, strict=True)
                    ],
                }
            )
    summary, detailed = vc.paired_analysis(tmp_path, rows, results)
    group = summary["frozen/default/seed-42/all"]
    assert [group[k] for k in ("corrected", "new_error", "remaining_error", "both_correct")] == [
        1,
        1,
        1,
        1,
    ]
    assert (
        summary["lora/default/three-seed-decisions/recoverable_corruption_human_VALID"]["new_error"]
        == 3
    )
    assert len(detailed) == 48


@pytest.fixture
def prepared(registry_case, tmp_path, monkeypatch):
    config, db = registry_case
    registry, _, _ = vc.v2_inventory(config, db)
    rows = [row(i, "test", i < 188) for i in range(400)]
    frozen = {"test_sha256": "fixture-sha", "selection_lock": {"prompt_files": {}}}
    test = tmp_path / "frozen"
    test.mkdir()
    for name in ("test.jsonl", "manifest.json", "freeze.json"):
        (test / name).write_text("fixture-only")
    monkeypatch.setattr(vc, "locked_registry", lambda _: (rows, frozen, registry))
    cache = tmp_path / "cached.json"
    cache.write_text("immutable cached results")
    results = []
    truth = [int(r["label"] == "GIBBERISH") for r in rows]
    for approach, seed in sorted(vc.EXPECTED):
        measured = {
            m: {"threshold": 0.5 if m == "default" else 0.7, **metrics(truth, truth)}
            for m in vc.MODES
        }
        records = [
            {
                **r,
                "prediction_default": y,
                "prediction_validation_locked": y,
                "p_gibberish": 0.9 if y else 0.1,
            }
            for r, y in zip(rows, truth, strict=True)
        ]
        results.append(
            {
                "version": "v1",
                "approach": approach,
                "seed": seed,
                "metrics": measured,
                "records": records,
                "timing": {
                    "batch_size": 2,
                    "warmup_batches": 3,
                    "model_load_seconds": 1,
                    "amortized_ms_per_example": 1,
                },
            }
        )
    results.append(
        {
            "version": "untuned",
            "approach": "untuned_prompt",
            "seed": 42,
            "metrics": {"zero_shot": {**metrics(truth, truth), "parsing_success_rate": 1}},
            "records": rows,
            "timing": {
                "batch_size": 1,
                "model_load_seconds": 2,
                "latency_ms_mean": 5,
                "generation_ms_mean": 4,
            },
        }
    )
    monkeypatch.setattr(
        vc, "load_reused", lambda *a: (copy.deepcopy(results), {str(cache): sha(cache)})
    )
    monkeypatch.setattr(vc, "preprocessing_check", lambda *a: {"identical": True})
    directory = tmp_path / "versioned-comparison"
    vc.prepare(directory, config, db, test, tmp_path / "v1", tmp_path / "baseline")
    return directory, rows, registry, cache


def test_prepare_is_immutable_and_locks_before_gpu(prepared):
    directory, _, _, cache = prepared
    assert (directory / "validation_comparison.md").exists()
    assert (directory / "threshold_lock.json").exists()
    vc.checked_protocol(directory)
    cache.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        vc.checked_protocol(directory)


def test_cuda_absence_never_creates_predictions(prepared, monkeypatch):
    directory, *_ = prepared
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU inference is disabled"):
        command.main(["evaluate", "--directory", str(directory)])
    assert not (directory / "benchmark").exists()


def test_new_mlflow_evaluation_only_loads_v2_fixture_models(prepared, tmp_path, monkeypatch):
    import mlflow

    from jais_guard.finetuning import final_test, tracking

    directory, rows, registry, cache = prepared
    before = cache.read_bytes()
    monkeypatch.chdir(tmp_path)
    sources = {r["source_runs"]["training_run_id"]: r for r in registry}
    original = mlflow.tracking.MlflowClient.get_run

    def get_run(self, rid):
        if rid in sources:
            r = sources[rid]
            return SimpleNamespace(
                info=SimpleNamespace(status="FINISHED"),
                data=SimpleNamespace(params={"approach": r["approach"], "seed": str(r["seed"])}),
            )
        return original(self, rid)

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "get_run", get_run)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=6)
    )
    monkeypatch.setattr(tracking, "environment", lambda: {"hardware": "CPU fixture"})
    visits = []

    def fake_infer(item, fixture_rows, batch, warmup):
        assert item["metadata"]["config"]["iteration"] == 2
        assert fixture_rows == rows and (batch, warmup) == (2, 3)
        visits.append((item["approach"], item["seed"]))
        return (
            [0.1 if r["label"] == "VALID" else 0.9 for r in rows],
            rows,
            {
                "batch_size": 2,
                "warmup_batches": 3,
                "model_load_seconds": 1,
                "amortized_ms_per_example": 1,
            },
        )

    monkeypatch.setattr(final_test, "timed_classifier", fake_infer)
    uri = "sqlite:///" + str(tmp_path / "target.db")
    command.main(["evaluate", "--directory", str(directory), "--tracking-uri", uri])
    assert len(visits) == 6 and set(visits) == vc.EXPECTED
    assert cache.read_bytes() == before
    output = directory / "benchmark"
    assert (output / "comparison.md").exists()
    assert json.loads((output / "completion.json").read_text())["cached_models_reused"] == 7
    all_metrics = json.loads((output / "metrics.json").read_text())
    assert len(all_metrics["groups"]) == 9
    assert "not an untouched" in all_metrics["scope"]
    assert len([r for r in mlflow.tracking.MlflowClient(tracking_uri=uri).search_runs(["1"])]) == 7
    with pytest.raises(ValueError, match="Execution already exists"):
        command.main(["evaluate", "--directory", str(directory), "--tracking-uri", uri])
