"""CPU-only checks: tiny random K2 Horizon, never pretrained model training."""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", tempfile.mkdtemp(prefix="jaisguard-hf-modules-"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch

from jais_guard.finetuning.data import collate, encode_rows, load_datasets
from jais_guard.finetuning.evaluation import metrics, save_report, threshold_report
from jais_guard.finetuning.model import (
    last_nonpadding,
    load_checkpoint,
    load_classifier,
    optimizer_for,
    predict_rows,
    save_checkpoint,
    seed_everything,
    verify_trainable,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = json.loads((ROOT / "configs/finetuning.json").read_text())
    for key in ("train_path", "validation_path"):
        cfg[key] = str(ROOT / cfg[key])
    return cfg


@pytest.fixture(scope="module")
def tiny_backbone_path(tmp_path_factory):
    """Use the cached official architecture with tiny dimensions and random weights."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    revision = "e51941b37091287fee4934e3f8a799e0d3fef37f"
    try:
        cfg = AutoConfig.from_pretrained(
            "IFM/Jais-3-0.9B", revision=revision, trust_remote_code=True, local_files_only=True
        )
    except OSError:
        pytest.skip("Pinned architecture not cached")
    cfg.hidden_size = 16
    cfg.intermediate_size = 32
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 1
    cfg.head_dim = 8
    cfg.rope_head_dim = 8
    cfg.mlp_only_layers = [0, 1]
    cfg.use_cache = False
    cfg._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = AutoModel.from_config(cfg, trust_remote_code=True)
    path = tmp_path_factory.mktemp("tiny-k2")
    model.save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(
        "IFM/Jais-3-0.9B", revision=revision, trust_remote_code=True, local_files_only=True
    )
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_config(config, tiny_backbone_path, tmp_path):
    config["model"] = str(tiny_backbone_path)
    config["output_dir"] = str(tmp_path / "outputs")
    return config


def test_real_exports_are_immutable_inputs(config):
    if not Path(config["train_path"]).exists():
        pytest.skip("Local exports unavailable")
    train, val, tm, vm = load_datasets(config)
    assert (len(train), len(val)) == (700, 200)
    assert sum(r["label_origin"] == "human" for r in train) == 567
    assert vm["annotation_complete"] and not tm["annotation_complete"]
    changed = copy.deepcopy(config)
    changed["train_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        load_datasets(changed)


def test_pooling_ignores_padding_and_rejects_empty():
    hidden = torch.arange(24).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert torch.equal(last_nonpadding(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 2]]))
    with pytest.raises(ValueError, match="mask"):
        last_nonpadding(hidden, torch.zeros_like(mask))


def test_threshold_ties_endpoints_and_default():
    truth = [0] * 99 + [1] * 101
    scores = [0.1] * 98 + [0.8] + [0.8] * 100 + [1.0]
    report = threshold_report(truth, scores)
    assert report["default"]["valid_false_rejections"] == 1
    assert report["exploratory"]["threshold"] == 0.8
    assert report["exploratory"]["gibberish_recall"] == 1
    assert report["sweep"][-1]["threshold"] > 1
    assert report["sweep"][-1]["valid_false_rejection_rate"] == 0
    assert report["sweep"][-1]["gibberish_recall"] == 0
    zero = threshold_report(truth, scores, limit=0)
    assert zero["exploratory"]["threshold"] == 1
    assert zero["default"] == report["default"]
    with pytest.raises(ValueError, match="nonfinite"):
        threshold_report([0, 1], [np.nan, 0.2])


def test_invalid_predictions_penalize_recall_without_false_rejection():
    report = metrics([0, 0, 1, 1], [-1, 0, -1, 1])
    assert report["accuracy"] == 0.5
    assert report["valid_false_rejection_rate"] == 0
    assert report["valid_acceptance_rate"] == 0.5
    assert report["valid_recall"] == report["gibberish_recall"] == 0.5
    assert report["confusion"] == [[1, 0, 1], [0, 1, 1]]
    assert metrics([1], [0])["valid_false_rejection_rate"] is None


def test_report_includes_errors_metadata_and_full_sweep(tmp_path):
    rows = [
        {
            "id": str(i),
            "input_text": "نص",
            "label": label,
            "dialect": "MSA",
            "dialect_status": "proposed_not_human_confirmed",
            "input_tokens": 3,
        }
        for i, label in enumerate(["VALID", "GIBBERISH"])
    ]
    report = save_report(tmp_path, rows, [0.9, 0.8])
    saved = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert saved[0]["dialect_status"] == "proposed_not_human_confirmed"
    assert saved[0]["prediction_default"] == 1
    assert saved[0]["prediction_exploratory"] == 0
    assert report["subgroups"]["input_length:1-8"]["default"]["examples"] == 2
    assert (tmp_path / "thresholds.csv").exists()
    assert (tmp_path / "confusion_default.png").exists()
    assert (tmp_path / "threshold_tradeoff.png").exists()


@pytest.mark.parametrize("approach", ["frozen", "lora"])
def test_tiny_k2_gradients_padding_and_checkpoint(tiny_config, tmp_path, approach):
    seed_everything(42)
    model, tokenizer = load_classifier(tiny_config, approach, "cpu")
    rows = [{"id": "a", "input_text": "مرحبا"}, {"id": "b", "input_text": "أريد حجز موعد جديد"}]
    tokens = encode_rows(rows, tokenizer, 64)
    with pytest.raises(ValueError, match="budget"):
        encode_rows(rows, tokenizer, 1)
    batch = collate(tokens, tokenizer, "cpu", pad_to=16)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = optimizer_for(model, tiny_config)
    model.train()
    assert model.backbone.training == (approach == "lora")
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(**batch), torch.tensor([0, 1]))
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), n
            else:
                assert p.grad is None
        optimizer.step()
    assert all(
        torch.equal(p, before[n]) for n, p in model.named_parameters() if not p.requires_grad
    )
    assert verify_trainable(model)["trainable"] > 0
    model.eval()
    with torch.inference_mode():
        expected = model(**batch)
        natural = model(**collate(tokens, tokenizer, "cpu"))
    torch.testing.assert_close(expected, natural, rtol=1e-5, atol=1e-6)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, tokenizer, tiny_config, {"seed": 42}, optimizer)
    restored, restored_tokenizer, _ = load_checkpoint(checkpoint)
    with torch.inference_mode():
        torch.testing.assert_close(expected, restored(**batch), rtol=1e-5, atol=1e-6)
    scores, logits = predict_rows(restored, restored_tokenizer, rows, 64, "cpu")
    assert len(scores) == 2 and logits.shape == (2, 2)
    assert not any("lm_head" in n for n, _ in restored.named_parameters())


def test_head_initialization_matches_across_approaches(tiny_config):
    frozen, _ = load_classifier(tiny_config, "frozen", "cpu", 42)
    lora, _ = load_classifier(tiny_config, "lora", "cpu", 42)
    torch.testing.assert_close(frozen.head.weight, lora.head.weight)


@pytest.mark.parametrize("approach", ["frozen", "lora"])
def test_local_mlflow_roundtrip_and_tiny_training(tiny_config, tmp_path, approach):
    mlflow = pytest.importorskip("mlflow")
    from jais_guard.finetuning.tracking import configure, log_setup
    from jais_guard.finetuning.training import train_one

    tiny_config["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tiny_config["experiment"] = "cpu-tiny-test"
    tiny_config["frozen"].update(epochs=2, patience=2, batch_size=2)
    tiny_config["lora"].update(epochs=2, patience=2, microbatch=1, accumulation=3)
    rows = [
        {
            "id": str(i),
            "input_text": text,
            "label": label,
            "label_origin": "human",
            "family_id": str(i),
        }
        for i, (text, label) in enumerate(
            [
                ("مرحبا", "VALID"),
                ("هسهسصث", "GIBBERISH"),
                ("طلب موعد", "VALID"),
                ("غغغغغ", "GIBBERISH"),
            ]
        )
    ]
    directory = tmp_path / "run"
    directory.mkdir()
    configure(tiny_config)
    # Keep the SQL backend and artifact store entirely inside pytest's temp directory.
    mlflow.set_experiment("cpu-tiny-test")
    client = mlflow.MlflowClient()
    exp = mlflow.get_experiment_by_name("cpu-tiny-test")
    # Create a second experiment with an explicit temporary artifact root.
    exp_id = mlflow.create_experiment(
        f"cpu-tiny-artifacts-{approach}", artifact_location=str(tmp_path / "artifacts")
    )
    seed_everything(42)
    with mlflow.start_run(experiment_id=exp_id) as run:
        log_setup(
            mlflow,
            tiny_config,
            directory,
            rows,
            rows,
            ({"version": "tiny-train"}, {"version": "tiny-val"}),
        )
        model, tokenizer = load_classifier(tiny_config, approach, "cpu")
        report = train_one(model, tokenizer, rows, rows, tiny_config, 42, directory, mlflow)
        assert "default" in report and "exploratory" in report
        loaded_uri = json.loads((directory / "logged_model.json").read_text())["model_uri"]
        expected, _ = predict_rows(model, tokenizer, rows, 64, "cpu")
        loaded = mlflow.pyfunc.load_model(loaded_uri)
        import pandas as pd

        predictions = loaded.predict(pd.DataFrame({"text": [r["input_text"] for r in rows]}))
        np.testing.assert_allclose(predictions["p_gibberish"], expected, rtol=1e-6)
        with pytest.raises(ValueError, match="Unsupported"):
            loaded.predict(pd.DataFrame({"text": ["hello"]}))
        assert len(client.get_metric_history(run.info.run_id, "train/loss")) == 4
        assert client.get_run(run.info.run_id).inputs.dataset_inputs
        assert client.list_artifacts(run.info.run_id, "evaluation")
    assert exp is not None
    mlflow.set_tracking_uri("file:///tmp/jaisguard-unused-test-tracking")


def test_tracking_refuses_remote(config):
    from jais_guard.finetuning.tracking import configure

    config["tracking_uri"] = "https://remote.example.com"
    with pytest.raises(ValueError, match="local"):
        configure(config)
