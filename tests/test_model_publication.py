"""Portable regressions for the standalone checkpoint release runtime."""

import ast
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jais_guard.finetuning.model import HiddenClassifier as OriginalClassifier
from jais_guard.publication import model_runtime as runtime
from jais_guard.publication.models import seal


def test_runtime_has_no_training_repository_imports():
    tree = ast.parse(inspect.getsource(runtime))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert not (node.module or "").startswith("jais_guard")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("jais_guard") for alias in node.names)


def test_standalone_pooling_and_trained_bias_match_original():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(hidden_size=4)

        def forward(self, input_ids, attention_mask, use_cache):
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
            return SimpleNamespace(last_hidden_state=hidden)

    original = OriginalClassifier(Backbone(), "frozen", 43)
    standalone = runtime.HiddenClassifier(Backbone(), "frozen", 43)
    with torch.no_grad():
        original.head.bias.copy_(torch.tensor([2.0, -3.0]))
        standalone.head.load_state_dict(original.head.state_dict())
    ids = torch.tensor([[1, 2, 0], [0, 3, 4]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert torch.equal(
        original(input_ids=ids, attention_mask=mask), standalone(input_ids=ids, attention_mask=mask)
    )
    with pytest.raises(ValueError, match="mask"):
        runtime.last_nonpadding(torch.zeros(1, 2, 4), torch.zeros(1, 2))


@pytest.fixture
def package(tmp_path):
    (tmp_path / "head.safetensors").write_bytes(b"synthetic")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "label2id": {"VALID": 0, "GIBBERISH": 1},
                "pooling": "last_nonpadding_final_hidden",
            }
        )
    )
    seal(tmp_path)
    return tmp_path


def test_integrity_checks_all_bytes(package):
    assert len(runtime.verify_package(package)["files"]) == 2
    (package / "head.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        runtime.verify_package(package)


def test_manifest_cannot_escape_package(package):
    (package / "manifest.json").write_text(json.dumps({"files": {"../outside": "x"}}))
    with pytest.raises(ValueError, match="escapes"):
        runtime.verify_package(package)


def test_incompatible_labels_fail_before_model_loading(package):
    (package / "config.json").write_text(json.dumps({"label2id": {"GIBBERISH": 0, "VALID": 1}}))
    seal(package)
    with pytest.raises(ValueError, match="label mapping"):
        runtime.JaisGuardClassifier.from_pretrained(package, device="cpu")


def test_explicit_reduced_precision_cannot_silently_use_cpu(package):
    with pytest.raises(ValueError, match="CUDA"):
        runtime.JaisGuardClassifier.from_pretrained(package, device="cpu", precision="fp16")


def test_decisions_use_saved_threshold_and_ge_boundary():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "attention_mask": [1]}

        def pad(self, encoded, **kwargs):
            class Batch(dict):
                def to(self, device):
                    return self

            return Batch(input_ids=torch.tensor([[1]]), attention_mask=torch.tensor([[1]]))

    class Model:
        def __call__(self, **kwargs):
            return torch.tensor([[0.0, 0.0]])

    classifier = runtime.JaisGuardClassifier()
    classifier.model = Model()
    classifier.tokenizer = Tokenizer()
    classifier.device = torch.device("cpu")
    classifier.dtype = torch.float32
    classifier.config = {"max_length": 64, "decision_threshold": 0.75}
    assert classifier.predict("كيف حالك")["class_id"] == 0
    assert classifier.predict("كيف حالك", operating_point="default")["class_id"] == 1
    with pytest.raises(ValueError, match="script"):
        classifier.predict("hello")
    with pytest.raises(ValueError, match="nonempty"):
        classifier.predict(" ")
