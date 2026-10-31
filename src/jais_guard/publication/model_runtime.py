"""Standalone JaisGuard checkpoint loader. No dependency on the training repository.

Derived from the project's MIT-licensed classifier implementation. AutoModel is
used only for the pinned base backbone; this is not an AutoModel sequence classifier.
"""

import contextlib
import hashlib
import json
import os
import random
import unicodedata
from pathlib import Path

import numpy as np
import torch
from torch import nn


def encode_rows(rows, tokenizer, max_length):
    encoded = []
    for row in rows:
        text = row["input_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected nonempty text")
        if any(c.isalpha() and "ARABIC" not in unicodedata.name(c, "") for c in text):
            raise ValueError("Unsupported script; no gibberish label assigned")
        item = tokenizer(text, add_special_tokens=True, truncation=False)
        if not 0 < len(item["input_ids"]) <= max_length:
            raise ValueError(f"Token budget exceeded or empty: {row['id']}")
        encoded.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]})
    return encoded


def collate(encoded, tokenizer, device, pad_to=None):
    return tokenizer.pad(
        encoded, padding="max_length" if pad_to else True, max_length=pad_to, return_tensors="pt"
    ).to(device)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def precision(device):
    if str(device).startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def autocast(device):
    return (
        torch.autocast("cuda", dtype=precision(device))
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def last_nonpadding(hidden, attention_mask):
    if hidden.shape[:2] != attention_mask.shape or not attention_mask.any(dim=1).all():
        raise ValueError("Invalid or empty attention mask")
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    indices = (
        positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(1).values
    )
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class HiddenClassifier(nn.Module):
    def __init__(self, backbone, approach, seed=42):
        super().__init__()
        if approach not in ("frozen", "lora"):
            raise ValueError("Unknown approach")
        self.backbone = backbone
        self.approach = approach
        # LoRA initialization consumes RNG; isolate head initialization to pair seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(backbone.config.hidden_size, 2, dtype=torch.float32)
        if approach == "frozen":
            self.backbone.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.approach == "frozen":
            self.backbone.eval()
        return self

    def features(self, **inputs):
        context = torch.no_grad() if self.approach == "frozen" else contextlib.nullcontext()
        with context:
            output = self.backbone(**inputs, use_cache=False)
        return last_nonpadding(output.last_hidden_state, inputs["attention_mask"]).float()

    def forward(self, **inputs):
        features = self.features(**inputs)
        # Keep the classifier math FP32 even while the backbone is autocast.
        with torch.autocast(features.device.type, enabled=False):
            return self.head(features.float())


def load_classifier(config, approach, device, seed=42):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Pinned tokenizer must define a padding token")
    backbone, loading = AutoModel.from_pretrained(
        config["model"],
        revision=config["revision"],
        code_revision=config["revision"],
        trust_remote_code=True,
        local_files_only=config.get("_local_files_only", True),
        dtype=precision(device),
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Backbone loading mismatch: {loading}")
    if any(not key.startswith("lm_head.") for key in loading.get("unexpected_keys", [])):
        raise ValueError(f"Unexpected backbone weights: {loading}")
    if any("lm_head" in n for n, _ in backbone.named_parameters()):
        raise ValueError("Classifier unexpectedly includes a language-model head")
    backbone.config.use_cache = False
    backbone.to(device)
    backbone.requires_grad_(False)
    if approach == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = config["lora"]
        targets = [
            n
            for n, module in backbone.named_modules()
            if isinstance(module, nn.Linear) and n.rsplit(".", 1)[-1] in cfg["target_modules"]
        ]
        expected = backbone.config.num_hidden_layers * len(cfg["target_modules"])
        if len(targets) != expected:
            raise ValueError(f"Expected {expected} LoRA targets, found {len(targets)}")
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=cfg["rank"],
                lora_alpha=cfg["alpha"],
                lora_dropout=cfg["dropout"],
                target_modules=cfg["target_modules"],
                bias="none",
            ),
        )
    model = HiddenClassifier(backbone, approach, seed).to(device)
    verify_trainable(model)
    return model, tokenizer


def verify_trainable(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any(
        not (n.startswith("head.") or (model.approach == "lora" and "lora_" in n)) for n in names
    ):
        raise ValueError(f"Unexpected trainable parameters: {names}")
    if model.approach == "lora" and not any("lora_" in n for n in names):
        raise ValueError("No trainable LoRA parameters")
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def restore_weights(model, directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    model.head.load_state_dict(load_file(str(directory / "head.safetensors")))
    if model.approach == "lora":
        from peft import set_peft_model_state_dict

        state = load_file(str(directory / "adapter/adapter_model.safetensors"))
        set_peft_model_state_dict(model.backbone, state)
    model.eval()


def verify_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Manifest path escapes model package")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Package integrity mismatch: {name}")
    return manifest


class JaisGuardClassifier:
    """Explicit checkpoint API; batch-one inference and validation-locked decisions."""

    @classmethod
    def from_pretrained(
        cls, model, *, revision=None, device="cuda", precision="historical", local_files_only=False
    ):
        directory = Path(model)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(
                snapshot_download(
                    repo_id=str(model), revision=revision, local_files_only=local_files_only
                )
            )
        verify_package(directory)
        cfg = json.loads((directory / "config.json").read_text())
        if cfg["label2id"] != {"VALID": 0, "GIBBERISH": 1}:
            raise ValueError("Unsupported label mapping")
        if cfg["pooling"] != "last_nonpadding_final_hidden":
            raise ValueError("Unsupported pooling")
        if precision not in ("historical", "bf16", "fp16"):
            raise ValueError("Choose historical, bf16 or fp16 precision")
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU debugging is explicit via device='cpu'")
        if precision in ("bf16", "fp16") and device.type != "cuda":
            raise ValueError("Explicit reduced-precision validation requires CUDA")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported by this GPU")
        seed_everything(cfg["seed"])
        config = dict(cfg["classifier_config"], _local_files_only=local_files_only)
        model, _ = load_classifier(config, cfg["approach"], device, cfg["seed"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        tokenizer.padding_side = "right"
        restore_weights(model, directory)
        model.eval().requires_grad_(False)
        model.backbone.gradient_checkpointing_disable()
        if precision == "fp16":
            # Match historical HF FP16: load original policy, then convert backbone.
            model.backbone.to(torch.float16)
        result = cls()
        result.model, result.tokenizer, result.config = model, tokenizer, cfg
        result.device = device
        result.dtype = (
            torch.float16
            if precision == "fp16"
            else torch.bfloat16
            if precision == "bf16"
            else next(model.backbone.parameters()).dtype
        )
        return result

    def predict(self, text, *, operating_point="validation_locked"):
        if operating_point not in ("validation_locked", "default"):
            raise ValueError("Unknown operating point")
        row = encode_rows([{"input_text": text}], self.tokenizer, self.config["max_length"])
        batch = collate(row, self.tokenizer, self.device)
        context = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), context:
            logits = self.model(**batch).float()
        values = logits.softmax(-1)[0].cpu().tolist()
        threshold = (
            self.config["decision_threshold"] if operating_point == "validation_locked" else 0.5
        )
        label = int(values[1] >= threshold)
        return {
            "class_id": label,
            "class": ["VALID", "GIBBERISH"][label],
            "probabilities": {"VALID": values[0], "GIBBERISH": values[1]},
            "threshold": threshold,
            "operating_point": operating_point,
            "precision": str(self.dtype),
        }
