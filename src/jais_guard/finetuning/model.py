"""Pinned K2 Horizon backbone, explicit pooling, and lightweight checkpoints."""

import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import collate, encode_rows
from .evaluation import write_json


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
        local_files_only=True,
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


def optimizer_for(model, config):
    cfg = config[model.approach]
    groups = []
    for head in (False, True):
        for decay in (False, True):
            parameters = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and n.startswith("head.") == head
                and (p.ndim >= 2 and not n.endswith("bias")) == decay
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": cfg["head_lr"] if head else cfg["adapter_lr"],
                        "weight_decay": config["weight_decay"] if decay else 0.0,
                    }
                )
    return torch.optim.AdamW(groups)


def save_checkpoint(
    directory, model, tokenizer, config, metadata, optimizer=None, scheduler=None, scaler=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.head.state_dict().items()},
        str(directory / "head.safetensors"),
    )
    if model.approach == "lora":
        model.backbone.save_pretrained(directory / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    write_json(
        directory / "classifier.json",
        {
            "config": config,
            "approach": model.approach,
            "seed": metadata.get("seed", 42),
            "label_mapping": {"VALID": 0, "GIBBERISH": 1},
            "pooling": "last_nonpadding_final_hidden",
            "metadata": metadata,
        },
    )
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "metadata": metadata,
            },
            directory / "training_state.pt",
        )


def restore_weights(model, directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    model.head.load_state_dict(load_file(str(directory / "head.safetensors")))
    if model.approach == "lora":
        from peft import set_peft_model_state_dict

        state = load_file(str(directory / "adapter/adapter_model.safetensors"))
        set_peft_model_state_dict(model.backbone, state)
    model.eval()


def load_checkpoint(directory, device="cpu"):
    from transformers import AutoTokenizer

    directory = Path(directory)
    saved = json.loads((directory / "classifier.json").read_text())
    model, _ = load_classifier(saved["config"], saved["approach"], device, saved["seed"])
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    tokenizer.padding_side = "right"
    restore_weights(model, directory)
    return model, tokenizer, saved


def predict_rows(model, tokenizer, rows, max_length, device, batch_size=2):
    encoded = encode_rows(rows, tokenizer, max_length)
    scores, logits = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = collate(encoded[start : start + batch_size], tokenizer, device)
            with autocast(device):
                output = model(**batch).float()
            logits.append(output.cpu())
            scores.extend(output.softmax(-1)[:, 1].cpu().tolist())
    return scores, torch.cat(logits)
