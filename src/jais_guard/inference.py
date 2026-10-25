"""Untuned Jais classification using Transformers and PyTorch."""

import unicodedata
from dataclasses import dataclass

from .prompts import build_prompt, parse_label


@dataclass(frozen=True)
class BaselineConfig:
    model: str
    revision: str
    reasoning_effort: str = "low"
    max_tokens: int = 256
    max_model_len: int = 1024

    def __post_init__(self):
        if self.reasoning_effort not in {"high", "medium", "low"}:
            raise ValueError("reasoning_effort must be high, medium, or low")
        if not 0 < self.max_tokens < self.max_model_len:
            raise ValueError("Require 0 < max_tokens < max_model_len")


class BaselineClassifier:
    def __init__(self, config: BaselineConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.revision, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config.model,
                revision=config.revision,
                trust_remote_code=True,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            .to(self.device)
            .eval()
        )

    def classify(self, text: str, variant: str = "system_only") -> dict:
        import torch

        if any(char.isalpha() and "ARABIC" not in unicodedata.name(char, "") for char in text):
            raise ValueError("Unsupported script for v1; no gibberish label assigned")
        cfg = self.config
        prompt = build_prompt(
            self.tokenizer, text, cfg.reasoning_effort, constrained=False, variant=variant
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens + cfg.max_tokens > cfg.max_model_len:
            raise ValueError("Input exceeds context budget; refusing silent truncation")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_tokens:]
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        stopped = bool(len(generated) and generated[-1].item() in (eos_ids or []))
        # Remove only the terminal EOS; preserve reasoning tags for strict parsing.
        answer_tokens = generated[:-1] if stopped else generated
        raw = self.tokenizer.decode(answer_tokens, skip_special_tokens=False)
        label = parse_label(raw, cfg.reasoning_effort, constrained=False) if stopped else None
        return {
            "prompt": prompt,
            "prediction": label,
            "raw_output": raw,
            "finish_reason": "stop" if stopped else "length",
            "generated_tokens": len(generated),
            "input_tokens": input_tokens,
        }
