"""Precision-only adaptation of the existing locked HF/FastAPI predictor."""

import time

from jais_guard.serving import Predictor


class FP16Predictor(Predictor):
    def __init__(self, manifest, model_key):
        import torch

        super().__init__(manifest, model_key)
        if self.item["approach"] not in ("frozen", "lora"):
            raise ValueError("FP16 benchmark adaptation supports classifiers only")
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.backbone.to(torch.float16)
        original_forward = self.model.forward

        def forward_fp16(*args, **kwargs):
            # Overrides the original predict_rows BF16 autocast, while the original
            # HiddenClassifier.forward still explicitly runs the saved head in FP32.
            with torch.autocast("cuda", dtype=torch.float16):
                return original_forward(*args, **kwargs)

        self.model.forward = forward_fp16
        torch.cuda.synchronize()
        self.load_seconds += time.perf_counter() - start
        self.load_memory = self.memory()

    def info(self):
        return {
            **super().info(),
            "serving_precision": "float16",
            "head_precision": "float32",
            "adaptation": "Original Predictor; FP16 backbone/autocast",
        }
