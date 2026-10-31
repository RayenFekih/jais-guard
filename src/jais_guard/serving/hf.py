"""Locked local inference shared by the API and batch-one latency benchmarks."""

import hashlib
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

DEFAULT_DIRECTORY = Path("results/latency_comparison/jaisguard-all-v2")


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_manifest(directory):
    from jais_guard.data.test_workflow import sha
    from jais_guard.finetuning.version_comparison import verify_files

    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    identity = manifest.get("identity")
    payload = {k: v for k, v in manifest.items() if k != "identity"}
    if identity != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest():
        raise ValueError("Serving manifest changed")
    verify_files(manifest["files"])
    if sha(directory / "quality.json") != manifest["quality_sha256"]:
        raise ValueError("Quality snapshot changed")
    return manifest


def frozen_inputs(manifest):
    from jais_guard.data.test_workflow import read_frozen_test, sha

    rows = read_frozen_test(manifest["test"])
    if len(rows) != 400 or sha(Path(manifest["test"]) / "test.jsonl") != manifest["test_sha256"]:
        raise ValueError("Frozen benchmark changed")
    return rows


def cuda_required():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run this command in WSL on the RTX 4050")


class Predictor:
    """One resident model, serialized calls, unchanged original inference policies."""

    def __init__(self, manifest, model_key):
        import torch

        from jais_guard.baselines.inference import BaselineClassifier, BaselineConfig
        from jais_guard.finetuning.model import load_checkpoint, seed_everything

        cuda_required()
        self.item = manifest["models"][model_key]
        self.model_key = model_key
        self.lock = threading.Lock()
        seed_everything(self.item["seed"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if self.item["approach"] == "untuned":
            settings = self.item["settings"]
            cfg = settings["baseline"]
            self.classifier = BaselineClassifier(
                BaselineConfig(
                    settings["model"],
                    settings["revision"],
                    cfg["reasoning_effort"],
                    cfg["max_tokens"],
                    cfg["max_model_len"],
                    local_files_only=True,
                )
            )
            self.model = self.classifier.model
            if self.model.generation_config.to_dict() != settings["generation_config"]:
                raise ValueError("Original generation configuration changed")
        else:
            self.model, self.tokenizer, saved = load_checkpoint(self.item["checkpoint"], "cuda")
            if saved != self.item["metadata"]:
                raise ValueError("Checkpoint metadata changed")
            self.model.requires_grad_(False)
            self.model.eval()
            self.model.backbone.gradient_checkpointing_disable()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.load_memory = self.memory()

    @staticmethod
    def memory():
        import torch

        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def info(self):
        import torch

        return {
            "model_key": self.model_key,
            "seed": self.item["seed"],
            "threshold": self.item.get("threshold"),
            "device": torch.cuda.get_device_name(0),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "dtype": str(next(self.model.parameters()).dtype),
            "model_load_seconds": self.load_seconds,
            "loading_memory": self.load_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    def predict(self, text):
        import torch

        from jais_guard.finetuning.model import predict_rows
        from jais_guard.finetuning.test_baseline import classify_timed

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected nonempty Arabic text")
        with self.lock:
            torch.cuda.synchronize()
            start = time.perf_counter()
            if self.item["approach"] == "untuned":
                result = classify_timed(
                    self.classifier, text, self.item["settings"]["baseline"]["variant"]
                )
                label = result["prediction_id"]
                probabilities = None  # Free generation supplies no binary posterior.
                raw = result["raw_output"]
                model_ms = result["generation_ms"]
            else:
                forward_times = []
                original = self.model.forward

                def timed_forward(*args, **kwargs):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    output = original(*args, **kwargs)
                    torch.cuda.synchronize()
                    forward_times.append((time.perf_counter() - begin) * 1000)
                    return output

                self.model.forward = timed_forward
                try:
                    scores, _ = predict_rows(
                        self.model,
                        self.tokenizer,
                        [{"id": "request", "input_text": text}],
                        self.item["metadata"]["config"]["max_length"],
                        "cuda",
                        batch_size=1,
                    )
                finally:
                    self.model.forward = original
                if len(forward_times) != 1:
                    raise ValueError("Expected exactly one classifier forward")
                model_ms = forward_times[0]
                score = float(scores[0])
                label = int(score >= self.item["threshold"])
                probabilities = {"VALID": 1 - score, "GIBBERISH": score}
                raw = None
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "model_key": self.model_key,
                "class_id": label,
                "class": {0: "VALID", 1: "GIBBERISH", -1: "INVALID"}[label],
                "probabilities": probabilities,
                "threshold": self.item.get("threshold"),
                "raw_output": raw,
                "inference_ms": elapsed,
                "model_ms": model_ms,
            }


def create_app(
    directory=None,
    model_key=None,
    predictor_factory=Predictor,
    checkpoint=None,
    operating_point="validation_locked",
):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    directory = Path(directory or os.environ.get("JAISGUARD_DIRECTORY", DEFAULT_DIRECTORY))

    class Request(BaseModel):
        text: str = Field(min_length=1, max_length=16000)

    @asynccontextmanager
    async def lifespan(app):
        selected_checkpoint = checkpoint or os.environ.get("JAISGUARD_CHECKPOINT")
        if selected_checkpoint:
            from jais_guard.workflows.public import checkpoint_manifest

            manifest = checkpoint_manifest(
                selected_checkpoint, os.environ.get("JAISGUARD_OPERATING_POINT", operating_point)
            )
        else:
            try:
                manifest = checked_manifest(directory)
            except FileNotFoundError as exc:
                raise ValueError(
                    "Serving artifacts unavailable: set JAISGUARD_CHECKPOINT to a trained best/ "
                    "directory, or JAISGUARD_DIRECTORY to a verified historical manifest"
                ) from exc
        key = (
            manifest["default_model"]
            if selected_checkpoint
            else model_key or os.environ.get("JAISGUARD_MODEL", manifest["default_model"])
        )
        app.state.predictor = predictor_factory(manifest, key)
        app.state.identity = manifest["identity"]
        yield
        del app.state.predictor

    app = FastAPI(title="JaisGuard local demo", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "manifest_identity": app.state.identity,
            **app.state.predictor.info(),
        }

    @app.post("/predict")
    def predict(request: Request):
        try:
            return app.state.predictor.predict(request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
