"""Separate real vLLM server, or the unchanged HF app with benchmark diagnostics."""

import time
from contextlib import asynccontextmanager

from jais_guard.serving import create_app as create_hf_app

from .artifacts import checked_export
from .parity import acceptance_reference, require_parity
from .runtime import AsyncClassifier, decision


def create_app(root, key, backend):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from transformers import AutoTokenizer

    from jais_guard.finetuning.data import encode_rows

    plan, artifact, receipt = checked_export(root, key)
    require_parity(root, key)
    if backend == "hf":
        # Reuse the actual existing application, its Predictor and serialization lock.
        if acceptance_reference(plan) == "hf-fp16":
            from .fp16 import FP16Predictor

            app = create_hf_app(plan["source_manifest"], key, predictor_factory=FP16Predictor)
        else:
            app = create_hf_app(plan["source_manifest"], key)
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            require_parity(root, key)
            async with original(app):
                yield

        app.router.lifespan_context = lifespan
    elif backend == "vllm":

        @asynccontextmanager
        async def lifespan(app):
            require_parity(root, key)
            app.state.tokenizer = AutoTokenizer.from_pretrained(
                artifact, local_files_only=True, trust_remote_code=True
            )
            app.state.tokenizer.padding_side = "right"
            app.state.classifier = AsyncClassifier(root, key)
            try:
                await app.state.classifier.ready()
                yield
            finally:
                app.state.classifier.close()

        app = FastAPI(title="JaisGuard — real vLLM pooling", lifespan=lifespan)

        class ClassificationRequest(BaseModel):
            input: str = Field(min_length=1, max_length=16000)

        @app.post("/classify")
        async def classify(request: ClassificationRequest):
            start = time.perf_counter()
            try:
                encoded = encode_rows(
                    [{"id": "request", "input_text": request.input}],
                    app.state.tokenizer,
                    plan["models"][key]["metadata"]["config"]["max_length"],
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            probs = await app.state.classifier.predict_tokens(encoded[0]["input_ids"])
            return {
                "model_key": key,
                **decision(probs, receipt["threshold"]),
                "inference_ms": (time.perf_counter() - start) * 1000,
                "backend": "vllm",
                "label_order": ["VALID", "GIBBERISH"],
            }

        @app.get("/health")
        async def health():
            await app.state.classifier.engine.check_health()
            return {"status": "ready", "model_key": key, "plan_identity": plan["identity"]}
    else:
        raise ValueError("Backend must be hf or vllm")

    @app.get("/benchmark/info")
    async def info():
        if backend == "hf":
            metadata = app.state.predictor.info()
        else:
            classifier = app.state.classifier
            metadata = {
                "model_load_seconds": classifier.load_seconds,
                "loading_memory": classifier.loading,
                "engine": plan["engine"],
            }
        return {
            **metadata,
            "backend": backend,
            "model_key": key,
            "plan_identity": plan["identity"],
            "export_identity": receipt["identity"],
            "threshold": receipt["threshold"],
            "label_order": ["VALID", "GIBBERISH"],
        }

    @app.post("/benchmark/reset")
    async def reset():
        if backend == "hf":
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            await app.state.classifier.reset_peak()
        return {"status": "reset"}

    @app.get("/benchmark/memory")
    async def memory():
        if backend == "hf":
            return app.state.predictor.memory()
        return await app.state.classifier.memory()

    return app
