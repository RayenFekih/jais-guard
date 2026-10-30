"""Actual vLLM engines and worker diagnostics; no HF inference fallback."""

import time

import numpy as np

from jais_guard.serving import cuda_required

from .artifacts import checked_export, tensor_hash


class WorkerDiagnostics:
    """Supported worker_extension_cls: measure the worker, not the API process."""

    def jaisguard_reset_peak(self):
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def jaisguard_status(self):
        import torch

        model = self.model_runner.get_model()
        classifier = model.classifier
        torch.cuda.synchronize()
        return {
            "model_class": type(model).__name__,
            "head_dtype": str(classifier.weight.dtype),
            "head_hashes": {n: tensor_hash(getattr(classifier, n)) for n in ("weight", "bias")},
            "vllm_module_count": sum(
                type(m).__module__.startswith("vllm.") for m in model.modules()
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        }


def audit_worker(statuses, receipt):
    if len(statuses) != 1:
        raise ValueError("Exactly one GPU worker is required on the 6 GB card")
    status = statuses[0]
    if (
        status["model_class"] != "TransformersForSequenceClassification"
        or status["head_dtype"] != "torch.float32"
        or status["head_hashes"] != receipt["head_hashes"]
        or status["vllm_module_count"] < 1
    ):
        raise ValueError("vLLM worker did not load the exact trained FP32 weight and bias")
    return status


def probabilities(values):
    values = np.asarray(values, dtype=float)
    if (
        values.shape != (2,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.isclose(values.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Expected VALID/GIBBERISH softmax probabilities in that order")
    return values.tolist()


def decision(values, threshold):
    values = probabilities(values)
    label = int(values[1] >= threshold)
    return {
        "class_id": label,
        "class": ["VALID", "GIBBERISH"][label],
        "default_class_id": int(values[1] >= 0.5),
        "probabilities": {"VALID": values[0], "GIBBERISH": values[1]},
        "threshold": threshold,
    }


def engine_kwargs(plan, artifact):
    from vllm.config import PoolerConfig

    return {
        **plan["engine"],
        "model": str(artifact),
        "tokenizer": str(artifact),
        "pooler_config": PoolerConfig(
            task="classify", seq_pooling_type="LAST", use_activation=True
        ),
        "hf_overrides": {"head_dtype": "float32"},
    }


class OfflineClassifier:
    def __init__(self, root, key):
        from vllm import LLM

        cuda_required()
        self.plan, artifact, self.receipt = checked_export(root, key)
        start = time.perf_counter()
        self.engine = LLM(**engine_kwargs(self.plan, artifact))
        self.loading = audit_worker(self.engine.collective_rpc("jaisguard_status"), self.receipt)
        self.load_seconds = time.perf_counter() - start

    def predict_tokens(self, ids):
        output = self.engine.classify([{"prompt_token_ids": ids}], use_tqdm=False)
        if len(output) != 1 or output[0].prompt_token_ids != ids:
            raise ValueError("vLLM changed the input token sequence")
        return probabilities(output[0].outputs.probs)

    def reset_peak(self):
        self.engine.collective_rpc("jaisguard_reset_peak")

    def memory(self):
        return audit_worker(self.engine.collective_rpc("jaisguard_status"), self.receipt)

    def close(self):
        # Installed 0.32.0 EngineCoreClient shutdown waits for its worker process.
        self.engine.llm_engine.engine_core.shutdown(timeout=20)


class AsyncClassifier:
    def __init__(self, root, key):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        cuda_required()
        self.plan, artifact, self.receipt = checked_export(root, key)
        self.start = time.perf_counter()
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**engine_kwargs(self.plan, artifact))
        )

    async def ready(self):
        await self.engine.check_health()
        self.loading = await self.memory()
        self.load_seconds = time.perf_counter() - self.start

    async def predict_tokens(self, ids):
        from uuid import uuid4

        from vllm import PoolingParams
        from vllm.outputs import ClassificationOutput

        final = None
        async for output in self.engine.encode(
            {"prompt_token_ids": ids},
            PoolingParams(task="classify", use_activation=True),
            request_id=uuid4().hex,
        ):
            final = output
        if final is None or final.prompt_token_ids != ids:
            raise ValueError("vLLM changed the input token sequence")
        return probabilities(ClassificationOutput.from_base(final.outputs).probs)

    async def memory(self):
        return audit_worker(await self.engine.collective_rpc("jaisguard_status"), self.receipt)

    async def reset_peak(self):
        await self.engine.collective_rpc("jaisguard_reset_peak")

    def close(self):
        self.engine.shutdown(timeout=20)
