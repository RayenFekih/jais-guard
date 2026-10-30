"""CPU verification of trained-weight export, vLLM contracts and fail-closed parity."""

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jais_guard.vllm_support import artifacts, benchmark, parity, runtime, server


def head_module(monkeypatch):
    fake = types.ModuleType("jais_guard.vllm_support.modeling_k2_horizon")

    class Base(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

    class Backbone(nn.Module):
        def __init__(self, config):
            super().__init__()

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=input_ids)

    fake.K2HorizonPreTrainedModel = Base
    fake.K2HorizonModel = Backbone
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    name = "jais_guard.vllm_support.modeling_jaisguard"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_head_bias_fp32_last_token_and_vllm_shape(monkeypatch):
    module = head_module(monkeypatch)
    model = module.JaisGuardForSequenceClassification(SimpleNamespace(hidden_size=2))
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.classifier.bias.copy_(torch.tensor([0.25, -0.5]))
    hidden = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], [[99.0, 99.0], [5.0, 6.0], [7.0, 8.0]]],
        dtype=torch.float16,
    )
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    logits = model(input_ids=hidden, attention_mask=mask).logits
    assert logits.dtype == torch.float32
    assert torch.equal(logits, torch.tensor([[3.25, 3.5], [7.25, 7.5]]))
    # The installed mixin prepends singleton sequence dimension before this call.
    assert torch.equal(model.classifier(hidden[:, 1:2]), torch.tensor([[3.25, 3.5], [5.25, 5.5]]))
    with pytest.raises(ValueError, match="exactly one"):
        model.classifier(hidden)


def test_threshold_probability_order_and_worker_audit():
    result = runtime.decision([0.1, 0.9], 0.95)
    assert result["class"] == "VALID"
    assert result["default_class_id"] == 1
    assert runtime.decision([0.1, 0.9], 0.9)["class"] == "GIBBERISH"
    for bad in ([1], [0, 0], [1.1, -0.1], [float("nan"), 1]):
        with pytest.raises(ValueError):
            runtime.probabilities(bad)
    receipt = {"head_hashes": {"weight": "w", "bias": "b"}}
    status = {
        "model_class": "TransformersForSequenceClassification",
        "head_dtype": "torch.float32",
        "head_hashes": receipt["head_hashes"],
        "vllm_module_count": 10,
    }
    assert runtime.audit_worker([status], receipt) == status
    for field, value in [
        ("head_dtype", "torch.float16"),
        ("head_hashes", {"weight": "w"}),
        ("vllm_module_count", 0),
        ("model_class", "HiddenClassifier"),
    ]:
        with pytest.raises(ValueError, match="exact trained"):
            runtime.audit_worker([{**status, field: value}], receipt)


def row(probs, name="same"):
    return {"id": name, "category": "clean", "token_ids": [1, 2], "probabilities": probs}


def test_parity_zero_mismatches_even_within_numerical_tolerance():
    policy = artifacts.POLICY
    result = parity.compare_probabilities([row([0.1, 0.9])], [row([0.099, 0.901])], 0.95, policy)
    assert result["passed"]
    assert result["max_abs_probability_difference"] == pytest.approx(0.001)
    mismatch = parity.compare_probabilities(
        [row([0.051, 0.949])], [row([0.049, 0.951])], 0.95, policy
    )
    assert not mismatch["passed"]
    assert mismatch["locked_prediction_mismatches"] == 1
    default = parity.compare_probabilities(
        [row([0.501, 0.499])], [row([0.499, 0.501])], 0.95, policy
    )
    assert not default["passed"]
    with pytest.raises(ValueError, match="IDs"):
        parity.compare_probabilities(
            [row([0.1, 0.9])], [row([0.1, 0.9], "different")], 0.95, policy
        )
    assert not parity.compare_probabilities([row([0.1, 0.9])], [row([0.2, 0.8])], 0.95, policy)[
        "passed"
    ]


@pytest.mark.parametrize("approach", ["frozen", "lora"])
def test_export_exact_weights_bias_and_lora_merge(monkeypatch, tmp_path, approach):
    from safetensors.torch import load_file, save_file
    from transformers import PretrainedConfig

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig(hidden_size=2, num_hidden_layers=1)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = nn.Module()
            self.layers[0].self_attn.q_proj = nn.Linear(2, 2, bias=False)
            self.layers[0].self_attn.v_proj = nn.Linear(2, 2, bias=False)

    backbone = Tiny()
    base = {"model." + n: t.clone() for n, t in backbone.state_dict().items()}
    expected = {k: v.clone() for k, v in base.items()}
    if approach == "lora":
        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone, LoraConfig(r=1, lora_alpha=2, target_modules=["q_proj", "v_proj"])
        )
        with torch.no_grad():
            for name, p in backbone.named_parameters():
                if "lora_" in name:
                    p.fill_(0.1)
        # Independently use PEFT's supported merge and compare every projection.
        # Save the unmerged adapter before that operation.
    checkpoint = tmp_path / "source/best"
    checkpoint.mkdir(parents=True)
    head = nn.Linear(2, 2)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.12345, -0.98765]))
    save_file(head.state_dict(), str(checkpoint / "head.safetensors"))
    if approach == "lora":
        backbone.save_pretrained(checkpoint / "adapter")
        peft_merged = backbone.merge_and_unload(safe_merge=True)
        expected = {"model." + k: v for k, v in peft_merged.state_dict().items()}
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "tokenizer/tokenizer.json").write_text("{}")
    metadata = {
        "label_mapping": {"VALID": 0, "GIBBERISH": 1},
        "pooling": "last_nonpadding_final_hidden",
        "config": {
            "model": "unused",
            "revision": "pinned",
            "lora": {"rank": 1, "alpha": 2, "target_modules": ["q_proj", "v_proj"]},
        },
    }
    (checkpoint / "classifier.json").write_text(json.dumps(metadata))
    before = {str(p): artifacts.sha(p) for p in checkpoint.rglob("*") if p.is_file()}
    item = {
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "threshold": 0.95,
        "approach": approach,
    }
    plan = {"identity": "plan", "models": {"key": item}}
    monkeypatch.setattr(artifacts, "checked_plan", lambda _: plan)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("modeling_k2_horizon.py", "configuration_k2_horizon.py"):
        (snapshot / name).write_text("# pinned code\n")
    (snapshot / "config.json").write_text(json.dumps({"hidden_size": 2, "num_hidden_layers": 1}))
    save_file({**base, "lm_head.weight": torch.ones(2, 2)}, str(snapshot / "model.safetensors"))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **kw: str(snapshot))
    receipt = artifacts.export_classifier(tmp_path / "new", "key")
    exported = tmp_path / "new/exports/key"
    index = json.loads((exported / "model.safetensors.index.json").read_text())
    weights = {}
    for name in set(index["weight_map"].values()):
        weights.update(load_file(str(exported / name)))
    assert torch.equal(weights["classifier.bias"], head.bias)
    assert torch.equal(weights["classifier.weight"], head.weight)
    for name, value in expected.items():
        assert torch.equal(weights[name], value)
    assert not any("score." in n or "lora_" in n or "lm_head" in n for n in weights)
    assert receipt["merged_lora_modules"] == (2 if approach == "lora" else 0)
    config = json.loads((exported / "config.json").read_text())
    assert config["head_dtype"] == "float32"
    assert config["id2label"] == {"0": "VALID", "1": "GIBBERISH"}
    assert config["pooler_config"]["seq_pooling_type"] == "LAST"
    assert before == {str(p): artifacts.sha(p) for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        artifacts.export_classifier(tmp_path / "new", "key")


def test_parity_gate_blocks_and_tampering(monkeypatch, tmp_path):
    plan, receipt = {"identity": "plan"}, {"identity": "export"}
    monkeypatch.setattr(parity, "checked_export", lambda *a: (plan, tmp_path, receipt))
    directory = tmp_path / "parity/key"
    directory.mkdir(parents=True)
    gate = {"passed": False, "plan_identity": "plan", "export_identity": "export", "files": {}}
    gate["identity"] = artifacts.seal(gate)
    (directory / "gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="not passed"):
        parity.require_parity(tmp_path, "key")
    gate["passed"] = True
    (directory / "gate.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        parity.require_parity(tmp_path, "key")


def test_offline_uses_llm_classify_token_ids(monkeypatch, tmp_path):
    calls = []
    receipt = {"head_hashes": {"weight": "w", "bias": "b"}}
    status = {
        "head_hashes": receipt["head_hashes"],
        "head_dtype": "torch.float32",
        "vllm_module_count": 4,
        "model_class": "TransformersForSequenceClassification",
    }

    class LLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def collective_rpc(self, method):
            return [status]

        def classify(self, prompts, use_tqdm):
            calls.append(prompts)
            return [
                SimpleNamespace(
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    outputs=SimpleNamespace(probs=[0.25, 0.75]),
                )
            ]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM))
    monkeypatch.setattr(runtime, "cuda_required", lambda: None)
    monkeypatch.setattr(runtime, "checked_export", lambda *a: ({}, tmp_path, receipt))
    monkeypatch.setattr(runtime, "engine_kwargs", lambda *a: {"runner": "pooling"})
    classifier = runtime.OfflineClassifier(tmp_path, "key")
    assert classifier.predict_tokens([0, 8, 9]) == [0.25, 0.75]
    assert calls == [{"runner": "pooling"}, [{"prompt_token_ids": [0, 8, 9]}]]


def test_async_endpoint_loads_once_and_uses_threshold(monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    plan = {"identity": "plan", "models": {"key": {"metadata": {"config": {"max_length": 64}}}}}
    monkeypatch.setattr(server, "checked_export", lambda *a: (plan, tmp_path, {"threshold": 0.95}))
    monkeypatch.setattr(server, "require_parity", lambda *a: {})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [0, 5], "attention_mask": [1, 1]}

    class Fake:
        def __init__(self, *args):
            calls.append("load")

        async def ready(self):
            pass

        async def predict_tokens(self, ids):
            calls.append(ids)
            return [0.1, 0.9]

        def close(self):
            calls.append("close")

    monkeypatch.setattr(server, "AsyncClassifier", Fake)
    app = server.create_app(tmp_path, "key", "vllm")

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as c:
                for _ in range(2):
                    response = await c.post("/classify", json={"input": "نص عربي"})
                    assert response.json()["class_id"] == 0
                    assert response.json()["default_class_id"] == 1
                assert (await c.post("/classify", json={"input": "Latin"})).status_code == 422
                assert (await c.post("/classify", json={"input": ""})).status_code == 422

    asyncio.run(exercise())
    assert calls == ["load", [0, 5], [0, 5], "close"]


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16])
def test_http_same_400_at_requested_concurrency(monkeypatch, tmp_path, concurrency):
    plan = {"identity": "plan", "parity_policy": artifacts.POLICY}
    monkeypatch.setattr(benchmark, "checked_plan", lambda _: plan)
    monkeypatch.setattr(benchmark, "require_parity", lambda *a: {})
    rows = [
        dict(row([0.1, 0.9], str(i)), category="benchmark", input_text="نص") for i in range(400)
    ]
    monkeypatch.setattr(benchmark, "inputs", lambda *a: rows)
    folder = tmp_path / "parity/key/hf-original"
    folder.mkdir(parents=True)
    (folder / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    active, peak, count = 0, 0, 0

    async def handle(request):
        nonlocal active, peak, count
        if request.url.path == "/benchmark/info":
            return httpx.Response(
                200,
                json={
                    "backend": "vllm",
                    "model_key": "key",
                    "plan_identity": "plan",
                    "threshold": 0.95,
                },
            )
        if request.url.path == "/benchmark/memory":
            return httpx.Response(200, json={"peak_allocated_bytes": 10, "peak_reserved_bytes": 20})
        if request.url.path == "/benchmark/reset":
            return httpx.Response(200, json={"status": "reset"})
        assert request.url.path == "/classify"
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(0)
        active -= 1
        count += 1
        return httpx.Response(
            200,
            json={
                "model_key": "key",
                "threshold": 0.95,
                "class_id": 0,
                "probabilities": {"VALID": 0.1, "GIBBERISH": 0.9},
                "inference_ms": 1,
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    result = asyncio.run(benchmark.http_run(tmp_path, "key", "vllm", concurrency))
    assert result["latency"]["samples"] == 400
    assert result["response_parity"]["passed"]
    assert peak == concurrency
    assert count == 403


def test_async_engine_uses_classification_pooling(monkeypatch, tmp_path):
    receipt = {"head_hashes": {"weight": "w", "bias": "b"}}
    status = {
        "head_hashes": receipt["head_hashes"],
        "head_dtype": "torch.float32",
        "vllm_module_count": 4,
        "model_class": "TransformersForSequenceClassification",
    }
    calls = []

    class Engine:
        async def check_health(self):
            pass

        async def collective_rpc(self, method):
            calls.append(method)
            return [status]

        async def encode(self, prompt, params, request_id):
            calls.append((prompt, params.task, params.use_activation))
            yield SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs="pooled")

        def shutdown(self, timeout=None):
            calls.append("shutdown")

    engine = Engine()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(PoolingParams=SimpleNamespace))
    monkeypatch.setitem(
        sys.modules, "vllm.engine.arg_utils", SimpleNamespace(AsyncEngineArgs=SimpleNamespace)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.async_llm",
        SimpleNamespace(AsyncLLM=SimpleNamespace(from_engine_args=lambda _: engine)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.outputs",
        SimpleNamespace(
            ClassificationOutput=SimpleNamespace(
                from_base=lambda _: SimpleNamespace(probs=[0.2, 0.8])
            )
        ),
    )
    monkeypatch.setattr(runtime, "cuda_required", lambda: None)
    monkeypatch.setattr(runtime, "checked_export", lambda *a: ({}, tmp_path, receipt))
    monkeypatch.setattr(runtime, "engine_kwargs", lambda *a: {"runner": "pooling"})

    async def exercise():
        classifier = runtime.AsyncClassifier(tmp_path, "key")
        await classifier.ready()
        assert await classifier.predict_tokens([0, 5]) == [0.2, 0.8]
        classifier.close()

    asyncio.run(exercise())
    assert calls == [
        "jaisguard_status",
        ({"prompt_token_ids": [0, 5]}, "classify", True),
        "shutdown",
    ]


def test_report_preserves_failed_parity_and_pending_results(monkeypatch, tmp_path):
    plan = {"identity": "plan", "models": {"key": {"threshold": 0.95}}}
    monkeypatch.setattr(benchmark, "checked_plan", lambda _: plan)
    gate_path = tmp_path / "parity/key/gate.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text("{}")
    failed = parity.compare_probabilities(
        [row([0.051, 0.949])], [row([0.049, 0.951])], 0.95, artifacts.POLICY
    )
    monkeypatch.setattr(
        benchmark,
        "checked_gate",
        lambda *a: {"passed": False, "comparisons": {"hf_vs_vllm": failed}},
    )
    path = benchmark.report(tmp_path)
    text = (path / "comparison.md").read_text()
    assert "within_strict_tolerance=False" in text
    assert "Completed HTTP workloads: 0/8" in text
    assert "pending" in text
    with pytest.raises(FileExistsError):
        benchmark.report(tmp_path)


def test_collection_refuses_cpu_before_creating_results(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA required"):
        parity.collect(tmp_path, "key", "vllm")
    assert not list(tmp_path.iterdir())


def test_worker_status_uses_plain_ipc_primitives(monkeypatch):
    from torch.torch_version import TorchVersion

    head = nn.Linear(2, 2)
    model = nn.Module()
    model.classifier = head
    worker = runtime.WorkerDiagnostics()
    worker.model_runner = SimpleNamespace(get_model=lambda: model)
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.13.0+cu132"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 20)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "fake GPU")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100)
    )
    status = worker.jaisguard_status()
    assert type(status["torch_version"]) is str
    assert status["torch_version"] == "2.13.0+cu132"

    def check(value):
        assert type(value) in (dict, str, int, float, bool, type(None))
        if isinstance(value, dict):
            for item in value.values():
                check(item)

    check(status)


def test_serialization_repair_preserves_exports_and_hf_references(monkeypatch, tmp_path):
    import hashlib

    from jais_guard.vllm_support import serialization_repair as fix

    source, destination = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    runtime_path = Path(runtime.__file__).resolve()
    old_source = runtime_path.read_text().replace(fix.NEW_VERSION, fix.OLD_VERSION)
    plan = {
        "models": {"key": {}},
        "files": {str(runtime_path): hashlib.sha256(old_source.encode()).hexdigest()},
    }
    plan["identity"] = artifacts.seal(plan)
    (source / "plan.json").write_text(json.dumps(plan))
    export = source / "exports/key"
    export.mkdir(parents=True)
    weights = export / "model.safetensors"
    weights.write_bytes(b"unchanged trained weights")
    receipt = {
        "model_key": "key",
        "plan_identity": plan["identity"],
        "base_files": {},
        "files": {str(weights): artifacts.sha(weights)},
    }
    receipt["identity"] = artifacts.seal(receipt)
    (export / "export.json").write_text(json.dumps(receipt))
    folder = source / "parity/key/hf-original"
    folder.mkdir(parents=True)
    records = [row([0.25, 0.75])]
    predictions = folder / "predictions.jsonl"
    predictions.write_text(json.dumps(records[0]) + "\n")
    completion = {
        "status": "complete",
        "model_key": "key",
        "backend": "hf-original",
        "plan_identity": plan["identity"],
        "export_identity": receipt["identity"],
        "samples": 1,
        "predictions_sha256": artifacts.sha(predictions),
    }
    (folder / "completion.json").write_text(json.dumps(completion))
    (source / "parity/key/vllm").mkdir()
    before = {str(p): artifacts.sha(p) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(fix, "inputs", lambda *a: records)
    result = fix.repair(source, destination)
    assert result["reused_hf"] == ["key/hf-original"]
    assert not (destination / "parity/key/vllm").exists()
    assert (
        destination / "parity/key/hf-original/predictions.jsonl"
    ).read_bytes() == predictions.read_bytes()
    assert before == {str(p): artifacts.sha(p) for p in source.rglob("*") if p.is_file()}
    migrated = json.loads((destination / "parity/key/hf-original/completion.json").read_text())
    assert migrated["plan_identity"] != plan["identity"]
    assert migrated["reused_from"]["sha256"] == artifacts.sha(folder / "completion.json")
    with pytest.raises(ValueError, match="new versioned"):
        fix.repair(source, destination)
    # A repair must not silently accept an unrelated code/weight change.
    plan["files"][str(runtime_path)] = "0" * 64
    plan["identity"] = artifacts.seal({k: v for k, v in plan.items() if k != "identity"})
    (source / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="differs beyond"):
        fix.validate_previous_plan(source)


@pytest.mark.parametrize("case", ["accepted", "bad_head", "fp16_mismatch"])
def test_user_authorized_fp16_gate_keeps_bf16_diagnostics(monkeypatch, tmp_path, case):
    plan = {
        "identity": "plan",
        "parity_policy": artifacts.POLICY,
        "acceptance": {"reference": "hf-fp16", "bf16_diagnostic_only": True},
    }
    receipt = {"identity": "export", "threshold": 0.95, "head_hashes": {"weight": "w", "bias": "b"}}
    status = {
        "model_class": "TransformersForSequenceClassification",
        "head_dtype": "torch.float32",
        "head_hashes": receipt["head_hashes"],
        "vllm_module_count": 20,
    }
    if case == "bad_head":
        status = {**status, "head_hashes": {"weight": "wrong", "bias": "b"}}
    monkeypatch.setattr(parity, "checked_export", lambda *a: (plan, tmp_path, receipt))
    monkeypatch.setattr(parity, "inputs", lambda *a: [row([0.1, 0.9])])
    for backend, probs in [
        ("hf-original", [0.1, 0.9]),
        ("hf-fp16", [0.049, 0.951]),
        ("vllm", [0.1, 0.9] if case == "fp16_mismatch" else [0.0495, 0.9505]),
    ]:
        folder = tmp_path / "parity/key" / backend
        folder.mkdir(parents=True)
        path = folder / "predictions.jsonl"
        path.write_text(json.dumps(row(probs)) + "\n")
        (folder / "completion.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "plan_identity": "plan",
                    "export_identity": "export",
                    "samples": 1,
                    "predictions_sha256": artifacts.sha(path),
                    "loading": status,
                }
            )
        )
    if case == "bad_head":
        with pytest.raises(ValueError, match="exact trained"):
            parity.verify(tmp_path, "key")
        return
    gate = parity.verify(tmp_path, "key")
    assert gate["passed"] == (case == "accepted")
    assert gate["trained_head_verified"]
    assert gate["acceptance_reference"] == "hf-fp16"
    assert gate["bf16_diagnostic_only"]
    if case == "accepted":
        assert not gate["comparisons"]["hf-original_vs_vllm"]["passed"]
        assert gate["comparisons"]["hf-original_vs_vllm"]["locked_prediction_mismatches"] == 1
        assert parity.acceptance_passes(plan, gate["comparisons"])
        assert not parity.acceptance_passes({}, gate["comparisons"])


def test_fp16_adapter_preserves_fp32_head_and_reuses_forward(monkeypatch):
    from contextlib import nullcontext

    from jais_guard.serving import Predictor
    from jais_guard.vllm_support.fp16 import FP16Predictor

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)
            self.head = nn.Linear(2, 2)

        def forward(self, value):
            return self.head(self.backbone(value).float())

    model = Model()
    original_head = model.head.weight.detach().clone()

    def load(self, manifest, key):
        self.item = {"approach": "lora"}
        self.model = model
        self.load_seconds = 1.0

    contexts = []
    monkeypatch.setattr(Predictor, "__init__", load)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(FP16Predictor, "memory", lambda _: {})
    monkeypatch.setattr(
        torch, "autocast", lambda device, dtype: contexts.append((device, dtype)) or nullcontext()
    )
    predictor = FP16Predictor({}, "key")
    result = predictor.model(torch.ones(1, 2, dtype=torch.float16))
    assert result.dtype == torch.float32
    assert model.backbone.weight.dtype == torch.float16
    assert model.head.weight.dtype == torch.float32
    assert torch.equal(model.head.weight, original_head)
    assert contexts == [("cuda", torch.float16)]


def test_comparison_chart_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    records = [
        {
            "model": "frozen",
            "backend": backend,
            "concurrency": c,
            "throughput_per_second": 10.0,
            "p95_ms": 100.0,
            "process_to_ready_seconds": 2.0,
            "peak_reserved_bytes": 2**30,
        }
        for backend in ("hf", "vllm")
        for c in (1, 4, 8, 16)
    ]
    benchmark.comparison_charts(tmp_path, records)
    for name in (
        "latency-throughput.png",
        "latency-throughput.svg",
        "startup-memory.png",
        "startup-memory.svg",
    ):
        assert (tmp_path / name).stat().st_size > 100
