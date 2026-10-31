"""CPU contracts: locked selection, API lifecycle, timing summaries and HTTP orchestration."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jais_guard.benchmarks import latency as bench
from jais_guard.serving import hf as serving


def test_validation_selection_ignores_test_quality():
    entries = [
        {"seed": 42, "test_accuracy": 1, "metadata": {"metadata": {"validation_loss": 0.3}}},
        {"seed": 44, "test_accuracy": 0, "metadata": {"metadata": {"validation_loss": 0.2}}},
    ]
    assert bench.select_validation(entries)["seed"] == 44
    entries[0]["metadata"]["metadata"]["validation_loss"] = 0.2
    assert bench.select_validation(entries)["seed"] == 42
    entries[0]["metadata"]["metadata"]["validation_loss"] = float("nan")
    with pytest.raises(ValueError):
        bench.select_validation(entries)


def test_summary_and_no_overwrite(tmp_path):
    result = bench.summary([10, 20, 30, 40])
    assert result["mean_ms"] == 25
    assert result["p50_ms"] == 25
    assert result["p95_ms"] == pytest.approx(38.5)
    assert result["p99_ms"] == pytest.approx(39.7)
    assert result["throughput_per_second"] == 40
    assert bench.summary([10, 20], 1)["throughput_per_second"] == 2
    for invalid in ([], [0], [float("nan")]):
        with pytest.raises(ValueError):
            bench.summary(invalid)
    bench.write_new(tmp_path / "a.json", {"original": True})
    with pytest.raises(FileExistsError):
        bench.write_new(tmp_path / "a.json", {})
    with pytest.raises(ValueError):
        bench.prepare(tmp_path)


def test_api_load_once_validation_and_invalid(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        serving, "checked_manifest", lambda _: {"default_model": "untuned", "identity": "id"}
    )

    class Fake:
        def __init__(self, manifest, key):
            loaded.append(key)

        def info(self):
            return {"model_key": "untuned"}

        def predict(self, text):
            if not text.strip() or text == "overbudget":
                raise ValueError("Invalid input")
            return {"class": "INVALID", "class_id": -1, "probabilities": None, "inference_ms": 2}

    app = serving.create_app(predictor_factory=Fake)
    # Keep fake CPU handlers on this loop; sandbox cross-thread wakeups are unavailable.
    import fastapi.routing

    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(fastapi.routing, "run_in_threadpool", inline)

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (await client.get("/health")).json()["status"] == "ready"
                for _ in range(2):
                    response = await client.post("/predict", json={"text": "نص عربي"})
                    assert response.json()["class_id"] == -1
                    assert response.json()["probabilities"] is None
                for body in (
                    {},
                    {"text": ""},
                    {"text": " "},
                    {"text": "overbudget"},
                    {"text": 123},
                ):
                    assert (await client.post("/predict", json=body)).status_code == 422

    asyncio.run(exercise())
    assert loaded == ["untuned"]


def test_manifest_tampering(tmp_path):
    from jais_guard.data.test_workflow import sha

    quality = tmp_path / "quality.json"
    quality.write_text("{}")
    artifact = tmp_path / "artifact"
    artifact.write_text("locked")
    manifest = {"files": {str(artifact): sha(artifact)}, "quality_sha256": sha(quality)}
    manifest["identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    bench.write_new(tmp_path / "manifest.json", manifest)
    serving.checked_manifest(tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        serving.checked_manifest(tmp_path)


def test_gpu_refuses_cpu(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA required"):
        bench.benchmark_gpu(tmp_path, "untuned")
    assert not list(tmp_path.iterdir())


def test_http_sequential_samples_and_identity(monkeypatch, tmp_path):
    manifest = {"identity": "locked", "warmup_inputs": 3, "test_sha256": "sha"}
    rows = [{"id": str(i), "input_text": "نص"} for i in range(400)]
    monkeypatch.setattr(bench, "checked_manifest", lambda _: manifest)
    monkeypatch.setattr(bench, "frozen_inputs", lambda _: rows)
    requests = []

    def handle(request):
        requests.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"model_key": "untuned", "manifest_identity": "locked"})
        assert json.loads(request.content)["text"] == "نص"
        return httpx.Response(
            200,
            json={"model_key": "untuned", "inference_ms": 1, "class_id": -1, "probabilities": None},
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(handle))
    )
    result = bench.benchmark_http(tmp_path, "untuned")
    assert result["latency"]["samples"] == 400
    assert result["server_inference"]["mean_ms"] == 1
    assert requests == ["/health"] + ["/predict"] * 403
    assert len((tmp_path / "http/untuned/samples.jsonl").read_text().splitlines()) == 400
    with pytest.raises(FileExistsError):
        bench.benchmark_http(tmp_path, "untuned")
    with pytest.raises(ValueError, match="restricted"):
        bench.benchmark_http(tmp_path, "untuned", url="https://api.typesafe.ai")


def test_report_pending_latency(monkeypatch, tmp_path):
    groups = {"jev/hosted/zero_shot": {k: {"mean": 0.5, "std": None} for k in bench.MEASURES}}
    bench.write_new(
        tmp_path / "quality.json",
        {
            "groups": groups,
            "scope": "seen benchmark",
            "jev_latency": {k: 200 for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")},
        },
    )
    monkeypatch.setattr(bench, "checked_manifest", lambda _: {"models": {}, "selected": {}})
    monkeypatch.setattr(bench, "charts", lambda *args: None)
    output = bench.report(tmp_path)
    text = (output / "comparison.md").read_text()
    assert "pending" in text
    assert "remote network overhead" in text
    assert (output / "comparison.csv").exists()
    with pytest.raises(FileExistsError):
        bench.report(tmp_path)


def test_predictor_locked_boundary_and_model_reuse(monkeypatch):
    import threading

    import torch

    from jais_guard.finetuning import model as model_module

    predictor = serving.Predictor.__new__(serving.Predictor)
    predictor.item = {
        "approach": "lora",
        "threshold": 0.9,
        "metadata": {"config": {"max_length": 64}},
    }
    predictor.model_key = "v2-lora-44"
    predictor.lock = threading.Lock()
    predictor.model = torch.nn.Linear(1, 2)
    predictor.tokenizer = object()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    scores = iter([0.9, 0.8])
    seen = []

    def predict_rows(model, tokenizer, rows, max_length, device, batch_size):
        seen.append((rows[0]["input_text"], max_length, device, batch_size))
        model(torch.ones(1, 1))
        return [next(scores)], None

    monkeypatch.setattr(model_module, "predict_rows", predict_rows)
    original_forward = predictor.model.forward
    first = predictor.predict("عربي")
    assert first["class_id"] == 1  # >=, not >; argmax would also be wrong at lower thresholds.
    assert first["probabilities"]["GIBBERISH"] == 0.9
    assert first["threshold"] == 0.9
    assert first["inference_ms"] >= first["model_ms"] > 0
    assert predictor.predict("نص")["class_id"] == 0
    assert predictor.model.forward == original_forward
    assert seen == [("عربي", 64, "cuda", 1), ("نص", 64, "cuda", 1)]


def test_http_wrong_server_and_failures_preserved(monkeypatch, tmp_path):
    manifest = {"identity": "locked", "warmup_inputs": 0, "test_sha256": "sha"}
    monkeypatch.setattr(bench, "checked_manifest", lambda _: manifest)
    monkeypatch.setattr(bench, "frozen_inputs", lambda _: [{"id": "1", "input_text": "نص"}])
    original = httpx.Client

    def wrong(request):
        return httpx.Response(200, json={"model_key": "other", "manifest_identity": "locked"})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(wrong))
    )
    with pytest.raises(ValueError, match="different model"):
        bench.benchmark_http(tmp_path, "untuned", execution="wrong")
    assert not (tmp_path / "wrong/untuned/metrics.json").exists()

    def failing(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"model_key": "untuned", "manifest_identity": "locked"})
        return httpx.Response(500, json={"detail": "failed"})

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(**kw, transport=httpx.MockTransport(failing))
    )
    with pytest.raises(httpx.HTTPStatusError):
        bench.benchmark_http(tmp_path, "untuned", execution="failed")
    assert not (tmp_path / "failed/untuned/metrics.json").exists()


def test_gpu_batch_one_warmup_loading_and_memory(monkeypatch, tmp_path):
    import torch

    manifest = {"identity": "id", "test_sha256": "sha", "warmup_inputs": 3}
    rows = [{"id": str(i), "input_text": "نص"} for i in range(400)]
    monkeypatch.setattr(bench, "cuda_required", lambda: None)
    monkeypatch.setattr(bench, "checked_manifest", lambda _: manifest)
    monkeypatch.setattr(bench, "frozen_inputs", lambda _: rows)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    resets = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: resets.append(True))
    calls = []

    class Fake:
        def __init__(self, manifest, key):
            calls.append(("load", key))

        def predict(self, text):
            calls.append(("predict", text))
            return {"inference_ms": 10, "model_ms": 8}

        def info(self):
            return {"model_load_seconds": 2}

        def memory(self):
            return {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200}

    monkeypatch.setattr(bench, "Predictor", Fake)
    result = bench.benchmark_gpu(tmp_path, "untuned")
    assert len(calls) == 404
    assert calls[0] == ("load", "untuned")
    assert resets == [True]
    assert result["latency"]["samples"] == 400
    assert result["latency"]["mean_ms"] == 10
    assert result["model_only"]["mean_ms"] == 8
    assert result["environment"]["model_load_seconds"] == 2
    assert result["memory"]["peak_reserved_bytes"] == 200
