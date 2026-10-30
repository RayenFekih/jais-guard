"""CPU-only Jev contract/resume/report checks. Every HTTP call uses MockTransport."""

import asyncio
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard import jev
from jais_guard import jev_reporting as reporting

ROOT = Path(__file__).resolve().parents[1]


def response(probability=0.8, model="jev-1.14.0"):
    return {
        "model": model,
        "answers": {"recoverable": {"type": "noul", "noul": probability}},
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }


@pytest.fixture
def protocol():
    rows = [
        {"id": "a", "input_text": "كيف أغير كلمة المرور", "label": "VALID"},
        {"id": "b", "input_text": "ضصض ضصض عغغ", "label": "GIBBERISH"},
    ]
    question = jev.load_question(ROOT / "configs/jev_benchmark.json")
    config = jev.manifest_config(
        "smoke",
        rows,
        question,
        "jev-1.14.0",
        {
            "concurrency": 1,
            "max_retries": 2,
            "timeout_seconds": 10,
            "backoff_seconds": 0.001,
            "max_backoff_seconds": 1,
        },
    )
    return config, rows


def run(tmp_path, protocol, handler, **kwargs):
    config, rows = protocol
    return asyncio.run(
        jev.execute(
            tmp_path, config, rows, "test-secret", transport=httpx.MockTransport(handler), **kwargs
        )
    )


@pytest.mark.parametrize("probability", [True, None, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_reject_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probability"):
        jev.validate_response(response(probability), "recoverable")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.pop("usage"),
        lambda r: r["usage"].update(input_tokens=True),
        lambda r: r["answers"]["recoverable"].update(type="binary"),
        lambda r: r["answers"].update(extra={}),
        lambda r: r.update(model="not-a-model"),
    ],
)
def test_schema(mutation):
    value = response()
    mutation(value)
    with pytest.raises(ValueError):
        jev.validate_response(value, "recoverable")


def test_payload_resume_and_no_secret_persistence(tmp_path, protocol):
    seen = []

    def handler(request):
        assert request.url == jev.ENDPOINT
        assert request.headers["authorization"] == "Bearer test-secret"
        data = json.loads(request.content)
        assert set(data) == {"state", "model", "questions"}
        assert data["state"] in [r["input_text"] for r in protocol[1]]
        assert data["questions"]["recoverable"]["type"] == "noul"
        seen.append(data)
        return httpx.Response(200, json={**response(), "untrusted_extra": "test-secret"})

    assert run(tmp_path, protocol, handler) == {"success": 2}
    assert run(tmp_path, protocol, handler) == {"success": 2}
    assert len(seen) == 2
    manifest, examples, attempts = jev.read_ledger(tmp_path)
    assert manifest["config"]["question"]["yes_threshold"] == 0.5
    assert len(attempts) == 2 and all(e["status"] == "success" for e in examples)
    assert all(b"test-secret" not in p.read_bytes() for p in tmp_path.iterdir() if p.is_file())


@pytest.mark.parametrize("status", [429, 529, 503])
def test_retry_transient(tmp_path, protocol, status):
    seen, slept = [], []

    def handler(request):
        seen.append(request)
        return (
            httpx.Response(status, headers={"retry-after-ms": "1"})
            if len(seen) == 1
            else httpx.Response(200, json=response())
        )

    async def sleep(delay):
        slept.append(delay)

    assert run(tmp_path, protocol, handler, sleep=sleep) == {"success": 2}
    assert len(seen) == 3 and slept == [0.001]
    assert len(jev.read_ledger(tmp_path)[2]) == 3


def test_permanent_auth_error_stops(tmp_path, protocol):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(401, json={"error": "test-secret"})

    assert run(tmp_path, protocol, handler) == {"failed": 1, "pending": 1}
    assert len(seen) == 1
    assert "test-secret" not in json.dumps(jev.read_ledger(tmp_path))


def test_timeout_bounded_retry(tmp_path, protocol):
    def handler(request):
        raise httpx.ReadTimeout("untrusted test-secret", request=request)

    result = run(tmp_path, protocol, handler)
    assert result == {"failed": 2}
    _, _, attempts = jev.read_ledger(tmp_path)
    assert len(attempts) == 6
    assert {a["error_type"] for a in attempts} == {"transport_error"}
    assert "test-secret" not in json.dumps(attempts)


def test_model_drift_and_bad_schema_stop(tmp_path, protocol):
    assert run(
        tmp_path, protocol, lambda _: httpx.Response(200, json=response(model="jev-1.13.0"))
    ) == {"failed": 1, "pending": 1}
    assert jev.read_ledger(tmp_path)[2][0]["error_type"] == "model_drift"


def test_alias_drift(tmp_path, protocol):
    protocol[0]["model"] = "jev-latest"
    seen = []

    def handler(_):
        seen.append(1)
        return httpx.Response(
            200, json=response(model="jev-1.14.0" if len(seen) == 1 else "jev-1.13.0")
        )

    assert run(tmp_path, protocol, handler) == {"success": 1, "failed": 1}


def test_uncertain_requires_explicit_retry(tmp_path, protocol):
    config, rows = protocol
    tmp_path.mkdir(exist_ok=True)
    db = jev.initialize(tmp_path, config, rows)
    with db:
        db.execute("UPDATE examples SET status='in_flight' WHERE id='a'")
        db.execute(
            "INSERT INTO attempts(id,number,started_at,error_type) VALUES "
            "('a',1,'before-crash','in_flight')"
        )
    db.close()
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["state"])
        return httpx.Response(200, json=response())

    assert run(tmp_path, protocol, handler) == {"uncertain": 1, "success": 1}
    assert seen == [rows[1]["input_text"]]
    assert run(tmp_path, protocol, handler, retry_uncertain=True) == {"success": 2}
    assert seen == [rows[1]["input_text"], rows[0]["input_text"]]


def test_failed_requires_explicit_retry(tmp_path, protocol):
    run(tmp_path, protocol, lambda _: httpx.Response(401))
    assert run(tmp_path, protocol, lambda _: httpx.Response(200, json=response())) == {
        "failed": 1,
        "success": 1,
    }
    assert run(
        tmp_path, protocol, lambda _: httpx.Response(200, json=response()), retry_failed=True
    ) == {"success": 2}


def test_changed_protocol_rejected(tmp_path, protocol):
    run(tmp_path, protocol, lambda _: httpx.Response(200, json=response()))
    protocol[0]["model"] = "jev-1.13.0"
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, protocol, lambda _: pytest.fail("No network for mismatched protocol"))


def test_exclusive_runner(tmp_path):
    with jev.runner_lock(tmp_path):
        with pytest.raises(ValueError, match="Another runner"):
            with jev.runner_lock(tmp_path):
                pytest.fail("Lock must fail")


def test_backoff():
    assert jev.retry_delay({}, 2, 1, 30) == 4
    assert jev.retry_delay({"retry-after": "8"}, 1, 1, 30) == 8
    assert jev.retry_delay({"retry-after-ms": "2000"}, 0, 1, 30) == 2
    assert jev.retry_delay({"retry-after": "120"}, 0, 1, 30) is None
    assert jev.retry_delay({"retry-after": "bad"}, 0, 1, 30) == 1


def cli_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_jev", ROOT / "scripts/benchmark_jev.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_dry_run_no_credentials_no_files(tmp_path, monkeypatch):
    cli = cli_module()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(
        httpx.AsyncClient, "post", lambda *a, **k: pytest.fail("Dry-run cannot call HTTP")
    )
    target = tmp_path / "new"
    assert (
        cli.main(["smoke", "--run-dir", str(target), "--text", "كيف أستعيد حسابي", "--dry-run"])
        == 0
    )
    assert not target.exists()


def test_paid_requires_key_and_pinned_evaluation(tmp_path, monkeypatch):
    cli = cli_module()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert (
        cli.main(
            [
                "smoke",
                "--run-dir",
                str(tmp_path / "new"),
                "--text",
                "كيف أستعيد حسابي",
                "--execute-paid",
            ]
        )
        == 2
    )
    assert cli.main(["evaluate", "--run-dir", str(tmp_path / "new"), "--execute-paid"]) == 2
    assert not (tmp_path / "new").exists()


def test_concurrency_is_bounded(tmp_path, protocol):
    config, rows = protocol
    config["execution"]["concurrency"] = 2
    rows.extend({"id": str(i), "input_text": "طلب عربي"} for i in range(4))
    config["rows_sha256"] = jev.digest(rows)
    config["examples"] = len(rows)
    active, maximum = 0, 0

    async def handler(_):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0.001)
        active -= 1
        return httpx.Response(200, json=response())

    assert run(tmp_path, protocol, handler) == {"success": 6}
    assert maximum == 2


def test_report_tie_invalid_denominator_usage_and_no_http(tmp_path, protocol, monkeypatch):
    config, rows = protocol
    config.update(
        stage="evaluate", test="dummy", test_files={}, comparison="cached", comparison_files={}
    )
    run(tmp_path, protocol, lambda _: httpx.Response(200, json=response(0.5)))
    db = sqlite3.connect(tmp_path / "requests.sqlite3")
    with db:
        db.execute(
            "UPDATE examples SET "
            "status='failed',response_json=NULL,last_error='http_error' WHERE "
            "id='b'"
        )
    db.close()
    monkeypatch.setattr(reporting, "frozen_rows", lambda _: rows)
    monkeypatch.setattr(reporting, "sha", lambda _: "test-sha")
    monkeypatch.setattr(
        reporting, "cached_comparison", lambda *a: {"groups": {}, "latency": {}, "files": {}}
    )
    monkeypatch.setattr(
        httpx.AsyncClient, "post", lambda *a, **k: pytest.fail("Report cannot call HTTP")
    )
    result, predictions = reporting.summarize(tmp_path)
    assert predictions[0]["prediction_id"] == 0  # yes probability exactly 0.5 is VALID
    assert predictions[1]["prediction_id"] == -1
    assert result["metrics"]["accuracy"] == 0.5
    assert result["metrics"]["confusion"] == [[1, 0, 0], [0, 0, 1]]
    assert result["usage_and_cost"]["input_tokens"] == 24
    assert result["usage_and_cost"]["estimated_total_cost_usd"] == pytest.approx(
        24 * 0.042 / 1_000_000
    )
    report = reporting.save_report(tmp_path)
    assert len((report / "predictions.jsonl").read_text().splitlines()) == 2
    assert (report / "report.md").exists()
    assert reporting.save_report(tmp_path) != report
    db = sqlite3.connect(tmp_path / "requests.sqlite3")
    with db:
        db.execute("UPDATE examples SET status='pending' WHERE id='b'")
    db.close()
    with pytest.raises(ValueError, match="Unfinished"):
        reporting.summarize(tmp_path)
    assert reporting.summarize(tmp_path, allow_incomplete=True)[0]["complete"] is False


def test_latency_summary():
    result = reporting.latency([1, 2, 3, 4, None])
    assert result["mean_ms"] == result["median_ms"] == result["p50_ms"] == 2.5
    assert result["p99_ms"] == pytest.approx(3.97)


def test_schema_failure_stops_and_never_stores_raw_body(tmp_path, protocol):
    assert run(
        tmp_path,
        protocol,
        lambda _: httpx.Response(200, json={"untrusted_error": "test-secret"}),
    ) == {"failed": 1, "pending": 1}
    _, _, attempts = jev.read_ledger(tmp_path)
    assert attempts[0]["error_type"] == "schema_error"
    assert "test-secret" not in json.dumps(attempts)


def test_retry_after_cap_pauses_without_retrying_early(tmp_path, protocol):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})

    assert run(tmp_path, protocol, handler) == {"failed": 1, "pending": 1}
    assert len(calls) == 1


def test_report_unknown_usage_is_not_zero_cost(tmp_path, protocol, monkeypatch):
    config, rows = protocol
    config.update(
        stage="evaluate", test="dummy", test_files={}, comparison="cached", comparison_files={}
    )
    run(tmp_path, protocol, lambda _: httpx.Response(401))
    monkeypatch.setattr(reporting, "frozen_rows", lambda _: rows)
    monkeypatch.setattr(reporting, "sha", lambda _: "test-sha")
    monkeypatch.setattr(
        reporting, "cached_comparison", lambda *a: {"groups": {}, "latency": {}, "files": {}}
    )
    result, _ = reporting.summarize(tmp_path, allow_incomplete=True)
    assert result["metrics"]["accuracy"] == 0
    assert result["metrics"]["valid_acceptance_rate"] == 0
    assert result["metrics"]["valid_false_rejection_rate"] == 0
    assert result["metrics"]["valid_invalid_outputs"] == 1
    assert result["usage_and_cost"]["estimated_total_cost_usd"] is None
    assert result["usage_and_cost"]["attempts_without_usage"] == 1
    assert result["reliability"]["http_error_rate_per_response"] == 1


def test_cached_threshold_mapping_uses_prior_lock_not_test_artifact(tmp_path, monkeypatch):
    directory = tmp_path / "benchmark"
    directory.mkdir()
    for name in ("protocol.json", "threshold_lock.json"):
        (tmp_path / name).write_text("{}")
    (directory / "completion.json").write_text(
        json.dumps({"status": "complete", "test_sha256": "test-sha"})
    )
    source = directory / "frozen-42"
    source.mkdir()
    for name in ("metrics.json", "predictions.jsonl"):
        (source / name).write_text("{}")
    entry = {
        "version": "v2",
        "approach": "frozen",
        "seed": 42,
        "metrics": {"validation_locked": {"threshold": 0.7}},
    }
    (directory / "metrics.json").write_text(json.dumps({"per_seed": [entry]}))
    monkeypatch.setattr(
        reporting,
        "checked_protocol",
        lambda _: (
            {"test_sha256": "test-sha", "v1_registry": []},
            {"checkpoints": [{"approach": "frozen", "seed": 42, "threshold": 0.9}]},
            [],
        ),
    )

    def measured(rows, records, threshold, archived, baseline=False):
        assert threshold == 0.9
        return {"validation_locked": {"threshold": threshold}}, []

    monkeypatch.setattr(reporting, "cached_metrics", measured)
    with pytest.raises(ValueError, match="Metric compatibility mismatch: threshold"):
        reporting.cached_comparison(directory, [], "test-sha")
