"""Offline Jev reporting; reuse and revalidate previous classifier outputs, never infer."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from jais_guard.benchmarks.jev import frozen_rows, now, read_ledger, validate_response
from jais_guard.data.test_workflow import sha, write_new
from jais_guard.finetuning.evaluation import metrics
from jais_guard.finetuning.version_comparison import cached_metrics, checked_protocol, metric_match

MEASURES = (
    "accuracy",
    "macro_f1",
    "gibberish_precision",
    "gibberish_recall",
    "valid_false_rejection_rate",
    "valid_acceptance_rate",
)
SCOPE = (
    "Exploratory head-to-head comparison on a previously examined, human-reviewed, "
    "constructed Arabic-input benchmark (400 inputs; 188 VALID, 212 GIBBERISH). "
    "This is not production traffic or an unbiased final evaluation. Jev and untuned Jais "
    "are zero-shot; JaisGuard was fine-tuned specifically for this task. "
    "No test-based threshold or model selection is performed."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cached_comparison(directory, rows, test_sha):
    """Recompute archived metrics against identical text/IDs/labels and locked thresholds."""
    directory = Path(directory).resolve()
    protocol, lock, locked_rows = checked_protocol(directory.parent)
    if rows != locked_rows or protocol["test_sha256"] != test_sha:
        raise ValueError("Cached comparison uses different frozen data")
    completion = read_json(directory / "completion.json")
    if completion["status"] != "complete" or completion["test_sha256"] != test_sha:
        raise ValueError("Cached comparison is incomplete or uses different data")
    archived = read_json(directory / "metrics.json")
    expected = {(v, a, s) for v in ("v1", "v2") for a in ("frozen", "lora") for s in (42, 43, 44)}
    actual = set()
    grouped = defaultdict(list)
    files = {str(p): sha(p) for p in directory.rglob("*") if p.is_file()}
    for name in ("protocol.json", "threshold_lock.json"):
        p = directory.parent / name
        files[str(p)] = sha(p)
    thresholds = {
        (version, item["approach"], item["seed"]): item["threshold"]
        for version, entries in (("v1", protocol["v1_registry"]), ("v2", lock["checkpoints"]))
        for item in entries
    }
    baseline_count = 0
    for entry in archived["per_seed"]:
        version, approach, seed = entry["version"], entry["approach"], entry["seed"]
        baseline = version == "untuned"
        if baseline:
            baseline_count += 1
        else:
            key = (version, approach, seed)
            if key in actual or key not in expected:
                raise ValueError("Unexpected or repeated cached seed")
            actual.add(key)
        source = (
            directory / f"{approach}-{seed}" if version == "v2" else Path(entry["source_directory"])
        )
        for name in ("predictions.jsonl", "metrics.json"):
            files[str((source / name).resolve())] = sha(source / name)
        locked = None if baseline else thresholds[(version, approach, seed)]
        measured, _ = cached_metrics(
            rows,
            read_lines(source / "predictions.jsonl"),
            locked,
            read_json(source / "metrics.json"),
            baseline=baseline,
        )
        for mode, values in measured.items():
            metric_match(entry["metrics"][mode], values)
            grouped[f"{version}/{approach}/{mode}"].append(values)
    if actual != expected or baseline_count != 1 or len(archived["per_seed"]) != 13:
        raise ValueError("Expected twelve cached classifiers and one untuned baseline")
    groups = {}
    for key, values in grouped.items():
        groups[key] = {}
        for measure in MEASURES:
            numbers = [r[measure] for r in values]
            summary = {
                "mean": float(np.mean(numbers)),
                "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
            }
            metric_match(archived["groups"][key][measure], summary)
            groups[key][measure] = summary
        matrices = np.array([r["confusion"] for r in values])
        groups[key]["confusion"] = {
            "mean_per_seed": matrices.mean(axis=0).tolist(),
            "sum_repeated_seed_decisions": matrices.sum(axis=0).tolist(),
        }
        if groups[key]["confusion"] != archived["groups"][key]["confusion"]:
            raise ValueError("Cached confusion aggregation differs")
    return {
        "groups": groups,
        "latency": archived["latency"],
        "files": files,
        "std_ddof": 1,
        "source": str(directory),
    }


def latency(values):
    values = [v for v in values if v is not None]
    if not values:
        return {k: None for k in ("mean_ms", "median_ms", "p50_ms", "p95_ms", "p99_ms")}
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        **{f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 95, 99)},
    }


def summarize(directory, allow_incomplete=False):
    manifest, examples, attempts = read_ledger(directory)
    config = manifest["config"]
    if config["stage"] != "evaluate":
        raise ValueError("Comparison reports require an evaluate run, not smoke")
    rows = frozen_rows(config["test"])
    if rows != [json.loads(e["row_json"]) for e in examples]:
        raise ValueError("Prediction checkpoint differs from frozen dataset")
    for path, expected in config["test_files"].items():
        if sha(path) != expected:
            raise ValueError("Frozen dataset changed since run initialization")
    cached = cached_comparison(config["comparison"], rows, sha(Path(config["test"]) / "test.jsonl"))
    if cached["files"] != config["comparison_files"]:
        raise ValueError("Stored comparison artifacts changed since run initialization")
    unfinished = [e for e in examples if e["status"] not in {"success", "failed"}]
    if unfinished and not allow_incomplete:
        raise ValueError(
            "Unfinished/uncertain inputs remain; resume or explicitly use --allow-incomplete"
        )
    by_id = defaultdict(list)
    for attempt in attempts:
        by_id[attempt["id"]].append(attempt)
    predictions = []
    for row, record in zip(rows, examples, strict=True):
        history = by_id[row["id"]]
        success = record["status"] == "success"
        response = (
            validate_response(
                json.loads(record["response_json"]), config["question"]["question_id"]
            )
            if success
            else None
        )
        if success and (
            not history
            or history[-1]["error_type"] is not None
            or history[-1]["response_json"] != record["response_json"]
        ):
            raise ValueError("Successful checkpoint has inconsistent attempt history")
        probability = (
            response["answers"][config["question"]["question_id"]]["noul"] if success else None
        )
        predicted = (0 if probability >= 0.5 else 1) if success else -1
        predictions.append(
            {
                "id": row["id"],
                "input_text": row["input_text"],
                "label": row["label"],
                "truth_id": int(row["label"] == "GIBBERISH"),
                "prediction_id": predicted,
                "prediction": {0: "VALID", 1: "GIBBERISH", -1: "INVALID"}[predicted],
                "p_yes": probability,
                "p_gibberish": 1 - probability if success else None,
                "model": response["model"] if success else None,
                "status": record["status"],
                "error": record["last_error"],
                "attempts": len(history),
                "retries": max(0, len(history) - 1),
                "api_latency_ms": history[-1]["latency_ms"] if success else None,
                "last_invocation_elapsed_ms": record["total_ms"],
                "usage": response["usage"] if success else None,
                "dialect": row.get("dialect"),
                "subtype": row.get("subtype"),
            }
        )
    measured = metrics(
        [p["truth_id"] for p in predictions], [p["prediction_id"] for p in predictions]
    )
    successful = [p for p in predictions if p["status"] == "success"]
    measured["response_success_rate"] = len(successful) / len(rows)
    measured["valid_invalid_outputs"] = sum(
        p["truth_id"] == 0 and p["prediction_id"] == -1 for p in predictions
    )
    measured["valid_invalid_rate"] = measured["valid_invalid_outputs"] / measured["valid_support"]
    http = [a for a in attempts if a["status_code"] is not None]
    http_errors = sum(not 200 <= a["status_code"] < 300 for a in http)
    usage = Counter(input_tokens=0, output_tokens=0)
    known_cost, priced_attempts = 0.0, 0
    usage_attempts = 0
    for attempt in attempts:
        if attempt["usage_json"]:
            tokens = json.loads(attempt["usage_json"])
            usage.update(tokens)
            usage_attempts += 1
            safe = json.loads(attempt["response_json"]) if attempt["response_json"] else None
            rate = config["question"]["pricing"]["models"].get(safe["model"]) if safe else None
            if rate:
                known_cost += sum(
                    tokens[f"{kind}_tokens"] * rate[f"{kind}_per_million"] / 1_000_000
                    for kind in ("input", "output")
                )
                priced_attempts += 1
    reliability = {
        "examples": len(rows),
        "successful_examples": len(successful),
        "statuses": dict(Counter(e["status"] for e in examples)),
        "total_api_requests_journaled": len(attempts),
        "http_responses": len(http),
        "http_errors": http_errors,
        "http_error_rate_per_response": http_errors / len(http) if http else None,
        "http_error_rate_per_attempt": http_errors / len(attempts) if attempts else None,
        "retries": sum(max(0, len(v) - 1) for v in by_id.values()),
        "attempt_errors": dict(Counter(a["error_type"] for a in attempts if a["error_type"])),
        "transport_errors": sum(a["error_type"] == "transport_error" for a in attempts),
        "unknown_outcome_attempts": sum(a["ended_at"] is None for a in attempts),
    }
    costs = {
        **dict(usage),
        "attempts_with_reported_usage": usage_attempts,
        "attempts_without_usage": len(attempts) - usage_attempts,
        "known_cost_subtotal_usd": known_cost,
        "estimated_total_cost_usd": known_cost if priced_attempts == len(attempts) else None,
        "unpriced_attempts": len(attempts) - priced_attempts,
        "pricing": config["question"]["pricing"],
        "note": "Token-based estimate only; unknown/error request billing, taxes and "
        "account terms are not inferred.",
    }
    result = {
        "scope": SCOPE,
        "created_at": now(),
        "complete": len(successful) == len(rows),
        "protocol_sha256": manifest["protocol_sha256"],
        "metrics": measured,
        "invalid_output_policy": "All 400 inputs remain in the denominator; "
        "failed/missing responses "
        "are INVALID (-1), wrong for accuracy and true-class recall/F1. "
        "VALID false rejection counts explicit GIBBERISH only; acceptance "
        "counts explicit VALID only. Invalid VALID rate is reported "
        "separately; acceptance is not necessarily 1 - FRR.",
        "confusion_axes": {
            "rows": ["VALID", "GIBBERISH"],
            "columns": ["VALID", "GIBBERISH", "INVALID"],
        },
        "latency": {
            "successful_request": latency([p["api_latency_ms"] for p in successful]),
            "all_completed_attempts": latency([a["latency_ms"] for a in attempts]),
            "last_invocation_per_example": latency(
                [p["last_invocation_elapsed_ms"] for p in predictions]
            ),
        },
        "latency_policy": "Client wall time for HTTP POST including network, server queue and "
        "first-connection TLS; excludes local preparation/reporting. Last "
        "invocation includes retries/backoff but not earlier invocations. No "
        "warm-up requests. Not equivalent to cached GPU model-only timings.",
        "reliability": reliability,
        "usage_and_cost": costs,
        "returned_models": dict(Counter(p["model"] for p in successful)),
        "comparison": cached,
    }
    return result, predictions


def formatted(value):
    return f"{value['mean']:.4f}" + (f" ± {value['std']:.4f}" if value["std"] is not None else "")


def save_report(directory, allow_incomplete=False):
    result, predictions = summarize(directory, allow_incomplete)
    target = (
        Path(directory)
        / "reports"
        / (now().replace(":", "").replace("+", "_") + "-" + uuid4().hex[:8])
    )
    target.mkdir(parents=True, exist_ok=False)
    write_new(target / "metrics.json", result)
    for name, records in (
        ("predictions.jsonl", predictions),
        ("errors.jsonl", [p for p in predictions if p["prediction_id"] != p["truth_id"]]),
    ):
        with (target / name).open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (target / "predictions.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    lines = [
        "# Jev versus JaisGuard",
        "",
        SCOPE,
        "",
        f"Complete: **{result['complete']}**. "
        "All 400 examples are scored, including failed responses.",
        "",
        "| Model / operating point | Accuracy | Macro F1 | G precision | G "
        "recall | VALID FRR | VALID acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    values = [formatted({"mean": result["metrics"][m], "std": None}) for m in MEASURES]
    lines.append("| Jev / fixed yes ≥ 0.5 | " + " | ".join(values) + " |")
    for name, group in result["comparison"]["groups"].items():
        lines.append("| " + name + " | " + " | ".join(formatted(group[m]) for m in MEASURES) + " |")
    lines.extend(
        [
            "",
            "Fine-tuned rows show means ± sample standard deviations across "
            "seeds 42, 43 and 44. Operating points were fixed before these API "
            "calls. No winner is selected.",
            "",
            "## Latency (milliseconds)",
            "",
            "Jev client API response latency: "
            + json.dumps(result["latency"]["successful_request"]),
            "",
            "| Cached local model | Mean classify ms | Load seconds |",
            "|---|---:|---:|",
        ]
    )
    for name, timing in result["comparison"]["latency"].items():
        lines.append(
            f"| {name} | {formatted(timing['classify_ms'])} | {formatted(timing['load_seconds'])} |"
        )
    lines.extend(
        [
            "",
            result["latency_policy"],
            "",
            "## Reliability, usage and cost",
            "",
            "```json",
            json.dumps(
                {k: result[k] for k in ("returned_models", "reliability", "usage_and_cost")},
                indent=2,
            ),
            "```",
            "",
            "Pricing snapshot: [TypeSafe "
            "models](https://docs.typesafe.ai/models), verified 2026-10-30. "
            "Usage includes all attempts with reported token counts; unknown "
            "billed usage is not estimated.",
            "",
            "## Errors and confusion matrices",
            "",
            result["invalid_output_policy"],
            "",
            "Jev confusion (rows VALID/GIBBERISH, columns VALID/GIBBERISH/INVALID):",
            "",
            "```json",
            json.dumps(result["metrics"]["confusion"]),
            "```",
            "",
            "Cached confusion matrices (mean per seed and sum of repeated seed "
            "decisions; the latter is not a larger independent test):",
            "",
            "```json",
            json.dumps(
                {k: v["confusion"] for k, v in result["comparison"]["groups"].items()}, indent=2
            ),
            "```",
            "",
            "See predictions.csv, predictions.jsonl and errors.jsonl for "
            "per-input probabilities, failures, dialect/subtype and latency. No "
            "API calls occur during reporting.",
            "",
            "## Practical trade-offs",
            "",
            "Jev requires no task training or local GPU, but sends input text to "
            "a third party and depends on network, service availability, rate "
            "limits and token pricing. A pinned version improves "
            "reproducibility; aliases can change. JaisGuard runs locally and "
            "avoids per-request API charges, but requires task-specific "
            "annotation/training and GPU resources. Cached local timings and "
            "measured API latency use different timing boundaries and hardware. "
            "This constructed benchmark does not establish production costs, "
            "dialect prevalence or production reliability.",
            "",
            "The manifest records the exact question, threshold, version, "
            "execution settings, input hashes and cached comparison hashes. "
            "Historical datasets, models and MLflow records are reused read-only.",
        ]
    )
    (target / "report.md").write_text("\n".join(lines) + "\n")
    return target
