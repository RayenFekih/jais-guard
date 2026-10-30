"""Small, file-based evaluation reports for the existing baseline runner."""

import csv
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from jais_guard.baselines.metrics import summarize


def evaluate(rows: list[dict]) -> dict:
    metrics = summarize(rows)
    counts = Counter((row["label"], row["prediction"]) for row in rows)
    tp = counts["GIBBERISH", "GIBBERISH"]
    fp = counts["VALID", "GIBBERISH"]
    tn = counts["VALID", "VALID"]
    fn = counts["GIBBERISH", "VALID"]
    invalid_valid = counts["VALID", None]
    invalid_gibberish = counts["GIBBERISH", None]
    valid_total = tn + fp + invalid_valid
    gibberish_total = tp + fn + invalid_gibberish

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    metrics.update(
        {
            "gibberish_precision": ratio(tp, tp + fp),
            "gibberish_recall": ratio(tp, gibberish_total),
            "valid_false_rejection_rate": ratio(fp, valid_total),
            "valid_acceptance_rate": ratio(tn, valid_total),
            # Invalid responses are false negatives for their true class in F1.
            "macro_f1": (
                (ratio(2 * tp, 2 * tp + fp + fn + invalid_gibberish) or 0)
                + (ratio(2 * tn, 2 * tn + fp + fn + invalid_valid) or 0)
            )
            / 2,
            "confusion": {
                "VALID": {"VALID": tn, "GIBBERISH": fp, "INVALID": invalid_valid},
                "GIBBERISH": {"VALID": fn, "GIBBERISH": tp, "INVALID": invalid_gibberish},
            },
        }
    )
    for field in ("input_tokens", "generated_tokens"):
        if all(field in row for row in rows):
            metrics[field] = {
                "total": sum(row[field] for row in rows),
                "mean": statistics.mean(row[field] for row in rows),
            }
    if all("latency_ms" in row for row in rows):
        times = [row["latency_ms"] for row in rows]
        metrics["latency_ms"] = {
            "mean": statistics.mean(times),
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=100, method="inclusive")[94]
            if len(times) > 1
            else times[0],
        }
    return metrics


def write_report(report: dict, directory: Path, jev_archive: Path) -> None:
    """Derive artifacts from saved predictions; never call a model."""
    directory.mkdir(parents=True, exist_ok=True)
    flat = []
    summaries = {}
    for variant, run in report["variants"].items():
        rows = run["results"]
        summaries[variant] = {"overall": evaluate(rows)}
        for field in ("query_type", "subtype", "dialect"):
            summaries[variant][field] = {
                value: evaluate([row for row in rows if row.get(field) == value])
                for value in sorted({row[field] for row in rows if field in row})
            }
        flat.extend({"variant": variant, **row} for row in rows)
    reference = next(iter(report["variants"].values()))["results"]
    if jev_archive.exists() and report["split"] == "dev":
        source = {row["id"]: row for row in reference}
        with zipfile.ZipFile(jev_archive) as archive:
            historical = [
                json.loads(line)
                for line in archive.read("jev_results/detailed_results.jsonl").decode().splitlines()
            ]
        matched = []
        for row in historical:
            if row["id"] not in source:
                continue
            original = source[row["id"]]
            if (
                row["input_text"] != original["text"]
                or row["true_is_gibberish"] != original["is_gibberish"]
            ):
                raise ValueError(f"Jev source mismatch: {row['id']}")
            matched.append(
                {
                    **original,
                    "prediction": None
                    if row["error_type"]
                    else ("GIBBERISH" if row["prob_gibberish"] >= 0.5 else "VALID"),
                    "latency_ms": row["latency_ms"],
                    "input_tokens": row["input_tokens"],
                    "generated_tokens": row["output_tokens"],
                }
            )
        if len(matched) != len(source):
            raise ValueError("Incomplete Jev ID match")
        summaries["jev_historical_threshold_0.5"] = {"overall": evaluate(matched)}
    (directory / "summary.json").write_text(
        json.dumps(
            {"run": {k: v for k, v in report.items() if k != "variants"}, "metrics": summaries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (directory / "detailed_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in flat)
    )
    with (directory / "detailed_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(dict.fromkeys(key for row in flat for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    def percent(value):
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "# JaisGuard baseline development evaluation",
        "",
        f"Device: `{report['device']}`. Split: `{report['split']}`. "
        f"Examples per variant: {len(reference)}.",
        "",
        "Model/configuration and package versions are in summary.json; "
        "rendered prompts and raw outputs are in detailed_results.jsonl/CSV.",
        "",
        "| Variant | Accuracy | Macro-F1 | Gibberish precision | Recall | Valid false rejection | "
        "Valid acceptance | Invalid | Wrong labels |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, groups in summaries.items():
        m = groups["overall"]
        values = [
            percent(m[key])
            for key in (
                "accuracy",
                "macro_f1",
                "gibberish_precision",
                "gibberish_recall",
                "valid_false_rejection_rate",
                "valid_acceptance_rate",
            )
        ]
        lines.append(
            f"| {name} | " + " | ".join(values) + f" | {m['invalid']} | {m['incorrect']} |"
        )
    lines += [
        "",
        "Invalid responses count as failures in accuracy, recall, acceptance and macro-F1. "
        "False rejection counts explicit GIBBERISH predictions on VALID inputs; invalid VALID "
        "responses are reported separately in the confusion tables. Undefined precision is n/a.",
        "",
        "## Latency and tokens",
        "",
        "| Variant | Mean ms | P50 ms | P95 ms | Input tokens | Generated tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, groups in summaries.items():
        m = groups["overall"]
        t = m["latency_ms"]
        lines.append(
            f"| {name} | {t['mean']:.1f} | {t['p50']:.1f} | {t['p95']:.1f} | "
            f"{m['input_tokens']['total']} | {m['generated_tokens']['total']} |"
        )
    lines += [
        "",
        "Jais timings include formatting, tokenization, generation and parsing, exclude "
        "model loading, and include the first request (no warmup). Calls are sequential. "
        "Jev timings are historical remote calls with unspecified timing boundaries. "
        "Hardware, prompts and inference interfaces differ; these are not comparable speed "
        "or cost measurements. No local cost estimate is made.",
    ]
    for variant, run in report["variants"].items():
        lines += [
            "",
            f"## {variant}",
            "",
            "Confusion: rows are truth, columns are prediction.",
            "",
            "| Truth | VALID | GIBBERISH | INVALID |",
            "| --- | --- | --- | --- |",
        ]
        for truth, predictions in summaries[variant]["overall"]["confusion"].items():
            lines.append(f"| {truth} | " + " | ".join(str(n) for n in predictions.values()) + " |")
        for field in ("query_type", "subtype", "dialect"):
            lines += [
                "",
                f"### By {field}",
                "",
                "| Group | N | Correct | Invalid | Wrong labels | Accuracy |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for value, m in summaries[variant][field].items():
                lines.append(
                    f"| {value} | {m['examples']} | {m['correct']} | {m['invalid']} | "
                    f"{m['incorrect']} | {percent(m['accuracy'])} |"
                )
        lines += [
            "",
            "### Failures",
            "",
            "| ID | Type | Truth | Prediction | Input |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in run["results"]:
            if row["prediction"] != row["label"]:
                kind = "invalid output" if row["prediction"] is None else "incorrect label"
                text = row["text"].replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {row['id']} | {kind} | {row['label']} | "
                    f"{row['prediction'] or 'INVALID'} | {text} |"
                )
    lines += [
        "",
        "This is development data used for prompt comparison, not a held-out test "
        "claim. Labels and manual-review flags are preserved; no test inference, tuning, "
        "fine-tuning or inference optimization was performed.",
        "",
    ]
    (directory / "summary.md").write_text("\n".join(lines))
