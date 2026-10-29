"""One evaluator for default, exploratory threshold, and prompt predictions."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth) or len(truth) != len(prediction):
        raise ValueError("Empty or mismatched predictions")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [-1, 0, 1]).all():
        raise ValueError("Invalid labels")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=[0, 1], zero_division=0
    )
    valid = truth == 0
    n = int(valid.sum())
    rejected = int(((prediction == 1) & valid).sum())
    rate = rejected / n if n else None
    # Wilson 95% interval: descriptive uncertainty, not a production guarantee.
    z = 1.959963984540054
    center = (rate + z * z / (2 * n)) / (1 + z * z / n) if n else None
    half = (
        z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n) if n else None
    )
    result = {
        "examples": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1.mean()),
        "gibberish_recall": float(recall[1]),
        "valid_false_rejections": rejected,
        "valid_support": n,
        "valid_false_rejection_rate": rate,
        "valid_frr_wilson95_low": center - half if n else None,
        "valid_frr_wilson95_high": center + half if n else None,
        "valid_acceptance_rate": float((prediction[valid] == 0).mean()) if n else None,
        "invalid": int((prediction == -1).sum()),
        "confusion_labels": ["VALID", "GIBBERISH", "INVALID"],
        "confusion": confusion_matrix(truth, prediction, labels=[0, 1, -1])[:2].tolist(),
    }
    for i, name in enumerate(("valid", "gibberish")):
        result.update(
            {
                f"{name}_precision": float(precision[i]),
                f"{name}_recall": float(recall[i]),
                f"{name}_f1": float(f1[i]),
                f"{name}_support": int(support[i]),
            }
        )
    return result


def threshold_report(truth, scores, limit=0.02):
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(truth) or not np.isfinite(scores).all():
        raise ValueError("Missing or nonfinite scores")
    if ((scores < 0) | (scores > 1)).any() or not 0 <= limit <= 1:
        raise ValueError("Invalid probability or FRR limit")
    # >= semantics: nextafter(max) is the accept-all endpoint, even if max == 1.
    thresholds = sorted(
        set([0.0, 0.5, *scores.tolist(), float(np.nextafter(scores.max(), np.inf))])
    )
    sweep = [{"threshold": t, **metrics(truth, (scores >= t).astype(int))} for t in thresholds]
    feasible = [
        r
        for r in sweep
        if r["valid_false_rejection_rate"] is not None and r["valid_false_rejection_rate"] <= limit
    ]
    if not feasible:
        raise ValueError("Threshold tuning needs VALID examples")
    selected = max(
        feasible,
        key=lambda r: (r["gibberish_recall"], -r["valid_false_rejections"], r["threshold"]),
    )
    return {
        "default": metrics(truth, (scores >= 0.5).astype(int)),
        "exploratory": selected,
        "exploratory_frr_limit": limit,
        "selection_data": "validation; not independent test performance",
        "sweep": sweep,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def save_report(directory, rows, scores=None, predictions=None, limit=0.02):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth = [int(r["label"] == "GIBBERISH") for r in rows]
    if scores is not None:
        report = threshold_report(truth, scores, limit)
        default = (np.asarray(scores) >= 0.5).astype(int)
        tuned = (np.asarray(scores) >= report["exploratory"]["threshold"]).astype(int)
        write_csv(
            directory / "thresholds.csv",
            [{k: v for k, v in r.items() if not isinstance(v, list)} for r in report["sweep"]],
        )
    else:
        default = np.asarray(predictions)
        tuned = default
        report = {"default": metrics(truth, default)}
    records, errors = [], []
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        record = {
            **row,
            "truth_id": truth[i],
            "p_gibberish": float(scores[i]) if scores is not None else None,
            "prediction_default": int(default[i]),
            "prediction_exploratory": int(tuned[i]),
        }
        records.append(record)
        if default[i] != truth[i] or tuned[i] != truth[i]:
            errors.append(record)
        for field in ("subtype", "dialect", "source_method", "label_origin"):
            groups[(field, str(row.get(field, "unknown")))].append(i)
        if "input_tokens" in row:
            length_bin = (
                "1-8"
                if row["input_tokens"] <= 8
                else ("9-16" if row["input_tokens"] <= 16 else "17+")
            )
            groups[("input_length", length_bin)].append(i)
    report["subgroups"] = {
        f"{field}:{value}": {
            "default": metrics([truth[i] for i in ix], default[ix]),
            "exploratory": metrics([truth[i] for i in ix], tuned[ix]),
        }
        for (field, value), ix in groups.items()
    }
    for name, items in (("predictions", records), ("errors", errors)):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in items)
        )
        # Preserve provenance in JSONL; flatten only useful inspection columns in CSV.
        write_csv(
            directory / f"{name}.csv",
            [
                {
                    k: r.get(k)
                    for k in (
                        "id",
                        "input_text",
                        "label",
                        "truth_id",
                        "p_gibberish",
                        "prediction_default",
                        "prediction_exploratory",
                        "subtype",
                        "dialect",
                        "dialect_status",
                        "source_method",
                        "label_origin",
                        "input_tokens",
                        "raw_output",
                        "parse_status",
                        "finish_reason",
                    )
                }
                for r in items
            ],
        )
    write_json(directory / "metrics.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in ("default", "exploratory"):
        if mode not in report:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        matrix = np.asarray(report[mode]["confusion"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(
            xticks=[0, 1, 2],
            xticklabels=["VALID", "GIBBERISH", "INVALID"],
            yticks=[0, 1],
            yticklabels=["VALID", "GIBBERISH"],
            xlabel="Prediction",
            ylabel="Truth",
            title=mode,
        )
        fig.tight_layout()
        fig.savefig(directory / f"confusion_{mode}.png")
        plt.close(fig)
    if scores is not None:
        fig, ax = plt.subplots()
        ax.plot(
            [r["valid_false_rejection_rate"] for r in report["sweep"]],
            [r["gibberish_recall"] for r in report["sweep"]],
            marker=".",
        )
        ax.set(
            xlabel="VALID false-rejection rate",
            ylabel="Gibberish recall",
            title="Validation threshold trade-off (exploratory)",
        )
        fig.savefig(directory / "threshold_tradeoff.png")
        plt.close(fig)
    summary = [
        "# Validation report",
        "",
        "Validation-selected results; no independent test estimate.",
        "",
        "| Operating point | Accuracy | Macro F1 | Gibberish recall | VALID FRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("default", "exploratory"):
        if mode in report:
            values = report[mode]
            frr = values["valid_false_rejection_rate"]
            frr_text = f"{frr:.4f}" if frr is not None else "N/A"
            summary.append(
                f"| {mode} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} "
                f"| {values['gibberish_recall']:.4f} | {frr_text} |"
            )
    if scores is not None:
        summary.extend(
            [
                "",
                f"Exploratory threshold: {report['exploratory']['threshold']:.8g}; "
                f"observed FRR limit {limit:.1%}. This is not a production requirement.",
                "See thresholds.csv and threshold_tradeoff.png for all operating points.",
            ]
        )
    summary.extend(
        [
            "",
            "## Error inspection",
            "",
            "All errors and provenance are in errors.jsonl / errors.csv. "
            "Rows below prioritize false rejection of VALID text. "
            "Dialect and difficulty metadata may be proposed rather than human-confirmed.",
            "",
        ]
    )
    ordered = sorted(errors, key=lambda r: r["truth_id"])
    for row in ordered[:20]:
        text = row["input_text"].replace("\n", " ")
        summary.append(
            f"- {row['id']}: truth={row['label']}, "
            f"default={row['prediction_default']}, "
            f"exploratory={row['prediction_exploratory']}: {text}"
        )
    if not ordered:
        summary.append("No observed validation errors at these operating points.")
    (directory / "report.md").write_text("\n".join(summary) + "\n")
    return report
