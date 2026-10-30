import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.baselines.report import evaluate, write_report


def test_invalid_predictions_are_failures_without_becoming_wrong_labels():
    rows = [
        {"label": "VALID", "prediction": None},
        {"label": "GIBBERISH", "prediction": "GIBBERISH"},
        {"label": "VALID", "prediction": "GIBBERISH"},
        {"label": "GIBBERISH", "prediction": "VALID"},
    ]
    metrics = evaluate(rows)
    assert metrics["accuracy"] == 0.25
    assert metrics["invalid"] == 1
    assert metrics["incorrect"] == 2
    assert metrics["gibberish_recall"] == 0.5
    assert metrics["valid_acceptance_rate"] == 0
    assert metrics["valid_false_rejection_rate"] == 0.5
    assert metrics["macro_f1"] == 0.25
    assert metrics["confusion"]["VALID"]["INVALID"] == 1
    invalid_gibberish = evaluate(
        [
            {"label": "GIBBERISH", "prediction": None},
            {"label": "VALID", "prediction": "VALID"},
        ]
    )
    assert invalid_gibberish["gibberish_precision"] is None
    assert invalid_gibberish["macro_f1"] == 0.5


def test_report_writes_inspectable_artifacts_without_jev(tmp_path):
    row = {
        "id": "sample",
        "text": "مرحبا",
        "label": "VALID",
        "prediction": "VALID",
        "raw_output": "</ifm|think_faster>VALID",
        "generated_tokens": 4,
        "input_tokens": 10,
        "latency_ms": 12,
        "subtype": "short_topic",
        "dialect": "MSA",
    }
    report = {
        "device": "cpu",
        "split": "smoke",
        "variants": {
            "explicit_user": {"results": [row]},
        },
    }
    write_report(report, tmp_path, tmp_path / "absent.zip")
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["metrics"]["explicit_user"]["overall"]["accuracy"] == 1
    assert (
        json.loads((tmp_path / "detailed_results.jsonl").read_text())["raw_output"]
        == row["raw_output"]
    )
    assert "Examples per variant: 1" in (tmp_path / "summary.md").read_text()
    assert (tmp_path / "detailed_results.csv").exists()


def test_binary_macro_f1_agrees_with_sklearn():
    from sklearn.metrics import f1_score

    rows = [
        {"label": truth, "prediction": prediction}
        for truth, prediction in [
            ("VALID", "VALID"),
            ("VALID", "GIBBERISH"),
            ("GIBBERISH", "GIBBERISH"),
            ("GIBBERISH", None),
        ]
    ]
    expected = f1_score(
        [row["label"] for row in rows],
        [row["prediction"] or "INVALID" for row in rows],
        labels=["VALID", "GIBBERISH"],
        average="macro",
        zero_division=0,
    )
    assert evaluate(rows)["macro_f1"] == pytest.approx(expected)
