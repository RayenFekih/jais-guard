import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jais_guard.baselines.metrics import summarize
from jais_guard.baselines.prompts import build_prompt, parse_label


@pytest.mark.parametrize(
    "effort,tag", [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")]
)
def test_reasoning_answer(effort, tag):
    assert parse_label(f"reasoning</ifm|{tag}>VALID", effort, False) == "VALID"
    assert parse_label("VALID", effort, False) is None


@pytest.mark.parametrize("raw", ["valid", "VALID because...", "GIBBERISH VALID", ""])
def test_invalid_output(raw):
    assert parse_label(raw, "low", True) is None


def test_invalid_counts_as_failure():
    assert summarize([{"label": "VALID", "prediction": None}]) == {
        "examples": 1,
        "correct": 0,
        "invalid": 1,
        "incorrect": 0,
        "accuracy": 0.0,
    }


@pytest.mark.integration
def test_official_cached_template():
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "IFM/Jais-3-0.9B",
            revision="e51941b37091287fee4934e3f8a799e0d3fef37f",
            local_files_only=True,
            trust_remote_code=True,
        )
    except OSError:
        pytest.skip("Pinned official tokenizer is not cached")
    for effort, tag in [("high", "think"), ("medium", "think_fast"), ("low", "think_faster")]:
        prompt = build_prompt(tokenizer, "نحب تذكرة", effort, False)
        assert prompt.endswith(f"<ifm|{tag}>\n")
        assert build_prompt(tokenizer, "نحب تذكرة", effort, True).endswith(f"</ifm|{tag}>")
        explicit = build_prompt(tokenizer, "نحب تذكرة", effort, False, "explicit_user")
        assert "<text>\nنحب تذكرة\n</text>" in explicit
        assert explicit.endswith(f"<ifm|{tag}>\n")


def test_incorrect_labels_are_separate_from_invalid():
    summary = summarize(
        [
            {"label": "VALID", "prediction": None},
            {"label": "VALID", "prediction": "GIBBERISH"},
            {"label": "VALID", "prediction": "VALID"},
        ]
    )
    assert summary["invalid"] == 1
    assert summary["incorrect"] == 1
    assert summary["correct"] == 1
