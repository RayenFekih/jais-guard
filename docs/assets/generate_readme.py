"""Rebuild README SVGs using only stdlib and the adjacent verified metric extract."""

import csv
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INK, MUTED, BLUE, TEAL = "#262626", "#686868", "#d1d1ce", "#343f3c"


def text(x, y, value, size=18, color=INK, weight=400):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}">{escape(str(value))}</text>'
    )


def rect(x, y, w, h, fill="#f3f3f1", stroke="none"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}"/>'
    )


def save(name, height, title, subtitle, body):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
        f'viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>'
        '<g font-family="DejaVu Sans, Arial, sans-serif">'
        + rect(0, 0, 1000, height, "#ffffff")
        + text(36, 47, title, 27, weight=700)
        + text(36, 77, subtitle, 16, MUTED)
        + "".join(body)
        + "</g></svg>\n"
    )
    (HERE / name).write_text(svg)


def box(x, y, w, title, sub="", active=False):
    return (
        rect(x, y, w, 76, "#eeefeb" if active else "#f3f3f1", TEAL if active else "#d9d9d5")
        + text(x + 14, y + 31, title, 18, TEAL if active else INK, 700)
        + text(x + 14, y + 56, sub, 14, MUTED)
    )


def arrow(x, y, end):
    return (
        f'<path d="M{x} {y} H{end - 7} m-6 -5 l7 5 -7 5" '
        'fill="none" stroke="#999b96" stroke-width="2"/>'
    )


def architecture():
    b = [text(36, 120, "01  ORIGINAL JAIS 3", 15, MUTED, 700)]
    for x, w, title, sub in [
        (36, 150, "Arabic text", "Tokenizer"),
        (216, 250, "Decoder backbone", "28 layers · hidden size 1,536"),
        (496, 175, "LM head", "Vocabulary logits"),
        (701, 263, "Token generation", "Autoregressive decoding"),
    ]:
        b.append(box(x, 138, w, title, sub))
    b += [
        arrow(186, 176, 210),
        arrow(466, 176, 490),
        arrow(671, 176, 695),
        text(36, 258, "02  JAISGUARD · REPLACE THE GENERATIVE OUTPUT PATH", 15, TEAL, 700),
    ]
    for x, w, title, sub, active in [
        (36, 150, "Arabic text", "Tokenizer · ≤64", False),
        (216, 250, "Same backbone", "Final hidden representations", False),
        (496, 215, "Last non-padding", "One 1,536-dim vector", False),
        (741, 223, "New linear head", "FP32 · 1,536 → 2", True),
    ]:
        b.append(box(x, 276, w, title, sub, active))
    b += [
        arrow(186, 314, 210),
        arrow(466, 314, 490),
        arrow(711, 314, 735),
        text(741, 380, "Softmax → VALID / GIBBERISH", 14, TEAL, 700),
        text(36, 426, "03  TWO TRAINING STRATEGIES", 15, MUTED, 700),
    ]
    for x, title, lines in [
        (
            36,
            "Frozen Backbone + Head",
            [
                "FROZEN   All transformer parameters",
                "TRAIN     Classification head only",
                "3,074 trainable parameters (including bias)",
            ],
        ),
        (
            514,
            "LoRA + Head",
            [
                "FROZEN   Original backbone weights",
                "TRAIN     q_proj / v_proj adapters + head",
                "Rank 8 · alpha 16 · dropout 0.05",
            ],
        ),
    ]:
        b += [rect(x, 445, 450, 154, "#f7f7f5"), text(x + 20, 477, title, 21, INK, 700)]
        b += [
            text(x + 20, 508 + i * 29, line, 16, TEAL if i == 1 else MUTED)
            for i, line in enumerate(lines)
        ]
    save(
        "architecture.svg",
        625,
        "From Generative LLM to Binary Classifier",
        "Shared Arabic transformer · task-specific output · controlled trainable parameters",
        b,
    )


def chart_frame(title, subtitle, maximum, ticks, unit, rows, footer):
    """Editorial horizontal bars with zero-origin axes and direct end labels."""
    start, width = 350, 490
    body = []
    for value in ticks:
        x = start + value / maximum * width
        body.append(f'<path d="M{x} 105 V450" stroke="#e9e9e6" stroke-width="1"/>')
        body.append(text(x - 5, 478, value, 14, MUTED))
    for i, row in enumerate(rows):
        y = 126 + i * 92
        length = row["value"] / maximum * width
        highlight = row.get("highlight", False)
        color = TEAL if highlight else BLUE
        body += [
            text(36, y + 8, row["name"], 18, INK, 600 if highlight else 400),
            text(36, y + 33, row["condition"], 13, MUTED),
            f'<rect x="{start}" y="{y - 11}" width="{length}" height="30" fill="{color}"/>',
            text(start + length + 12, y + 11, row["label"], 18, INK, 600),
        ]
    body += [text(start, 510, unit, 15, MUTED)]
    body += [text(36, 552 + i * 25, line, 14, MUTED) for i, line in enumerate(footer)]
    return body


def quality():
    rows = json.loads((HERE / "readme-metrics.json").read_text())["quality"]
    chart_rows = [
        dict(
            r,
            value=r["macro_f1"] * 100,
            label=f"{r['macro_f1'] * 100:.2f}%",
            highlight=r["name"] == "LoRA + Head",
        )
        for r in rows
    ]
    subtitle = "Same 400 historical inputs · individual validation-selected checkpoints"
    body = chart_frame(
        "Classification quality",
        subtitle,
        100,
        [0, 25, 50, 75, 100],
        "Macro F1 (%) · higher is better",
        chart_rows,
        [
            "Frozen and LoRA: v2, locked thresholds, BF16 batch 2; not three-seed means.",
            "Untuned Jais and hosted Jev: original zero-shot protocols.",
        ],
    )
    save("quality.svg", 600, "Classification quality", subtitle, body)


def latency():
    with (ROOT / "docs/benchmarks/quality-latency/comparison.csv").open() as f:
        evidence = {row["group"]: row for row in csv.DictReader(f)}
    specs = [
        ("Jev", "Remote API · Jev 1.14.0", "jev/hosted/zero_shot", "jev_remote_mean_ms"),
        (
            "Untuned Jais baseline",
            "Local GPU · zero-shot generation",
            "untuned/untuned_prompt/zero_shot",
            "local_gpu_mean_ms",
        ),
        (
            "Frozen Backbone + Head",
            "Local GPU · v2 seed 43",
            "v2/frozen/validation_locked",
            "local_gpu_mean_ms",
        ),
        ("LoRA + Head", "Local GPU · v2 seed 44", "v2/lora/validation_locked", "local_gpu_mean_ms"),
    ]
    rows = []
    for name, condition, key, metric in specs:
        value = float(evidence[key][metric])
        rows.append(
            dict(
                name=name,
                condition=condition,
                value=value,
                label=f"{value:.2f} ms",
                highlight=name == "LoRA + Head",
            )
        )
    subtitle = "Mean end-to-end latency · 400 inputs per run · lower is better"
    body = chart_frame(
        "Inference latency",
        subtitle,
        250,
        [0, 50, 100, 150, 200, 250],
        "Mean latency (ms)",
        rows,
        [
            "Local: RTX 4050 Laptop GPU, 6 GB · batch 1 · tokenization and transfers included.",
            "Jev includes networking; different deployment paths preclude a controlled speed test.",
            "Local: 3 warmup inputs, no HTTP. Jev: no warmup. Loading excluded.",
            "Dark bar identifies the published LoRA checkpoint; frozen has the lowest latency.",
        ],
    )
    save("latency.svg", 650, "Inference latency", subtitle, body)


if __name__ == "__main__":
    architecture()
    quality()
    latency()
