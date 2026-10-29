"""Independent-test evaluation at precommitted operating points. Never tunes thresholds."""

import gc
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jais_guard.test_workflow import read_frozen_test, sha

from .evaluation import metrics, write_csv, write_json

MEASURES = (
    "accuracy",
    "macro_f1",
    "gibberish_recall",
    "valid_false_rejection_rate",
    "valid_acceptance_rate",
    "valid_precision",
    "valid_recall",
    "gibberish_precision",
)
DESCRIPTION = (
    "Human-reviewed, constructed Arabic-input benchmark; not observed production traffic. "
    "One authoring source; dialect hints are not independently verified. "
    "Class balance and corruption mix are designed, not measured production frequencies."
)


def locked_registry(folder):
    """Fail closed on missing/duplicate entries, unpinned loader files or ambiguous thresholds."""
    folder = Path(folder)
    rows = read_frozen_test(folder)  # Verifies export, manifests, checkpoint and prompt hashes.
    frozen = json.loads((folder / "freeze.json").read_text())
    if frozen["version"] != "test-400-v1" or not frozen.get(
        "evaluation_allowed_only_with_locked_protocol"
    ):
        raise ValueError("Expected the locked test-400-v1 evaluation protocol")
    if Counter(r["label"] for r in rows) != {"VALID": 188, "GIBBERISH": 212}:
        raise ValueError("Unexpected frozen test composition")
    if len({r["id"] for r in rows}) != 400 or len({r["family_id"] for r in rows}) != 400:
        raise ValueError("Duplicate frozen ID or scored family")
    registry = []
    seen = set()
    for entry in frozen["selection_lock"]["checkpoints"]:
        key = (entry["approach"], entry["seed"])
        if key in seen:
            raise ValueError("Duplicate locked checkpoint")
        seen.add(key)
        files = {Path(p).resolve(): h for p, h in entry["files"].items()}
        metadata_paths = [p for p in files if p.name == "classifier.json"]
        threshold_paths = [p for p in files if p.name == "threshold.json"]
        if len(metadata_paths) != 1 or len(threshold_paths) != 1:
            raise ValueError("Ambiguous locked classifier/threshold mapping")
        metadata_path, threshold_path = metadata_paths[0], threshold_paths[0]
        checkpoint = metadata_path.parent
        if threshold_path != checkpoint.parent / "threshold.json":
            raise ValueError("Threshold does not belong to the locked checkpoint")
        required = {metadata_path, checkpoint / "head.safetensors", threshold_path}
        required.update(p.resolve() for p in (checkpoint / "tokenizer").rglob("*") if p.is_file())
        if entry["approach"] == "lora":
            required |= {
                checkpoint / "adapter/adapter_config.json",
                checkpoint / "adapter/adapter_model.safetensors",
            }
        if not (checkpoint / "tokenizer/tokenizer.json").is_file() or not required <= files.keys():
            raise ValueError("Checkpoint loader would consume unpinned files")
        metadata = json.loads(metadata_path.read_text())
        config = metadata["config"]
        if (metadata["approach"], metadata["seed"]) != key:
            raise ValueError("Locked checkpoint identity mismatch")
        if (
            metadata["label_mapping"] != {"VALID": 0, "GIBBERISH": 1}
            or metadata["pooling"] != "last_nonpadding_final_hidden"
        ):
            raise ValueError("Unexpected classifier label mapping or pooling")
        for field in ("model", "revision", "train_sha256", "validation_sha256"):
            if config[field] != entry[field]:
                raise ValueError(f"Locked checkpoint configuration mismatch: {field}")
        for split in ("train", "validation"):
            if sha(config[f"{split}_path"]) != entry[f"{split}_sha256"]:
                raise ValueError(f"Selected {split} export changed")
        threshold = json.loads(threshold_path.read_text())
        value = threshold.get("exploratory")
        if (
            threshold.get("default") != 0.5
            or threshold.get("selection_data") != "validation"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= np.nextafter(1.0, np.inf)
        ):
            raise ValueError("Invalid locked validation-selected threshold")
        if not isinstance(config["max_length"], int) or config["max_length"] < 1:
            raise ValueError("Invalid locked token budget")
        registry.append(
            {
                "approach": key[0],
                "seed": key[1],
                "checkpoint": str(checkpoint),
                "threshold": value,
                "threshold_path": str(threshold_path),
                "metadata": metadata,
                "locked_files": {str(p): h for p, h in files.items()},
                "source_parent_run_id": checkpoint.parent.parent.name,
            }
        )
    if seen != {(a, s) for a in ("frozen", "lora") for s in (42, 43, 44)}:
        raise ValueError("Exactly the six preselected checkpoints are required")
    identities = {
        (
            r["metadata"]["config"]["model"],
            r["metadata"]["config"]["revision"],
            r["metadata"]["config"]["train_sha256"],
            r["metadata"]["config"]["validation_sha256"],
        )
        for r in registry
    }
    if len(identities) != 1:
        raise ValueError("Locked runs do not share model and dataset versions")
    return rows, frozen, sorted(registry, key=lambda r: (r["approach"], r["seed"]))


def token_preflight(rows, registry):
    """Tokenization only; rejects over-budget inputs without altering text or token policy."""
    from transformers import AutoTokenizer

    from .data import encode_rows

    summaries, cache = {}, {}
    for item in registry:
        checkpoint = Path(item["checkpoint"])
        key = (
            sha(checkpoint / "tokenizer/tokenizer.json"),
            sha(checkpoint / "tokenizer/tokenizer_config.json"),
            item["metadata"]["config"]["max_length"],
        )
        if key not in cache:
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint / "tokenizer", local_files_only=True
            )
            encoded = encode_rows(rows, tokenizer, key[-1])
            lengths = [len(r["input_ids"]) for r in encoded]
            cache[key] = {
                "max_tokens": max(lengths),
                "median_tokens": float(np.median(lengths)),
                "max_length": key[-1],
                "truncation": False,
            }
        summaries[f"{item['approach']}-{item['seed']}"] = cache[key]
    return summaries


def fixed_report(directory, rows, scores, threshold):
    """Compute only 0.5 and the saved validation threshold; no sweep/selection API."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    scores = np.asarray(scores, dtype=float)
    if (
        len(scores) != len(rows)
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Missing or invalid probabilities")
    truth = np.array([int(r["label"] == "GIBBERISH") for r in rows])
    predictions = {
        "default": (scores >= 0.5).astype(int),
        "validation_locked": (scores >= threshold).astype(int),
    }
    report = {
        mode: {"threshold": 0.5 if mode == "default" else threshold, **metrics(truth, pred)}
        for mode, pred in predictions.items()
    }
    report["threshold_selection"] = "Previously locked validation data only; no test tuning"
    records = [
        {
            **r,
            "truth_id": int(truth[i]),
            "p_gibberish": float(scores[i]),
            **{f"prediction_{mode}": int(pred[i]) for mode, pred in predictions.items()},
        }
        for i, r in enumerate(rows)
    ]
    errors = [
        r for r in records if any(r[f"prediction_{mode}"] != r["truth_id"] for mode in predictions)
    ]
    for name, items in (("predictions", records), ("errors", errors)):
        (directory / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in items)
        )
        keys = (
            "id",
            "input_text",
            "label",
            "truth_id",
            "p_gibberish",
            "prediction_default",
            "prediction_validation_locked",
            "subtype",
            "dialect",
            "input_tokens",
            "batch_latency_ms",
            "amortized_latency_ms",
        )
        write_csv(directory / f"{name}.csv", [{k: r.get(k) for k in keys} for r in items])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        for field in ("subtype", "dialect", "source_method", "label_origin"):
            groups[(field, str(row.get(field, "unknown")))].append(i)
        groups[
            (
                "characters",
                "1-15"
                if len(row["input_text"]) <= 15
                else "16-40"
                if len(row["input_text"]) <= 40
                else "41+",
            )
        ].append(i)
    report["subgroups"] = {
        f"{field}:{value}": {
            mode: metrics(truth[ix], pred[ix]) for mode, pred in predictions.items()
        }
        for (field, value), ix in groups.items()
    }
    write_json(directory / "metrics.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in predictions:
        fig, ax = plt.subplots(figsize=(5, 3))
        matrix = np.array(report[mode]["confusion"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set(
            xticks=[0, 1, 2],
            xticklabels=["VALID", "GIBBERISH", "INVALID"],
            yticks=[0, 1],
            yticklabels=["VALID", "GIBBERISH"],
            xlabel="Prediction",
            ylabel="Human truth",
            title=mode,
        )
        fig.tight_layout()
        fig.savefig(directory / f"confusion_{mode}.png")
        plt.close(fig)
    return report, records


def timed_classifier(item, rows, batch_size, warmup):
    """One GPU model at a time. Batch latency includes tokenization, transfer and scoring."""
    import torch

    from .data import encode_rows
    from .model import load_checkpoint, predict_rows, seed_everything

    device = "cuda"
    model = tokenizer = None
    seed_everything(item["seed"])
    inputs = [{"id": r["id"], "input_text": r["input_text"]} for r in rows]
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model, tokenizer, saved = load_checkpoint(item["checkpoint"], device)
        model.eval()
        model.requires_grad_(False)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        load_peak = torch.cuda.max_memory_allocated()
        for _ in range(warmup):
            predict_rows(
                model,
                tokenizer,
                inputs[:batch_size],
                saved["config"]["max_length"],
                device,
                batch_size,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        lengths = [
            len(r["input_ids"])
            for r in encode_rows(inputs, tokenizer, saved["config"]["max_length"])
        ]
        scores, enriched, batches = [], [], []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            torch.cuda.synchronize()
            begin = time.perf_counter()
            values, _ = predict_rows(
                model, tokenizer, batch, saved["config"]["max_length"], device, batch_size
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            batches.append(elapsed)
            scores.extend(values)
            enriched.extend(
                {
                    **rows[i],
                    "input_tokens": lengths[i],
                    "batch_latency_ms": elapsed,
                    "amortized_latency_ms": elapsed / len(batch),
                }
                for i in range(start, start + len(batch))
            )
        total = sum(batches)
        timing = {
            "model_load_seconds": load_seconds,
            "warmup_batches": warmup,
            "batch_size": batch_size,
            "steady_state_seconds": total / 1000,
            "steady_state_batch_ms_mean": float(np.mean(batches)),
            "steady_state_batch_ms_p50": float(np.percentile(batches, 50)),
            "steady_state_batch_ms_p95": float(np.percentile(batches, 95)),
            "amortized_ms_per_example": total / len(rows),
            "examples_per_second": len(rows) / (total / 1000),
            "load_peak_allocated_bytes": load_peak,
            "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "backbone_dtype": str(next(model.backbone.parameters()).dtype),
            "max_length": saved["config"]["max_length"],
            "truncation": False,
            "latency_scope": "Synchronized end-to-end batches: tokenize, pad, H2D, forward, "
            "softmax and D2H. Excludes loading, warmup, reporting and MLflow. "
            "Per-example values are amortized, not single-request latency.",
        }
        return scores, enriched, timing
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def summarize(directory, results, all_records):
    directory = Path(directory)
    summary = {}
    lines = [
        "# Independent test evaluation",
        "",
        DESCRIPTION,
        "",
        "400 human-reviewed examples: 188 VALID / 212 GIBBERISH. "
        "All six locked checkpoints are reported; no test-based selection or tuning.",
        "",
        (
            "| Approach | Seed | Operating point | Threshold | Accuracy | Macro F1 | G recall | "
            "VALID FRR | VALID acceptance |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for mode in ("default", "validation_locked"):
            m = result["metrics"][mode]
            lines.append(
                f"| {result['approach']} | {result['seed']} | {mode} | {m['threshold']:.8g} | "
                + " | ".join(f"{m[k]:.4f}" for k in MEASURES[:5])
                + " |"
            )
    for approach in ("frozen", "lora"):
        group = [r for r in results if r["approach"] == approach]
        if {r["seed"] for r in group} != {42, 43, 44} or len(group) != 3:
            raise ValueError("Cannot summarize incomplete/duplicate seeds")
        for mode in ("default", "validation_locked"):
            summary[f"{approach}/{mode}"] = {
                key: {
                    "mean": float(np.mean([r["metrics"][mode][key] for r in group])),
                    "std": float(np.std([r["metrics"][mode][key] for r in group], ddof=1)),
                }
                for key in MEASURES
            }
    lines += [
        "",
        "## Across seeds",
        "",
        "Means ± sample standard deviations (ddof=1), three seeds on the same test examples; "
        "these are not independent test samples or confidence intervals.",
        "",
        (
            "| Approach / operating point | Accuracy | Macro F1 | G recall | "
            "VALID FRR | VALID acceptance |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        lines.append(
            f"| {name} | "
            + " | ".join(f"{values[k]['mean']:.4f} ± {values[k]['std']:.4f}" for k in MEASURES[:5])
            + " |"
        )
    lines += [
        "",
        "## Latency",
        "",
        "Models run sequentially. Loading and warmup are excluded from steady-state timings. "
        "Batch timings include tokenization and transfers; per-example values are amortized. "
        "Loading may benefit from OS/Hugging Face caches; model order is fixed, not randomized.",
        "",
        "| Approach | Seed | Load seconds | Mean batch ms | P95 batch ms | Amortized ms/input |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        t = result["timing"]
        lines.append(
            f"| {result['approach']} | {result['seed']} | {t['model_load_seconds']:.3f} | "
            f"{t['steady_state_batch_ms_mean']:.3f} | {t['steady_state_batch_ms_p95']:.3f} | "
            f"{t['amortized_ms_per_example']:.3f} |"
        )
    error_analysis = []
    for record in all_records:
        entry = {k: record[k] for k in ("id", "input_text", "label", "subtype", "dialect")}
        for mode in ("default", "validation_locked"):
            wrong = [p for p in record["runs"] if p[f"prediction_{mode}"] != record["truth_id"]]
            entry[f"errors_{mode}"] = [f"{p['approach']}-{p['seed']}" for p in wrong]
        if entry["errors_default"] or entry["errors_validation_locked"]:
            error_analysis.append(entry)
    error_analysis.sort(key=lambda r: (r["label"] != "VALID", -len(r["errors_default"]), r["id"]))
    write_json(directory / "error_analysis.json", error_analysis)
    lines += [
        "",
        "## Error analysis",
        "",
        "See error_analysis.json for every error and per-run errors.jsonl/CSV for scores. "
        "Subgroup metrics are descriptive, often small, and use proposed dialect/subtype hints. "
        "Short inputs, construction artifacts and length imbalance limit generalization. "
        "Confusion matrices use rows VALID/GIBBERISH and columns VALID/GIBBERISH/INVALID.",
        "",
    ]
    for entry in error_analysis[:20]:
        text = entry["input_text"].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"- {entry['id']} ({entry['label']}): {text}; "
            f"default errors={len(entry['errors_default'])}/6, "
            f"validation-locked errors={len(entry['errors_validation_locked'])}/6"
        )
    write_json(
        directory / "metrics.json",
        {"benchmark": DESCRIPTION, "runs": results, "across_seeds": summary, "std_ddof": 1},
    )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary
