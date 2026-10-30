"""Validation-locked, paired regression analysis on a previously examined benchmark."""

import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from jais_guard.baselines.prompts import build_prompt, parse_diagnostic
from jais_guard.data.annotations import ROOT
from jais_guard.data.test_workflow import now, read_jsonl, sha, write_new

from .data import encode_rows, load_datasets
from .evaluation import metrics, write_csv, write_json
from .final_test import MEASURES, locked_registry
from .test_baseline import INVALID_POLICY

SCOPE = (
    "Previously examined, human-reviewed constructed Arabic-input benchmark. "
    "Version regression analysis on a seen benchmark; not an untouched estimate for iteration 2. "
    "No test-based threshold, checkpoint, seed or winner selection."
)
MODES = ("default", "validation_locked")
EXPECTED = {(a, s) for a in ("frozen", "lora") for s in (42, 43, 44)}


def identity_rows(gold, predictions):
    mapping = {r["id"]: r for r in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {r["id"] for r in gold}:
        raise ValueError("Prediction IDs differ, are missing, or are duplicated")
    ordered = []
    for row in gold:
        p = mapping[row["id"]]
        for field in ("id", "input_text", "label", "partition", "family_id", "subtype", "dialect"):
            if p.get(field) != row.get(field):
                raise ValueError(f"Prediction identity mismatch: {row['id']} {field}")
        if p.get("truth_id", int(row["label"] == "GIBBERISH")) != int(row["label"] == "GIBBERISH"):
            raise ValueError("Cached numeric truth disagrees with frozen labels")
        ordered.append(p)
    return ordered


def metric_match(expected, computed):
    for key, value in computed.items():
        if key not in expected:
            raise ValueError(f"Missing archived metric: {key}")
        previous = expected[key]
        if isinstance(value, float):
            if not isinstance(previous, (int, float)) or not math.isclose(
                value, previous, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"Metric compatibility mismatch: {key}")
        elif value != previous:
            raise ValueError(f"Metric compatibility mismatch: {key}")


def cached_metrics(rows, records, threshold, archived, baseline=False):
    ordered = identity_rows(rows, records)
    truth = [int(r["label"] == "GIBBERISH") for r in rows]
    if baseline:
        if archived["unparseable_policy"] != INVALID_POLICY:
            raise ValueError("Baseline invalid-output policy differs")
        for record in ordered:
            parsed = parse_diagnostic(
                record["raw_output"], "low", record["finish_reason"] == "stop"
            )
            if (
                record["prediction_id"]
                != {"VALID": 0, "GIBBERISH": 1}.get(parsed["prediction"], -1)
                or record["parse_status"] != parsed["parse_status"]
            ):
                raise ValueError("Archived baseline disagrees with the original strict parser")
        measured = metrics(truth, [p["prediction_id"] for p in ordered])
        measured["parsing_success_rate"] = sum(
            p["parse_status"] == "success" for p in ordered
        ) / len(rows)
        measured["valid_invalid_outputs"] = sum(
            p["prediction_id"] == -1 and y == 0 for p, y in zip(ordered, truth, strict=True)
        )
        if dict(Counter(p["parse_status"] for p in ordered)) != archived["parsing_status_counts"]:
            raise ValueError("Baseline parsing status counts differ")
        metric_match(archived["default"], measured)
        return {"zero_shot": measured}, ordered
    scores = np.array([p["p_gibberish"] for p in ordered], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid archived probabilities")
    result = {}
    for mode, value in (("default", 0.5), ("validation_locked", threshold)):
        pred = (scores >= value).astype(int).tolist()
        field = f"prediction_{mode}"
        if any(p[field] != value for p, value in zip(ordered, pred, strict=True)):
            raise ValueError("Cached prediction does not use the locked operating threshold")
        measured = metrics(truth, pred)
        metric_match(archived[mode], measured)
        result[mode] = {"threshold": value, **measured}
    return result, ordered


def source_training_run(db_path, parent, approach, seed, experiment, expected_hashes=None):
    """Read existing MLflow records without migrations, writes or run resumption."""
    with sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True) as db:
        row = db.execute(
            "SELECT r.status,e.name FROM runs r JOIN experiments e "
            "ON r.experiment_id=e.experiment_id WHERE r.run_uuid=?",
            (parent,),
        ).fetchone()
        if row != ("FINISHED", experiment):
            raise ValueError("Wrong experiment or unfinished training parent")
        candidates = [
            r[0]
            for r in db.execute(
                "SELECT run_uuid FROM tags WHERE key='mlflow.parentRunId' AND value=?", (parent,)
            )
        ]
        matched = []
        for rid in candidates:
            status = db.execute("SELECT status FROM runs WHERE run_uuid=?", (rid,)).fetchone()[0]
            params = dict(db.execute("SELECT key,value FROM params WHERE run_uuid=?", (rid,)))
            if (
                status == "FINISHED"
                and params.get("approach") == approach
                and params.get("seed") == str(seed)
            ):
                if expected_hashes and any(params.get(k) != v for k, v in expected_hashes.items()):
                    raise ValueError("MLflow training dataset hashes disagree with checkpoint")
                matched.append({"parent_run_id": parent, "training_run_id": rid})
    if len(matched) != 1:
        raise ValueError("Cannot uniquely map best checkpoint to finished MLflow training run")
    return matched[0]


def v2_inventory(config_path, mlflow_db):
    """Validation artifacts only. Never opens any version-2 benchmark predictions."""
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    for field in ("train_path", "validation_path", "output_dir"):
        cfg[field] = str((ROOT / cfg[field]).resolve())
    if cfg.get("iteration") != 2 or cfg.get("validation_review_complete") is not True:
        raise ValueError("Iteration-two validation is not approved")
    _, validation, tm, vm = load_datasets(cfg)
    registry, files, seen = [], {str(config_path): sha(config_path)}, set()
    for approach, seed in sorted(EXPECTED):
        paths = sorted(Path(cfg["output_dir"]).glob(f"*/{approach}-{seed}/best/classifier.json"))
        if len(paths) != 1:
            raise ValueError(
                f"Expected one best checkpoint for {approach}-{seed}, found {len(paths)}"
            )
        metadata_path = paths[0]
        directory = metadata_path.parents[1]
        saved = json.loads(metadata_path.read_text())
        key = (saved["approach"], saved["seed"])
        if key != (approach, seed) or key in seen:
            raise ValueError("Checkpoint seed/architecture mismatch or duplicate")
        seen.add(key)
        for field, expected in cfg.items():
            if saved["config"].get(field) != expected:
                raise ValueError(
                    f"Checkpoint differs from iteration-two config: {approach}-{seed} {field}"
                )
        if (
            saved["label_mapping"] != {"VALID": 0, "GIBBERISH": 1}
            or saved["pooling"] != "last_nonpadding_final_hidden"
        ):
            raise ValueError("Checkpoint classifier semantics mismatch")
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text())
        if (
            not history
            or [h["epoch"] for h in history] != list(range(1, len(history) + 1))
            or any(h["seed"] != seed or not math.isfinite(h["validation_loss"]) for h in history)
        ):
            raise ValueError("Invalid training history")
        best = min(history, key=lambda h: h["validation_loss"])
        if saved["metadata"] != best:
            raise ValueError("Best checkpoint does not match first minimum validation-loss epoch")
        threshold_path = directory / "threshold.json"
        threshold = json.loads(threshold_path.read_text())
        value = threshold.get("exploratory")
        if (
            threshold.get("default") != 0.5
            or threshold.get("selection_data") != "validation"
            or isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= np.nextafter(1.0, np.inf)
        ):
            raise ValueError("Threshold is not a valid validation-selected operating point")
        report_path = directory / "evaluation/metrics.json"
        predictions_path = directory / "evaluation/predictions.jsonl"
        report = json.loads(report_path.read_text())
        predictions = read_jsonl(predictions_path)
        # Original validation artifacts call the locked operating point 'exploratory'.
        adapted = [
            {**p, "prediction_validation_locked": p["prediction_exploratory"]} for p in predictions
        ]
        cached_metrics(
            validation,
            adapted,
            value,
            {"default": report["default"], "validation_locked": report["exploratory"]},
        )
        if (
            report["exploratory"]["threshold"] != value
            or report["selection_data"] != "validation; not independent test performance"
        ):
            raise ValueError("Validation report/threshold selection provenance mismatch")
        logits = np.array([[p["logit_valid"], p["logit_gibberish"]] for p in predictions])
        # The artifact rows retain validation order; labels are read from those same rows.
        truth = np.array([int(p["label"] == "GIBBERISH") for p in predictions])
        loss = float(
            np.mean(
                np.logaddexp(logits[:, 0], logits[:, 1]) - logits[np.arange(len(logits)), truth]
            )
        )
        probabilities = np.exp(logits[:, 1] - np.logaddexp(logits[:, 0], logits[:, 1]))
        if not np.allclose(probabilities, [p["p_gibberish"] for p in predictions], atol=1e-6):
            raise ValueError("Archived validation logits and probabilities disagree")
        if not math.isclose(loss, best["validation_loss"], rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("Selected-epoch validation loss disagrees with archived final logits")
        checkpoint = metadata_path.parent
        pinned = [p for p in checkpoint.rglob("*") if p.is_file() and p.name != "training_state.pt"]
        pinned += [
            threshold_path,
            history_path,
            report_path,
            predictions_path,
            directory / "evaluation/report.md",
            directory / "evaluation/thresholds.csv",
        ]
        fingerprints = {str(p.resolve()): sha(p) for p in pinned}
        files.update(fingerprints)
        source = source_training_run(
            mlflow_db,
            directory.parent.name,
            approach,
            seed,
            cfg["experiment"],
            {k: cfg[k] for k in ("train_sha256", "validation_sha256")},
        )
        registry.append(
            {
                "approach": approach,
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "threshold": value,
                "threshold_path": str(threshold_path.resolve()),
                "selected_epoch": best["epoch"],
                "metadata": saved,
                "source_runs": source,
                "locked_files": fingerprints,
                "validation_metrics": report,
                "validation_directory": str(directory / "evaluation"),
            }
        )
    for split in ("train", "validation"):
        files[cfg[f"{split}_path"]] = sha(cfg[f"{split}_path"])
        manifest = Path(cfg[f"{split}_path"]).with_name("manifest.json")
        files[str(manifest)] = sha(manifest)
    return registry, files, {"train": tm, "validation": vm}


def validation_comparison(directory, registry):
    records = []
    lines = [
        "# Iteration-two validation comparison",
        "",
        "250 reviewed validation examples only; thresholds/epochs are already selected. "
        "No benchmark predictions were read for this comparison.",
        "",
        (
            "| Approach | Seed | Best epoch | Mode | Threshold | Accuracy | Macro F1 | "
            "G recall | VALID FRR | VALID acceptance |"
        ),
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = {}
    for item in registry:
        for mode, source in (("default", "default"), ("validation_locked", "exploratory")):
            m = item["validation_metrics"][source]
            records.append({"approach": item["approach"], "seed": item["seed"], "mode": mode, **m})
            threshold = 0.5 if mode == "default" else item["threshold"]
            lines.append(
                (
                    f"| {item['approach']} | {item['seed']} | {item['selected_epoch']} | "
                    f"{mode} | {threshold:.9g} | "
                )
                + " | ".join(f"{m[k]:.4f}" for k in MEASURES[:5])
                + " |"
            )
    lines += ["", "Sample standard deviation across three seeds (ddof=1):", ""]
    for approach in ("frozen", "lora"):
        for mode in MODES:
            group = [r for r in records if r["approach"] == approach and r["mode"] == mode]
            name = f"{approach}/{mode}"
            summaries[name] = {
                k: {
                    "mean": float(np.mean([r[k] for r in group])),
                    "std": float(np.std([r[k] for r in group], ddof=1)),
                }
                for k in MEASURES
            }
            lines.append(
                f"- {name}: "
                + "; ".join(
                    f"{k}={summaries[name][k]['mean']:.4f} ± {summaries[name][k]['std']:.4f}"
                    for k in MEASURES[:5]
                )
            )
    write_json(
        directory / "validation_comparison.json",
        {"per_seed": records, "across_seeds": summaries, "std_ddof": 1},
    )
    (directory / "validation_comparison.md").write_text("\n".join(lines) + "\n")


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Locked comparison input changed: {path}")


def load_reused(rows, frozen, v1_registry, v1_folder, baseline_folder):
    v1_folder, baseline_folder = Path(v1_folder).resolve(), Path(baseline_folder).resolve()
    reused, files = [], {}
    for folder in (v1_folder, baseline_folder):
        complete = json.loads((folder / "completion.json").read_text())
        if complete["status"] != "complete" or complete["test_sha256"] != frozen["test_sha256"]:
            raise ValueError("Archived result is incomplete or uses a different test dataset")
        files.update({str(p): sha(p) for p in folder.rglob("*") if p.is_file()})
    if json.loads((v1_folder / "completion.json").read_text())["models_evaluated"] != 6:
        raise ValueError("Need all six completed iteration-one test evaluations")
    for item in v1_registry:
        path = v1_folder / f"{item['approach']}-{item['seed']}"
        report, records = cached_metrics(
            rows,
            read_jsonl(path / "predictions.jsonl"),
            item["threshold"],
            json.loads((path / "metrics.json").read_text()),
        )
        timing = json.loads((path / "timing.json").read_text())
        reused.append(
            {
                "version": "v1",
                "approach": item["approach"],
                "seed": item["seed"],
                "metrics": report,
                "records": records,
                "timing": timing,
                "source_directory": str(path),
                "environment": json.loads((v1_folder / "environment.json").read_text()),
            }
        )
    report, records = cached_metrics(
        rows,
        read_jsonl(baseline_folder / "predictions.jsonl"),
        None,
        json.loads((baseline_folder / "metrics.json").read_text()),
        baseline=True,
    )
    settings = json.loads((baseline_folder / "validation_settings.json").read_text())
    first = v1_registry[0]["metadata"]["config"]
    if settings["model"] != first["model"] or settings["revision"] != first["revision"]:
        raise ValueError("Untuned baseline model/revision differs")
    verify_files(settings["files"])
    reused.append(
        {
            "version": "untuned",
            "approach": "untuned_prompt",
            "seed": 42,
            "metrics": report,
            "records": records,
            "timing": json.loads((baseline_folder / "timing.json").read_text()),
            "source_directory": str(baseline_folder),
            "environment": json.loads((baseline_folder / "environment.json").read_text()),
        }
    )
    return reused, files


def preprocessing_check(rows, v1_registry, v2_registry, reused):
    from transformers import AutoTokenizer

    reference = None
    tokenizer = None
    evidence, cache = {}, {}
    for item in v1_registry + v2_registry:
        path = Path(item["checkpoint"]) / "tokenizer"
        key = tuple(sha(p) for p in sorted(path.rglob("*")) if p.is_file())
        budget = item["metadata"]["config"]["max_length"]
        cache_key = (key, budget)
        if cache_key not in cache:
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
            tok.padding_side = "right"
            encoded = encode_rows(rows, tok, budget)
            fingerprint = (
                __import__("hashlib")
                .sha256(json.dumps(encoded, sort_keys=True).encode())
                .hexdigest()
            )
            cache[cache_key] = (fingerprint, [len(r["input_ids"]) for r in encoded])
            if tokenizer is None:
                tokenizer = tok
        fingerprint, lengths = cache[cache_key]
        current = (fingerprint, budget)
        if reference is None:
            reference = current
        if current != reference:
            raise ValueError("Classifier preprocessing differs across iterations/checkpoints")
        evidence[str(item["checkpoint"])] = {
            "encoded_ids_sha256": fingerprint,
            "max_length": budget,
            "padding": "right",
            "truncation": False,
        }
    for item in reused:
        if item["version"] == "v1":
            if any(
                p["input_tokens"] != n
                for p, n in zip(item["records"], next(iter(cache.values()))[1], strict=True)
            ):
                raise ValueError(
                    "Archived classifier tokenization differs from current preprocessing"
                )
        else:
            settings = json.loads(
                (Path(item["source_directory"]) / "validation_settings.json").read_text()
            )
            cfg = settings["baseline"]
            for p in item["records"]:
                prompt = build_prompt(
                    tokenizer, p["input_text"], cfg["reasoning_effort"], False, cfg["variant"]
                )
                if p["prompt"] != prompt or p["input_tokens"] != len(
                    tokenizer(prompt, add_special_tokens=False)["input_ids"]
                ):
                    raise ValueError("Cached untuned baseline prompt/preprocessing differs")
    return {
        "classifiers": evidence,
        "raw_inputs_ids_labels": "identical, verified by ID",
        "baseline": "Original chat wrapper/parser verified; differs from hidden-state models",
    }


def prepare(directory, config, mlflow_db, test, v1_folder, baseline_folder):
    """Validation-only selection lock is written before any benchmark-result loading."""
    import shutil

    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Comparison version already exists; never overwrite locks or results")
    protected = (
        ROOT / "data",
        ROOT / "results/fine_tuning",
        ROOT / "results/fine_tuning_iteration2",
        Path(v1_folder).resolve(),
        Path(baseline_folder).resolve(),
    )
    if any(directory == p or p in directory.parents for p in protected):
        raise ValueError("Comparison must use a separate output directory")
    registry, files, manifests = v2_inventory(config, mlflow_db)
    directory.mkdir(parents=True)
    validation_comparison(directory, registry)
    for item in registry:
        shutil.copytree(
            item["validation_directory"],
            directory / "validation" / f"{item['approach']}-{item['seed']}",
        )
    # This lock contains only validation-selected choices and their immutable source hashes.
    write_new(
        directory / "threshold_lock.json",
        {
            "created_at": now(),
            "policy": "Six validation-loss best checkpoints; default and saved val thresholds",
            "selection_data": "v2 validation only; no v2 benchmark predictions read",
            "checkpoints": registry,
            "files": files,
            "dataset_manifests": manifests,
        },
    )
    rows, frozen, old = locked_registry(test)
    if any(
        r["metadata"]["config"][k] != old[0]["metadata"]["config"][k]
        for r in registry
        for k in ("model", "revision")
    ):
        raise ValueError("Iterations do not use the same pinned Jais backbone")
    reused, cached_files = load_reused(rows, frozen, old, v1_folder, baseline_folder)
    preprocessing = preprocessing_check(rows, old, registry, reused)
    timing_settings = {
        (r["timing"]["batch_size"], r["timing"]["warmup_batches"])
        for r in reused
        if r["version"] == "v1"
    }
    if len(timing_settings) != 1:
        raise ValueError("Iteration-one classifiers have incompatible timing settings")
    batch, warmup = next(iter(timing_settings))
    source_files = {
        **cached_files,
        **{
            str(Path(test).resolve() / n): sha(Path(test) / n)
            for n in ("test.jsonl", "manifest.json", "freeze.json")
        },
    }
    for item in old:
        source_files.update(item["locked_files"])
    for path in frozen["selection_lock"]["prompt_files"]:
        source_files[path] = sha(path)
    # Pin inference/metrics implementation as well as the source data and weights.
    source_files.update(
        {
            str(p): sha(p)
            for p in (
                Path(__file__).resolve(),
                ROOT / "scripts/compare_versions.py",
                Path(__file__).with_name("model.py"),
                Path(__file__).with_name("evaluation.py"),
                Path(__file__).with_name("data.py"),
                Path(__file__).with_name("final_test.py"),
            )
        }
    )
    for p in directory.rglob("*"):
        if p.is_file() and p.name != "threshold_lock.json":
            source_files[str(p)] = sha(p)
    write_new(
        directory / "reused_metrics.json",
        [{k: v for k, v in r.items() if k != "records"} for r in reused],
    )
    write_new(directory / "preprocessing.json", preprocessing)
    protocol = {
        "scope": SCOPE,
        "test": str(Path(test).resolve()),
        "test_sha256": frozen["test_sha256"],
        "v1_results": str(Path(v1_folder).resolve()),
        "baseline_results": str(Path(baseline_folder).resolve()),
        "threshold_lock_sha256": sha(directory / "threshold_lock.json"),
        "files": source_files,
        "batch_size": batch,
        "warmup_batches": warmup,
        "experiment": "JaisGuard/version-regression-v1-v2",
        "classifiers_to_evaluate": 6,
        "cached_models_to_rerun": 0,
        "v1_registry": old,
    }
    write_new(directory / "protocol.json", protocol)
    return {
        "directory": str(directory),
        "validation_reports_verified": 6,
        "selected_epochs": {f"{r['approach']}-{r['seed']}": r["selected_epoch"] for r in registry},
        "thresholds": {f"{r['approach']}-{r['seed']}": r["threshold"] for r in registry},
        "cached_models_verified": 7,
        "inference_run": False,
    }


def checked_protocol(directory):
    directory = Path(directory).resolve()
    protocol = json.loads((directory / "protocol.json").read_text())
    if sha(directory / "threshold_lock.json") != protocol["threshold_lock_sha256"]:
        raise ValueError("Validation threshold lock changed")
    lock = json.loads((directory / "threshold_lock.json").read_text())
    verify_files(lock["files"])
    verify_files(protocol["files"])
    rows, frozen, _ = locked_registry(protocol["test"])
    if frozen["test_sha256"] != protocol["test_sha256"]:
        raise ValueError("Frozen benchmark identity changed")
    return protocol, lock, rows


def paired_analysis(directory, rows, results):
    mapping = {
        (r["version"], r["approach"], r["seed"]): r for r in results if r["version"] != "untuned"
    }
    detailed, groups = [], defaultdict(Counter)
    for approach, seed in sorted(EXPECTED):
        before, after = mapping[("v1", approach, seed)], mapping[("v2", approach, seed)]
        old_records = identity_rows(rows, before["records"])
        new_records = identity_rows(rows, after["records"])
        for mode in MODES:
            for row, old, new in zip(rows, old_records, new_records, strict=True):
                truth = int(row["label"] == "GIBBERISH")
                a, b = old[f"prediction_{mode}"], new[f"prediction_{mode}"]
                status = (
                    "both_correct"
                    if a == truth and b == truth
                    else "corrected"
                    if a != truth and b == truth
                    else "new_error"
                    if a == truth and b != truth
                    else "remaining_error"
                )
                record = {
                    "id": row["id"],
                    "input_text": row["input_text"],
                    "label": row["label"],
                    "subtype": row["subtype"],
                    "dialect": row["dialect"],
                    "approach": approach,
                    "seed": seed,
                    "mode": mode,
                    "transition": status,
                    "prediction_v1": a,
                    "prediction_v2": b,
                    "p_gibberish_v1": old["p_gibberish"],
                    "p_gibberish_v2": new["p_gibberish"],
                    "threshold_v1": before["metrics"][mode]["threshold"],
                    "threshold_v2": after["metrics"][mode]["threshold"],
                    "valid_false_rejection_v1": truth == 0 and a == 1,
                    "valid_false_rejection_v2": truth == 0 and b == 1,
                }
                detailed.append(record)
                strata = ["all", f"subtype:{row['subtype']}", f"dialect:{row['dialect']}"]
                if truth == 0:
                    strata.append("human_VALID")
                    if row["subtype"] == "recoverable_corruption":
                        strata.append("recoverable_corruption_human_VALID")
                    if row["dialect"] in ("Tunisian", "Egyptian", "Levantine", "Gulf"):
                        strata.append("dialectal_human_VALID")
                for stratum in strata:
                    for group in (
                        f"{approach}/{mode}/seed-{seed}/{stratum}",
                        f"{approach}/{mode}/three-seed-decisions/{stratum}",
                    ):
                        groups[group][status] += 1
                        groups[group]["decisions"] += 1
                        groups[group]["valid_false_rejections_v1"] += int(truth == 0 and a == 1)
                        groups[group]["valid_false_rejections_v2"] += int(truth == 0 and b == 1)
    summaries = {
        k: {
            name: int(v.get(name, 0))
            for name in (
                "decisions",
                "corrected",
                "new_error",
                "remaining_error",
                "both_correct",
                "valid_false_rejections_v1",
                "valid_false_rejections_v2",
            )
        }
        for k, v in groups.items()
    }
    (directory / "paired_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in detailed)
    )
    write_csv(directory / "paired_predictions.csv", detailed)
    write_json(directory / "paired_summary.json", summaries)
    return summaries, detailed


def comparison_report(directory, rows, results):
    """No ranks or winners; each operating point remains separately identifiable."""
    directory = Path(directory)
    aggregate = {}
    table = []
    ordered_groups = [
        ("untuned", "untuned_prompt"),
        ("v1", "frozen"),
        ("v1", "lora"),
        ("v2", "frozen"),
        ("v2", "lora"),
    ]
    for version, approach in ordered_groups:
        entries = [r for r in results if r["version"] == version and r["approach"] == approach]
        baseline = version == "untuned"
        if (
            not baseline and ({r["seed"] for r in entries} != {42, 43, 44} or len(entries) != 3)
        ) or (baseline and len(entries) != 1):
            raise ValueError(
                "Comparison needs baseline plus all three seeds for four classifier groups"
            )
        for mode in ("zero_shot",) if baseline else MODES:
            name = f"{version}/{approach}/{mode}"
            aggregate[name] = {
                key: {
                    "mean": float(np.mean([r["metrics"][mode][key] for r in entries])),
                    "std": None
                    if baseline
                    else float(np.std([r["metrics"][mode][key] for r in entries], ddof=1)),
                }
                for key in MEASURES
            }
            matrices = np.array([r["metrics"][mode]["confusion"] for r in entries])
            aggregate[name]["confusion"] = {
                "mean_per_seed": matrices.mean(axis=0).tolist(),
                "sum_repeated_seed_decisions": matrices.sum(axis=0).tolist(),
            }
            table.append(
                {
                    "group": f"{version} {approach}",
                    "mode": mode,
                    **{k: v for k, v in aggregate[name].items() if k != "confusion"},
                }
            )
    paired, detailed = paired_analysis(directory, rows, results)
    lines = [
        "# JaisGuard version-to-version regression comparison",
        "",
        SCOPE,
        "",
        "400 identical reviewed records: 188 VALID / 212 GIBBERISH. "
        "Iteration 2 was motivated by this already observed benchmark. "
        "All six v2 models are evaluated; v1 and untuned predictions are reused, not rerun.",
        "",
        "Values are means ± sample SD (ddof=1) across three seeds; the baseline is one fixed run. "
        "Seed repetitions are not independent benchmark samples or confidence intervals.",
        "",
        (
            "| Group | Operating point | Accuracy | Macro F1 | G recall | "
            "VALID FRR | VALID acceptance |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        values = [
            f"{r[k]['mean']:.4f}" + ("" if r[k]["std"] is None else f" ± {r[k]['std']:.4f}")
            for k in MEASURES[:5]
        ]
        lines.append(f"| {r['group']} | {r['mode']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Confusion matrices",
        "",
        "Rows: true VALID/GIBBERISH; columns: predicted VALID/GIBBERISH/INVALID. "
        "Per-seed matrices and mean-per-seed matrices are in metrics.json. "
        "Summed classifier matrices count each example three times.",
        "",
    ]
    confusion_dir = directory / "confusions"
    confusion_dir.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, values in aggregate.items():
        matrix = np.array(values["confusion"]["mean_per_seed"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(matrix, cmap="Blues")
        for (i, j), v in np.ndenumerate(matrix):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center")
        ax.set(
            xticks=[0, 1, 2],
            xticklabels=["VALID", "GIBBERISH", "INVALID"],
            yticks=[0, 1],
            yticklabels=["VALID", "GIBBERISH"],
            title=name,
        )
        fig.tight_layout()
        fig.savefig(confusion_dir / (name.replace("/", "_") + ".png"))
        plt.close(fig)
        lines.append(f"- {name}: {matrix.tolist()}")
    lines += [
        "",
        "## Latency",
        "",
        "Cached timings are preserved, not remeasured. Classifiers use synchronized batches "
        "including tokenization/transfers; per-input times are amortized batch times. "
        "Untuned baseline uses batch 1 with autoregressive generation and records actual "
        "per-input generation time. These are different workloads. Loading/warmup excluded "
        "from steady-state times; hardware/software/cache/order effects remain.",
        "",
        "| Group | Load seconds | Classify ms/input | Generation ms/input | Batch size |",
        "|---|---:|---:|---:|---:|",
    ]
    latency = {}
    for version, approach in ordered_groups:
        entries = [r for r in results if r["version"] == version and r["approach"] == approach]
        baseline = version == "untuned"
        time_key = "latency_ms_mean" if baseline else "amortized_ms_per_example"
        values = {}
        for name, key in (("load_seconds", "model_load_seconds"), ("classify_ms", time_key)):
            vector = [r["timing"][key] for r in entries]
            values[name] = {
                "mean": float(np.mean(vector)),
                "std": None if baseline else float(np.std(vector, ddof=1)),
            }
        generation = entries[0]["timing"].get("generation_ms_mean") if baseline else None
        latency[f"{version}/{approach}"] = {**values, "generation_ms": generation}

        def show(stat):
            return f"{stat['mean']:.3f}" + ("" if stat["std"] is None else f" ± {stat['std']:.3f}")

        lines.append(
            (
                f"| {version} {approach} | {show(values['load_seconds'])} | "
                f"{show(values['classify_ms'])} | "
            )
            + (
                f"{generation if generation is not None else 'n/a'} | "
                f"{entries[0]['timing']['batch_size']} |"
            )
        )
    lines += [
        "",
        "## Paired changes",
        "",
        "Same architecture and seed are paired across versions, separately at default and "
        "their own validation-selected thresholds. Tuned-point changes combine model and "
        "validation-operating-point differences. Thresholds were not tuned on this test. "
        "Counts below pool three seed decisions (1200 for 'all'), not 1200 distinct inputs. "
        "Per-seed and proposed-dialect/subtype breakdowns are in paired_summary.json; "
        "all 4800 paired decisions are in paired_predictions.jsonl/CSV.",
        "",
        (
            "| Approach | Mode | Subset | Decisions | Corrected | New errors | "
            "Remaining errors | VALID rejects v1 → v2 |"
        ),
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for approach in ("frozen", "lora"):
        for mode in MODES:
            for group in ("all", "recoverable_corruption_human_VALID", "dialectal_human_VALID"):
                p = paired.get(f"{approach}/{mode}/three-seed-decisions/{group}", Counter())
                lines.append(
                    (
                        f"| {approach} | {mode} | {group} | {p.get('decisions', 0)} | "
                        f"{p.get('corrected', 0)} | "
                    )
                    + f"{p.get('new_error', 0)} | {p.get('remaining_error', 0)} | "
                    + (
                        f"{p.get('valid_false_rejections_v1', 0)} → "
                        f"{p.get('valid_false_rejections_v2', 0)} |"
                    )
                )
    lines += [
        "",
        "Representative recoverable/dialect VALID changes (full records in paired files):",
        "",
    ]
    focus = [
        p
        for p in detailed
        if p["label"] == "VALID"
        and p["transition"] != "both_correct"
        and (
            p["subtype"] == "recoverable_corruption"
            or p["dialect"] in ("Tunisian", "Egyptian", "Levantine", "Gulf")
        )
    ]
    focus.sort(
        key=lambda p: (p["transition"] != "new_error", p["id"], p["approach"], p["seed"], p["mode"])
    )
    for p in focus[:30]:
        lines.append(
            f"- {p['id']} {p['approach']}-{p['seed']} {p['mode']} "
            f"{p['transition']}: {p['input_text']}"
        )
    baseline = next(r for r in results if r["version"] == "untuned")
    b = baseline["metrics"]["zero_shot"]
    lines += [
        "",
        "## Interpretation limits",
        "",
        SCOPE,
        "",
        f"Untuned parsing success {b['parsing_success_rate']:.2%}, INVALID outputs {b['invalid']}. "
        + INVALID_POLICY,
        "",
        "Construction and length cues can affect all methods; dialect/subtype hints are "
        "proposed rather than independently adjudicated tags. No significance claim, "
        "causal attribution, production-traffic estimate or test-selected winner is made.",
    ]
    write_json(
        directory / "metrics.json",
        {
            "scope": SCOPE,
            "std_ddof": 1,
            "groups": aggregate,
            "latency": latency,
            "per_seed": [{k: v for k, v in r.items() if k != "records"} for r in results],
        },
    )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    return aggregate
