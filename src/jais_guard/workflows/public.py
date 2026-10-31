"""Pinned public-data adapter and checkpoint workflows, separate from historical exports."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REPO_ID = "RayenFekih/arabic-gibberish-detection"
REVISION = "c3e739a396dcf4ae982633f19dc24555dab837d4"
MODEL = "IFM/Jais-3-0.9B"
MODEL_REVISION = "e51941b37091287fee4934e3f8a799e0d3fef37f"
COUNTS = {
    "train": {"VALID": 708, "GIBBERISH": 790},
    "validation": {"VALID": 120, "GIBBERISH": 130},
    "test": {"VALID": 188, "GIBBERISH": 212},
}
# Verified upload receipt; only these exact published bytes are accepted.
FILES = {
    "README.md": "2d73b0ab24c2ee3e30090b3914bf378a6a433a3f0fa725187e0318fde04bfdf8",
    "metadata.json": "b26303d0571adbc034feeaf3337b7822747d4067d3cfcbb878bcff9cae84c8c8",
    "data/train.parquet": "856cb75dbd8ab9a4c2eba9f00c077162c4b640449c1c32ed14be5dfe813c5c99",
    "data/validation.parquet": "06a7385ee7f50aabbfb0fc5fb025fcbc60c2ac1c93c0896b9f85ee929a93d57b",
    "data/test.parquet": "003493dd220523f7387e87b5751b66ecac2d149dcb3347ac2e3960367742cfbc",
    "LICENSES/ATTRIBUTION.md": "203dd333b262ac28e8b04f1519f02beefac2c6be543c68678cdd62e9d50e4e5f",
    "LICENSES/Apache-2.0.txt": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "LICENSES/CC-BY-4.0.txt": "c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def public_row(row, split, index):
    """Adapt public fields; IDs are public positions, not original candidate/family IDs."""
    label = row["label_name"]
    if type(row["label"]) is not int or {"VALID": 0, "GIBBERISH": 1}.get(label) != row["label"]:
        raise ValueError("Public label mapping changed")
    origin = row["label_origin"]
    if origin not in ("human-reviewed", "audited-provisional"):
        raise ValueError("Unknown public label origin")
    if split != "train" and origin != "human-reviewed":
        raise ValueError("Public evaluation data must be human-reviewed")
    if not isinstance(row["text"], str) or not row["text"].strip():
        raise ValueError("Empty public text")
    return {
        **row,
        "id": f"public-v1:{split}:{index}",
        "input_text": row["text"],
        "label": label,
        "is_gibberish": row["label"] == 1,
        "partition": split,
        # This unique row key is not evidence of source-family independence.
        "family_id": f"public-row:{split}:{index}",
        "family_id_scope": "public-row-only; original family IDs are not published",
        "review_status": origin,
        "annotation": {"flagged": False},
    }


def export_public(dataset, destination, counts=COUNTS):
    """Create new adapter exports. Never writes historical data or existing directories."""
    destination = Path(destination).resolve()
    rows_by_split = {}
    for split, expected in counts.items():
        rows = [public_row(row, split, i) for i, row in enumerate(dataset[split])]
        if Counter(r["label"] for r in rows) != Counter(expected):
            raise ValueError(f"Unexpected public {split} counts")
        rows_by_split[split] = rows
    destination.mkdir(parents=True, exist_ok=False)
    for split, rows in rows_by_split.items():
        folder = destination / split
        folder.mkdir()
        path = folder / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        write_json(
            folder / "manifest.json",
            {
                "version": f"public-v1-{split}",
                "partition": split,
                "complete": True,
                "sha256": sha(path),
                "annotation_complete": split != "train",
                "source": {"repo_id": REPO_ID, "revision": REVISION},
                "family_independence": "not established by public row IDs",
            },
        )
    config = {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "train_path": str(destination / "train/train.jsonl"),
        "validation_path": str(destination / "validation/validation.jsonl"),
        "train_sha256": sha(destination / "train/train.jsonl"),
        "validation_sha256": sha(destination / "validation/validation.jsonl"),
        "train_counts": counts["train"],
        "validation_counts": counts["validation"],
        "max_length": 64,
        "seeds": [42, 43, 44],
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_clip": 1.0,
        "exploratory_frr": 0.02,
        "output_dir": str(destination / "runs"),
        "tracking_uri": "sqlite:///" + str(destination / "mlflow.db"),
        "experiment": "JaisGuard/public-v1-new-experiments",
        "dataset_revision": REVISION,
        "historical_reproduction": False,
        "frozen": {"epochs": 30, "patience": 5, "batch_size": 32, "head_lr": 0.001},
        "lora": {
            "epochs": 8,
            "patience": 2,
            "microbatch": 2,
            "accumulation": 8,
            "adapter_lr": 0.0001,
            "head_lr": 0.001,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
    }
    write_json(destination / "training.json", config)
    return config


def prepare(destination, source=None):
    if Path(destination).exists():
        raise FileExistsError(f"Refusing existing public export directory: {destination}")
    from datasets import Dataset, DatasetDict
    from huggingface_hub import snapshot_download

    source = (
        Path(source)
        if source
        else Path(
            snapshot_download(
                REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=list(FILES)
            )
        )
    )
    for name, expected in FILES.items():
        if sha(source / name) != expected:
            raise ValueError(f"Published file checksum mismatch: {name}")
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["label_mapping"] != {"VALID": 0, "GIBBERISH": 1}:
        raise ValueError("Published label mapping changed")
    # Read local Parquet with a writable adjacent cache; no further network calls.
    dataset = DatasetDict(
        {
            split: Dataset.from_parquet(
                str(source / "data" / f"{split}.parquet"),
                cache_dir=str(Path(destination).resolve().parent / ".hf-cache"),
            )
            for split in COUNTS
        }
    )
    for split in COUNTS:
        if dataset[split].features["label"].names != ["VALID", "GIBBERISH"]:
            raise ValueError("Published ClassLabel order changed")
    config = export_public(dataset, destination)
    import shutil

    # Keep the exact published card and attribution package beside adapted exports.
    for name in ["README.md", "metadata.json", *[n for n in FILES if n.startswith("LICENSES/")]]:
        target = Path(destination) / "publication" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return config


def configuration(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for field in ("train_path", "validation_path", "output_dir"):
        config[field] = str((path.parent / config[field]).resolve())
    return config


def checkpoint_manifest(checkpoint, operating_point="validation_locked"):
    checkpoint = Path(checkpoint).resolve()
    saved = json.loads((checkpoint / "classifier.json").read_text())
    if saved["label_mapping"] != {"VALID": 0, "GIBBERISH": 1}:
        raise ValueError("Unsupported checkpoint label mapping")
    if saved["pooling"] != "last_nonpadding_final_hidden":
        raise ValueError("Unsupported checkpoint pooling")
    if operating_point == "default":
        threshold = 0.5
    else:
        path = checkpoint.parent / "threshold.json"
        if not path.exists():
            raise ValueError("Saved validation threshold missing; select --operating-point default")
        locked = json.loads(path.read_text())
        if locked.get("selection_data") != "validation":
            raise ValueError("Threshold must be selected on validation")
        threshold = locked["exploratory"]
    # A validation-selected accept-all threshold can be nextafter(1, +inf).
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not (0 <= threshold <= math.nextafter(1.0, math.inf))
    ):
        raise ValueError("Invalid saved threshold")
    item = {
        "approach": saved["approach"],
        "seed": saved["seed"],
        "checkpoint": str(checkpoint),
        "metadata": saved,
        "threshold": threshold,
    }
    locked_files = {
        str(path.relative_to(checkpoint)): sha(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file() and path.name != "training_state.pt"
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "files": locked_files,
                "threshold": threshold,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {"identity": identity, "models": {"checkpoint": item}, "default_model": "checkpoint"}


def evaluate(checkpoint, data, output, operating_point="validation_locked", device="cuda"):
    from jais_guard.finetuning.data import read_export
    from jais_guard.finetuning.evaluation import metrics
    from jais_guard.finetuning.model import load_checkpoint, predict_rows

    path = Path(data)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    # Counts come from the rows but checksums and semantic validation remain mandatory.
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    rows, _ = read_export(
        path, manifest["sha256"], manifest["partition"], dict(Counter(row["label"] for row in raw))
    )
    item = checkpoint_manifest(checkpoint, operating_point)["models"]["checkpoint"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing existing evaluation directory: {output}")
    model, tokenizer, saved = load_checkpoint(checkpoint, device)
    scores, _ = predict_rows(model, tokenizer, rows, saved["config"]["max_length"], device)
    prediction = [int(score >= item["threshold"]) for score in scores]
    report = metrics([int(r["is_gibberish"]) for r in rows], prediction)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", report)
    write_json(
        output / "protocol.json",
        {
            "checkpoint_identity": checkpoint_manifest(checkpoint, operating_point)["identity"],
            "dataset_sha256": sha(path),
            "operating_point": operating_point,
            "threshold": item["threshold"],
            "threshold_selection": "validation; never evaluation",
        },
    )
    (output / "predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "truth": int(row["is_gibberish"]),
                    "p_gibberish": score,
                    "prediction": pred,
                }
            )
            + "\n"
            for row, score, pred in zip(rows, scores, prediction, strict=True)
        )
    )
    return report
