"""Read-only, checksum-verified access to the approved exports."""

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

LABELS = {"VALID": 0, "GIBBERISH": 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_export(path, expected_hash, partition, expected_counts):
    path = Path(path)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    actual_hash = digest(path)
    if actual_hash != expected_hash or manifest["sha256"] != actual_hash:
        raise ValueError(f"Export checksum mismatch: {path}")
    if manifest["partition"] != partition or not manifest["complete"]:
        raise ValueError("Incomplete or wrong-partition export")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if Counter(row["label"] for row in rows) != Counter(expected_counts):
        raise ValueError("Unexpected class counts")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs")
    for row in rows:
        if row["partition"] != partition or row["label"] not in LABELS:
            raise ValueError("Invalid partition or final label")
        if row["annotation"]["flagged"] or not row["input_text"].strip():
            raise ValueError("Flagged or empty example")
        if row["is_gibberish"] != (row["label"] == "GIBBERISH"):
            raise ValueError("Inconsistent final label")
        if row["annotation"].get("label") not in (None, row["label"]):
            raise ValueError("Final label disagrees with human annotation")
        if partition == "validation" and row["review_status"] != "human-reviewed":
            raise ValueError("Validation must be fully human-reviewed")
        if any(c.isalpha() and "ARABIC" not in unicodedata.name(c, "") for c in row["input_text"]):
            raise ValueError(f"Unsupported script: {row['id']}")
    return rows, manifest


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def expected_counts(config, partition):
    defaults = {
        "train": {"VALID": 359, "GIBBERISH": 341},
        "validation": {"VALID": 99, "GIBBERISH": 101},
    }
    counts = config.get(f"{partition}_counts", defaults[partition])
    if (
        not isinstance(counts, dict)
        or set(counts) != set(LABELS)
        or any(type(value) is not int or value <= 0 for value in counts.values())
    ):
        raise ValueError(f"Invalid {partition} class counts")
    return counts


def load_datasets(config):
    train, tm = read_export(
        config["train_path"], config["train_sha256"], "train", expected_counts(config, "train")
    )
    val, vm = read_export(
        config["validation_path"],
        config["validation_sha256"],
        "validation",
        expected_counts(config, "validation"),
    )
    for key in ("id", "family_id"):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f"Train/validation overlap: {key}")
    if {normalized_text(r["input_text"]) for r in train} & {
        normalized_text(r["input_text"]) for r in val
    }:
        raise ValueError("Train/validation normalized text overlap")
    return train, val, tm, vm


def encode_rows(rows, tokenizer, max_length):
    encoded = []
    for row in rows:
        text = row["input_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected nonempty text")
        if any(c.isalpha() and "ARABIC" not in unicodedata.name(c, "") for c in text):
            raise ValueError("Unsupported script; no gibberish label assigned")
        item = tokenizer(text, add_special_tokens=True, truncation=False)
        if not 0 < len(item["input_ids"]) <= max_length:
            raise ValueError(f"Token budget exceeded or empty: {row['id']}")
        encoded.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]})
    return encoded


def collate(encoded, tokenizer, device, pad_to=None):
    return tokenizer.pad(
        encoded, padding="max_length" if pad_to else True, max_length=pad_to, return_tensors="pt"
    ).to(device)
