"""Load and inspect the existing evaluation dataset without running a model."""

import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data/gibberish.jsonl"
STRING_FIELDS = (
    "id",
    "input_text",
    "split",
    "subtype",
    "dialect",
    "query_type",
    "difficulty",
    "label_rationale",
)
REVIEW_HINTS = {
    "scrambled_words": "Check whether the intended words are still recoverable.",
    "arabic_like_nonsense": "Check for real dialect words or unfamiliar vocabulary.",
    "incoherent_sentence": "Check for a meaningful metaphor, fiction, or unusual question.",
    "noisy_but_meaningful": "Check that meaning survives the surrounding noise.",
    "short_topic": "Check meaning without requiring a complete sentence.",
}


def load_dataset(path: Path = DATA_PATH, split: str | None = None) -> list[dict]:
    """Validate the entire file before selecting a split; never shuffle or relabel."""
    if split not in {None, "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")
    rows = []
    ids = set()
    texts = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Line {line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number}: expected an object")
        for field in STRING_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Line {line_number}: missing or empty string field {field}")
        if type(row.get("is_gibberish")) is not bool:
            raise ValueError(f"Line {line_number}: is_gibberish must be a JSON boolean")
        if row["split"] not in {"dev", "test"}:
            raise ValueError(f"Line {line_number}: unknown split {row['split']}")
        # Compare whitespace-normalized texts, but preserve the original text in the record.
        text_key = " ".join(row["input_text"].split())
        if row["id"] in ids:
            raise ValueError(f"Line {line_number}: duplicate ID {row['id']}")
        if text_key in texts:
            raise ValueError(f"Line {line_number}: duplicate text ({row['id']})")
        ids.add(row["id"])
        texts.add(text_key)
        rows.append(
            {
                **row,
                "text": row["input_text"],
                "label": "GIBBERISH" if row["is_gibberish"] else "VALID",
                "review_reason": REVIEW_HINTS.get(row["subtype"]),
            }
        )
    counts = Counter(row["split"] for row in rows)
    if counts != {"dev": 40, "test": 160}:
        raise ValueError(f"Expected dev=40 and test=160; found {dict(counts)}")
    return [row for row in rows if split is None or row["split"] == split]


def main() -> None:
    rows = load_dataset()
    print("Validated 200 records: dev=40, test=160; no duplicate IDs or texts.")
    for split in ("dev", "test"):
        selected = [row for row in rows if row["split"] == split]
        print(f"\n{split}:")
        for field in ("label", "subtype", "dialect", "query_type"):
            print(f"  {field}: {dict(Counter(row[field] for row in selected))}")
    print("\nDevelopment review candidates (heuristic hints, labels unchanged):")
    for row in rows:
        if row["split"] == "dev" and row["review_reason"]:
            print(f"{row['id']} [{row['label']}]: {row['input_text']}")
            print(f"  {row['review_reason']} Original rationale: {row['label_rationale']}")


if __name__ == "__main__":
    main()
