"""Small smoke-test summary; invalid predictions count as failures."""


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No evaluation examples")
    return {
        "examples": len(rows),
        "correct": sum(row["prediction"] == row["label"] for row in rows),
        "invalid": sum(row["prediction"] is None for row in rows),
        "incorrect": sum(
            row["prediction"] is not None and row["prediction"] != row["label"] for row in rows
        ),
        "accuracy": sum(row["prediction"] == row["label"] for row in rows) / len(rows),
    }
