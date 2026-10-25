"""Inspect untuned Jais predictions on a few Arabic examples."""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from jais_guard.data import load_dataset
    from jais_guard.inference import BaselineClassifier, BaselineConfig
    from jais_guard.metrics import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Use the existing 40-example dev split")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/baseline.json").read_text())
    rows = (
        load_dataset(split="dev")
        if args.dev
        else [
            json.loads(line)
            for line in (ROOT / "data/eval/smoke.jsonl").read_text().splitlines()
            if line.strip()
        ]
    )
    classifier = BaselineClassifier(BaselineConfig(**config))
    print(f"Device: {classifier.device}", flush=True)
    filename = "dev_prompt_comparison.json" if args.dev else "prompt_comparison.json"
    output = ROOT / "results/baseline" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"config": config, "device": classifier.device, "variants": {}}
    for variant in ("system_only", "explicit_user"):
        results = []
        for row in rows:
            result = {**row, **classifier.classify(row["text"], variant=variant)}
            results.append(result)
            print(f"\nVariant: {variant} | Input: {row['text']}")
            print(f"Raw output:\n{result['raw_output']}")
            print(f"Expected: {row['label']} | Predicted: {result['prediction']}")
            print(
                f"Finish: {result['finish_reason']} | Tokens: {result['generated_tokens']}",
                flush=True,
            )
            # Save after each example so interrupted runs remain inspectable.
            report["variants"][variant] = {"summary": summarize(results), "results": results}
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\n{variant}: {summarize(results)}", flush=True)
    print(f"Saved to {output}")
    return (
        0
        if all(run["summary"]["correct"] == len(rows) for run in report["variants"].values())
        else 1
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        sys.exit(main())
    except Exception:
        logging.exception("Baseline smoke test failed")
        sys.exit(1)
