"""Inspect untuned Jais predictions on a few Arabic examples."""

import argparse
import importlib.metadata
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import torch

    from jais_guard.baselines.inference import BaselineClassifier, BaselineConfig
    from jais_guard.baselines.metrics import summarize
    from jais_guard.baselines.report import write_report
    from jais_guard.data.legacy import load_dataset

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
    report = {
        "config": config,
        "device": classifier.device,
        "hardware": torch.cuda.get_device_name(0) if classifier.device == "cuda" else "CPU",
        "dtype": str(classifier.model.dtype),
        "split": "dev" if args.dev else "smoke",
        "started_utc": datetime.now(UTC).isoformat(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "generation_config": classifier.model.generation_config.to_dict(),
        "generation_overrides": {
            "do_sample": False,
            "max_new_tokens": config["max_tokens"],
            "pad_token_id": classifier.tokenizer.eos_token_id,
        },
        "variants": {},
    }
    for variant in ("system_only", "explicit_user"):
        results = []
        for row in rows:
            if classifier.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = {**row, **classifier.classify(row["text"], variant=variant)}
            if classifier.device == "cuda":
                torch.cuda.synchronize()
            result["latency_ms"] = (time.perf_counter() - start) * 1000
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
    report["completed_utc"] = datetime.now(UTC).isoformat()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_report(report, output.parent / output.stem, ROOT / "data/jev_results.zip")
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
