"""Project reviewed aggregate evidence from local originals into a NEW directory.

No training, inference, API requests or original-file writes. Review the explicit
file list before publication; this script does not grant clearance to other files.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
dest = args.output.resolve()
dest.mkdir(parents=True, exist_ok=False)
entries = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def register(target, source, transform):
    entries[str(target.relative_to(dest))] = {
        "sha256": sha(target),
        "source": str(source.relative_to(root)),
        "source_sha256": sha(source),
        "transformation": transform,
    }


def copy(source, target, sanitize=False):
    source = root / source
    target = dest / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if sanitize:
        target.write_text(source.read_text().replace(str(root) + "/", "[original-checkout]/"))
    else:
        shutil.copyfile(source, target)
    register(
        target,
        source,
        "absolute checkout prefix replaced; numbers unchanged"
        if sanitize
        else "byte-identical copy",
    )


v = "results/vllm_serving/jaisguard-v4"
latency_root = "results/latency_comparison/jaisguard-all-v2/report-final"
for name in [
    "comparison.md",
    "comparison.csv",
    "metrics.json",
    "latency-throughput.png",
    "latency-throughput.svg",
    "startup-memory.png",
    "startup-memory.svg",
]:
    copy(f"{v}/report/{name}", f"vllm/{name}")
for name in [
    "comparison.md",
    "comparison.csv",
    "latency.csv",
    "quality.png",
    "quality.svg",
    "latency.png",
    "latency.svg",
]:
    copy(f"{latency_root}/{name}", f"quality-latency/{name}", name == "comparison.md")
# Only aggregate metric sections; exclude per-row IDs and private absolute paths.
src = root / latency_root / "metrics.json"
d = json.loads(src.read_text())
target = dest / "quality-latency/aggregates.json"
target.write_text(
    json.dumps({"comparison": d["comparison"], "selected": d["selected"]}, indent=2) + "\n"
)
register(target, src, "only comparison and selected aggregate sections")
gates = []
for model in ["v2-frozen-43", "v2-lora-44"]:
    src = root / v / "parity" / model / "gate.json"
    d = json.loads(src.read_text())
    g = {
        k: d[k]
        for k in [
            "model_key",
            "passed",
            "acceptance_reference",
            "bf16_diagnostic_only",
            "trained_head_verified",
            "threshold",
            "policy",
        ]
    }
    g["comparisons"] = {
        key: {k: v for k, v in value.items() if k != "mismatch_ids"}
        for key, value in d["comparisons"].items()
    }
    gates.append(g)
target = dest / "vllm/parity-summary.json"
target.write_text(json.dumps(gates, indent=2) + "\n")
register(
    target,
    root / v / "parity/v2-frozen-43/gate.json",
    "aggregate projection of both gates; excludes IDs and private paths",
)
entries["vllm/parity-summary.json"]["additional_source"] = {
    "path": str(Path(v) / "parity/v2-lora-44/gate.json"),
    "sha256": sha(root / v / "parity/v2-lora-44/gate.json"),
}
timings = {}
sources = []
for path in sorted((root / v / "http").glob("*/*/c*/samples.json")):
    key = "/".join(path.relative_to(root / v / "http").parts[:-1])
    timings[key] = [r["http_ms"] for r in json.loads(path.read_text())]
    sources.append({"path": str(path.relative_to(root)), "sha256": sha(path)})
target = dest / "vllm/http-timings.json"
target.write_text(json.dumps(timings, indent=2) + "\n")
register(
    target,
    root / v / "http/v2-frozen-43/hf/c1/samples.json",
    "http_ms only from all 16 workloads; drops text, tokens, IDs and probabilities",
)
entries["vllm/http-timings.json"]["sources"] = sources
(dest / "allowlist.json").write_text(
    json.dumps(
        {
            "base_revision": "ea1986db76bc180783c0afefc8cffa319da9473b",
            "review": (
                "aggregate metrics, figures and timing-only observations; "
                "no original datasets or private notes"
            ),
            "files": entries,
        },
        indent=2,
    )
    + "\n"
)
print("Prepared", len(entries), "explicitly allowlisted files")
