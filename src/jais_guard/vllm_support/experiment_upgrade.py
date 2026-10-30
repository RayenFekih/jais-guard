"""New, explicitly authorized FP16-equivalence experiment; preserve all prior runs."""

import json
import shutil
from pathlib import Path

from jais_guard.latency_benchmark import write_new
from jais_guard.serving import read_json

from .artifacts import checked_export, seal, sha, verify_hashes
from .parity import inputs, verify

# Exact source versions that produced the already verified v3 observations.
PREVIOUS_HASHES = {
    "parity.py": "32c089881b514ef5587762bcc974f6ccff1673592906588563b80537cac12b3b",
    "server.py": "2ac11d8523787a929de242d67353d3d614f44d8706142715c387c98084eaba65",
    "benchmark.py": "8683b6bcc63bfa5030caa5944d7001cfc23c20fdf27e1cd57d08c5027b50d85a",
}


def upgrade(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new versioned destination")
    old = read_json(source / "plan.json")
    if seal({k: v for k, v in old.items() if k != "identity"}) != old["identity"]:
        raise ValueError("Previous plan changed")
    changed = {str(Path(__file__).with_name(n).resolve()): h for n, h in PREVIOUS_HASHES.items()}
    for path, expected in old["files"].items():
        if path in changed:
            if expected != changed[path]:
                raise ValueError("Unrecognized previous experiment source")
        elif sha(path) != expected:
            raise ValueError(f"Previous artifact changed: {path}")
    plan = {**old, "files": dict(old["files"])}
    plan.pop("identity")
    for path in Path(__file__).parent.glob("*.py"):
        plan["files"][str(path.resolve())] = sha(path)
    launcher = Path("scripts/prepare_vllm_fp16_experiment.py").resolve()
    plan["files"][str(launcher)] = sha(launcher)
    plan["acceptance"] = {
        "reference": "hf-fp16",
        "bf16_diagnostic_only": True,
        "verified_trained_head_and_bias_required": True,
        "zero_default_and_locked_mismatches_required": True,
        "authorization": "User explicitly accepted BF16 drift for this FP16 engineering experiment",
        "hf_benchmark_precision": "float16",
    }
    plan["experiment_provenance"] = {
        "previous_plan": str(source / "plan.json"),
        "previous_plan_sha256": sha(source / "plan.json"),
        "threshold_weights_labels_tolerances_changed": False,
    }
    plan["identity"] = seal(plan)
    destination.mkdir(parents=True)
    write_new(destination / "plan.json", plan)
    reused, gates = [], {}
    for key in plan["models"]:
        old_artifact = source / "exports" / key
        old_receipt = read_json(old_artifact / "export.json")
        if (
            seal({k: v for k, v in old_receipt.items() if k != "identity"})
            != old_receipt["identity"]
            or old_receipt["plan_identity"] != old["identity"]
        ):
            raise ValueError("Previous export receipt changed")
        verify_hashes(old_receipt["files"])
        verify_hashes(old_receipt["base_files"])
        artifact = destination / "exports" / key
        artifact.mkdir(parents=True)
        files = {}
        for path, expected in old_receipt["files"].items():
            if Path(path).parent.resolve() != old_artifact:
                raise ValueError("Unexpected export location")
            target = artifact / Path(path).name
            shutil.copyfile(path, target)
            if sha(target) != expected:
                raise ValueError("Copied export differs")
            files[str(target)] = expected
        receipt = {**old_receipt, "plan_identity": plan["identity"], "files": files}
        receipt.pop("identity")
        receipt["copied_from"] = {
            "path": str(old_artifact / "export.json"),
            "sha256": sha(old_artifact / "export.json"),
        }
        receipt["identity"] = seal(receipt)
        write_new(artifact / "export.json", receipt)
        checked_export(destination, key)
        expected_inputs = None
        complete = []
        for backend in ("hf-original", "hf-fp16", "vllm"):
            previous = source / "parity" / key / backend
            if not (previous / "completion.json").exists():
                continue
            completion = read_json(previous / "completion.json")
            predictions = previous / "predictions.jsonl"
            if (
                completion["status"] != "complete"
                or completion["model_key"] != key
                or completion["backend"] != backend
                or completion["plan_identity"] != old["identity"]
                or completion["export_identity"] != old_receipt["identity"]
                or completion["predictions_sha256"] != sha(predictions)
            ):
                raise ValueError("Previous reference is incomplete or incompatible")
            records = [json.loads(line) for line in predictions.read_text().splitlines()]
            expected_inputs = expected_inputs or inputs(destination, key)
            if len(records) != len(expected_inputs) or completion["samples"] != len(
                expected_inputs
            ):
                raise ValueError("Reference count changed")
            for a, b in zip(records, expected_inputs, strict=True):
                if any(a[field] != b[field] for field in ("id", "category", "token_ids")):
                    raise ValueError("Reference input/order/tokenization changed")
            folder = destination / "parity" / key / backend
            folder.mkdir(parents=True)
            shutil.copyfile(predictions, folder / "predictions.jsonl")
            completion.update(
                plan_identity=plan["identity"],
                export_identity=receipt["identity"],
                reused_from={
                    "path": str(previous / "completion.json"),
                    "sha256": sha(previous / "completion.json"),
                },
            )
            write_new(folder / "completion.json", completion)
            reused.append(f"{key}/{backend}")
            complete.append(backend)
        if len(complete) == 3:
            gates[key] = verify(destination, key)["passed"]
    result = {
        "source": str(source),
        "destination": str(destination),
        "reused": reused,
        "acceptance_reference": "hf-fp16",
        "gates": gates,
        "gpu_inference_run": False,
        "numeric_tolerances_changed": False,
    }
    write_new(destination / "upgrade.json", result)
    return result
