"""Preserve completed work after the diagnostics-only TorchVersion IPC fix."""

import hashlib
import shutil
from pathlib import Path

from jais_guard.latency_benchmark import write_new
from jais_guard.serving import read_json

from .artifacts import checked_export, seal, sha, verify_hashes
from .parity import inputs

OLD_VERSION = '"torch_version": torch.__version__,'
NEW_VERSION = '"torch_version": str(torch.__version__),'


def validate_previous_plan(root):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if seal({k: v for k, v in plan.items() if k != "identity"}) != plan["identity"]:
        raise ValueError("Previous plan changed")
    runtime = Path(__file__).with_name("runtime.py").resolve()
    source = runtime.read_text()
    if source.count(NEW_VERSION) != 1 or OLD_VERSION in source:
        raise ValueError("Expected the one-line diagnostics serialization repair")
    previous = source.replace(NEW_VERSION, OLD_VERSION)
    if hashlib.sha256(previous.encode()).hexdigest() != plan["files"].get(str(runtime)):
        raise ValueError("Runtime differs beyond the approved TorchVersion-to-string fix")
    verify_hashes({p: h for p, h in plan["files"].items() if Path(p).resolve() != runtime})
    return plan, runtime


def repair(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Destination must be a new versioned directory")
    previous, runtime = validate_previous_plan(source)
    plan = {**previous, "files": dict(previous["files"])}
    plan.pop("identity")
    plan["files"][str(runtime)] = sha(runtime)
    for p in (Path(__file__).resolve(), Path("scripts/repair_vllm_serialization.py").resolve()):
        plan["files"][str(p)] = sha(p)
    plan["repair_provenance"] = {
        "previous_plan": str(source / "plan.json"),
        "previous_plan_sha256": sha(source / "plan.json"),
        "change": "Worker TorchVersion metadata converted to plain string; inference unchanged",
        "model_threshold_tokenizer_engine_and_tolerance_changes": False,
    }
    plan["identity"] = seal(plan)
    destination.mkdir(parents=True)
    write_new(destination / "plan.json", plan)
    reused = []
    for key in plan["models"]:
        old_artifact = source / "exports" / key
        old_receipt = read_json(old_artifact / "export.json")
        if (
            seal({k: v for k, v in old_receipt.items() if k != "identity"})
            != old_receipt["identity"]
            or old_receipt["plan_identity"] != previous["identity"]
            or old_receipt["model_key"] != key
        ):
            raise ValueError("Previous export receipt changed")
        verify_hashes(old_receipt["files"])
        verify_hashes(old_receipt["base_files"])
        new_artifact = destination / "exports" / key
        new_artifact.mkdir(parents=True)
        files = {}
        for filename, expected in old_receipt["files"].items():
            filename = Path(filename)
            if filename.parent.resolve() != old_artifact.resolve():
                raise ValueError("Export file is outside its artifact directory")
            target = new_artifact / filename.name
            shutil.copyfile(filename, target)
            if sha(target) != expected:
                raise ValueError("Copied export changed")
            files[str(target)] = expected
        receipt = {**old_receipt, "plan_identity": plan["identity"], "files": files}
        receipt.pop("identity")
        receipt["copied_from"] = {
            "path": str(old_artifact / "export.json"),
            "sha256": sha(old_artifact / "export.json"),
        }
        receipt["identity"] = seal(receipt)
        write_new(new_artifact / "export.json", receipt)
        checked_export(destination, key)
        expected_inputs = None
        for backend in ("hf-original", "hf-fp16"):
            old_folder = source / "parity" / key / backend
            completion_path = old_folder / "completion.json"
            if not completion_path.exists():
                continue  # Failed or absent attempts are retained only in the old directory.
            completion = read_json(completion_path)
            predictions = old_folder / "predictions.jsonl"
            if (
                completion["status"] != "complete"
                or completion["model_key"] != key
                or completion["backend"] != backend
                or completion["plan_identity"] != previous["identity"]
                or completion["export_identity"] != old_receipt["identity"]
                or completion["predictions_sha256"] != sha(predictions)
            ):
                raise ValueError("Cannot reuse an unverified HF reference")
            import json

            records = [json.loads(line) for line in predictions.read_text().splitlines()]
            expected_inputs = expected_inputs or inputs(destination, key)
            if completion["samples"] != len(expected_inputs) or len(records) != len(
                expected_inputs
            ):
                raise ValueError("Reference input count changed")
            for record, row in zip(records, expected_inputs, strict=True):
                if any(record[k] != row[k] for k in ("id", "category", "token_ids")):
                    raise ValueError("Reference IDs, ordering or tokenization changed")
            new_folder = destination / "parity" / key / backend
            new_folder.mkdir(parents=True)
            shutil.copyfile(predictions, new_folder / "predictions.jsonl")
            migrated = {
                **completion,
                "plan_identity": plan["identity"],
                "export_identity": receipt["identity"],
                "reused_from": {"path": str(completion_path), "sha256": sha(completion_path)},
            }
            write_new(new_folder / "completion.json", migrated)
            reused.append(f"{key}/{backend}")
    result = {
        "source": str(source),
        "destination": str(destination),
        "reused_hf": reused,
        "gpu_inference_run": False,
        "parity_passed": False,
        "failed_vllm_attempts_copied": False,
    }
    write_new(destination / "repair.json", result)
    return result
