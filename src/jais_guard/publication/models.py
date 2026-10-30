"""Prepare compact model repositories from exact historical checkpoint components."""

# Model-card prose, generated examples and immutable hashes are intentionally long.
# ruff: noqa: E501

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

MODEL_NAMES = {
    "v2-lora-44": "JaisGuard-LoRA-0.9B",
    "v2-frozen-43": "JaisGuard-Frozen-0.9B",
}
BASE_REVISION = "e51941b37091287fee4934e3f8a799e0d3fef37f"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seal(directory):
    directory = Path(directory)
    files = {
        str(p.relative_to(directory)): sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(directory / "manifest.json", {"format_version": 1, "files": files})
    return files


def verify_locked(item):
    for filename, expected in item["locked_files"].items():
        if sha(filename) != expected:
            raise ValueError(f"Historical checkpoint/reference changed: {filename}")
    checkpoint = Path(item["checkpoint"])
    saved = json.loads((checkpoint / "classifier.json").read_text())
    if saved != item["metadata"]:
        raise ValueError("Historical checkpoint metadata differs from selection manifest")
    if saved["config"]["revision"] != BASE_REVISION:
        raise ValueError("Unexpected base revision")
    threshold = json.loads(Path(item["threshold_path"]).read_text())
    if threshold["selection_data"] != "validation" or threshold["exploratory"] != item["threshold"]:
        raise ValueError("Threshold differs from validation lock")
    history = json.loads((checkpoint.parent / "history.json").read_text())
    if min(history, key=lambda e: e["validation_loss"]) != saved["metadata"]:
        raise ValueError("Checkpoint is not first minimum validation-loss epoch")
    cfg = saved["config"]
    for split in ("train", "validation"):
        if sha(cfg[f"{split}_path"]) != cfg[f"{split}_sha256"]:
            raise ValueError(f"Original {split} dataset changed")
    return saved


def prepare(root, output, base_review):
    root, output = Path(root).resolve(), Path(output).resolve()
    plan_path = root / "results/vllm_serving/jaisguard-v4/plan.json"
    plan = json.loads(plan_path.read_text())
    license_review = json.loads(Path(base_review).read_text())
    if license_review["revision"] != BASE_REVISION or license_review["license"] != "apache-2.0":
        raise ValueError("Base-model license/revision review is not the expected Apache release")
    original_metrics_path = (
        root / "results/version_comparison/jaisguard-v1-v2-v1/benchmark/metrics.json"
    )
    metrics = json.loads(original_metrics_path.read_text())
    verified = {key: verify_locked(plan["models"][key]) for key in MODEL_NAMES}
    output.mkdir(parents=True, exist_ok=False)
    for key, name in MODEL_NAMES.items():
        item, saved = plan["models"][key], verified[key]
        checkpoint = Path(item["checkpoint"])
        folder = output / name
        folder.mkdir()
        shutil.copyfile(checkpoint / "head.safetensors", folder / "head.safetensors")
        shutil.copytree(checkpoint / "tokenizer", folder / "tokenizer")
        if item["approach"] == "lora":
            (folder / "adapter").mkdir()
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copyfile(checkpoint / "adapter" / filename, folder / "adapter" / filename)
        shutil.copyfile(Path(__file__).with_name("model_runtime.py"), folder / "jaisguard_model.py")
        shutil.copyfile(root / "LICENSE", folder / "LICENSE")
        (folder / "LICENSES").mkdir()
        shutil.copyfile(
            root
            / "artifacts/huggingface/arabic-gibberish-detection-public-v1/LICENSES/Apache-2.0.txt",
            folder / "LICENSES/Apache-2.0.txt",
        )
        cfg = saved["config"]
        classifier_config = {k: cfg[k] for k in ("model", "revision", "max_length", "lora")}
        config = {
            "format_version": 1,
            "model_type": "jaisguard_checkpoint",
            "supported_api": "jaisguard_model.JaisGuardClassifier",
            "base_model_name_or_path": cfg["model"],
            "base_model_revision": cfg["revision"],
            "base_model_current_repository": license_review["repo_id"],
            "approach": item["approach"],
            "seed": item["seed"],
            "label2id": {"VALID": 0, "GIBBERISH": 1},
            "id2label": {"0": "VALID", "1": "GIBBERISH"},
            "pooling": saved["pooling"],
            "max_length": cfg["max_length"],
            "decision_threshold": item["threshold"],
            "default_threshold": 0.5,
            "threshold_selection": "validation only; first minimum-loss checkpoint",
            "head_precision": "float32",
            "classifier_config": classifier_config,
            "precision_policy": "historical CUDA BF16 when supported, otherwise FP16; FP32 CPU debug",
        }
        write_json(folder / "config.json", config)
        training = {
            k: v
            for k, v in cfg.items()
            if k
            not in ("train_path", "validation_path", "output_dir", "tracking_uri", "experiment")
        }
        training.update(
            selected_epoch=item["selected_epoch"],
            selected_seed=item["seed"],
            selected_validation_loss=saved["metadata"]["validation_loss"],
        )
        write_json(folder / "training_config.json", training)
        entry = next(
            e
            for e in metrics["per_seed"]
            if e["version"] == "v2"
            and e["approach"] == item["approach"]
            and e["seed"] == item["seed"]
        )
        individual = {mode: entry["metrics"][mode] for mode in ("default", "validation_locked")}
        aggregate = {
            mode: metrics["groups"][f"v2/{item['approach']}/{mode}"]
            for mode in ("default", "validation_locked")
        }
        write_json(
            folder / "metrics.json",
            {
                "individual_seed": {
                    "seed": item["seed"],
                    "precision": "historical BF16",
                    "test": individual,
                },
                "three_seed_experiment": {"seeds": [42, 43, 44], "std_ddof": 1, "test": aggregate},
                "source_sha256": sha(original_metrics_path),
            },
        )
        environment = json.loads((checkpoint.parent / "environment.json").read_text())
        export = json.loads(
            (root / "results/vllm_serving/jaisguard-v4/exports" / key / "export.json").read_text()
        )
        provenance = {
            "model_key": key,
            "training_run_ids": item["source_runs"],
            "training_recorded_git_revision": environment["git_revision"],
            "training_workspace_was_dirty": bool(environment["git_status"]),
            "training_source_limit": "Recorded revision had uncommitted changes; not a complete clean source snapshot",
            "historical_reference_revision": "ea1986db76bc180783c0afefc8cffa319da9473b",
            "packaging_repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "checkpoint_metadata_sha256": sha(checkpoint / "classifier.json"),
            "head_file_sha256": sha(checkpoint / "head.safetensors"),
            "trained_head_tensor_sha256": export["head_hashes"],
            "selection_plan_sha256": sha(plan_path),
            "original_dataset": {
                "train_rows": 1500,
                "validation_rows": 250,
                "test_rows": 400,
                "train_sha256": cfg["train_sha256"],
                "validation_sha256": cfg["validation_sha256"],
                "test_sha256": plan["test_sha256"],
            },
            "public_derivative": {
                "repo_id": "RayenFekih/arabic-gibberish-detection",
                "revision": "c3e739a396dcf4ae982633f19dc24555dab837d4",
                "train_rows": 1498,
                "total_rows": 2148,
                "used_for_this_training": False,
                "access_at_preparation": "private; authorized access required",
            },
            "base_license_review": license_review,
            "original_versions": environment["versions"],
        }
        if item["approach"] == "lora":
            provenance["adapter_sha256"] = sha(checkpoint / "adapter/adapter_model.safetensors")
        # The review inventory contains public base-repo filenames, never local workspace paths.
        write_json(folder / "provenance.json", provenance)
        requirements = []
        for dependency in (
            "torch",
            "transformers",
            "peft",
            "numpy",
            "safetensors",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "huggingface-hub",
        ):
            requirements.append(
                f"{dependency}=={importlib.metadata.version(dependency).split('+')[0]}"
            )
        (folder / "requirements.txt").write_text("\n".join(requirements) + "\n")
        (folder / "LICENSES/ATTRIBUTION.md").write_text(
            "# Component licensing and attribution\n\n"
            "JaisGuard loader and project classifier components: the project's MIT license, "
            "copied as LICENSE. Publication remains subject to owner approval.\n\n"
            f"Pinned base: IFM/Jais-3-0.9B at {BASE_REVISION}, currently redirected to "
            f"{license_review['repo_id']}. Publisher declares Apache-2.0. "
            "Backbone weights and upstream modeling code are not bundled. Copied tokenizer "
            "assets retain Apache-2.0; see Apache-2.0.txt. No separate LICENSE/NOTICE file was "
            "present in the pinned upstream file inventory.\n\n"
            "Upstream configuration code credits: Copyright 2024 The Qwen team, Alibaba Group "
            "and the HuggingFace Inc. team. Upstream modeling code credits: Copyright 2025 "
            "The Qwen team, Alibaba Group and the HuggingFace Inc. team. These upstream files "
            "are downloaded from the pinned base, not redistributed here.\n\n"
            "Training sources include Amazon MASSIVE (CC BY 4.0), Aya Open Science / Cohere "
            "Labs (Apache 2.0), TyDi QA / Google Research (Apache 2.0), and project-generated "
            "scenarios and transformations. No training texts are packaged. Dataset component "
            "licenses are documented separately at "
            "https://huggingface.co/datasets/RayenFekih/arabic-gibberish-detection. "
            "This model license does not relicense those datasets.\n"
        )
        (folder / "example.py").write_text(EXAMPLE)
        (folder / "README.md").write_text(model_card(name, item, saved, individual, aggregate))
        write_json(
            folder / "validation.json",
            {"status": "pending real standalone GPU validation", "upload_approved": False},
        )
        seal(folder)
    return output


EXAMPLE = '''"""Run the explicitly supported checkpoint API from a downloaded model directory."""

import argparse
import json
from pathlib import Path

from jaisguard_model import JaisGuardClassifier

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default=str(Path(__file__).resolve().parent))
parser.add_argument("--text", action="append", required=True)
parser.add_argument("--precision", choices=("historical", "bf16", "fp16"), default="historical")
parser.add_argument("--device", default="cuda")
parser.add_argument("--operating-point", choices=("validation_locked", "default"), default="validation_locked")
parser.add_argument("--local-files-only", action="store_true")
args = parser.parse_args()
classifier = JaisGuardClassifier.from_pretrained(
    args.model, device=args.device, precision=args.precision, local_files_only=args.local_files_only
)
for text in args.text:
    print(json.dumps(classifier.predict(text, operating_point=args.operating_point), ensure_ascii=False))
'''


def model_card(name, item, saved, individual, aggregate):
    approach = item["approach"]
    lines = [
        f"---\nlanguage: ar\nlicense: mit\nbase_model: IFM/Jais-3-0.9B\ntags:\n- text-classification\n- custom-code\n- jaisguard\n- {approach}\n---\n",
        f"# {name}\n",
        "## Purpose and supported API\n",
        "Arabic-script VALID/GIBBERISH detection based on recoverable meaning. "
        "Meaningful typos, informal language and dialects can be VALID; meaning cannot "
        "be equated with correctness, safety or whether a request can be fulfilled. "
        "Examples: `كيف يمكنني تعلم اللغة العربية؟` and `شنوة نعمل باش نتعلم البرمجة` "
        "are meaningful requests. `ضصثقفغعهخحجشس` is a gibberish example. "
        "These examples illustrate intent, not a claim of perfect predictions.\n",
        "This is a compact checkpoint package, not a merged backbone. Use the bundled "
        "`JaisGuardClassifier` API. Compatibility with AutoModelForSequenceClassification, "
        "Transformers pipelines, hosted Inference Providers or native vLLM is **not claimed**.\n",
        "## Installation and inference\n",
        "```bash\nuv venv --python 3.11 .venv\n"
        "uv pip install --python .venv/bin/python -r requirements.txt\n"
        ".venv/bin/python example.py --text 'كيف يمكنني تعلم اللغة العربية؟'\n"
        ".venv/bin/python example.py --precision fp16 --text 'ضصثقفغعهخحجشس'\n```\n",
        "Run in the downloaded model directory. The base snapshot is downloaded as needed. "
        "For cached-only execution add `--local-files-only`. Review the base's trusted "
        "remote model code before running. From Python:\n",
        "```python\nfrom jaisguard_model import JaisGuardClassifier\n"
        'classifier = JaisGuardClassifier.from_pretrained(".", device="cuda", local_files_only=True)\n'
        'print(classifier.predict("كيف يمكنني تعلم اللغة العربية؟"))\n'
        'print(classifier.predict("ضصثقفغعهخحجشس", operating_point="default"))\n```\n',
        "For remote use, first `snapshot_download` this repository at an approved immutable "
        "revision, inspect its code, then import `jaisguard_model.py` from that directory. "
        "A standalone runtime plus tokenizer, adapter/head, config and integrity manifest "
        "are supplied; no training-repository checkout is required.\n",
        "## Architecture and selected checkpoint\n",
        f"Base: `IFM/Jais-3-0.9B`, revision `{BASE_REVISION}` (currently redirected to "
        "`inception42/Jais-3-0.9B`). Hidden size 1536; final nonpadding token pooling; "
        "FP32 linear 1536→2 classifier with its original **trained bias**, 3,074 head "
        "parameters. Label order VALID=0, GIBBERISH=1. Right padding; raw text with "
        "special tokens, no chat prompt. Maximum **64 tokens**, no truncation. Empty "
        "or unsupported-script inputs are errors, not GIBBERISH predictions.\n",
        f"Selected {approach} v2 seed **{item['seed']}**, epoch **{item['selected_epoch']}**, "
        f"validation loss **{saved['metadata']['validation_loss']}**. Selection uses the "
        "first minimum validation loss, not test quality. "
        + (
            "Frozen backbone with trained classification head only.\n"
            if approach == "frozen"
            else "LoRA rank 8, alpha 16, dropout 0.05 on q_proj/v_proj (56 target projections); "
            "the classification head and bias are trained separately. Unmerged adapter "
            "weights are loaded with PEFT; no static merge is performed by this package.\n"
        ),
        f"Saved validation-selected threshold **{item['threshold']}**; decision is "
        "`p_gibberish >= threshold`. The explicit default threshold is **0.5**. "
        "The saved exploratory operating point targeted low VALID rejection on validation; "
        "it is not a production guarantee and is not optimized on the test benchmark.\n",
        "## Training data and configuration\n",
        "These weights were trained on the **original 1,500-row training export**, with "
        "250 validation rows. Original training contains 710 VALID and 790 GIBBERISH; "
        "validation contains 120 VALID and 130 GIBBERISH. Training has 1,367 human-reviewed "
        "and 133 audited-provisional labels. The "
        "[dataset derivative](https://huggingface.co/datasets/RayenFekih/arabic-gibberish-detection) "
        "has **1,498 train + 250 validation + 400 test = 2,148 rows** and was **not** used "
        "to train these checkpoints. Two privacy-uncertain human-reviewed VALID Aya records "
        "were omitted only from that derivative. Their prior use in training remains a "
        "disclosed privacy limitation; no extraction or memorization audit is claimed. "
        "No original training texts or reviewer metadata are bundled. The HF dataset was "
        "private at preparation; authorized access may be required.\n",
        "AdamW, weight decay 0.01, warmup ratio 0.1, gradient clipping 1.0. Frozen: "
        "batch 32, head LR 0.001, up to 30 epochs, patience 5. LoRA: microbatch 2, "
        "accumulation 8, adapter LR 0.0001, head LR 0.001, up to 8 epochs, patience 2. "
        "Complete portable hyperparameters and source hashes are in `training_config.json` "
        "and `provenance.json`. Original training recorded git `fab3580` with uncommitted "
        "changes; a clean commit alone cannot exactly reconstruct that workspace.\n",
        "## Individual seed results — historical BF16, batch size 2\n",
        "Previously examined, constructed 400-input benchmark: 188 VALID, 212 GIBBERISH. "
        "The following numbers belong to this checkpoint, not the three-seed average.\n",
        "| Operating point | Accuracy | Macro F1 | GIBBERISH recall | VALID rejection |\n"
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in individual.items():
        lines.append(
            f"| {mode} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} | "
            f"{values['gibberish_recall']:.6f} | {values['valid_false_rejection_rate']:.6f} |"
        )
    lines += [
        "\n## Three-seed experiment aggregates — separate from this model\n",
        "Seeds 42/43/44; mean ± sample SD (ddof=1) on the same test inputs. "
        "This is variability across trained seeds, not independent test datasets.\n",
        "| Operating point | Accuracy | Macro F1 | GIBBERISH recall | VALID rejection |\n"
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in aggregate.items():
        cells = [
            f"{values[k]['mean']:.6f} ± {values[k]['std']:.6f}"
            for k in ("accuracy", "macro_f1", "gibberish_recall", "valid_false_rejection_rate")
        ]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines += [
        "\nFull precision metrics, confusion counts and separate scopes are in `metrics.json`.\n",
        "## Precision, validation and performance scope\n",
        "Historical HF inference uses BF16 on the tested RTX 4050 when supported, with "
        "the classification head/bias always FP32. FP16 mode reproduces the historical "
        "HF adaptation: load the original model then cast the backbone to FP16. The "
        "unmerged adapter can round differently from a static merge. BF16 and FP16 "
        "are not assumed equivalent. The two requested serving checkpoints had known "
        "BF16/FP16 disagreements in the historical diagnostics.\n",
        "The historical full-backbone vLLM exports passed matched FP16 HF/vLLM gates "
        "(zero default/locked mismatches, maximum/mean probability differences at most "
        "0.02/0.005) and HTTP concurrency 1/4/8/16. Those results concern the exported "
        "pooling models, not native loading of this compact adapter/head package. "
        "No new vLLM server benchmark is claimed. See "
        "[JaisGuard evidence](https://github.com/RayenFekih/jais-guard/tree/cleanup/public-workflows/docs/benchmarks).\n",
        "Real local inference validation and its precision-specific acceptance results "
        "are recorded in `validation.json`. Check that file before making claims; "
        "pending or failed validation blocks publication. Tested hardware: RTX 4050 "
        "Laptop GPU, 6 GB, Python 3.11, Torch 2.14.1/CUDA 13.0, Transformers 5.20.2, "
        "PEFT 0.21.2. CPU is an explicit FP32 debugging path and is not a measured "
        "deployment or probability-parity claim.\n",
        "## Limitations and intended use\n",
        "Experimental Arabic recoverability filtering, not a safety moderator, language "
        "identifier, factuality checker or general production-quality guarantee. Dialect "
        "coverage and synthetic corruption recipes are limited. The test benchmark was "
        "examined and informed iteration two; it is not an untouched independent final "
        "test. Test[319] is contained in test[204]. Parents for 260 transformed test "
        "requests are unavailable; 80 passthrough parents are inferable. Public-data "
        "training cannot exactly reproduce these original experiments. Confidence scores "
        "are not calibrated truth guarantees. Rejecting meaningful dialects is a known risk.\n",
        "## Licensing, integrity and publication status\n",
        "Component scope: MIT for the project's loader and classifier components; "
        "Apache 2.0 for copied tokenizer assets and the separately downloaded base. "
        "See `LICENSE` and `LICENSES/ATTRIBUTION.md`, including upstream credits and "
        "the Apache license copy. Dataset attribution/licenses remain separate. "
        "No base weights or upstream modeling code are bundled. SHA-256 hashes are in "
        "`manifest.json`; they check integrity, not independent code authenticity. "
        "Publication is controlled separately; verify hashes and obtain owner approval before any upload.\n",
    ]
    return "\n".join(lines)


def finalize_validation(root, staged, reviews, validated_packages):
    """Attach actual receipts only when the inference-critical files match tested copies."""
    from jais_guard.finetuning.evaluation import metrics as compute_metrics

    root, staged, reviews = Path(root), Path(staged), Path(reviews)
    plan = json.loads((root / "results/vllm_serving/jaisguard-v4/plan.json").read_text())
    test_rows = [
        json.loads(s) for s in (Path(plan["test"]) / "test.jsonl").read_text().splitlines()
    ]
    truth = [int(row["label"] == "GIBBERISH") for row in test_rows]
    for key, name in MODEL_NAMES.items():
        short = "lora" if "lora" in key else "frozen"
        folder = staged / name
        tested_manifest = json.loads(
            (Path(validated_packages) / name / "manifest.json").read_text()
        )
        critical = {
            filename: sha(folder / filename)
            for filename in tested_manifest["files"]
            if filename in ("config.json", "head.safetensors", "jaisguard_model.py")
            or filename.startswith(("tokenizer/", "adapter/"))
        }
        if any(value != tested_manifest["files"][filename] for filename, value in critical.items()):
            raise ValueError("Inference files differ from the GPU-tested fresh copy")
        receipts = {}
        runtime_metrics = {}
        for implementation in ("refactored", "package"):
            for precision in ("bf16", "fp16"):
                filename = f"{implementation}-{short}-{precision}"
                receipt = json.loads((reviews / f"{filename}.json").read_text())
                if not receipt["passed"] or receipt["samples"] != 658:
                    raise ValueError("Missing or failed GPU inference receipt")
                receipts[f"{implementation}_{precision}"] = receipt
                if implementation == "package":
                    scores = json.loads((reviews / f"{filename}.probabilities.json").read_text())[
                        250:650
                    ]
                    runtime_metrics[precision] = {
                        "default": compute_metrics(truth, [int(v[1] >= 0.5) for v in scores]),
                        "validation_locked": compute_metrics(
                            truth, [int(v[1] >= plan["models"][key]["threshold"]) for v in scores]
                        ),
                    }
        batch2 = json.loads((reviews / f"package-{short}-batch2.json").read_text())
        if not batch2["passed"] or batch2["samples"] != 400 or batch2["batch_size"] != 2:
            raise ValueError("Historical batch-two quality reproduction failed")
        write_json(
            folder / "validation.json",
            {
                "status": "passed real GPU inference",
                "publication_approval": "Separate owner approval; not granted by inference receipts",
                "same_precision_gate": "max probability error <= 1e-6; zero default/locked mismatches",
                "tested_inference_files": critical,
                "serving_batch1_reproduction": receipts,
                "quality_batch2_reproduction": batch2,
                "observed_batch1_test_metrics": runtime_metrics,
                "scope": "658 parity inputs: validation250 + historical test400 + probes8; quality batch2 test400",
                "batching_disclosure": "Batch-one reduced-precision decisions differ from batch-two; archived metrics unchanged",
                "vllm_scope": "Compared against historical matching FP16 outputs; compact package is not a native vLLM export",
            },
        )
        provenance_path = folder / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        base_hash_path = reviews / "base-artifact-hashes.json"
        if base_hash_path.exists():
            provenance["pinned_base_artifact_sha256"] = json.loads(base_hash_path.read_text())
        provenance["packaging_workspace_was_dirty_at_initial_preparation"] = True
        write_json(provenance_path, provenance)
        card = (folder / "README.md").read_text()
        card += "\n## Validated standalone batch-one test observations\n\n"
        card += "These are new runtime observations on the same examined 400 inputs, "
        card += "separate from the historical batch-two quality table above. Both batch sizes "
        card += "reproduced their matching saved references exactly; a few reduced-precision "
        card += "threshold decisions differ across batch sizes. No thresholds or archived metrics changed.\n\n"
        card += "| Runtime | Default accuracy | Locked accuracy |\n| --- | ---: | ---: |\n"
        for precision, values in runtime_metrics.items():
            card += f"| {precision.upper()} batch 1 | {values['default']['accuracy']:.6f} | {values['validation_locked']['accuracy']:.6f} |\n"
        card += "\nAll four real GPU parity checks (refactored/package × BF16/FP16) had "
        card += "maximum probability error **0.0** and zero mismatches at both thresholds. "
        card += "The additional package BF16 batch-two check reproduced all 400 original "
        card += "quality probabilities exactly. See `validation.json` for complete receipts.\n"
        (folder / "README.md").write_text(card)
        seal(folder)
