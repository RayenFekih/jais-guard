"""Immutable plans and exact CPU exports; no GPU model or vLLM import here."""

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from jais_guard.benchmarks.latency import write_new
from jais_guard.serving.hf import DEFAULT_DIRECTORY, checked_manifest, frozen_inputs, read_json

DEFAULT_ROOT = Path("results/vllm_serving/jaisguard-v2")
VLLM_SITE = Path(".venv-vllm/lib/python3.11/site-packages")
ENGINE = dict(
    model_impl="transformers",
    runner="pooling",
    convert="classify",
    trust_remote_code=True,
    dtype="float16",
    enforce_eager=True,
    max_model_len=1024,
    gpu_memory_utilization=0.75,
    max_num_seqs=16,
    max_num_batched_tokens=1024,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    seed=42,
    worker_extension_cls="jais_guard.vllm_support.runtime.WorkerDiagnostics",
)
POLICY = {
    "max_abs_probability_difference": 0.02,
    "mean_abs_probability_difference": 0.005,
    "require_all_default_and_locked_classes_match": True,
}
PROBES = [
    ("clean", "كيف يمكنني تعلم اللغة العربية؟"),
    ("clean", "هذا نص عربي مفهوم وواضح"),
    ("dialect", "شنوة نعمل باش نتعلم البرمجة"),
    ("dialect", "عايز اعرف الجو عامل إيه"),
    ("dialect", "شو فينا نعمل اليوم"),
    ("recoverable_corruption", "كيفيم كنني تعلم اللغه العربيه"),
    ("gibberish", "ضصثقفغعهخحجشس"),
    ("gibberish", "قلم نافذة يركض أزرق لماذا كرسي"),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def inspect_installation():
    dist = importlib.metadata.Distribution.at(VLLM_SITE / "vllm-0.32.0.dist-info")
    sources = {}
    for name in (
        "entrypoints/pooling/offline.py",
        "entrypoints/pooling/classify/api_router.py",
        "model_executor/models/transformers/pooling.py",
        "model_executor/models/transformers/legacy.py",
        "config/pooler.py",
    ):
        path = VLLM_SITE / "vllm" / name
        sources[str(path.resolve())] = sha(path)
    return {
        "version": dist.version,
        "environment": str(Path(".venv-vllm").resolve()),
        "interfaces": ["LLM.classify", "AsyncLLM.encode(task=classify)", "/classify"],
        "sources": sources,
        "native_lora_status": "Not asserted for classification. Use verified static PEFT merge.",
        "bias_issue": "0.32.0 LegacyMixin drops score.bias. Export uses classifier.bias instead.",
    }


def prepare(root=DEFAULT_ROOT, source=DEFAULT_DIRECTORY, keys=None):
    root = Path(root)
    if root.exists():
        raise ValueError("Use a new versioned results directory")
    parent = checked_manifest(source)
    frozen_inputs(parent)
    keys = keys or [parent["selected"]["v2/frozen"], parent["selected"]["v2/lora"]]
    models = {}
    files = {str((Path(source) / "manifest.json").resolve()): sha(Path(source) / "manifest.json")}
    files.update(parent["files"])
    for key in keys:
        item = parent["models"][key]
        if item["approach"] not in ("frozen", "lora"):
            raise ValueError("Only trained classifiers can be exported")
        models[key] = item
        files.update(item["locked_files"])
    for path in (Path("scripts/vllm_classifier.py"), Path("scripts/serve_vllm_classifier.py")):
        files[str(path.resolve())] = sha(path)
    for path in Path(__file__).parent.glob("*.py"):
        files[str(path.resolve())] = sha(path)
    for path in (Path(parent["test"]) / n for n in ("test.jsonl", "manifest.json", "freeze.json")):
        files[str(path.resolve())] = sha(path)
    installation = inspect_installation()
    files.update(installation["sources"])
    plan = {
        "models": models,
        "files": files,
        "engine": ENGINE,
        "parity_policy": POLICY,
        "installation": installation,
        "source_manifest": str(Path(source).resolve()),
        "source_identity": parent["identity"],
        "test": parent["test"],
        "test_sha256": parent["test_sha256"],
        "probes": PROBES,
        "selection": "Reuse validation-only minimum-loss selections from source manifest",
    }
    plan["identity"] = seal(plan)
    root.mkdir(parents=True)
    write_new(root / "plan.json", plan)
    return plan


def verify_hashes(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Pinned artifact changed: {path}")


def checked_plan(root):

    plan = read_json(Path(root) / "plan.json")
    if seal({k: v for k, v in plan.items() if k != "identity"}) != plan["identity"]:
        raise ValueError("vLLM plan changed")
    verify_hashes(plan["files"])
    return plan


def checked_export(root, key):

    plan = checked_plan(root)
    artifact = Path(root) / "exports" / key
    receipt = read_json(artifact / "export.json")
    if receipt["plan_identity"] != plan["identity"] or receipt["model_key"] != key:
        raise ValueError("Export belongs to a different selection")
    if seal({k: v for k, v in receipt.items() if k != "identity"}) != receipt["identity"]:
        raise ValueError("Export receipt changed")
    verify_hashes(receipt["files"])
    verify_hashes(receipt.get("base_files", {}))
    return plan, artifact.resolve(), receipt


def export_classifier(root, key):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    plan = checked_plan(root)
    item = plan["models"][key]
    artifact = Path(root) / "exports" / key
    artifact.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(item["checkpoint"])
    saved = read_json(checkpoint / "classifier.json")
    if saved != item["metadata"] or saved["label_mapping"] != {"VALID": 0, "GIBBERISH": 1}:
        raise ValueError("Saved classifier metadata differs from locked selection")
    if saved["pooling"] != "last_nonpadding_final_hidden":
        raise ValueError("Unexpected original pooling policy")
    snapshot = Path(
        snapshot_download(
            saved["config"]["model"], revision=saved["config"]["revision"], local_files_only=True
        )
    )
    config = read_json(snapshot / "config.json")
    hidden_size = config["hidden_size"]
    original_head = load_file(str(checkpoint / "head.safetensors"))
    if (
        set(original_head) != {"weight", "bias"}
        or original_head["weight"].shape != (2, hidden_size)
        or original_head["bias"].shape != (2,)
        or any(t.dtype != torch.float32 for t in original_head.values())
    ):
        raise ValueError("Expected the actual two-class FP32 weight AND bias")
    # Stream base tensors and merge one projection at a time. No full CPU/GPU
    # model is constructed; this avoids a multi-GB FP32 model allocation.
    adapters = {}
    if item["approach"] == "lora":
        ac = read_json(checkpoint / "adapter/adapter_config.json")
        cfg = saved["config"]["lora"]
        unsupported = (
            ac.get("fan_in_fan_out"),
            ac.get("use_dora"),
            ac.get("use_rslora"),
            ac.get("rank_pattern"),
            ac.get("alpha_pattern"),
            ac.get("modules_to_save"),
            ac.get("lora_bias"),
            ac.get("layer_replication"),
        )
        if (
            any(unsupported)
            or ac.get("bias") != "none"
            or ac["r"] != cfg["rank"]
            or ac["lora_alpha"] != cfg["alpha"]
            or set(ac["target_modules"]) != set(cfg["target_modules"])
        ):
            raise ValueError("Unsupported LoRA variant; refusing approximate adapter merge")
        state = load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
        for name, value in state.items():
            if not name.startswith("base_model.model."):
                raise ValueError("Unexpected adapter module prefix")
            for suffix, kind in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
                if name.endswith(suffix):
                    target = "model." + name.removeprefix("base_model.model.").removesuffix(suffix)
                    adapters.setdefault(target + ".weight", {})[kind] = value.float()
                    break
            else:
                raise ValueError("Unexpected non-LoRA adapter tensor")
        expected = {
            f"model.layers.{i}.self_attn.{target}.weight"
            for i in range(config["num_hidden_layers"])
            for target in cfg["target_modules"]
        }
        if set(adapters) != expected or any(set(pair) != {"A", "B"} for pair in adapters.values()):
            raise ValueError("Missing or unexpected adapter projection")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        base_paths = sorted({snapshot / n for n in read_json(index_path)["weight_map"].values()})
    else:
        base_paths = [snapshot / "model.safetensors"]
    weight_map, shard, pending_bytes, shard_index, seen, merged = {}, {}, 0, 0, set(), []
    total_size = 0

    def flush():
        nonlocal shard, pending_bytes, shard_index
        if not shard:
            return
        filename = f"model-{shard_index:05d}.safetensors"
        save_file(shard, str(artifact / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})
        shard, pending_bytes, shard_index = {}, 0, shard_index + 1

    for path in base_paths:
        with safe_open(path, framework="pt", device="cpu") as base:
            for name in base.keys():
                if name.startswith("lm_head."):
                    continue
                if not name.startswith("model.") or name in seen:
                    raise ValueError(f"Unexpected/duplicate base tensor: {name}")
                seen.add(name)
                tensor = base.get_tensor(name)
                if name in adapters:
                    pair = adapters[name]
                    delta = pair["B"] @ pair["A"]
                    if delta.shape != tensor.shape or pair["A"].shape[0] != cfg["rank"]:
                        raise ValueError("Adapter projection shape mismatch")
                    # Exact standard PEFT formula; FP32, no approximation or quantization.
                    tensor = tensor.float() + delta * (cfg["alpha"] / cfg["rank"])
                    if not torch.isfinite(tensor).all():
                        raise ValueError("Nonfinite merged adapter weights")
                    merged.append(name)
                size = tensor.numel() * tensor.element_size()
                total_size += size
                if pending_bytes + size > 64 * 2**20:
                    flush()
                shard[name] = tensor.contiguous()
                pending_bytes += size
                if pending_bytes >= 64 * 2**20:
                    flush()
    if set(merged) != set(adapters):
        raise ValueError("An adapter did not match a base weight")
    shard.update({"classifier." + n: value for n, value in original_head.items()})
    flush()
    total_size += sum(v.numel() * v.element_size() for v in original_head.values())
    write_new(
        artifact / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    # Reopen the stored head: both tensors must survive serialization unchanged.
    for name, value in original_head.items():
        location = weight_map["classifier." + name]
        with safe_open(artifact / location, framework="pt") as stored:
            if not torch.equal(stored.get_tensor("classifier." + name), value):
                raise ValueError("Exported trained head changed")
    config.update(
        architectures=["JaisGuardForSequenceClassification"],
        num_labels=2,
        id2label={0: "VALID", 1: "GIBBERISH"},
        label2id={"VALID": 0, "GIBBERISH": 1},
        problem_type="single_label_classification",
        head_dtype="float32",
        use_cache=False,
    )
    config["auto_map"] = {
        "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
        "AutoModel": "modeling_k2_horizon.K2HorizonModel",
        "AutoModelForSequenceClassification": (
            "modeling_jaisguard.JaisGuardForSequenceClassification"
        ),
    }
    config["pooler_config"] = {
        "task": "classify",
        "seq_pooling_type": "LAST",
        "use_activation": True,
    }
    write_new(artifact / "config.json", config)
    for name in ("configuration_k2_horizon.py", "modeling_k2_horizon.py"):
        shutil.copyfile(snapshot / name, artifact / name)
    shutil.copyfile(
        Path(__file__).with_name("modeling_jaisguard.py"), artifact / "modeling_jaisguard.py"
    )
    for path in (checkpoint / "tokenizer").iterdir():
        if not path.is_file() or (artifact / path.name).exists():
            raise ValueError("Unexpected tokenizer export contents")
        shutil.copyfile(path, artifact / path.name)
    receipt = {
        "plan_identity": plan["identity"],
        "model_key": key,
        "threshold": item["threshold"],
        "label_order": ["VALID", "GIBBERISH"],
        "pooling": "LAST nonpadding input token",
        "head_hashes": {n: tensor_hash(v) for n, v in original_head.items()},
        "merged_lora_modules": len(merged),
        "merged_names": sorted(merged),
        "backbone_export_dtype": "Original base dtype; merged LoRA projections FP32",
        "head_dtype": "float32",
        "base_revision": saved["config"]["revision"],
        "base_files": {str(p.resolve()): sha(p) for p in base_paths + [snapshot / "config.json"]},
        "parity_status": "NOT RUN; export alone is not conversion success",
        "files": {str(p.resolve()): sha(p) for p in artifact.iterdir() if p.is_file()},
    }
    receipt["identity"] = seal(receipt)
    write_new(artifact / "export.json", receipt)
    checked_export(root, key)
    return receipt


def check_export_structure(root, key):
    """Check vLLM's HF architecture-discovery contract on meta, without inference."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from .parity import inputs

    plan, artifact, receipt = checked_export(root, key)
    config = AutoConfig.from_pretrained(artifact, trust_remote_code=True, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(
            config, trust_remote_code=True, dtype=torch.float16
        )
    if (
        tuple(model.classifier.weight.shape) != (2, config.hidden_size)
        or tuple(model.classifier.bias.shape) != (2,)
        or model.classifier.weight.dtype != torch.float32
        or config.id2label != {0: "VALID", 1: "GIBBERISH"}
    ):
        raise ValueError("Sequence-classification discovery contract differs")
    rows = inputs(root, key)
    checks = Path(root) / "structure-checks"
    checks.mkdir(exist_ok=True)
    result = {
        "model_key": key,
        "plan_identity": plan["identity"],
        "export_identity": receipt["identity"],
        "architecture": type(model).__name__,
        "head_shape": list(model.classifier.weight.shape),
        "bias_shape": [2],
        "head_dtype": "float32",
        "tokenizer_identical_inputs": len(rows),
        "max_tokens": max(len(r["token_ids"]) for r in rows),
        "transformers_version": importlib.metadata.version("transformers"),
        "status": "CPU structure/tokenizer verified; GPU parity NOT RUN",
    }
    write_new(checks / f"{key}.json", result)
    return result
