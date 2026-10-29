"""Disposable GPU feasibility checks; never invokes the experiment training loop."""

import gc
import math
import tempfile
import time
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import LABELS, collate, encode_rows
from .evaluation import write_json
from .model import (
    autocast,
    load_checkpoint,
    load_classifier,
    optimizer_for,
    precision,
    save_checkpoint,
    seed_everything,
    verify_trainable,
)
from .tracking import environment


def gpu_smoke(config, approach, microbatch, train, validation):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not visible here. Execute this smoke command in your WSL terminal."
        )
    seed_everything(42)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_classifier(config, approach, "cuda", 42)
    counts = verify_trainable(model)
    # Training examples only; never adapt even a disposable model to validation labels.
    examples = [next(r for r in train if r["label"] == label) for label in ("VALID", "GIBBERISH")]
    examples = examples[:microbatch]
    encoded = encode_rows(examples, tokenizer, config["max_length"])
    batch = collate(encoded, tokenizer, "cuda", pad_to=config["max_length"])
    labels = torch.tensor([LABELS[r["label"]] for r in examples], device="cuda")
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=precision("cuda") == torch.float16)
    frozen_versions = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    train_before = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    # Two updates expose lazy optimizer allocations and test initially zero LoRA branches.
    losses = []
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            loss = F.cross_entropy(model(**batch), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite smoke loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for name, param in model.named_parameters():
            if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                raise AssertionError(f"Missing or nonfinite gradient: {name}")
            if not param.requires_grad and param.grad is not None:
                raise AssertionError(f"Frozen parameter acquired gradient: {name}")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    for name, param in model.named_parameters():
        if not param.requires_grad and param._version != frozen_versions[name]:
            raise AssertionError(f"Frozen parameter changed: {name}")
    changed = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and not torch.equal(p.detach().cpu(), train_before[n])
    ]
    if not any(n.startswith("head.") for n in changed):
        raise AssertionError("Head did not update")
    if approach == "lora" and not any("lora_" in n for n in changed):
        raise AssertionError("Adapters did not update")
    model.eval()
    with torch.inference_mode(), autocast("cuda"):
        reference = model(**batch).float().cpu()
        unpadded = model(**collate(encoded, tokenizer, "cuda")).float().cpu()
    torch.testing.assert_close(reference, unpadded, rtol=0.03, atol=0.03)
    free, total = torch.cuda.mem_get_info()
    memory = {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_gib": free / 2**30,
        "total_gib": total / 2**30,
    }
    output = Path(config["output_dir"]) / "smoke" / f"{approach}-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="jaisguard-smoke-") as temp:
        save_checkpoint(temp, model, tokenizer, config, {"seed": 42, "smoke_only": True})
        # Free the first model before reloading; no two-backbone VRAM requirement.
        del optimizer, model, batch, labels, loss, scaler, param
        gc.collect()
        torch.cuda.empty_cache()
        reloaded, reloaded_tokenizer, _ = load_checkpoint(temp, "cuda")
        with torch.inference_mode(), autocast("cuda"):
            restored = (
                reloaded(
                    **collate(encoded, reloaded_tokenizer, "cuda", pad_to=config["max_length"])
                )
                .float()
                .cpu()
            )
        torch.testing.assert_close(reference, restored, rtol=1e-4, atol=1e-4)
        del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    passed = memory["free_gib"] >= 1.0 and all(math.isfinite(v) for v in losses)
    report = {
        "status": "PASS" if passed else "INSUFFICIENT_HEADROOM",
        "approach": approach,
        "microbatch": microbatch,
        "sequence_length": config["max_length"],
        "precision": str(precision("cuda")),
        "attention": "sdpa",
        "parameters": counts,
        "losses": losses,
        "updates": 2,
        "padding_invariance": True,
        "checkpoint_reload": True,
        "base_parameters_unchanged": True,
        "changed_trainable_tensors": len(changed),
        "memory": memory,
        "seconds": time.perf_counter() - started,
        "environment": environment(),
        "dataset_hashes": {k: config[k] for k in ("train_sha256", "validation_sha256")},
        "validation_examples": len(validation),
        "validation_used_for_updates": False,
    }
    write_json(output / "smoke.json", report)
    print(
        f"{report['status']}: {approach}, microbatch={microbatch}, seq=64, "
        f"trainable={counts['trainable']:,}, free={memory['free_gib']:.2f} GiB",
        flush=True,
    )
    print(
        "PASS: finite gradients, frozen weights, optimizer updates, padding, checkpoint reload"
        f"\n{output}"
    )
    if not passed:
        raise RuntimeError(
            "Less than 1 GiB headroom. Retry --microbatch 1; do not start full runs."
        )
