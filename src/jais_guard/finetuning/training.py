"""Single-device training with exact accumulation and validation-only selection."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import LABELS, collate, encode_rows
from .evaluation import metrics, save_report, write_json
from .model import (
    autocast,
    optimizer_for,
    predict_rows,
    restore_weights,
    save_checkpoint,
    verify_trainable,
)
from .tracking import log_classifier, log_metrics


def tokenizer_identity(tokenizer):
    state = json.loads(tokenizer.backend_tokenizer.to_str())
    # Runtime padding/truncation is set by each batch; exclude it from tokenizer identity.
    state.pop("padding", None)
    state.pop("truncation", None)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def feature_cache(model, tokenizer, rows, config, partition, device):
    identity = {
        "model": config["model"],
        "revision": config["revision"],
        "dataset": config[f"{partition}_sha256"],
        "max_length": config["max_length"],
        "tokenizer": tokenizer_identity(tokenizer),
        "padding_side": tokenizer.padding_side,
        "pooling": "last_nonpadding_final_hidden",
        "dtype": str(next(model.backbone.parameters()).dtype),
        "attention": "sdpa",
        "feature_batch_size": config.get("feature_batch_size", 2),
        "torch_version": str(torch.__version__),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(config["output_dir"]) / "feature_cache" / f"{key}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["ids"] != [r["id"] for r in rows]:
            raise ValueError("Feature-cache identity mismatch")
        return saved["features"]
    encoded = encode_rows(rows, tokenizer, config["max_length"])
    features = []
    model.eval()
    # no_grad rather than inference_mode: cached tensors will feed a trainable head.
    with torch.no_grad():
        feature_batch = config.get("feature_batch_size", 2)
        for start in range(0, len(rows), feature_batch):
            with autocast(device):
                value = model.features(
                    **collate(encoded[start : start + feature_batch], tokenizer, device)
                )
            features.append(value.cpu().float())
    result = torch.cat(features)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "ids": [r["id"] for r in rows], "features": result}, path)
    return result


def train_one(model, tokenizer, train, val, config, seed, directory, mlflow):
    cfg = config[model.approach]
    device = next(model.head.parameters()).device
    train_y = torch.tensor([LABELS[r["label"]] for r in train])
    val_y = torch.tensor([LABELS[r["label"]] for r in val])
    encoded = encode_rows(train, tokenizer, config["max_length"])
    train_x = val_x = None
    if model.approach == "frozen":
        train_x = feature_cache(model, tokenizer, train, config, "train", device)
        val_x = feature_cache(model, tokenizer, val, config, "validation", device)
    micro = cfg.get("microbatch", cfg.get("batch_size"))
    accumulation = cfg.get("accumulation", 1)
    batches = math.ceil(len(train) / micro)
    updates_per_epoch = math.ceil(batches / accumulation)
    total_updates = cfg["epochs"] * updates_per_epoch
    optimizer = optimizer_for(model, config)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, math.ceil(total_updates * config["warmup_ratio"]), total_updates
    )
    from .model import precision

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and precision(device) == torch.float16)
    )
    best, bad_epochs, global_step = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train), generator=generator).tolist()
        model.train()
        epoch_sum = 0.0
        # Accumulation groups divide by their actual number of examples, including the tail.
        for group_start in range(0, batches, accumulation):
            optimizer.zero_grad(set_to_none=True)
            start = group_start * micro
            group = order[start : min(len(train), (group_start + accumulation) * micro)]
            group_loss = 0.0
            for offset in range(0, len(group), micro):
                ix = group[offset : offset + micro]
                if train_x is not None:
                    logits = model.head(train_x[ix].to(device))
                else:
                    batch = collate([encoded[i] for i in ix], tokenizer, device)
                    with autocast(device):
                        logits = model(**batch)
                loss_sum = F.cross_entropy(logits.float(), train_y[ix].to(device), reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss_sum / len(group)).backward()
                group_loss += loss_sum.item()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["gradient_clip"],
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            epoch_sum += group_loss
            mlflow.log_metrics(
                {
                    "train/loss": group_loss / len(group),
                    "train/gradient_norm": float(norm),
                    **{
                        f"train/lr_group_{i}": g["lr"] for i, g in enumerate(optimizer.param_groups)
                    },
                },
                step=global_step,
            )
        model.eval()
        with torch.inference_mode():
            if val_x is not None:
                val_logits = model.head(val_x.to(device)).cpu()
            else:
                _, val_logits = predict_rows(
                    model, tokenizer, val, config["max_length"], device, micro
                )
        val_loss = F.cross_entropy(val_logits, val_y).item()
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        measured = metrics(val_y.tolist(), (val_logits.softmax(-1)[:, 1] >= 0.5).int().tolist())
        measured["loss"] = val_loss
        log_metrics(mlflow, "validation_default", measured, epoch + 1)
        entry = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "seed": seed,
            "train_loss": epoch_sum / len(train),
            "validation_loss": val_loss,
        }
        history.append(entry)
        write_json(directory / "history.json", history)
        save_checkpoint(
            directory / "last", model, tokenizer, config, entry, optimizer, scheduler, scaler
        )
        if val_loss < best:
            best, bad_epochs = val_loss, 0
            save_checkpoint(
                directory / "best", model, tokenizer, config, entry, optimizer, scheduler, scaler
            )
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch + 1} train_loss={entry['train_loss']:.5f} "
            f"val_loss={val_loss:.5f} step={global_step}",
            flush=True,
        )
        if bad_epochs >= cfg["patience"]:
            break
    restore_weights(model, directory / "best")
    if val_x is not None:
        with torch.inference_mode():
            logits = model.head(val_x.to(device)).cpu()
            scores = logits.softmax(-1)[:, 1].tolist()
    else:
        scores, logits = predict_rows(model, tokenizer, val, config["max_length"], device, micro)
    report_rows = [
        {
            **row,
            "input_tokens": len(tokens["input_ids"]),
            "logit_valid": float(logits[i, 0]),
            "logit_gibberish": float(logits[i, 1]),
        }
        for i, (row, tokens) in enumerate(
            zip(val, encode_rows(val, tokenizer, config["max_length"]), strict=True)
        )
    ]
    report = save_report(
        directory / "evaluation", report_rows, scores, limit=config["exploratory_frr"]
    )
    write_json(
        directory / "threshold.json",
        {
            "default": 0.5,
            "exploratory": report["exploratory"]["threshold"],
            "selection_data": "validation",
            "production_requirement": False,
        },
    )
    for name in ("default", "exploratory"):
        log_metrics(mlflow, name, report[name])
    mlflow.log_metrics(
        {
            "resources/seconds": time.perf_counter() - started,
            **{f"parameters/{k}": v for k, v in verify_trainable(model).items()},
        }
    )
    if device.type == "cuda":
        mlflow.log_metrics(
            {
                "resources/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "resources/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
    mlflow.log_artifacts(str(directory))
    uri = log_classifier(mlflow, directory / "best", report["exploratory"]["threshold"])
    write_json(directory / "logged_model.json", {"model_uri": uri})
    mlflow.log_dict({"model_uri": uri}, "logged_model.json")
    return report
