"""Small public workflow entry point; historical experiment commands remain in scripts/."""

import argparse
import json
import os
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare-data", help="Download and verify the pinned public derivative")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--source", type=Path, help="Existing exact public release (offline)")
    sub.add_parser("cache-model", help="Download the pinned backbone/tokenizer for offline loading")
    inspect = sub.add_parser("inspect", help="Validate a prepared training configuration")
    inspect.add_argument("--config", type=Path, required=True)
    train = sub.add_parser("train", help="Train a NEW model; this command requires CUDA")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--approach", choices=("frozen", "lora"), required=True)
    train.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    for name in ("predict", "evaluate", "serve"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument(
            "--operating-point",
            choices=("default", "validation_locked"),
            default="validation_locked",
        )
        if name != "serve":
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "predict":
            command.add_argument("--text", required=True)
        elif name == "evaluate":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8000)
    benchmark = sub.add_parser("verify-evidence", help="Verify reviewed public evidence hashes")
    benchmark.add_argument("--directory", type=Path, default=Path("docs/benchmarks"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
    from . import public_workflow as workflow

    if args.command == "cache-model":
        from huggingface_hub import snapshot_download

        print(snapshot_download(workflow.MODEL, revision=workflow.MODEL_REVISION))
    elif args.command == "prepare-data":
        config = workflow.prepare(args.output, args.source)
        print(
            json.dumps(
                {
                    "config": str(args.output.resolve() / "training.json"),
                    "historical_reproduction": config["historical_reproduction"],
                }
            )
        )
    elif args.command in ("inspect", "train"):
        from .finetuning.data import load_datasets

        config = workflow.configuration(args.config)
        train, validation, tm, vm = load_datasets(config)
        if args.command == "inspect":
            print(
                json.dumps(
                    {
                        "train": len(train),
                        "validation": len(validation),
                        "train_sha256": tm["sha256"],
                        "validation_sha256": vm["sha256"],
                    }
                )
            )
            return
        from .serving import cuda_required

        cuda_required()
        from .finetuning.model import load_classifier, seed_everything
        from .finetuning.tracking import configure, log_setup
        from .finetuning.training import train_one

        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        mlflow = configure(config)
        with mlflow.start_run(run_name=f"public-{args.approach}-{args.seed}") as run:
            directory = Path(config["output_dir"]) / run.info.run_id
            directory.mkdir(exist_ok=False)
            log_setup(mlflow, config, directory, train, validation, (tm, vm))
            seed_everything(args.seed)
            model, tokenizer = load_classifier(config, args.approach, "cuda", args.seed)
            train_one(model, tokenizer, train, validation, config, args.seed, directory, mlflow)
            print(directory)
    elif args.command == "predict":
        from .finetuning.model import load_checkpoint, predict_rows

        item = workflow.checkpoint_manifest(args.checkpoint, args.operating_point)["models"][
            "checkpoint"
        ]
        model, tokenizer, saved = load_checkpoint(args.checkpoint, args.device)
        scores, _ = predict_rows(
            model,
            tokenizer,
            [{"input_text": args.text}],
            saved["config"]["max_length"],
            args.device,
        )
        score = float(scores[0])
        label = int(score >= item["threshold"])
        print(
            json.dumps(
                {
                    "class_id": label,
                    "class": ["VALID", "GIBBERISH"][label],
                    "probabilities": {"VALID": 1 - score, "GIBBERISH": score},
                    "threshold": item["threshold"],
                }
            )
        )
    elif args.command == "evaluate":
        print(
            json.dumps(
                workflow.evaluate(
                    args.checkpoint, args.data, args.output, args.operating_point, args.device
                )
            )
        )
    elif args.command == "serve":
        import uvicorn

        from .serving import create_app

        uvicorn.run(
            create_app(checkpoint=args.checkpoint, operating_point=args.operating_point),
            host=args.host,
            port=args.port,
            workers=1,
        )
    else:
        from .evidence import verify

        print(json.dumps(verify(args.directory)))


if __name__ == "__main__":
    main()
