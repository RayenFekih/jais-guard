"""Explicit local MLflow tracking; no autologging or external evaluators."""

import importlib.metadata
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from .evaluation import write_json


def configure(config):
    import mlflow

    uri = config["tracking_uri"]
    if uri.startswith(("http://", "https://")):
        if urlparse(uri).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("This pipeline only permits local MLflow tracking")
    elif not uri.startswith("sqlite:///"):
        raise ValueError("Use localhost MLflow or a local SQLite URI")
    os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["experiment"])
    return mlflow


def environment():
    import torch

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    versions = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "mlflow",
        "numpy",
        "scikit-learn",
        "datasets",
        "accelerate",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "versions": versions,
        "torch_cuda": torch.version.cuda,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "determinism": "seeded; deterministic algorithms warn_only=True; kernel limits apply",
    }


def log_setup(mlflow, config, directory, train, val, manifests):
    import pandas as pd

    write_json(directory / "config.json", config)
    write_json(directory / "environment.json", environment())
    mlflow.log_dict(config, "configuration.json")
    mlflow.log_dict(environment(), "environment.json")
    for name, rows, manifest in zip(("train", "validation"), (train, val), manifests, strict=True):
        frame = pd.DataFrame(
            [
                {k: row.get(k) for k in ("id", "input_text", "label", "label_origin", "family_id")}
                for row in rows
            ]
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=str(Path(config[f"{name}_path"]).resolve()),
            name=manifest["version"],
            targets="label",
        )
        mlflow.log_input(dataset, context="training" if name == "train" else "validation")
        mlflow.log_dict(manifest, f"datasets/{name}_manifest.json")
        mlflow.log_params(
            {f"{name}_version": manifest["version"], f"{name}_sha256": config[f"{name}_sha256"]}
        )


def log_metrics(mlflow, prefix, values, step=None):
    mlflow.log_metrics(
        {
            f"{prefix}/{k}": float(v)
            for k, v in values.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        step=step,
    )


def log_genai(mlflow, records):
    """Optional deterministic scoring of cached outputs; never repeats inference."""
    from mlflow.genai.scorers import scorer

    @scorer
    def correct(outputs, expectations):
        return outputs["prediction"] == expectations["label"]

    @scorer
    def invalid_output(outputs):
        return outputs["prediction"] == -1

    # This is supplementary row-level inspection, not the aggregate metric source.
    return mlflow.genai.evaluate(
        data=[
            {
                "inputs": {"id": r["id"], "text": r["input_text"]},
                "outputs": {"prediction": r["prediction_default"]},
                "expectations": {"label": r["truth_id"]},
            }
            for r in records
        ],
        scorers=[correct, invalid_output],
    )


def log_classifier(mlflow, checkpoint, threshold):
    import mlflow.pyfunc
    import pandas as pd
    from mlflow.models import infer_signature

    class LocalClassifier(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import torch

            from jais_guard.finetuning.model import load_checkpoint

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.tokenizer, self.saved = load_checkpoint(
                context.artifacts["checkpoint"], self.device
            )

        def predict(self, context, model_input, params=None):
            import numpy as np

            from jais_guard.finetuning.model import predict_rows

            text = model_input["text"].tolist()
            if any(not isinstance(t, str) or not t.strip() for t in text):
                raise ValueError("Expected nonempty Arabic-script text")
            import unicodedata

            if any(
                c.isalpha() and "ARABIC" not in unicodedata.name(c, "") for t in text for c in t
            ):
                raise ValueError("Unsupported script; no gibberish label assigned")
            rows = [{"id": str(i), "input_text": t} for i, t in enumerate(text)]
            if not rows:
                return pd.DataFrame(
                    columns=["p_gibberish", "prediction_default", "prediction_exploratory"]
                )
            scores, _ = predict_rows(
                self.model, self.tokenizer, rows, self.saved["config"]["max_length"], self.device
            )
            scores = np.asarray(scores)
            return pd.DataFrame(
                {
                    "p_gibberish": scores,
                    "prediction_default": (scores >= 0.5).astype(int),
                    "prediction_exploratory": (scores >= threshold).astype(int),
                }
            )

    # Exclude optimizer/RNG state from the inference package.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / "checkpoint"
        shutil.copytree(checkpoint, target, ignore=shutil.ignore_patterns("training_state.pt"))
        example = pd.DataFrame({"text": ["مرحبا"]})
        output = pd.DataFrame(
            {"p_gibberish": [0.1], "prediction_default": [0], "prediction_exploratory": [0]}
        )
        info = mlflow.pyfunc.log_model(
            name="classifier",
            python_model=LocalClassifier(),
            artifacts={"checkpoint": str(target)},
            code_paths=[str(Path(__file__).parents[1])],
            signature=infer_signature(example, output),
            pip_requirements=[
                f"{name}=={importlib.metadata.version(name)}"
                for name in (
                    "mlflow",
                    "torch",
                    "transformers",
                    "peft",
                    "numpy",
                    "pandas",
                    "safetensors",
                    "accelerate",
                    "scikit-learn",
                )
            ],
            metadata={
                "backbone_source": "pinned local Hugging Face cache; not bundled",
                "threshold_exploratory": threshold,
            },
        )
    return info.model_uri
