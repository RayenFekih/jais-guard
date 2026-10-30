"""HF serving entry point; implementation lives in :mod:`jais_guard.serving.hf`."""

from .hf import create_app

__all__ = ["create_app"]
