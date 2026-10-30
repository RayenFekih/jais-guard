"""Single-worker GPU inference demo; select model via JAISGUARD_MODEL."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from jais_guard.serving.hf import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
