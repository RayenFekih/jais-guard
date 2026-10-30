# Jais Fine-Tuning

A Python workspace for Jais inference and fine-tuning experiments, managed with [`uv`](https://docs.astral.sh/uv/).

## Prerequisites

- Python **3.11** (pinned in `.python-version` / `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

## Installing uv

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal afterwards so `uv` is on your `PATH`, or verify with:

```bash
uv --version
```

## Setting up the project

From the project root (same command on Windows, macOS, and Linux):

```bash
uv sync
```

This creates a `.venv` virtual environment and installs dependencies from `pyproject.toml`/`uv.lock` (including the `dev` group, e.g. `ruff`). For a production-only install (no dev dependencies):

```bash
uv sync --no-dev
```

## Activating the virtual environment

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**

```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux (bash/zsh):**

```bash
source .venv/bin/activate
```

To deactivate on any platform, run `deactivate`.

You don't strictly need to activate the environment to run things — `uv run <command>` (e.g. `uv run python main.py`) runs a command inside the project's environment without activating it.

## Running the service

`main.py` is currently an empty placeholder; no service is implemented yet.

```bash
python main.py
```

or, without activating the environment first:

```bash
uv run python main.py
```

## Configuration

Application settings live in [src/config.py](src/config.py) (a Pydantic `Config` model). Local secrets/environment variables can be added to a `.env` file (copy `.env.example` to `.env`) and loaded via `python-dotenv`; `.env` itself is gitignored.

## Linting

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting, configured in `ruff.toml`.

```bash
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
main.py           # application entrypoint
src/
  config.py       # application settings (Pydantic model)
  logger.py       # logging configuration
data/             # static/reference data
tests/            # test scripts
logs/             # runtime logs (gitignored)
docs/             # project documentation
```

## Deployment

The [Dockerfile](Dockerfile) builds from `python:3.11-slim` and installs dependencies via `uv sync --frozen --no-dev` (from `pyproject.toml`/`uv.lock`). Its service command requires an implemented `main:app` and the `gunicorn` dependency before it can run.

```bash
docker build -t project-template .
```

## Push to GitHub

Create an empty GitHub repository without a README, license, or `.gitignore`, then run these commands from the project root in a terminal with write access to `.git`:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Initial commit"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Replace `OWNER/REPOSITORY` with your GitHub owner and repository name. Review the staged files before committing. Virtual environments, environment secrets, local agent/cloud configuration, logs, and generated model weights are excluded by `.gitignore`; `uv.lock` and `.env.example` are included.

## JaisGuard baseline inference

Initial `VALID` / `GIBBERISH` inference, using Transformers and PyTorch in `.venv`:

```bash
.venv/bin/python scripts/smoke_baseline.py
```

See [baseline setup and research conditions](docs/baseline.md) for the repository
inspection, official reasoning controls, CPU checks, configuration, output format,
and validation limits. This is Phase 1 diagnostic functionality; no training is
implemented.

## Local dataset annotation

The 100-candidate, data-only pilot is ready for blind review:

```bash
.venv/bin/python -m streamlit run apps/annotate.py
```

Open http://localhost:8501. See [annotation workflow and source provenance](docs/annotation.md).
Labels, notes, flags and progress persist locally in SQLite. No model is loaded.

## Hidden-representation fine-tuning

The frozen-head and LoRA pipelines, CPU tests, local MLflow tracking, and disposable
GPU smoke checks are documented in [fine-tuning workflow](docs/finetuning.md).
Full experiments require separate approval; begin with the WSL smoke commands there.

The next [evaluation phase](docs/evaluation-phase.md) adds the versioned untuned
validation prompt baseline and the isolated, human-reviewed `test-400-v1` preparation
and freeze workflow. It does not evaluate unfinished test candidates.

For the complete cached-quality comparison, batch-1 GPU benchmarks and local
FastAPI/httpx demo across untuned Jais, frozen v1/v2 and LoRA v1/v2, see
[the latency and serving workflow](docs/latency-serving.md). It preserves Jev's
existing remote API measurements and all historical experiments.
