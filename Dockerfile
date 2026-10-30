FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MLFLOW_ENABLE_TELEMETRY=false
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra finetune
EXPOSE 8000
# Explicitly select a mounted checkpoint or historical manifest. One resident GPU model.
CMD ["uvicorn", "jais_guard.serving:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
