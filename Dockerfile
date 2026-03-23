# syntax=docker/dockerfile:1.7

FROM python:3.14-slim AS builder

# Install uv binaries from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /uvx /bin/

WORKDIR /app

# Needed for packages that compile native extensions (e.g. llama-cpp-python).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first for better layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --compile-bytecode

# Copy project files and install the project into the same virtualenv.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --compile-bytecode


FROM python:3.14-slim AS runtime

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy prebuilt environment and app files, then switch to non-root.
COPY --from=builder --chown=appuser:appuser /app /app
RUN mkdir -p /data && chown -R appuser:appuser /data

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV VECTOR_DB_PATH=/data/vectors.db
ENV EMBEDDING_MODEL_PATH=/data/bge-small-en-v1.5-q8_0.gguf

VOLUME ["/data"]

EXPOSE 8000

# Keep healthcheck lightweight and dependency-free.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=5)" || exit 1

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
