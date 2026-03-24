# syntax=docker/dockerfile:1.7

FROM python:3.14-slim

WORKDIR /app

# Build + runtime dependencies for native Python extensions.
# Keeping these in one stage avoids missing shared-object issues.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    libstdc++6 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Install uv directly in this stage.
RUN pip install --no-cache-dir uv==0.5.5

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

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

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

USER appuser

CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
