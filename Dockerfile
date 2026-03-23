# Build stage - install uv and sync dependencies
FROM python:3.14-slim AS build

WORKDIR /app

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy dependency metadata first (best for layer caching)
COPY pyproject.toml uv.lock ./

# Sync dependencies without project code
RUN UV_COMPILE_BYTECODE=1 UV_NO_DEV=1 uv sync --frozen

# Copy application code
COPY . .

# Runtime stage
FROM python:3.14-slim

WORKDIR /app

# Copy uv from build stage (faster than installing in each build)
COPY --from=build /root/.local /root/.local
ENV PATH="/root/.local/bin/:$PATH"

# Copy virtual environment from build stage
COPY --from=build /app/.venv /app/.venv

# Copy application code
COPY . .

# Create data directory for model + database
RUN mkdir -p /app/data

# Bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

EXPOSE 8000

CMD ["uv", "run", "main.py"]
