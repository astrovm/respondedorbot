FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM python:3.14-slim AS builder
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.14-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY . .

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "run_polling.py"]
