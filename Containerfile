FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM rust:1.98.0-slim AS rust-builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --locked --workspace --release

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
COPY --from=rust-builder \
    /app/target/release/librespondedorbot_rs.so \
    /app/respondedorbot_rs.so
COPY --from=rust-builder /app/target/release/botd /usr/local/bin/botd

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "run_polling.py"]
