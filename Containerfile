FROM rust:1.98.0-slim AS chef
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && cargo install cargo-chef --version 0.1.78 --locked

FROM chef AS planner
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS rust-builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --locked --release --package botd --recipe-path recipe.json

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --locked --release -p botd

FROM debian:trixie-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopus0 \
    libssl3t64 \
    libzstd1 \
    zlib1g \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder /app/target/release/botd /usr/local/bin/botd

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

CMD ["/usr/local/bin/botd"]
