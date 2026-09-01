FROM rust:1.98.0-slim AS rust-builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

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

COPY deploy/certificates/supabase-root-2021-ca.crt /usr/local/share/ca-certificates/supabase-root-2021-ca.crt
RUN update-ca-certificates \
    && test -L /etc/ssl/certs/supabase-root-2021-ca.pem

COPY --from=rust-builder /app/target/release/botd /usr/local/bin/botd

RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

CMD ["/usr/local/bin/botd"]
