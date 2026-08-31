# respondedorbot

An AI Telegram bot that plays "el gordo", a blunt Argentine character that replies in lowercase using Argentine slang. The application is one native Rust binary.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- Streaming AI chat with conversation memory, web search, tools, and provider fallback
- Crypto, stock, ETF, index, fund, futures, dollar, BCRA, weather, Polymarket, and Hacker News data
- Audio transcription, image description, summaries, and memory compaction
- Telegram Stars billing, shared group credits, transfers, charge history, and reconciliation
- Durable scheduled tasks with leases, recurrence, cancellation, and restart recovery
- Telegram callbacks, payments, media, command localization, and supported-link repair

## Build and run

Requirements: Rust 1.98, PostgreSQL, Redis Stack with RediSearch, and FFmpeg.

```bash
cp .env.example .env
# Configure the required secrets and either BOT_SYSTEM_PROMPT or workspace/SOUL.md.
cargo build --locked --release -p botd
set -a
. ./.env
set +a
./target/release/botd --check-config
./target/release/botd
```

The process owns Telegram polling, background price refresh, memory compaction, billing reconciliation, and scheduled-task execution. Stop it with `SIGINT` or `SIGTERM`; shutdown waits for background workers.

## Configuration

| Required variable | Purpose |
| --- | --- |
| `TELEGRAM_TOKEN` | Bot token from BotFather |
| `TELEGRAM_USERNAME` | Bot username, with or without `@` |
| `SUPABASE_POSTGRES_URL` | PostgreSQL billing and chat-configuration database |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `OPENROUTER_API_KEY` | OpenRouter chat, vision, summary, and fallback key |
| `BOT_SYSTEM_PROMPT` | Complete personality prompt; may instead come from `workspace/SOUL.md` and `workspace/RULES.md` |

Redis defaults to `localhost:6379`. Optional provider, reconciliation, polling, maintenance, and monitoring settings are documented in [.env.example](.env.example).

## Commands

| Command | Main aliases | Description |
| --- | --- | --- |
| `/ask` | `/pregunta`, `/che`, `/gordo` | AI chat |
| `/resumen` | `/summary`, `/tldr` | Conversation summary |
| `/transcribe` | `/describe` | Audio transcription or image description |
| `/prices` | `/price`, `/precios`, `/precio`, `/c` | Crypto and traditional-market prices |
| `/crypto` | `/criptos` | Crypto-only prices and conversions |
| `/clima` | `/weather` | Current weather |
| `/dolar` | `/dollar`, `/usd` | Dollar rates |
| `/petroleo` | `/oil` | Oil prices |
| `/acciones` | `/stocks` | Stock prices |
| `/eleccion` | `/elections` | Polymarket elections |
| `/devo`, `/rulo` |  | Arbitrage calculations |
| `/powerlaw`, `/rainbow`, `/satoshi` | `/sat`, `/sats` | Bitcoin reference models |
| `/bcra` | `/variables` | BCRA variables |
| `/random`, `/convertbase`, `/comando`, `/time` | `/command` | Utilities |
| `/config`, `/language` | `/settings`, `/idioma` | Chat settings and language |
| `/topup`, `/balance`, `/charges`, `/transfer` | `/history`, `/gastos` | Credits and billing |
| `/tarea`, `/tareas` | `/task`, `/tasks` | Create, list, and cancel tasks |
| `/gm`, `/gn`, `/help`, `/instance` |  | Greetings and bot information |

`/prices` resolves crypto through CoinMarketCap, then unresolved symbols and company names through Yahoo Finance. Full Solana/EVM addresses and `$ticker` messages use token cards where available.

## Architecture

```text
botd             composition, polling, workers, scheduling, orchestration
  -> bot-adapters Telegram, HTTP, Redis, PostgreSQL, providers, media
       -> bot-core deterministic domain types, parsing, routing, and state machines
```

The dependency direction is enforced by the Cargo workspace. Untrusted payloads are decoded at adapter boundaries. Core routing, billing, AI, and scheduling behavior uses typed states and actions.

## Tests and quality gates

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

Integration tests use synthetic data. Set `TEST_REDIS_URL`, `TEST_POSTGRES_URL`, and `TEST_DATABASE_URL` to exercise Redis Stack and PostgreSQL paths. CI enforces line coverage of 95% for `bot-core`, 85% for `bot-adapters`, and 80% for the I/O-heavy `botd` composition crate.

## Container and deployment

```bash
podman build --tag respondedorbot:local .
mkdir -p ~/.config/containers/systemd ~/.config/systemd/user ~/respondedorbot/workspace
cp quadlets/* ~/.config/containers/systemd/
cp systemd/respondedorbot-maintenance.* systemd/respondedorbot-podman-prune.* ~/.config/systemd/user/
cp .env.example ~/respondedorbot/.env
# Create ~/respondedorbot/workspace/SOUL.md and RULES.md, or set BOT_SYSTEM_PROMPT.
systemctl --user daemon-reload
systemctl --user enable --now respondedorbot-maintenance.timer respondedorbot-podman-prune.timer
systemctl --user start respondedorbot-redis.service respondedorbot.service
```

The runtime image contains `/usr/local/bin/botd`, FFmpeg, and native shared libraries. It contains no scripting-language runtime. Run maintenance with `podman exec systemd-respondedorbot /usr/local/bin/botd --maintenance`.

CI publishes `latest` and immutable `sha-<full-commit-sha>` images. Roll back by pinning the Quadlet `Image=` line to a previously verified SHA tag, reloading user units, and restarting the service. Never run two pollers with the same Telegram token.

## Repository layout

- `crates/bot-core` — deterministic application and domain behavior
- `crates/bot-adapters` — external-system implementations
- `crates/botd` — native executable and composition root
- `contracts` — language-neutral compatibility and persistence fixtures
- `docs/rust-migration` — migration decisions and final verification record
- `quadlets`, `systemd` — deployment and maintenance units
- `Containerfile` — Rust-only production image
