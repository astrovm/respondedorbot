# Respondedorbot

An AI Telegram bot with an Argentine personality.

It streams replies, remembers conversations, understands media, tracks markets,
manages AI credits, and runs scheduled tasks.

**Try it:** [t.me/respondedorbot](https://t.me/respondedorbot)

## What it can do

- Chat with AI, memory, web search, tools, and provider fallback
- Transcribe audio and describe images
- Summarize conversations
- Show crypto, market, dollar, BCRA, weather, and Polymarket data
- Manage Telegram Stars, AI credits, transfers, and charge history
- Create recurring or one-time scheduled tasks
- Repair supported links and handle localized Telegram commands

## Run it locally

### 1. Install the requirements

- Rust 1.98
- PostgreSQL
- Redis Stack with RediSearch
- FFmpeg

### 2. Create the configuration

```bash
cp .env.example .env
```

Edit `.env` and set these values:

| Variable | What it is for |
| --- | --- |
| `TELEGRAM_TOKEN` | Telegram bot token from BotFather |
| `TELEGRAM_USERNAME` | Bot username, with or without `@` |
| `SUPABASE_POSTGRES_URL` | PostgreSQL database URL |
| `COINMARKETCAP_KEY` | Crypto market data |
| `OPENROUTER_API_KEY` | AI chat, vision, summaries, and fallback |

The bot also needs a personality prompt. Choose one option:

- Set `BOT_SYSTEM_PROMPT` in `.env`.
- Create `workspace/SOUL.md` and, optionally, `workspace/RULES.md`.

Redis uses `localhost:6379` by default. See [.env.example](.env.example) for
optional providers, monitoring, polling, and maintenance settings.

### 3. Build and check the configuration

```bash
cargo build --locked --release -p botd

set -a
. ./.env
set +a

./target/release/botd --check-config
```

### 4. Start the bot

```bash
./target/release/botd
```

Stop it with `Ctrl+C`. The bot waits for background work to finish before it
exits.

> [!IMPORTANT]
> For Supabase, use the session pooler on port `5432` with `sslmode=require`.
> Do not use the transaction pooler on port `6543`.

## Main commands

### AI and media

| Command | Purpose |
| --- | --- |
| `/ask`, `/pregunta`, `/che`, `/gordo` | Chat with AI |
| `/resumen`, `/summary`, `/tldr` | Summarize the conversation |
| `/transcribe`, `/describe` | Transcribe audio or describe images |

### Data

| Command | Purpose |
| --- | --- |
| `/prices`, `/precios`, `/c` | Crypto and traditional markets |
| `/crypto`, `/criptos` | Crypto prices and conversions |
| `/clima`, `/weather` | Current weather |
| `/dolar`, `/dollar`, `/usd` | Dollar rates |
| `/acciones`, `/stocks` | Stock prices |
| `/petroleo`, `/oil` | Oil prices |
| `/eleccion`, `/elections` | Polymarket elections |
| `/bcra`, `/variables` | BCRA variables |
| `/devo`, `/rulo` | Arbitrage calculations |
| `/powerlaw`, `/rainbow`, `/satoshi` | Bitcoin reference models |

### Bot tools

| Command | Purpose |
| --- | --- |
| `/config`, `/settings` | Chat settings |
| `/language`, `/idioma` | Language settings |
| `/topup`, `/balance` | Add or check AI credits |
| `/charges`, `/history`, `/gastos` | Credit history |
| `/transfer` | Move credits to a group |
| `/tarea`, `/task`, `/tareas`, `/tasks` | Manage scheduled tasks |
| `/random`, `/convertbase`, `/comando`, `/time` | Utilities |
| `/gm`, `/gn`, `/help`, `/instance` | Greetings and bot information |

`/prices` checks CoinMarketCap first, then Yahoo Finance for unresolved symbols
and company names. Full Solana/EVM addresses and `$ticker` messages use token
cards when available.

## Test it

Run the same checks used by pull requests:

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

Integration tests use synthetic data. Set `TEST_REDIS_URL`,
`TEST_POSTGRES_URL`, and `TEST_DATABASE_URL` to include Redis Stack and
PostgreSQL tests.

Coverage requirements:

- `bot-core`: 95%
- `bot-adapters`: 95%
- `botd`: 95%

More detail: [Testing](docs/TESTING.md)

## How the code is organized

```text
botd          Starts the app and connects all services
  |
  +-- bot-adapters   Telegram, HTTP, Redis, PostgreSQL, AI, and media
        |
        +-- bot-core   Parsing, routing, state machines, and domain rules
```

Dependencies point toward `bot-core`. External payloads are decoded in
`bot-adapters`, while deterministic behavior stays in `bot-core`.

| Path | Contents |
| --- | --- |
| `crates/bot-core` | Domain behavior and state machines |
| `crates/bot-adapters` | External service implementations |
| `crates/botd` | Executable and composition root |
| `docs` | Architecture, persistence, billing, and testing |
| `quadlets`, `systemd` | Deployment and maintenance units |
| `Containerfile` | Rust-only production image |

Read more: [Architecture](docs/ARCHITECTURE.md) ·
[Billing](docs/BILLING.md) · [Persistence](docs/PERSISTENCE.md)

## Deploy with Podman

<details>
<summary>Show deployment commands</summary>

```bash
podman build --tag respondedorbot:local .

mkdir -p ~/.config/containers/systemd
mkdir -p ~/.config/systemd/user
mkdir -p ~/respondedorbot/workspace

cp quadlets/* ~/.config/containers/systemd/
cp systemd/respondedorbot-maintenance.* ~/.config/systemd/user/
cp systemd/respondedorbot-podman-prune.* ~/.config/systemd/user/
cp .env.example ~/respondedorbot/.env

podman run --rm --env-file ~/respondedorbot/.env \
  -v ~/respondedorbot/workspace:/app/workspace:ro \
  respondedorbot:local /usr/local/bin/botd --check-config

systemctl --user daemon-reload
systemctl --user enable --now respondedorbot-maintenance.timer
systemctl --user enable --now respondedorbot-podman-prune.timer
systemctl --user start respondedorbot-redis.service
systemctl --user start respondedorbot.service
```

Before starting, edit `~/respondedorbot/.env` and add the personality files to
`~/respondedorbot/workspace`, or set `BOT_SYSTEM_PROMPT`.

</details>

The runtime image contains `botd`, FFmpeg, and native shared libraries. CI
publishes `latest` and immutable `sha-<full-commit-sha>` images.

Before removing old persisted formats, inspect and migrate them with the same
image that will run the bot:

```bash
# Report what would change. This does not write data.
podman exec systemd-respondedorbot \
  /usr/local/bin/botd --migrate-legacy

# After taking a database backup, apply the reported changes.
podman exec systemd-respondedorbot \
  /usr/local/bin/botd --migrate-legacy --apply
```

The command is idempotent and prints one JSON report. It upgrades Redis record
versions, rebuilds scheduled-task indexes, and removes obsolete PostgreSQL chat
configuration fields. Missed task occurrences are not executed.

To roll back, pin the Quadlet `Image=` setting to a verified SHA tag, reload
the user units, and restart the service.

> [!WARNING]
> Never run two pollers with the same Telegram token.
