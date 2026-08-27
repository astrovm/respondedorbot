# respondedorbot

An AI Telegram bot that plays "el gordo", a blunt Argentine character that replies in lowercase using Argentine slang.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- AI chat with a configurable personality, streaming responses, web search, and conversation memory
- AI tools for price lookups, calculations, web pages, and scheduled tasks
- Crypto, stock, ETF, index, fund, futures, dollar, and BCRA data
- Weather, Polymarket elections, and Hacker News
- Audio transcription and AI image description
- Conversation summaries and automatic memory compaction
- Telegram Stars billing for AI usage, with shared group credits
- Automatic fixes for supported social links

## Quick Start

```bash
uv sync --locked
cp .env.example .env
# Edit .env with your keys
uv run --locked python run_polling.py
```

## Configuration

| Variable | Description |
| --- | --- |
| `BOT_SYSTEM_PROMPT` | Complete AI personality prompt |
| `BOT_TRIGGER_WORDS` | Comma-separated keywords that trigger responses in groups |
| `TELEGRAM_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_USERNAME` | Bot username |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis cache (requires RediSearch) |
| `SUPABASE_POSTGRES_URL` | Pooled Supabase Postgres URL for AI credits |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `GROQ_API_KEY` | Paid Groq API key for transcription |
| `GROQ_FREE_API_KEY` | Optional free-tier Groq key for transcription |
| `OPENROUTER_API_KEY` | OpenRouter API key for chat/vision |
| `FIRECRAWL_API_KEY` | Firecrawl API key for direct web search |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

### AI providers

| Use | Provider | Model |
|---|---|---|
| Chat | OpenRouter | `~deepseek/deepseek-v4-flash-latest` |
| Vision | OpenRouter | `google/gemini-3.1-flash-lite-preview` |
| Transcription | Groq, then OpenRouter | `whisper-large-v3`, then `google/gemini-3.1-flash-lite-preview` |
| Summary | OpenRouter | `~deepseek/deepseek-v4-flash-latest` |

Text responses stream to Telegram. Tool calls run when their arguments are complete, then the response continues.

## Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/ask` | `/pregunta`, `/che`, `/gordo` | AI chat |
| `/resumen` | `/summary`, `/tldr` | Stream conversation summary |
| `/transcribe` | `/describe` | Transcribe audio / describe image |
| `/prices` | `/price`, `/precios`, `/precio`, `/c`, `/presio(s)`, `/bresio(s)`, `/brecio(s)` | Crypto, stock, ETF, index, fund, and futures prices |
| `/crypto` | `/criptos` | Crypto-only prices and conversions |
| `/clima` | `/weather` | Current weather for a city or location |
| `/dolar` | `/dollar`, `/usd` | Dollar rates (CriptoYa) |
| `/petroleo` | `/oil` | Oil prices |
| `/acciones` | `/stocks` | Stock prices |
| `/eleccion` | `/elecciones`, `/election`, `/elections` | Global Polymarket elections by liquidity |
| `/devo` | - | Arbitrage calculator (tarjeta vs crypto) |
| `/rulo` | - | Dollar arbitrage chains |
| `/powerlaw` | - | Bitcoin power law |
| `/rainbow` | - | Bitcoin rainbow chart |
| `/satoshi` | `/sat`, `/sats` | Satoshi value |
| `/bcra` | `/variables` | BCRA economic variables |
| `/random` | - | Random choice or number |
| `/convertbase` | - | Number base conversion |
| `/comando` | `/command` | Convert text to a Telegram command |
| `/time` | - | Unix timestamp |
| `/config` | `/configs`, `/settings` | Chat settings (admin only in groups) |
| `/language` | `/idioma` | Change the bot language |
| `/topup` | - | Buy AI credits with Telegram Stars |
| `/balance` | - | Show credit balance |
| `/charges` | `/history`, `/gastos` | Show credit charges |
| `/transfer` | - | Transfer credits to group |
| `/tarea`, `/tareas` | `/task`, `/tasks` | List tasks, or create one when followed by text |
| `/gm` | - | Good morning GIF |
| `/gn` | - | Good night GIF |
| `/help` | - | Command reference |
| `/instance` | - | Instance name |

`/prices` checks CoinMarketCap first and sends unresolved symbols or company names to Yahoo Finance. Use `stock:META` or `crypto:META` when both providers have the same symbol. A complete Solana/EVM address still opens a token card; `$ticker` opens a token card first and falls back to `/prices` when no token is found.

## How it works

`api/index.py` creates the services used by the Telegram handlers. OpenRouter handles AI chat, vision, and summaries. Groq handles transcription, with OpenRouter as a fallback.

Chat history is stored in Redis and indexed with RediSearch. The bot compacts history after 40 new messages and keeps the latest 25. Summaries are updated from the previous summary and the new messages.

Market data, weather, random choices, Hacker News, and command information are passed to the AI only when requested. AI credit reservations are settled after each response and refunded on failure.

## Project layout

- `api/` - application code
  - `api/admin/` - admin commands, reporting, authorization
  - `api/ai/` - AI orchestration, prompting, pricing, response cleanup
  - `api/billing/` - credits, settlement, billing commands, Stars callbacks
  - `api/bot/` - Telegram adapter, handlers, routing, streaming, chat config
  - `api/cache/` - HTTP and Redis caching
  - `api/core/` - configuration, constants, logging
  - `api/links/` - URL metadata, replacement, and enrichment
  - `api/markets/` - crypto, dollar, stocks, Polymarket, weather
  - `api/media/` - image, audio, video, transcription, media cache
  - `api/memory/` - chat history, retrieval, compaction, summaries
  - `api/providers/` - AI providers and fallback chains
  - `api/tasks/` - task execution and scheduling
  - `api/tools/` - AI tools for prices, calculations, web fetches, and tasks
  - `api/services/` - persistence and low-level external adapters
  - `api/utils/` - reusable helpers
  - `api/index.py` - application composition root and compatibility exports
- `quadlets/` - Podman Quadlet container definitions
- `systemd/` - systemd service and timer units
- `run_polling.py` - bot entrypoint
- `run_maintenance.py` - maintenance entrypoint
- `tests/` - test suite
- `Containerfile` - container image definition

## Deployment (Podman + systemd)

### Prerequisites (Debian/Ubuntu)

```bash
sudo apt install -y podman uidmap dbus-user-session slirp4netns fuse-overlayfs
sudo useradd -m -s /bin/bash respondedorbot
sudo loginctl enable-linger respondedorbot
```

### Setup (as `respondedorbot` user)

```bash
git clone https://github.com/astrovm/respondedorbot
cd respondedorbot

mkdir -p ~/.config/containers/systemd
cp quadlets/* ~/.config/containers/systemd/

mkdir -p ~/respondedorbot/workspace
cp .env.example ~/respondedorbot/.env
# Create ~/respondedorbot/workspace/SOUL.md and RULES.md manually.
# Edit ~/respondedorbot/.env - set REDIS_HOST=respondedorbot-redis
# Quadlet Redis uses redis-stack-server because the bot needs RediSearch (FT.CREATE / FT.SEARCH)

export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus

systemctl --user daemon-reload
systemctl --user start respondedorbot-redis.service
systemctl --user start respondedorbot.service
```

The bot mounts `~/respondedorbot/workspace` read-only at `/app/workspace`.
Create `SOUL.md` and `RULES.md` on the VPS before starting the service. You can
also set the complete prompt with `BOT_SYSTEM_PROMPT`.

The Redis Quadlet uses `redis/redis-stack-server:7.4.0-v8` because chat memory
requires RediSearch. Redis does not auto-update. Test `FT.CREATE` and
`FT.SEARCH` before changing the pinned version. Redis settings are passed
through `REDIS_ARGS`.

### Image publishing and rollback

CI runs Ruff, mypy, and the tests before building the image. A successful push
to `main` publishes:

- `ghcr.io/astrovm/respondedorbot:latest` for Podman auto-updates.
- `ghcr.io/astrovm/respondedorbot:sha-<full-commit-sha>` for rollback.

To roll back the VPS to a tested commit:

```bash
ROLLBACK_SHA=<full-commit-sha>
podman pull "ghcr.io/astrovm/respondedorbot:sha-${ROLLBACK_SHA}"
sed -i \
  "s|^Image=.*|Image=ghcr.io/astrovm/respondedorbot:sha-${ROLLBACK_SHA}|" \
  ~/.config/containers/systemd/respondedorbot.container
systemctl --user daemon-reload
systemctl --user restart respondedorbot.service
systemctl --user status respondedorbot.service --no-pager
```

The SHA tag prevents Podman auto-update from moving the bot forward. After the
problem is fixed, return to automatic releases:

```bash
sed -i \
  "s|^Image=.*|Image=ghcr.io/astrovm/respondedorbot:latest|" \
  ~/.config/containers/systemd/respondedorbot.container
systemctl --user daemon-reload
systemctl --user restart respondedorbot.service
```

### Persist across reboots

Some distributions cannot enable Quadlet-generated units with `systemctl --user enable`. Use symlinks instead:

```bash
mkdir -p ~/.config/systemd/user/default.target.wants
ln -sf ~/.config/containers/systemd/respondedorbot.container \
  ~/.config/systemd/user/default.target.wants/respondedorbot.container
ln -sf ~/.config/containers/systemd/respondedorbot-redis.container \
  ~/.config/systemd/user/default.target.wants/respondedorbot-redis.container
systemctl --user daemon-reload
```

### Maintenance timers

```bash
cp systemd/respondedorbot-maintenance.* ~/.config/systemd/user/
cp systemd/respondedorbot-podman-prune.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now respondedorbot-maintenance.timer
systemctl --user enable --now respondedorbot-podman-prune.timer
```

### Useful commands

```bash
journalctl --user -fu respondedorbot.service
systemctl --user status respondedorbot.service --no-pager
systemctl --user stop respondedorbot.service respondedorbot-redis.service
systemctl --user enable --now podman-auto-update.timer
podman exec systemd-respondedorbot python /app/run_maintenance.py
```

## Tests

```bash
uv run --locked pytest -q
```
