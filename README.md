# respondedorbot

An AI-powered Telegram bot playing "el gordo" — a blunt, politically incorrect Argentine character who answers everything in a single lowercase phrase using Argentine slang.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- **AI chat**: configurable personality with web search, powered by Qwen via OpenRouter with Groq fallback
- **Streaming responses**: AI replies stream token-by-token to Telegram (when no tools are active)
- **Chat memory with RediSearch**: persistent conversation history, full-text search, automatic compaction
- **Incremental summaries**: `/resumen` streams conversation summaries using Minimax, with automatic context compaction
- **Agentic tools**: AI can call tools (price lookup, calculator, web fetch, task scheduling) via function calling
- **Market data**: `/prices`, `/usd`, `/petroleo`, `/devo`, `/powerlaw`, `/rainbow`, `/rulo`, `/eleccion`
- **BCRA economic data**: `/bcra`, `/variables`
- **Media**: audio transcription (Whisper) and image description with Groq free→paid→OpenRouter fallback
- **Scheduled tasks**: `/tareas`, `/tasks` — create, list, and delete one-shot or recurring reminders via AI or inline buttons
- **AI credits billing**: Telegram Stars (`/topup`, `/balance`, `/transfer`)
- **Link enrichment**: URLs get metadata injected into AI context; social links auto-replaced (fxTwitter, fixupx, etc.)
- **Context injection**: market data, weather, Hacker News top stories, and Buenos Aires time in every system prompt
- **Response cleanup**: deduplication, prefix stripping, identity leak prevention

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
python run_polling.py
```

## Configuration

| Variable | Description |
| --- | --- |
| `BOT_SYSTEM_PROMPT` | Complete AI personality prompt |
| `BOT_TRIGGER_WORDS` | Comma-separated keywords that trigger responses in groups |
| `TELEGRAM_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_USERNAME` | Bot username |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis cache (requires RediSearch) |
| `SUPABASE_POSTGRES_URL` | Pooled Supabase Postgres URL (for AI credits) |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `GROQ_API_KEY` | Paid Groq API key for vision/transcription |
| `GROQ_FREE_API_KEY` | Optional free-tier Groq key for vision/transcribe |
| `OPENROUTER_API_KEY` | OpenRouter API key for chat/vision |
| `CF_AIG_TOKEN` | Cloudflare AI Gateway token forwarded to OpenRouter requests |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

### Provider contract

**Chat provider chain** (tries in order):
1. OpenRouter (`qwen/qwen3.6-plus`) — primary
2. Groq (`llama-3.3-70b-versatile`) — fallback with cooldown backoff

**Vision**:
- Groq free → Groq paid → OpenRouter (`meta-llama/llama-4-scout`)

**Transcription**:
- Groq free → Groq paid (no OpenRouter fallback)

**Summary**:
- OpenRouter (`minimax/minimax-m2.7`) exclusively

**Streaming**: token streaming only when no tools/web-search are active. Tool-enabled requests return complete responses.

## Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `/ask` | `/pregunta`, `/che`, `/gordo` | AI chat (streaming) |
| `/resumen` | `/summary` | Stream conversation summary |
| `/transcribe` | `/describe` | Transcribe audio / describe image |
| `/prices` | `/price`, `/precios`, `/precio`, `/presio(s)`, `/bresio(s)`, `/brecio(s)` | Crypto prices |
| `/dolar` | `/dollar`, `/usd` | Dollar rates (CriptoYa) |
| `/petroleo` | `/oil` | Oil prices |
| `/acciones` | `/stocks` | Stock prices |
| `/eleccion` | - | Polymarket Argentina election odds |
| `/devo` | - | Arbitrage calculator (tarjeta vs crypto) |
| `/rulo` | - | Dollar arbitrage chains |
| `/powerlaw` | - | Bitcoin power law |
| `/rainbow` | - | Bitcoin rainbow chart |
| `/satoshi` | `/sat`, `/sats` | Satoshi value |
| `/bcra` | `/variables` | BCRA economic variables |
| `/random` | - | Random choice or number |
| `/convertbase` | - | Number base conversion |
| `/comando` | `/command` | Text → Telegram command |
| `/time` | - | Unix timestamp |
| `/config` | - | Chat settings (admin only in groups) |
| `/topup` | - | Buy AI credits with Telegram Stars |
| `/balance` | - | Show credit balance |
| `/transfer` | - | Transfer credits to group |
| `/tareas` | `/tasks` | Manage scheduled reminders |
| `/gm` | - | Good morning GIF |
| `/gn` | - | Good night GIF |
| `/help` | - | Command reference |
| `/instance` | - | Instance name |

## Architecture

### Provider abstraction (`api/providers/`)

- `ProviderChain` — tries providers in order until one succeeds
- `OpenRouterProvider` — streaming + completion, primary chat model
- `GroqChatProvider` — completion only, cooldown-aware fallback

### Streaming (`api/streaming.py`)

`TelegramMessageStreamer` edits Telegram messages every 400ms or 15+ new chars. Token streaming from `OpenRouterProvider.stream()` when no tools active. Falls back to complete response for tool-enabled requests.

### AI service (`api/ai_service.py`)

`AIService` orchestrates credit reservation → model call → billing settlement:
- **Reserve**: holds worst-case credits before AI call
- **Settle**: calculates actual cost, charges/refunds difference
- **Refund**: full return on failure, fallback, or empty response

### Chat memory compaction

- `COMPACTION_THRESHOLD = 20` — compact when delta > 20 messages
- `COMPACTION_KEEP = 15` — retain last 15 messages
- Incremental summaries from delta messages + prior summary
- RediSearch index (`idx:chat_messages`) for full-text search and RAG retrieval

### Billing & credits

- **User credits** — personal balance
- **Group credits** — shared pool subsidizing creditless users
- **Onboarding** — 3 free credits for new users
- **Hourly limit** — `creditless_user_hourly_limit` caps free messages per user per hour
- **Credit packs** (Telegram Stars): 50→2500 credits with 50% bonus tiers

### Response pipeline (`api/ai_pipeline.py`)

Sequential cleanup:
1. Remove "gordo:" prefix
2. Strip echoed context strings
3. Remove identity leak prefixes (`@user:`)
4. Deduplicate consecutive lines/sentences

### Context injection

Every system prompt includes:
- **Market**: top 3 cryptos + dollar rates (oficial, blue, mep, tarjeta, usdt)
- **Weather**: Buenos Aires temp, rain probability, cloud cover
- **Hacker News**: top 5 stories (title, points, comments)
- **Time**: current Buenos Aires datetime

## Project layout

- `api/` - application code
  - `api/providers/` - AI provider abstraction (OpenRouter, Groq, ProviderChain)
  - `api/streaming.py` - Telegram token streaming
  - `api/ai_service.py` - AI conversation orchestration and billing
  - `api/ai_pipeline.py` - response cleanup pipeline
  - `api/ai_billing.py` - credit reservation/settlement/refund
  - `api/message_handler.py` - message routing, command dispatch, billing integration
  - `api/message_state.py` - Redis chat history, RediSearch indexing, compaction markers
  - `api/command_registry.py` - command definitions and aliases
  - `api/tools/` - agentic tool registry (crypto, calculator, web fetch, tasks)
  - `api/index.py` - core bot logic, commands, provider integration
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

mkdir -p ~/respondedorbot
cp .env.example ~/respondedorbot/.env
# Edit ~/respondedorbot/.env — set REDIS_HOST=respondedorbot-redis
# Quadlet Redis uses redis-stack-server because the bot needs RediSearch (FT.CREATE / FT.SEARCH)

export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus

systemctl --user daemon-reload
systemctl --user start respondedorbot-redis.service
systemctl --user start respondedorbot.service
```

The bundled Redis Quadlet uses `redis/redis-stack-server`, not plain `redis`, because chat memory search and compaction require RediSearch commands. The unit intentionally does not override the container command; it passes Redis tuning through `REDIS_ARGS` so the image can boot Redis Stack with its modules enabled.

### Persist across reboots

`systemctl --user enable` fails on Quadlet-generated units on some distros — use symlinks instead:

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
pytest -q
```
