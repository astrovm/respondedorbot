# respondedorbot

An AI-powered Telegram bot playing "el gordo" — a blunt, politically incorrect Argentine character who answers everything in a single lowercase phrase using Argentine slang.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- **AI chat**: configurable personality powered by Groq, responds to trigger words in groups
- **Market data**: `/prices`, `/usd`, `/petroleo`, `/devo`, `/powerlaw`, `/rainbow`
- **BCRA economic data**: `/bcra`, `/variables`
- **Media**: audio transcription (Whisper) and image description (vision) via `/transcribe`
- **Web search**: `/buscar` / `/search` using Groq Compound
- **Utilities**: `/random`, `/convertbase`, `/time`, `/gm`, `/gn`
- **AI credits billing**: Telegram Stars (`/topup`, `/balance`, `/transfer`)
- **Link enrichment**: URLs get metadata injected into AI context

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

Copy `.env.example` and fill in the values:

| Variable | Description |
| --- | --- |
| `BOT_SYSTEM_PROMPT` | Complete AI personality prompt |
| `BOT_TRIGGER_WORDS` | Comma-separated keywords that trigger responses in groups |
| `TELEGRAM_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_USERNAME` | Bot username |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis cache |
| `SUPABASE_POSTGRES_URL` | Pooled Supabase Postgres URL (for AI credits) |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `GROQ_API_KEY` | Paid Groq API key |
| `GROQ_FREE_API_KEY` | Optional free-tier key (tried first, falls back to paid on 429) |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

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
# Edit ~/respondedorbot/.env
# REDIS_HOST=respondedorbot-redis
# REDIS_PORT=6379
# REDIS_PASSWORD=

export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus

systemctl --user daemon-reload
systemctl --user start respondedorbot-redis.service
systemctl --user start respondedorbot.service
```

### Persist across reboots

`systemctl --user enable` on Quadlet-generated units fails on some distros.
Use symlinks instead:

```bash
mkdir -p ~/.config/systemd/user/default.target.wants

ln -sf ~/.config/containers/systemd/respondedorbot.container \
  ~/.config/systemd/user/default.target.wants/respondedorbot.container

ln -sf ~/.config/containers/systemd/respondedorbot-redis.container \
  ~/.config/systemd/user/default.target.wants/respondedorbot-redis.container

systemctl --user daemon-reload
```

### Useful commands

```bash
# Logs
journalctl --user -fu respondedorbot.service

# Status
systemctl --user status respondedorbot.service --no-pager

# Stop
systemctl --user stop respondedorbot.service respondedorbot-redis.service

# Auto-update images
systemctl --user enable --now podman-auto-update.timer
```

## Tests

```bash
pytest -q
```
