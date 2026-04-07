# respondedorbot

An AI-powered Telegram bot playing "el gordo" — a blunt, politically incorrect Argentine character who answers everything in a single lowercase phrase using Argentine slang.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- **AI chat**: configurable personality powered by Groq with OpenRouter fallback for chat/vision when local Groq is exhausted
- **Market data**: `/prices`, `/usd`, `/petroleo`, `/devo`, `/powerlaw`, `/rainbow`
- **BCRA economic data**: `/bcra`, `/variables`
- **Media**: audio transcription (Whisper) tries Groq free then paid; image description also falls back to OpenRouter
- **Web search**: `/buscar` / `/search` using free Groq first, then paid Groq
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

| Variable | Description |
| --- | --- |
| `BOT_SYSTEM_PROMPT` | Complete AI personality prompt |
| `BOT_TRIGGER_WORDS` | Comma-separated keywords that trigger responses in groups |
| `TELEGRAM_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_USERNAME` | Bot username |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis cache |
| `SUPABASE_POSTGRES_URL` | Pooled Supabase Postgres URL (for AI credits) |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `GROQ_API_KEY` | Paid Groq API key used after free Groq for compound flows |
| `GROQ_FREE_API_KEY` | Optional free-tier key for free Groq chat/vision/transcribe |
| `OPENROUTER_API_KEY` | OpenRouter API key used for chat/vision fallback |
| `CF_AIG_TOKEN` | Cloudflare AI Gateway token forwarded to OpenRouter requests |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

### Provider contract

- all scopes try free Groq first, then paid Groq
- chat and vision also fall back to OpenRouter if both Groq accounts are exhausted
- compound and transcription stop at paid Groq (no OpenRouter fallback)

## Project layout

- `api/` - application code
- `quadlets/` - Podman Quadlet container definitions
- `systemd/` - systemd service and timer units
- `run_polling.py` - bot entrypoint
- `run_maintenance.py` - maintenance entrypoint (run inside the container)
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

export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=${XDG_RUNTIME_DIR}/bus

systemctl --user daemon-reload
systemctl --user start respondedorbot-redis.service
systemctl --user start respondedorbot.service
```

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
