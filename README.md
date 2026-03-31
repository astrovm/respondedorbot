# respondedorbot

An AI-powered Telegram bot that plays the role of "el gordo", a politically incorrect Argentine character inspired by the classic "atendedor de boludos" and "viejo inimputable" memes. He's blunt, unfiltered, and answers everything in a single lowercase phrase using Argentine slang. Think of him as the guy at the ciber who spent too long on Taringa and the deep web.

Beyond the attitude, he actually knows his stuff: crypto, hacking, Linux, gaming, psychiatry, economics, and internet culture from the golden age of forums and Flash games. If the question is real, the answer is real.

**[t.me/respondedorbot](https://t.me/respondedorbot)**

## Features

- **AI chat**: configurable personality powered by Groq, responds to trigger words in groups
- **Market data**: `/prices`, `/usd`, `/petroleo`, `/devo`, `/powerlaw`, `/rainbow`
- **BCRA economic data**: `/bcra`, `/variables` (base monetaria, inflation, dollar rates, reserves, etc.)
- **Media handling**: audio transcription (Whisper) and image description (vision) via `/transcribe`
- **Web search**: `/buscar` / `/search` using Groq Compound for real-time info
- **Utilities**: `/random`, `/convertbase`, `/time`, `/gm`, `/gn`
- **AI credits billing**: Telegram Stars integration (`/topup`, `/balance`, `/transfer`)
- **Link enrichment**: URLs in messages get fetched metadata injected into AI context

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
python run_polling.py
```

## Project Structure

```text
api/index.py            # Core command handlers and AI logic
api/bot_ptb.py          # python-telegram-bot polling runtime
run_polling.py          # Polling entrypoint
api/message_handler.py  # Message flow, billing, rate-limit gating
tests/                  # pytest suite
```

## Configuration

Copy `.env.example` and fill in the values. Key variables:

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
| `GROQ_FREE_API_KEY` | Optional free-tier Groq key (tried first, falls back to paid) |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` GIFs |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

## Docker Deployment

Build and run with Docker:

```bash
docker build -t respondedorbot .
docker run --env-file .env respondedorbot
```

Or use the pre-built image from GitHub Container Registry:

```bash
docker run --env-file .env ghcr.io/astrovm/respondedorbot:latest
```

## AI Credits Billing

AI responses cost credits. Users get onboarding credits on first interaction, then recharge with Telegram Stars.

- `/topup` - buy credits with Stars (private chat only)
- `/balance` - check credits (personal in DM, personal + group in groups)
- `/transfer <amount>` - move credits from personal to group balance

In groups, personal balance is spent first, then group balance.

## Groq Routing

If both `GROQ_FREE_API_KEY` and `GROQ_API_KEY` are set, the bot tries the free key first for all Groq calls. On rate limit (429) or local budget exhaustion, it retries with the paid key.

## Configuration Options

| Variable | Description | Default |
| --- | --- | --- |
| `PTB_ALLOWED_UPDATES` | Comma-separated update types to receive | `message,callback_query,pre_checkout_query` |
| `PTB_DROP_PENDING_UPDATES` | Drop pending updates on startup | `true` |

## Tests

```bash
pytest -q
```

## Architecture

The bot uses **python-telegram-bot v20+** with polling mode:
- Automatic offset tracking and duplicate prevention
- Built-in flood wait handling and error recovery
- Async handlers with sync code bridging for the existing codebase
- No webhook required — works behind NAT and firewalls
