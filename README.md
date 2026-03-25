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
flask --app api/index run --host 0.0.0.0 --port 8080
```

## Project Structure

```text
api/index.py            # Flask entrypoint, command handlers, AI logic
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
| `WEBHOOK_AUTH_KEY` | Webhook authentication key |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | Redis cache |
| `SUPABASE_POSTGRES_URL` | Pooled Supabase Postgres URL (for AI credits) |
| `COINMARKETCAP_KEY` | CoinMarketCap API key |
| `GROQ_API_KEY` | Paid Groq API key |
| `GROQ_FREE_API_KEY` | Optional free-tier Groq key (tried first, falls back to paid) |
| `GIPHY_API_KEY` | Giphy API key for `/gm` and `/gn` GIFs |
| `FUNCTION_URL` | Deployment URL |
| `ADMIN_CHAT_ID` | Telegram chat ID for error reports |
| `FRIENDLY_INSTANCE_NAME` | Instance name for admin reports |

## AI Credits Billing

AI responses cost credits. Users get onboarding credits on first interaction, then recharge with Telegram Stars.

- `/topup` - buy credits with Stars (private chat only)
- `/balance` - check credits (personal in DM, personal + group in groups)
- `/transfer <amount>` - move credits from personal to group balance

In groups, personal balance is spent first, then group balance.

## Groq Routing

If both `GROQ_FREE_API_KEY` and `GROQ_API_KEY` are set, the bot tries the free key first for all Groq calls. On rate limit (429) or local budget exhaustion, it retries with the paid key. Webhook retries are deduped per Telegram update to avoid double-billing.

## Webhook Setup

```text
# Set webhook
{FUNCTION_URL}/?update_webhook=true&key={WEBHOOK_AUTH_KEY}

# Check webhook
{FUNCTION_URL}/?check_webhook=true&key={WEBHOOK_AUTH_KEY}
```

## Tests

```bash
pytest -q
```
