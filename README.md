# Telegram Bot Framework

Configurable Telegram bot built with Flask. It handles chat replies, media transcription/vision, market commands, web lookups, and AI credit billing.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
flask --app api/index run --host 0.0.0.0 --port 8080
```

## Main Files

- `api/index.py`: Flask entrypoint and most bot logic.
- `api/message_handler.py`: message flow, billing, and rate-limit gating.
- `tests/`: pytest suite.

## Features

- AI chat replies with configurable personality and trigger words.
- Market and utility commands such as `/prices`, `/usd`, `/petroleo`, `/devo`, `/powerlaw`, `/rainbow`, `/random`, `/convertbase`, and `/time`.
- Media handling for image description and audio transcription.
- Built-in web lookup support through `/buscar` / `/search` and provider-triggered search when the bot needs fresh information.
- AI credit billing with `/topup`, `/balance`, and `/transfer`.

### AI Credits Billing (Telegram Stars)

- AI responses are billed with credits (default: `1.0` credit per AI response).
- New users receive onboarding credits once (default: `3.0`).
- In groups, spending priority is: personal balance first, then group balance.
- `/topup`: recharge credits with Telegram Stars (private chat).
- `/balance`: in private shows personal balance; in groups shows personal + group balance.
- `/transfer <amount>`: transfer credits from personal balance to group balance.

### Web Search and Tools

- `/buscar <consulta>` or `/search <query>`: quick web searches using DuckDuckGo. No API keys required. Returns up to 10 results with titles and links.
- In AI conversations, the bot may do a web lookup on its own when it needs up-to-date information.
- It can also request to read a specific page with the `fetch_url` tool, which downloads any http/https URL and returns the plain text so the bot can quote passages in its responses.

## Required Config

```bash
BOT_SYSTEM_PROMPT="your bot personality"
BOT_TRIGGER_WORDS=bot,assistant,help

TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_USERNAME=your_bot_username
WEBHOOK_AUTH_KEY=your_webhook_authentication_key

REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
FUNCTION_URL=https://your-app.vercel.app

DATABASE_URL=postgresql://user:password@your-neon-host/neondb?sslmode=require

COINMARKETCAP_KEY=your_coinmarketcap_key
GROQ_API_KEY=your_paid_groq_api_key
WEBHOOK_MAX_RUNTIME_SECONDS=60
WEBHOOK_RETRY_SAFETY_MARGIN_SECONDS=30
WEBHOOK_IDEMPOTENCY_TTL_SECONDS=180
WEBHOOK_FORCE_PAID_RETRY_TTL_SECONDS=300

# AI Credits Billing (always enabled; defaults shown)
AI_ONBOARDING_CREDITS=3.0
AI_STARS_PACKS_JSON='[{"id":"p50","credits":50.0,"xtr":25},{"id":"p100","credits":100.0,"xtr":50},{"id":"p250","credits":250.0,"xtr":125},{"id":"p500","credits":500.0,"xtr":250},{"id":"p1000","credits":1000.0,"xtr":500},{"id":"p2500","credits":2500.0,"xtr":1250}]'

# Monitoring (Required)
ADMIN_CHAT_ID=your_telegram_chat_id
FRIENDLY_INSTANCE_NAME=My_Bot_Instance
```

Use `.env.example` for the full list, including billing defaults.

## Groq Routing

- `GROQ_FREE_API_KEY` is optional.
- If both Groq keys are set, the bot tries the free account first for chat, compound, vision, and transcription.
- If the free account is locally out of budget or Groq returns `429`, the same request is retried with `GROQ_API_KEY`.
- Webhook retries are deduped per Telegram update. If a Compound request fails on `free` and the webhook is too close to the runtime limit, the bot aborts on purpose so Telegram retries and the next attempt can start directly on `paid`.
- Users are still billed normally; key selection only changes which Groq account serves the request.

## Billing

- AI replies are billed with credits.
- Actual settlement uses model/tool usage data when available.
- Internal fallback responses are refunded.
- In groups, spending priority is personal balance first, then group balance.

## Tests

```bash
pytest -q
```

## Webhook

- Set webhook:
  `{FUNCTION_URL}/?update_webhook=true&key={WEBHOOK_AUTH_KEY}`
- Check webhook:
  `{FUNCTION_URL}/?check_webhook=true&key={WEBHOOK_AUTH_KEY}`
