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

ADMIN_CHAT_ID=your_telegram_chat_id
FRIENDLY_INSTANCE_NAME=My_Bot_Instance
```

Use `.env.example` for the full list, including billing defaults.

## Groq Routing

- `GROQ_FREE_API_KEY` is optional.
- If both Groq keys are set, the bot tries the free account first for chat, compound, vision, and transcription.
- If the free account is locally out of budget or Groq returns `429`, the same request is retried with `GROQ_API_KEY`.
- Users are still billed normally; key selection only changes which Groq account serves the request.

## Billing

- AI replies are billed with credits.
- Default reserve is controlled by `AI_CREDITS_PER_RESPONSE`.
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
