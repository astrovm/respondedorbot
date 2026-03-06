# Telegram Bot Framework

A configurable Telegram bot framework written in Python/Flask that can be customized with different personalities and behaviors. The bot runs 24/7 responding to messages in Telegram with various useful commands.

## Configuration Setup

This bot is configured entirely through environment variables, making it easy to deploy and customize without touching the code.

### Quick Start:

1. Copy `.env.example` to `.env` and configure your environment variables
2. Deploy to your preferred platform
3. Configure the webhook URL

### Local Development:

1. `cp .env.example .env`
2. Fill in your actual API keys and configuration in `.env`
3. `flask --app api/index run --host 0.0.0.0 --port 8080`

### Required Bot Configuration:

- **BOT_SYSTEM_PROMPT**: Complete AI personality prompt that defines the bot's character
- **BOT_TRIGGER_WORDS**: Comma-separated keywords that trigger responses in group chats

## Features

- AI-powered conversations with configurable personality
- Cryptocurrency prices with `/prices` command
- Currency exchange rates with `/dolar`, `/dollar`, or `/usd`
- Arbitrage calculator with `/devo`
- Bitcoin analysis with `/powerlaw` and `/rainbow`
- Random choices with `/random`
- Text to command conversion with `/comando` or `/command`
- Base number conversion with `/convertbase`
- Unix timestamp with `/time`
- And many more commands - use `/help` for complete list

### AI Credits Billing (Telegram Stars)

- AI responses are billed with credits (default: `1` credit per AI response).
- New users receive onboarding credits once (default: `3`).
- In groups, spending priority is: personal balance first, then group balance.
- `/topup`: recharge credits with Telegram Stars (private chat).
- `/balance`: in private shows personal balance; in groups shows personal + group balance.
- `/transfer <amount>`: transfer credits from personal balance to group balance.

### Web Search and Tools

- `/buscar <consulta>` or `/search <query>`: quick web searches using DuckDuckGo. No API keys required. Returns up to 10 results with titles and links.
- In AI conversations, the bot may do a web lookup on its own when it needs up-to-date information.
- It can also request to read a specific page with the `fetch_url` tool, which downloads any http/https URL and returns the plain text so the bot can quote passages in its responses.

## Testing

Run the unit test suite to verify functionality:

```bash
pytest test.py -q
```

## Deployment

### Required Environment Variables:

```bash
# Bot Configuration (Required)
BOT_SYSTEM_PROMPT=Your complete bot personality prompt here
BOT_TRIGGER_WORDS=bot,assistant,help

# Telegram Configuration (Required)
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_USERNAME=your_bot_username
WEBHOOK_AUTH_KEY=your_webhook_authentication_key

# Infrastructure (Required)
REDIS_HOST=your_redis_host
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
FUNCTION_URL=https://your-app.vercel.app

# Postgres / Neon (Required for AI credits billing)
DATABASE_URL=postgresql://user:password@your-neon-host/neondb?sslmode=require
# Optional fallback vars if DATABASE_URL is unset:
# PGHOST=your-neon-host
# PGPORT=5432
# PGDATABASE=neondb
# PGUSER=your_user
# PGPASSWORD=your_password

# APIs (Required)
GROQ_API_KEY=your_groq_api_key
COINMARKETCAP_KEY=your_coinmarketcap_key

# AI Credits Billing (always enabled; defaults shown)
AI_CREDITS_PER_RESPONSE=1
AI_ONBOARDING_CREDITS=3
AI_STARS_PACKS_JSON='[{"id":"p100","credits":100,"xtr":50},{"id":"p250","credits":250,"xtr":125},{"id":"p500","credits":500,"xtr":250},{"id":"p1000","credits":1000,"xtr":500},{"id":"p2500","credits":2500,"xtr":1250}]'

# Monitoring (Required)
ADMIN_CHAT_ID=your_telegram_chat_id
FRIENDLY_INSTANCE_NAME=My_Bot_Instance
```

## Webhook Setup

Set the webhook URL:
`{function_url}/?update_webhook=true&key={webhook_auth_key}`

Check webhook status:
`{function_url}/?check_webhook=true&key={webhook_auth_key}`
## Cache TTLs

The bot caches data to reduce latency and API usage. Default TTLs:

- Prices (CoinMarketCap): 5 minutes
- Dollar rates (CriptoYa): 5 minutes
- BCRA variables: 5 minutes
- ITCRM (BCRA spreadsheet):
  - If latest business-day value present: cache until 15:00 next business day (UTC-3)
  - Otherwise: 30 minutes
- Weather (Open-Meteo): 30 minutes
- Web search: 5 minutes
- Media (audio/image transcriptions/descriptions): 7 days

Notes:
- `cached_requests` stores a JSON payload with an internal timestamp and refreshes when the cache age exceeds the configured expiration. It also writes optional hourly snapshots for some endpoints.
- Redis JSON convenience helpers are used to read/write JSON consistently.
