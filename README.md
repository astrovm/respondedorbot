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
- Currency exchange rates with `/dolar` or `/dollar`
- Arbitrage calculator with `/devo`
- Bitcoin analysis with `/powerlaw` and `/rainbow`
- Random choices with `/random`
- Text to command conversion with `/comando` or `/command`
- Base number conversion with `/convertbase`
- Unix timestamp with `/time`
- And many more commands - use `/help` for complete list

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

# APIs (Required)
OPENROUTER_API_KEY=your_openrouter_key
CLOUDFLARE_API_KEY=your_cloudflare_key
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
COINMARKETCAP_KEY=your_coinmarketcap_key

# Monitoring (Required)
ADMIN_CHAT_ID=your_telegram_chat_id
FRIENDLY_INSTANCE_NAME=My_Bot_Instance
```

## Webhook Setup

Set the webhook URL:
`{function_url}/?update_webhook=true&key={webhook_auth_key}`

Check webhook status:
`{function_url}/?check_webhook=true&key={webhook_auth_key}`
