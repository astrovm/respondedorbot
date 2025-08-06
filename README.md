# Telegram Bot Framework

A configurable Telegram bot framework written in Python/Flask that can be customized with different personalities and behaviors. The bot runs 24/7 responding to messages in Telegram with various useful commands.

## Configuration Setup

This bot uses external configuration files to separate the codebase from specific personalities or commercial information. 

### Quick Start:
1. Copy `bot_config.example.json` to `bot_config.json`
2. Customize the configuration with your bot's personality
3. Set up the required environment variables (see below)
4. Deploy to your preferred platform

### Bot Configuration Structure

The `bot_config.json` file contains only two essential fields:

- **trigger_words**: Keywords that trigger bot responses in group chats
- **system_prompt**: The complete AI personality prompt that defines the bot's character

### Fallback Behavior

If no `bot_config.json` is found, the bot will:
- Use generic helpful assistant personality
- Look for `BOT_NAME` and `CREATOR_NAME` environment variables
- Default to basic functionality without specific personality traits

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

```
TELEGRAM_TOKEN: Your Telegram bot token
TELEGRAM_USERNAME: Bot username for mention detection  
ADMIN_CHAT_ID: Chat ID for admin error reports
COINMARKETCAP_KEY: CoinMarketCap API key for crypto prices
REDIS_HOST: Redis server host
REDIS_PORT: Redis server port
REDIS_PASSWORD: Redis password (if required)
CURRENT_FUNCTION_URL: Current deployment URL
MAIN_FUNCTION_URL: Main deployment URL
FRIENDLY_INSTANCE_NAME: Instance name for reports
OPENROUTER_API_KEY: OpenRouter API key for AI responses
CLOUDFLARE_API_KEY: Cloudflare Workers AI key (fallback)
CLOUDFLARE_ACCOUNT_ID: Cloudflare account ID
WEBHOOK_AUTH_KEY: Key for webhook authentication
```

### Optional Environment Variables (for fallback when no bot_config.json):
```
BOT_NAME: Default bot name
CREATOR_NAME: Creator/developer name
```

## Webhook Setup

Set the webhook URL:
`{function_url}/?update_webhook=true&key={webhook_auth_key}`

Check webhook status:
`{function_url}/?check_webhook=true&key={webhook_auth_key}`

## License

MIT License - Use it however you want.

## Credits

Open source Telegram bot framework. Contributions welcome!
