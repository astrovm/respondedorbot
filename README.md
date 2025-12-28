# Telegram Bot (Rust / Axum)

A configurable Telegram bot built with Rust and Axum. The service listens on `0.0.0.0:8080`, handles Telegram webhooks, and responds with AI-powered messages plus utility commands.

## Quick Start

1. Copy `.env.example` to `.env` and fill in your secrets and API keys.
2. Install Rust (stable) and ensure `cargo` is available.
3. Run the bot locally:
   ```bash
   cargo run
   ```
4. Configure the Telegram webhook once the server is reachable:
   - Set webhook: `{FUNCTION_URL}/?update_webhook=true&key={WEBHOOK_AUTH_KEY}`
   - Check webhook: `{FUNCTION_URL}/?check_webhook=true&key={WEBHOOK_AUTH_KEY}`

## Features

- AI-powered conversations with configurable personality (`BOT_SYSTEM_PROMPT` / `BOT_TRIGGER_WORDS`)
- Cryptocurrency prices with `/prices`
- Currency exchange rates with `/dolar`, `/dollar`, or `/usd`
- Arbitrage calculator with `/devo`
- Bitcoin analysis with `/powerlaw` and `/rainbow`
- Random choices with `/random`
- Text-to-command conversion with `/comando` or `/command`
- Base number conversion with `/convertbase`
- Unix timestamp with `/time`
- Link rewriting for social sites
- Media transcription/description via `/transcribe`

### Web Search and Tools

- `/buscar <consulta>` or `/search <query>`: DuckDuckGo search, returning up to 10 results.
- AI conversations may request tools like `web_search` or `fetch_url` for current data and quoting sources.

## Configuration

All configuration is via environment variables (see `.env.example` for the full list). Required keys include:

- Bot: `BOT_SYSTEM_PROMPT`, `BOT_TRIGGER_WORDS`
- Telegram: `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`, `WEBHOOK_AUTH_KEY`
- Infrastructure: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `FUNCTION_URL`
- APIs: `OPENROUTER_API_KEY`, `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `COINMARKETCAP_KEY`
- Monitoring: `ADMIN_CHAT_ID`, `FRIENDLY_INSTANCE_NAME`

## Development

- Run the server: `cargo run`
- Tests: `cargo test`
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`

## Cache TTLs

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
- Cached payloads include internal timestamps and refresh when staleness exceeds the TTL.
- Redis JSON helpers keep read/write of cached payloads consistent.
