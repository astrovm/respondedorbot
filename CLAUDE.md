# CLAUDE.md

Guidance for working with this repository using Claude Code.

## Project Overview

This is a configurable Telegram bot implemented in Rust with Axum. It listens for Telegram webhooks, routes commands (prices, dollar rates, powerlaw/rainbow charts, random choices, agent memory, media transcription/description), and generates AI responses using OpenRouter/Cloudflare. Redis backs caching, rate limiting, and chat history.

## Development Commands

```bash
# Run locally (expects env vars from .env/.env.example)
cargo run

# Tests
cargo test

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

## Architecture

- **src/main.rs**: Axum server, webhook query handlers (`check_webhook`, `update_webhook`, `update_dollars`, `run_agent`), message dispatch, rate limiting, media handling, AI flow, and link rewriting.
- **src/agent/**: Autonomous agent memory management, prompt shaping, and repetition detection.
- **src/commands.rs**: Command parsing, `/help`, `/random`, `/comando` helpers.
- **src/market.rs**: Prices, dollar rates, powerlaw/rainbow, base conversion.
- **src/bcra.rs**: BCRA variables via the official API (no scraping).
- **src/tools.rs**: Web search/fetch helpers.
- **src/message_utils.rs**: Chat history formatting, AI message building, post-processing.
- **tests/**: Coverage for commands, tools, market, agent, link rewriting, and messaging utilities.

## Environment Variables

Documented in `.env.example` and README. Key values:
- Telegram: `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`, `WEBHOOK_AUTH_KEY`
- Bot config: `BOT_SYSTEM_PROMPT`, `BOT_TRIGGER_WORDS`
- Infra: `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `FUNCTION_URL`
- APIs: `OPENROUTER_API_KEY`, `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `COINMARKETCAP_KEY`
- Monitoring: `ADMIN_CHAT_ID`, `FRIENDLY_INSTANCE_NAME`

## Rate Limiting

- Global: 1024 requests/hour
- Per chat: 128 requests/10 minutes
- Implemented in `src/main.rs` (`rate_limited` / `handle_rate_limit`).

## Webhook Setup

- Set webhook: `{FUNCTION_URL}/?update_webhook=true&key={WEBHOOK_AUTH_KEY}`
- Check webhook: `{FUNCTION_URL}/?check_webhook=true&key={WEBHOOK_AUTH_KEY}`

## Bot Configuration

Behavior is defined entirely by environment variables (`BOT_SYSTEM_PROMPT`, `BOT_TRIGGER_WORDS`, etc.), keeping personality private while retaining a public codebase.
