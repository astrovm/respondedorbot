# Telegram Bot (Rust / Cloudflare Workers)

A configurable Telegram bot built for Cloudflare Workers (Rust + Wasm). It handles Telegram webhooks and responds with AI-powered messages plus utility commands.

## Quick Start

1. Copy `.env.example` to `.env` and fill in your secrets and API keys.
2. Install Rust (stable) and ensure `cargo` is available.
3. Run the Worker locally:
   ```bash
   wrangler dev
   ```
4. Configure the Telegram webhook once the Worker is reachable:
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
- Hosting: `FUNCTION_URL` (public URL of your Cloudflare Worker)
- APIs: `OPENROUTER_API_KEY`, `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `COINMARKETCAP_KEY`, `GROQ_API_KEY`
- Monitoring: `ADMIN_CHAT_ID`, `FRIENDLY_INSTANCE_NAME`

### Cloudflare Worker bindings

- `BOT_KV` (KV Namespace): Required for caching, rate limiting, and storing the Telegram webhook secret token. The Worker code expects this exact binding name.

### Cloudflare Workers deployment

The included `wrangler.toml` defines the Worker name (`respondedorbot`), entry file (`build/worker/shim.mjs`), and the `BOT_KV` binding produced by `worker-build`.

1. Install the Cloudflare CLI: `npm install -g wrangler`.
2. Create the KV namespace binding expected by the code:
   ```bash
   wrangler kv:namespace create BOT_KV
   wrangler kv:namespace create BOT_KV --preview
   ```
   Update `wrangler.toml` with the generated IDs if they differ.
3. Build and deploy the Worker:
   ```bash
   wrangler publish
   ```
4. Set your public Worker URL in `FUNCTION_URL` (e.g., `https://<worker>.<account>.workers.dev` or your custom domain).
5. Configure the Telegram webhook to point to the Worker (after deployment):
   ```bash
   curl -X GET "$FUNCTION_URL/?update_webhook=true&key=$WEBHOOK_AUTH_KEY"
   curl -X GET "$FUNCTION_URL/?check_webhook=true&key=$WEBHOOK_AUTH_KEY"
   ```
   The Worker sets the Telegram `secret_token` to `WEBHOOK_AUTH_KEY`, which must match the `X-Telegram-Bot-Api-Secret-Token` header on incoming updates.

## Development

- Run locally: `wrangler dev`
- Build: `cargo build --target wasm32-unknown-unknown --features worker --no-default-features`
- Lint: `cargo clippy --target wasm32-unknown-unknown --features worker -- -D warnings`
- Tests (requires `wasm-bindgen-test-runner`): `cargo test`
  - Install once: `cargo install wasm-bindgen-cli`

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
