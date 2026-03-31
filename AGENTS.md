# Repository Guidelines

## Project Overview

This is a configurable Telegram bot written in Python that can be customized with different personalities and characteristics. The bot provides cryptocurrency prices, currency exchange rates, AI-powered conversations, BCRA economic data, audio/image transcription, and various utility commands. The bot's personality and behavior are fully configurable through environment variables.

## Project Structure & Module Organization

- `api/index.py`: Core handlers, AI integration, and command routing.
- `api/bot_ptb.py`: python-telegram-bot polling runtime and async handlers.
- `api/message_handler.py`: Message flow orchestration, billing, and rate-limit gating.
- `api/ai_pipeline.py`: AI response cleanup, normalization, and tool execution.
- `api/chat_context.py`: Chat history management and context building.
- `api/chat_settings.py`: Per-chat configuration and settings persistence.
- `api/random_replies.py`: Random reply generation for group chats.
- `api/command_registry.py`: Command registration and routing system.
- `api/credit_units.py`: Credit unit conversions for AI billing.
- `api/ai_billing.py`: AI credits billing system with Telegram Stars integration.
- `api/groq_billing.py`: Groq-specific billing calculations.
- `api/config.py`: Application-wide configuration helpers and Redis client setup.
- `api/services/`: Service modules (BCRA, Redis helpers, credits database).
- `api/utils/`: Utility modules (caching, formatting, HTTP, links).
- `run_polling.py`: Polling entrypoint.
- `tests/`: pytest-based test suite organized by module.
- `benchmark_bot.py`: Script to benchmark LLM responses against the bot's personality.
- `requirements.txt`: Python runtime dependencies.
- `.env.example` / `.env`: Configuration template and local overrides (do not commit secrets).
- `README.md`, `CLAUDE.md`: Usage and personality guidance.

## Build, Test, and Development Commands

- Create env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run locally: `python run_polling.py` (reads env from `.env`).
- Run tests: `pytest tests/` or `python -m pytest tests/ -v`.
- Run specific tests: `python -m pytest tests/test_ai_pipeline.py::test_ask_ai_with_provider_success -v`.
- Lint/format (optional): adhere to PEP 8; use your editor's formatter.

## Architecture

### Core Components

**Main Application (`api/index.py`):**
- Core command routing and AI integration
- Redis-based caching system for API responses and chat history
- Groq integration for AI conversations via Cloudflare AI Gateway
- Multiple command handlers for crypto prices, currency rates, utilities
- Rate limiting with per-chat and global quotas

**Message Handler (`api/message_handler.py`):**
- Entry point for all incoming messages
- Billing validation and credit deduction
- Command routing and AI gating

**AI Pipeline (`api/ai_pipeline.py`):**
- AI response post-processing and cleanup
- Tool call parsing and execution
- Compound tool integration (web search, browser automation)

**Key Functions:**
- `handle_msg()` - Main message processing pipeline at `api/index.py`
- `ask_ai()` - AI conversation handler with multi-provider fallback
- `cached_requests()` - Generic API caching wrapper
- `get_or_refresh_bcra_variables()` - Fetches BCRA economic data via official API
- `handle_transcribe_with_message()` - Audio/image transcription handler

### Data Flow
1. Telegram update (PTB polling) → `api/bot_ptb.py` async handlers → `api/message_handler.py`
2. Message processing: text extraction → command parsing → billing check → rate limiting → handler execution
3. AI responses: chat history retrieval → message building → Cloudflare AI Gateway → Groq API → response caching
4. All API calls go through `cached_requests()` with configurable TTL

### Dependencies
- **Redis**: Caching layer for API responses and chat history
- **OpenAI SDK**: Client library used to call Groq's OpenAI-compatible API through Cloudflare AI Gateway
- **Requests**: HTTP client for external APIs
- **python-telegram-bot**: Polling runtime and Telegram update handling
- **Supabase Postgres**: AI credits persistence (required for AI functionality — no fallback)

### External APIs
- **Telegram Bot API**: Message sending/receiving and file downloads
- **CoinMarketCap**: Cryptocurrency prices (requires `COINMARKETCAP_KEY`)
- **CriptoYa**: Argentine peso exchange rates
- **Groq (via Cloudflare AI Gateway)**: AI model access + built-in tools (requires `GROQ_API_KEY` and `CF_AIG_BASE_URL`)
- **BCRA**: Economic variables retrieved through the official API (https://api.bcra.gob.ar/estadisticas/v4.0)
- **Open-Meteo**: Weather data for Buenos Aires

### Environment Variables
Required environment variables are documented in `.env.example`. Critical ones:

**Bot Configuration:**
- `BOT_SYSTEM_PROMPT`: Complete AI personality prompt
- `BOT_TRIGGER_WORDS`: Comma-separated keywords that trigger responses in groups
- `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`: Bot authentication

**Infrastructure:**
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Cache configuration
- `SUPABASE_POSTGRES_URL`: Pooled Supabase Postgres URL for credits persistence (required for AI — bot will not process AI requests without this)

**AI & APIs:**
- `GROQ_API_KEY`: Paid Groq API key
- `GROQ_FREE_API_KEY`: Optional free-tier Groq key (tried first, falls back to paid)
- `CF_AIG_BASE_URL`: Cloudflare AI Gateway base URL (e.g., `https://gateway.ai.cloudflare.com/v1/ACCOUNT_ID/GATEWAY_NAME/compat`)
- `CF_AIG_TOKEN`: Cloudflare AI Gateway authorization token (optional, for authenticated gateways)
- `COINMARKETCAP_KEY`: CoinMarketCap API key
- `GIPHY_API_KEY`: Giphy API key for `/gm` and `/gn` commands

**Monitoring:**
- `ADMIN_CHAT_ID`: Telegram chat ID for error reports
- `FRIENDLY_INSTANCE_NAME`: Instance identification for admin reports

**⚠️ Security Note:** Never commit `CF_AIG_BASE_URL`, `CF_AIG_TOKEN`, or any API keys to the repository. These contain sensitive account identifiers and must be configured via environment variables in your deployment platform.

### Rate Limiting
- Global: 1024 requests/hour
- Per chat: 128 requests/10 minutes
- Groq API: Account-specific limits with local rate limit tracking
- Implemented in `check_rate_limit()` and `_reserve_groq_rate_limit()`

### Error Handling
- All errors are reported to admin via `admin_report()`
- Graceful fallbacks for API failures
- Redis connection failures raise exceptions to prevent silent failures
- Groq API errors trigger automatic fallback between free and paid accounts

### AI Credits Billing
Users consume credits for AI responses:
- Onboarding credits granted on first interaction
- Recharge via Telegram Stars (`/topup` command)
- Check balance with `/balance` (personal in DM, personal + group in groups)
- Transfer credits between personal and group balances with `/transfer`
- Credits deducted per token/audio second/tool usage

## Bot Configuration
The bot's personality and behavior are configured entirely through environment variables:

- `BOT_SYSTEM_PROMPT`: Complete AI personality prompt that defines the bot's character
- `BOT_TRIGGER_WORDS`: Comma-separated keywords that trigger responses in group chats

This approach allows the codebase to remain public while keeping specific bot personalities private and makes deployment much simpler across different platforms.

## Coding Style & Naming Conventions

- Python 3; 4‑space indentation (enforced via `.editorconfig`).
- Names: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants and env keys.
- Keep functions small and pure where possible; prefer helpers in `api/index.py`.
- Docstrings for public helpers; concise inline comments only where intent isn't obvious.
- Type hints encouraged for public APIs.

## Testing Guidelines

- Framework: `pytest` with simple `assert` style.
- File layout: Tests organized in `tests/` directory, mirroring module structure where applicable.
- Naming: test functions start with `test_...` and describe behavior, e.g., `test_should_gordo_respond_mentions`.
- Run: `pytest -q`; aim to cover new code paths and error handling.
- Comprehensive mocking of external dependencies (Redis, APIs, file system, Telegram API).
- Tests run independently without external service dependencies.

## Commit & Pull Request Guidelines

- Messages: short, imperative subject lines (e.g., "Add /usd alias"), optional body for rationale/impact.
- Scope changes narrowly; separate refactors from feature changes.
- PRs include: summary, linked issues, config notes (env vars, webhooks), and logs/screenshots for user-visible changes.
- Ensure `pytest` passes and the app starts with sample `.env` before requesting review.
- Never include sensitive URLs, tokens, or API keys in commits.

## Security & Configuration Tips

- **Never commit secrets**: Use `.env.example` to document required vars, keep actual values in deployment platform.
- **Cloudflare AI Gateway**: The gateway base URL contains your Cloudflare account ID — keep this private and configure via `CF_AIG_BASE_URL` env var.
- **API Keys**: All API keys (`GROQ_API_KEY`, `COINMARKETCAP_KEY`, etc.) must be configured via environment variables.
- **Redis**: Handle connection failures gracefully; cache via Redis when available.
- **Rate Limiting**: Implement both local and upstream rate limiting to prevent abuse.
- **Error Reporting**: Admin reports include sanitized context; never include raw API keys or tokens in error messages.

## Recent Major Features

**Cloudflare AI Gateway:**
- All Groq API requests route through Cloudflare AI Gateway
- Configurable via `CF_AIG_BASE_URL` and optional `CF_AIG_TOKEN`
- Model names prefixed with `groq/` for gateway routing

**BCRA Economic Data (/bcra, /variables):**
- Uses the official BCRA statistics API (https://api.bcra.gob.ar/estadisticas/v4.0)
- Extracts 11 key variables: Base monetaria, Inflación (mensual/interanual/esperada), TAMAR, BADLAR, Dólar (minorista/mayorista), UVA, CER, Reservas
- Caches responses in Redis for 5 minutes and persists Dólar Mayorista history

**Audio/Image Transcription (/transcribe):**
- Must be used as reply to messages containing audio, images, or stickers
- Audio transcription via Groq Whisper
- Image description via Groq vision model
- 7-day Redis caching for both audio and image processing
- Automatic file download from Telegram servers

**AI Credits Billing:**
- Telegram Stars integration for purchasing AI credits
- Per-user and per-group credit tracking
- Automatic credit deduction based on token usage and tool calls

**Web Search (/buscar, /search):**
- Groq Compound integration for real-time web search
- Automatic tool execution with synthesized responses
- Results cached to reduce redundant searches

**Link Enrichment:**
- URLs in messages automatically fetched for metadata
- Link context injected into AI conversations
- Metadata cached to improve response times
