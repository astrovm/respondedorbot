# Repository Guidelines

## Project Overview

This is a configurable Telegram bot written in Python/Flask that can be customized with different personalities and characteristics. The bot provides cryptocurrency prices, currency exchange rates, AI-powered conversations, BCRA economic data, audio/image transcription, and various utility commands. The bot's personality and behavior are fully configurable through environment variables.

## Project Structure & Module Organization

- `api/index.py`: Flask app and Telegram webhook handler; most bot logic lives here.
- `test.py`: Pytest-based unit tests for `api.index` helpers.
- `benchmark_bot.py`: Script to benchmark LLM responses against the bot's personality.
- `requirements.txt`: Python runtime dependencies.
- `.env.example` / `.env`: Configuration template and local overrides (do not commit secrets).
- `README.md`, `CLAUDE.md`: Usage and personality guidance.
- BCRA economic variables are retrieved via the official BCRA API (helpers in `api/index.py`); avoid web scraping.

## Build, Test, and Development Commands

- Create env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run locally: `flask --app api/index run --host 0.0.0.0 --port 8080` (reads env from `.env`).
- Run tests: `pytest test.py` or `python -m pytest test.py -v`.
- Run specific tests: `python -m pytest test.py::test_convert_to_command -v`.
- Lint/format (optional): adhere to PEP 8; use your editor's formatter.

## Architecture

### Core Components

**Main Application (`api/index.py`):**
- Flask web server handling Telegram webhooks
- Redis-based caching system for API responses and chat history
- Groq integration for AI conversations
- Multiple command handlers for crypto prices, currency rates, utilities

**Key Functions:**
- `handle_msg()` - Main message processing pipeline at api/index.py:2166
- `ask_ai()` - AI conversation handler at api/index.py:1184
- `cached_requests()` - Generic API caching wrapper at api/index.py:115
- `get_or_refresh_bcra_variables()` - Fetches BCRA economic data via official API at api/index.py:1015
- `handle_transcribe_with_message()` - Audio/image transcription handler at api/index.py:792

### Data Flow
1. Telegram webhook → `responder()` → `process_request_parameters()` → `handle_msg()`
2. Message processing: text extraction → command parsing → rate limiting → handler execution
3. AI responses: chat history retrieval → message building → Groq API → response caching
4. All API calls go through `cached_requests()` with configurable TTL

### Dependencies
- **Flask**: Web framework for webhook handling
- **Redis**: Caching layer for API responses and chat history
- **OpenAI SDK**: client library used to call Groq's OpenAI-compatible API
- **Requests**: HTTP client for external APIs
- **Cryptography**: For webhook security tokens

### External APIs
- **Telegram Bot API**: Message sending/receiving and file downloads
- **CoinMarketCap**: Cryptocurrency prices (requires `COINMARKETCAP_KEY`)
- **CriptoYa**: Argentine peso exchange rates
- **Groq**: AI model access + built-in tools (requires `GROQ_API_KEY`)
- **BCRA**: Economic variables retrieved through the official API (https://api.bcra.gob.ar/estadisticas/v4.0)
- **Open-Meteo**: Weather data for Buenos Aires

### Environment Variables
Required environment variables are documented in README.md. Critical ones:
- `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`: Bot authentication
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Cache configuration
- `COINMARKETCAP_KEY`, `GROQ_API_KEY`: API access
- `ADMIN_CHAT_ID`: Error reporting destination
- `WEBHOOK_AUTH_KEY`: Webhook authentication key
- `FUNCTION_URL`: Deployment URL
- `FRIENDLY_INSTANCE_NAME`: Instance identification for reports

### Rate Limiting
- Global: 1024 requests/hour
- Per chat: 128 requests/10 minutes
- Implemented in `check_rate_limit()` at api/index.py:1825

### Error Handling
- All errors are reported to admin via `admin_report()` at api/index.py:1099
- Graceful fallbacks for API failures
- Redis connection failures raise exceptions to prevent silent failures

### Recent Major Features

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

### Webhook Setup
To configure the Telegram webhook:
- Set webhook: `{function_url}/?update_webhook=true&key={webhook_auth_key}`
- Check webhook: `{function_url}/?check_webhook=true&key={webhook_auth_key}`

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

## Testing Guidelines

- Framework: `pytest` with simple `assert` style.
- File layout: colocated tests in `test.py` for now; add new tests near changed behavior.
- Naming: test functions start with `test_...` and describe behavior, e.g., `test_should_gordo_respond_mentions`.
- Run: `pytest -q`; aim to cover new code paths and error handling.
- Comprehensive mocking of external dependencies (Redis, APIs, file system).
- Tests run independently without external service dependencies.

## Commit & Pull Request Guidelines

- Messages: short, imperative subject lines (e.g., "Add /usd alias"), optional body for rationale/impact.
- Scope changes narrowly; separate refactors from feature changes.
- PRs include: summary, linked issues, config notes (env vars, webhooks), and logs/screenshots for user-visible changes.
- Ensure `pytest` passes and the app starts with sample `.env` before requesting review.

## Security & Configuration Tips

- Never commit secrets; use `.env.example` to document required vars (see `README.md`).
- Validate `WEBHOOK_AUTH_KEY` usage when touching webhook paths; avoid logging secrets.
- Networked features rely on `REDIS_*`, Groq, and other API keys—handle failures gracefully and cache via Redis when available.
