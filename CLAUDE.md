# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a configurable Telegram bot written in Python/Flask that can be customized with different personalities and characteristics. The bot provides cryptocurrency prices, currency exchange rates, AI-powered conversations, BCRA economic data, audio/image transcription, and various utility commands. The bot's personality and behavior are fully configurable through external configuration files.

## Development Commands

### Running the Application
```bash
# Start the Flask application locally
flask --app api/index run --host 0.0.0.0 --port 8080

# Run tests
python -m pytest test.py -v

```

### Testing
```bash
# Run all tests
python test.py

# Run specific test functions
python -m pytest test.py::test_convert_to_command -v
python -m pytest test.py::test_handle_msg -v
```

## Architecture

### Core Components

**Main Application (`api/index.py`):**
- Flask web server handling Telegram webhooks
- Redis-based caching system for API responses and chat history
- OpenAI/OpenRouter integration for AI conversations
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
3. AI responses: chat history retrieval → message building → OpenRouter API → response caching
4. All API calls go through `cached_requests()` with configurable TTL

### Dependencies
- **Flask**: Web framework for webhook handling
- **Redis**: Caching layer for API responses and chat history
- **OpenAI**: AI conversation capabilities via OpenRouter
- **Requests**: HTTP client for external APIs
- **Cryptography**: For webhook security tokens

### External APIs
- **Telegram Bot API**: Message sending/receiving and file downloads
- **CoinMarketCap**: Cryptocurrency prices (requires `COINMARKETCAP_KEY`)
- **CriptoYa**: Argentine peso exchange rates
- **OpenRouter**: AI model access (requires `OPENROUTER_API_KEY`)
- **Groq**: AI model access + built-in tools (requires `GROQ_API_KEY`)
- **BCRA**: Economic variables retrieved through the official API (https://api.bcra.gob.ar/estadisticas/v4.0)
- **Open-Meteo**: Weather data for Buenos Aires

### Environment Variables
Required environment variables are documented in README.md. Critical ones:
- `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`: Bot authentication
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Cache configuration
- `COINMARKETCAP_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`: API access
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

### Testing Strategy
- Comprehensive test suite in `test.py` covering all major functions
- Mocked external dependencies (Redis, APIs, file system)
- Edge case testing for message parsing, rate limiting, error conditions
- Tests run independently without external service dependencies

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
