# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RespondedorBot is a Telegram bot written in Python/Flask that operates as "el gordo" - an Argentine bot character based on the "atendedor de boludos" meme. The bot provides cryptocurrency prices, currency exchange rates, AI-powered conversations, and various utility commands.

## Development Commands

### Running the Application
```bash
# Start the Flask application locally
flask --app api/index run --host 0.0.0.0 --port 8080

# Run tests
python -m pytest test.py -v

# Build and run with Docker
docker build -t respondedorbot .
docker run -p 8080:8080 respondedorbot

# Deploy with Docker Compose
docker-compose up -d
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
- `handle_msg()` - Main message processing pipeline at api/index.py:1432
- `ask_ai()` - AI conversation handler at api/index.py:809
- `cached_requests()` - Generic API caching wrapper at api/index.py:60
- `get_prices()` - Cryptocurrency price fetcher at api/index.py:248
- `get_dollar_rates()` - Argentine peso exchange rates at api/index.py:472

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
- **Telegram Bot API**: Message sending/receiving
- **CoinMarketCap**: Cryptocurrency prices (requires `COINMARKETCAP_KEY`)
- **CriptoYa**: Argentine peso exchange rates
- **OpenRouter**: AI model access (requires `OPENROUTER_API_KEY`)
- **Open-Meteo**: Weather data for Buenos Aires

### Environment Variables
Required environment variables are documented in README.md. Critical ones:
- `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`: Bot authentication
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`: Cache configuration
- `COINMARKETCAP_KEY`, `OPENROUTER_API_KEY`: API access
- `ADMIN_CHAT_ID`: Error reporting destination

### Rate Limiting
- Global: 1024 requests/hour
- Per chat: 128 requests/10 minutes
- Implemented in `check_rate_limit()` at api/index.py:1357

### Error Handling
- All errors are reported to admin via `admin_report()` at api/index.py:722
- Graceful fallbacks for API failures
- Redis connection failures raise exceptions to prevent silent failures

### Testing Strategy
- Comprehensive test suite in `test.py` covering all major functions
- Mocked external dependencies (Redis, APIs, file system)
- Edge case testing for message parsing, rate limiting, error conditions
- Tests run independently without external service dependencies

## Character and Content
The bot operates as "el gordo" - an Argentine character with specific personality traits and language patterns. When modifying conversation logic, maintain the established character voice and Argentine Spanish vernacular present in the system prompts and responses.