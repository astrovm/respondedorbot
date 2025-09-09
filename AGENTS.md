# Repository Guidelines

## Project Structure & Module Organization
- `api/index.py`: Flask app and Telegram webhook handler; most bot logic lives here.
- `test.py`: Pytest-based unit tests for `api.index` helpers.
- `benchmark_bot.py`: Script to benchmark LLM responses against the bot’s personality.
- `requirements.txt`: Python runtime dependencies.
- `.env.example` / `.env`: Configuration template and local overrides (do not commit secrets).
- `README.md`, `CLAUDE.md`: Usage and personality guidance.
- BCRA economic variables are retrieved via the official BCRA API (helpers in `api/index.py`); avoid web scraping.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Run locally: `flask --app api/index run --host 0.0.0.0 --port 8080` (reads env from `.env`).
- Run tests: `pytest -q` (discovers tests in `test.py`).
- Lint/format (optional): adhere to PEP 8; use your editor’s formatter.

## Coding Style & Naming Conventions
- Python 3; 4‑space indentation (enforced via `.editorconfig`).
- Names: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants and env keys.
- Keep functions small and pure where possible; prefer helpers in `api/index.py`.
- Docstrings for public helpers; concise inline comments only where intent isn’t obvious.

## Testing Guidelines
- Framework: `pytest` with simple `assert` style.
- File layout: colocated tests in `test.py` for now; add new tests near changed behavior.
- Naming: test functions start with `test_...` and describe behavior, e.g., `test_should_gordo_respond_mentions`.
- Run: `pytest -q`; aim to cover new code paths and error handling.

## Commit & Pull Request Guidelines
- Messages: short, imperative subject lines (e.g., "Add /usd alias"), optional body for rationale/impact.
- Scope changes narrowly; separate refactors from feature changes.
- PRs include: summary, linked issues, config notes (env vars, webhooks), and logs/screenshots for user-visible changes.
- Ensure `pytest` passes and the app starts with sample `.env` before requesting review.

## Security & Configuration Tips
- Never commit secrets; use `.env.example` to document required vars (see `README.md`).
- Validate `WEBHOOK_AUTH_KEY` usage when touching webhook paths; avoid logging secrets.
- Networked features rely on `REDIS_*`, OpenRouter, and other API keys—handle failures gracefully and cache via Redis when available.

