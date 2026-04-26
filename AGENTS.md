## Commands

```bash
ruff check api/ tests/          # lint
mypy api/                        # typecheck
python -m pytest -q              # tests (no Redis needed - mocked in conftest)
```

CI runs all three in that order. Pre-commit runs them too (`ruff --fix`, `mypy api/`, `pytest -q`).

To run a single test file: `python -m pytest tests/test_ai_pipeline.py -q`

## Stack

- Python 3.14, dependencies in `requirements.txt` (no Poetry/uv/lockfile)
- Lint: ruff (rules in `pyproject.toml`, not defaults - many ignores including `I001`, `E501`, `S101`)
- Types: mypy with `ignore_missing_imports = true`, `disallow_untyped_defs = false`
- Tests: pytest, no coverage enforcement
- Bot framework: python-telegram-bot v22
- AI: OpenRouter (chat/vision), Groq (transcription, with OpenRouter fallback)
- Storage: Redis (requires RediSearch module - redis-stack-server, not plain redis) + Supabase Postgres (credits)
- Deployment: Podman quadlets + systemd, not Docker

## Architecture

**Entrypoints:**
- `run_polling.py` - bot main loop (loads `.env`, starts polling + price refresh + task scheduler)
- `run_maintenance.py` - cron job (Redis cleanup, cache pruning)

**`api/index.py` is the god module (~5700 lines):** handlers, commands, provider wiring, market data, billing helpers. It has an `F401` ruff exception because it re-exports symbols used by `bot_ptb.py` and other modules. Do not clean up "unused" imports in this file without checking callers.

**Key modules:**
- `api/config.py` - env loading, Redis connection factory, bot config (reads `workspace/SOUL.md` + `workspace/RULES.md`)
- `api/bot_ptb.py` - python-telegram-bot async adapter (calls sync handlers from `index.py`)
- `api/message_handler.py` - message routing, command dispatch
- `api/ai_service.py` - AI orchestration (credit reserve → model call → settle)
- `api/ai_pipeline.py` - response cleanup (prefix stripping, dedup, identity leak prevention)
- `api/ai_billing.py` - credit reservation/settlement/refund
- `api/streaming.py` - token-by-token Telegram message editing
- `api/providers/` - `ProviderChain` (tries providers in order), `OpenRouterProvider`, `GroqProvider`
- `api/message_state.py` - Redis chat history, RediSearch indexing, compaction markers
- `api/tools/` - agentic tool registry (crypto, calculator, web fetch, task scheduler)
- `api/services/` - `bcra.py`, `credits_db.py` (Supabase), `maintenance.py`, `redis_helpers.py`
- `api/utils/` - formatting, HTTP helpers, link enrichment, YouTube transcript

**Bot personality** lives in `workspace/SOUL.md` (character) and `workspace/RULES.md` (response rules). These are loaded by `api/config.py` at startup. The env var `BOT_SYSTEM_PROMPT` overrides both.

**Tests** (`tests/`): `conftest.py` sets up a `_FastFailRedis` that raises on any Redis call, and monkeypatches `config_redis`, `complete_with_providers`, and `time.sleep`. Tests run fully offline with no external services.

## mypy exceptions

These modules have `ignore_errors = true` in `pyproject.toml` - they are intentionally untyped:
`api.index`, `api.message_handler`, `api.ai_billing`, `api.provider_runtime`, `api.services.bcra`, `api.services.maintenance`, `api.agent_tools`, `api.utils.youtube_transcript`

Do not add type annotations that would require changes in these files without removing the ignore first.

## Gotchas

- `.env` loading is manual (inline parser with `dotenv` fallback). No framework auto-loads it.
- `api/index.py` has ruff `F401` exception - unused imports are intentional re-exports.
- Redis must be redis-stack-server (RediSearch `FT.CREATE`/`FT.SEARCH` used for chat memory).
- `python-telegram-bot` is pinned `<23.0` and `APScheduler` pinned `<4.0`.
- Tests have many per-file ruff ignores in `pyproject.toml` - do not refactor test files to remove these without understanding why.
- Container image uses Podman, not Docker (`Containerfile`, not `Dockerfile`).

## Conventions

- Code and comments in English. User-facing strings (bot personality, command responses) stay in Spanish/Argentine slang.
- Remove unused imports/variables/dead code from files you edit.
- No `as any`, `@ts-ignore` equivalent in Python (`# type: ignore` only where mypy config already allows it).
- All imports at top of file. None inside functions unless breaking circular deps.
- Plain hyphens and straight quotes only. No decorative Unicode.

## Git

- Ask before pushing every time.
- No batch commit+push. No force push or hard reset without approval.
- Never `git commit --amend` unless explicitly asked.
- Commit messages in English.

## Debugging

- Read code before explaining. Prove with direct evidence: failing test, reproduced run, or concrete probe.
- Reproduce before fixing runtime or external issues.
- Unproven concerns are risks, not bugs. Say so if not reproduced.

## Verification

- Smallest proof first, then broader checks.
- No "fixed/safe/ready" claims without fresh command output.
- Fix every issue you encounter in code you changed. Pre-existing issues in untouched code are not your responsibility.

## Tests

- Run before committing or declaring work complete. No exceptions.
- A failing test is a blocking issue. Fix it before moving on.
- Tests need no external services (Redis mocked, providers monkeypatched).

## Dependencies

- When adding a dependency, verify the actual latest version from the registry. Never rely on model memory.
- Environment variables only for secrets and external credentials. Hardcode sensible defaults for everything else.
