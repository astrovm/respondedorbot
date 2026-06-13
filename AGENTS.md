## Commands

```bash
uv run --locked ruff check api/ tests/  # lint
uv run --locked mypy api/               # typecheck
uv run --locked python -m pytest -q     # tests (no Redis needed - mocked in conftest)
```

CI runs all three in that order. Pre-commit runs them too (`ruff --fix`, `mypy api/`, `pytest -q`).

To run a single test file: `uv run --locked python -m pytest tests/test_ai_pipeline.py -q`

## Stack

- Python 3.14, dependencies in `pyproject.toml`, locked by `uv.lock`
- Lint: ruff (rules in `pyproject.toml`, not defaults - many ignores including `I001`, `E501`, `S101`)
- Types: mypy strict across `api/`; only APScheduler's two untyped modules ignore missing imports
- Tests: pytest, no coverage enforcement
- Bot framework: python-telegram-bot v22
- AI: OpenRouter (chat/vision), Groq (transcription, with OpenRouter fallback)
- Storage: Redis (requires RediSearch module - redis-stack-server, not plain redis) + Supabase Postgres (credits)
- Deployment: Podman quadlets + systemd, not Docker

## Architecture

**Entrypoints:**
- `run_polling.py` - bot main loop (loads `.env`, starts polling + price refresh + task scheduler)
- `run_maintenance.py` - cron job (Redis cleanup, cache pruning)

`api/index.py` is the composition root. It builds long-lived services and keeps
public compatibility exports used by entrypoints and tests. Its `F401`
exception is intentional; check callers before removing imports.

**Domain packages:**
- `api/core/` - environment loading, Redis configuration, constants, logging
- `api/admin/` - admin commands, reporting, authorization service
- `api/ai/` - request preparation, orchestration, pricing, prompt and response pipeline
- `api/billing/` - credit reservation/settlement, commands, Stars callbacks
- `api/bot/` - Telegram adapter, routing, handlers, streaming, chat configuration
- `api/cache/` - shared HTTP and Redis cache mechanics
- `api/links/` - URL extraction, metadata, replacement, and link services
- `api/markets/` - crypto, dollar, stocks, Polymarket, weather, BCRA formatting
- `api/media/` - image, audio, video, transcription, and media cache
- `api/memory/` - chat history, RediSearch retrieval, compaction, summaries
- `api/providers/` - provider clients, fallback chain, cooldowns, usage extraction
- `api/tasks/` - task execution and APScheduler integration
- `api/tools/` - agentic tool registry and tool implementations
- `api/services/` - persistence and low-level external service adapters
- `api/utils/` - reusable formatting, HTTP, caching, and transcript helpers

**Bot personality** lives outside Git in `workspace/SOUL.md` (character) and `workspace/RULES.md` (response rules). In production, the VPS directory `~/respondedorbot/workspace` is mounted read-only at `/app/workspace`. These are loaded by `api/core/config.py` at startup. The env var `BOT_SYSTEM_PROMPT` overrides both.

**Tests** (`tests/`): `conftest.py` sets up a `_FastFailRedis` that raises on any Redis call, and monkeypatches `config_redis`, `complete_with_providers`, and `time.sleep`. Tests run fully offline with no external services.

## mypy exceptions

Application code is checked with `strict = true`. Missing imports are ignored
only for `apscheduler.jobstores.redis` and
`apscheduler.schedulers.background`, because APScheduler 3 has no type metadata.

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
