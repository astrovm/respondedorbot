# Repository Guidelines

## Project Structure & Module Organization

- `src/main.rs`: Axum server entrypoint and Telegram webhook handling.
- `src/agent`, `src/commands`, `src/market`, etc.: Core bot features and helpers.
- `tests/`: Rust integration/unit tests covering commands, tools, and helpers.
- `.env.example` / `.env`: Configuration template and local overrides (do not commit secrets).
- `README.md`, `CLAUDE.md`: Usage and personality guidance.
- BCRA economic variables are retrieved via the official BCRA API (helpers in `src/bcra`); avoid web scraping.

## Build, Test, and Development Commands

- Install toolchain: stable Rust (edition 2021).
- Run locally: `cargo run` (reads env from `.env`) — binds to `0.0.0.0:8080`.
- Run tests: `cargo test`.
- Lint: `cargo clippy --all-targets --all-features -- -D warnings`.
- Format (optional): `cargo fmt`.

## Coding Style & Naming Conventions

- Rust 2021; 4‑space indentation (enforced via `.editorconfig`).
- Names: `snake_case` for functions/modules, `PascalCase` for types, `UPPER_SNAKE_CASE` for constants and env keys.
- Keep functions small and focused; prefer helpers in `src` modules.
- Doc comments for public helpers; concise inline comments where intent isn’t obvious.

## Testing Guidelines

- Framework: Rust `cargo test`.
- File layout: colocated Rust tests under `tests/`; add new tests near changed behavior when possible.
- Naming: test functions start with `test_...` and describe behavior, e.g., `test_should_gordo_respond_mentions`.
- Aim to cover new code paths and error handling; prefer deterministic tests (mock external IO).

## Commit & Pull Request Guidelines

- Messages: short, imperative subject lines (e.g., "Add /usd alias"), optional body for rationale/impact.
- Scope changes narrowly; separate refactors from feature changes.
- PRs include: summary, linked issues, config notes (env vars, webhooks), and logs/screenshots for user-visible changes.
- Ensure `cargo test` and `cargo clippy --all-targets --all-features -- -D warnings` pass before requesting review.

## Security & Configuration Tips

- Never commit secrets; use `.env.example` to document required vars (see `README.md`).
- Validate `WEBHOOK_AUTH_KEY` usage when touching webhook paths; avoid logging secrets.
- Networked features rely on `REDIS_*`, OpenRouter, and other API keys—handle failures gracefully and cache via Redis when available.
