# Rust Test Plan

## Required pull-request gates

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

CI supplies PostgreSQL and Redis Stack and sets `TEST_POSTGRES_URL`, `TEST_DATABASE_URL`, and `TEST_REDIS_URL`. Tests use synthetic identities, fixed clocks, injected randomness, fake HTTP transports, and reserved database ranges.

## Coverage gates

- `bot-core`: at least 95% line coverage.
- `bot-adapters`: at least 85% line coverage.
- `botd`: at least 80% line coverage; critical dispatcher, AI transaction,
  task-execution, and Telegram-streaming modules remain above 90%.
- Routing, billing, and scheduling require state-transition and failure-path assertions regardless of percentages.

## Test layers

1. Unit and property tests cover parsing, formatting, routing, accounting, idempotency, and state machines.
2. Contract fixtures preserve command, billing, cache, task, Telegram, and provider wire formats.
3. Adapter tests use fake HTTP transports and real Redis/PostgreSQL where persistence semantics matter.
4. Integration tests cover billing replay/refund policy, scheduler claiming/advancement, Redis state, PostgreSQL schema and transactions, and real FFmpeg normalization.
5. Lifecycle tests cover polling offsets, retries, worker failure reporting, graceful shutdown, cancellation, and restart-safe claims.
6. Container verification builds the release image, confirms `botd` is executable, and confirms no scripting-language interpreter or virtual environment is present.

## Critical scenarios

- Telegram: malformed updates, unsupported events, callbacks, pre-checkout, payments, captioned and replied media commands, 429, conflict, timeouts, and file limits.
- AI: streaming fragmentation, partial tool calls, repeated rounds, provider fallback, malformed payloads, cancellation, usage reconciliation, and delivery failure.
- Billing: payer choice, concurrent reservations/transfers, replay, exact settlement, refunds, debt, interrupted work, and retention.
- Memory: ordering, TTL repair, search, compaction thresholds, lease contention, retries, dead letters, and stale markers.
- Tasks: delay/interval/cron recurrence, time zones, competing owners, occurrence replay, cancellation, restart reconstruction, AI billing, and delivery.
- External data: fresh/stale/missing cache, malformed and partial provider responses, SSRF/redirect rejection, and deterministic rendering.

## Release record

Before retirement, the complete compatibility suite passed 1,298 tests. After cutover, the native workspace passed 697 tests with Redis and PostgreSQL integrations, formatting, check, Clippy, release compilation, and Rust-only image inspection. Measured line coverage is 95.31% for `bot-core`, 85.75% for `bot-adapters`, and 82.88% for `botd`. CI retains floors of 95%, 85%, and 80% respectively.
