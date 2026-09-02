# Testing

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
- `bot-adapters`: at least 95% line coverage.
- `botd`: at least 95% line coverage.
- Routing, billing, and scheduling require state-transition and failure-path assertions regardless of percentages.

## Test layers

1. Unit and property tests cover parsing, formatting, routing, accounting, idempotency, and state machines.
2. External-format fixtures preserve provider payload and persisted task-record compatibility.
3. Adapter tests use fake HTTP transports and real Redis/PostgreSQL where persistence semantics matter.
4. Integration tests cover billing replay/refund policy, scheduler claiming/advancement, Redis state, PostgreSQL schema and transactions, and real FFmpeg normalization.
5. Lifecycle tests cover polling offsets, retries, worker failure reporting, graceful shutdown, cancellation, and restart-safe claims.
6. Container verification builds the release image and confirms `botd` is executable.

## Critical scenarios

- Telegram: malformed updates, unsupported events, callbacks, pre-checkout, payments, captioned and replied media commands, 429, conflict, timeouts, and file limits.
- AI: streaming fragmentation, partial tool calls, repeated rounds, provider fallback, malformed payloads, cancellation, usage reconciliation, and delivery failure.
- Billing: payer choice, concurrent reservations/transfers, replay, exact settlement, refunds, debt, interrupted work, and retention.
- Memory: ordering, TTL repair, search, compaction thresholds, lease contention, retries, dead letters, and stale markers.
- Tasks: delay/interval/cron recurrence, time zones, competing owners, occurrence replay, cancellation, restart reconstruction, AI billing, and delivery.
- External data: fresh/stale/missing cache, malformed and partial provider responses, SSRF/redirect rejection, and deterministic rendering.
