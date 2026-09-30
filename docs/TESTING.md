# Testing

## Required pull-request gates

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

CI supplies PostgreSQL and Redis Stack and sets `TEST_DATABASE_URL`, `TEST_REDIS_HOST`, and `TEST_REDIS_PORT`. Tests that need a service return early when its variable is missing, so set all three locally to run them (plus `TEST_REDIS_PASSWORD` if your Redis needs one); in CI a guard test fails if any is missing. Tests use synthetic identities, fixed clocks, injected randomness, fake HTTP transports, and reserved database ranges. Tests that rerun global billing migrations do so in a schema of their own, so the suite runs in parallel and can be rerun against the same database.

## Coverage gates

- `bot-core`: 100% line coverage.
- `bot-adapters`: 100% line coverage.
- `botd`: 100% line coverage.
- Routing, billing, and scheduling require state-transition and failure-path assertions regardless of percentages.

CI requires zero uncovered executable source lines in each crate:

```bash
cargo llvm-cov --locked --workspace --all-features --no-report
cargo llvm-cov report -p bot-core --fail-uncovered-lines 0 --show-missing-lines
cargo llvm-cov report -p bot-adapters --fail-uncovered-lines 0 --show-missing-lines
cargo llvm-cov report -p botd --fail-uncovered-lines 0 --show-missing-lines
```

The source-line check avoids a [Rust/LLVM reporting bug](https://github.com/rust-lang/rust/issues/137524)
that can count uncovered generic instantiations in the summary even when all
exported source lines are covered. It still rejects any uncovered source line.

## Test layers

1. Unit and property tests cover parsing, formatting, routing, accounting, idempotency, and state machines.
2. External-format fixtures preserve provider payload and persisted task-record compatibility.
3. Adapter tests use fake HTTP transports and real Redis/PostgreSQL where persistence semantics matter.
4. Integration tests cover billing replay/refund policy, scheduler claiming/advancement, Redis state, PostgreSQL schema and transactions, and real FFmpeg normalization.
5. Lifecycle tests cover polling offsets, retries, worker failure reporting, graceful shutdown, cancellation, and restart-safe claims.
6. Container verification builds the release image and confirms `botd` is executable.
7. Legacy-data migration tests cover dry-run behavior, idempotent application,
   record versioning, task reconstruction, and conditional Redis writes.

## Critical scenarios

- Telegram: malformed updates, unsupported events, callbacks, pre-checkout, payments, captioned and replied media commands, 429, conflict, timeouts, and file limits.
- AI: streaming fragmentation, partial tool calls, repeated rounds, provider fallback, malformed payloads, cancellation, usage reconciliation, and delivery failure.
- Billing: payer choice, concurrent reservations/transfers, replay, exact settlement, refunds, debt, interrupted work, and retention.
- Memory: ordering, TTL repair, search, compaction thresholds, lease contention, retries, dead letters, and stale markers.
- Tasks: delay/interval/cron recurrence, time zones, competing owners, occurrence replay, cancellation, restart reconstruction, AI billing, and delivery.
- External data: fresh/stale/missing cache, malformed and partial provider responses, SSRF/redirect rejection, and deterministic rendering.
