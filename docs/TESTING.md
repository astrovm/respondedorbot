# Testing

## Required pull-request gates

```bash
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets --all-features
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

CI supplies PostgreSQL and Redis 8 and sets `TEST_DATABASE_URL`, `TEST_REDIS_HOST`, and `TEST_REDIS_PORT`. Tests that need a service return early when its variable is missing, so set all three locally to run them (plus `TEST_REDIS_PASSWORD` if your Redis needs one); in CI a guard test fails if any is missing. Tests use synthetic identities, fixed clocks, injected randomness, fake HTTP transports, and reserved database ranges. Tests that rerun global billing migrations do so in a schema of their own, so the suite runs in parallel and can be rerun against the same database.

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

## Fuzzing

`fuzz/` holds [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets for everything that reads untrusted input: Telegram updates, text users type, link repair, fetched web pages, OpenRouter replies (whole and streamed), Polymarket search responses, and stored task records. CI runs each one for a minute. Run them longer locally:

```bash
cd fuzz
cargo +nightly fuzz run user_text corpus/user_text seeds/user_text -- -max_total_time=900
```

## Mutation testing

[cargo-mutants](https://mutants.rs/) changes `bot-core` in small ways, such as flipping a comparison or emptying a function, and runs the crate's tests for each change. A change no test notices is a missed mutant: code whose behavior nothing checks. Only `bot-core` is mutated: its tests are fast and need no PostgreSQL, Redis, FFmpeg or network, while the `bot-adapters` and `botd` tests that need them return early without them.

```bash
scripts/mutants.sh diff origin/main   # the bot-core lines this branch changed
scripts/mutants.sh shard 3/64         # the 4th of 64 equal parts of every mutant
```

Results land in `build/mutants/mutants.out` (`missed.txt`, `caught.txt`, `timeout.txt` and a log per mutant). Settings live in `.cargo/mutants.toml`.

It never blocks a merge: `CI / Required` doesn't wait for it. On a pull request that changes `bot-core`, the `Mutants` workflow mutates the changed lines and keeps one comment listing what no test caught (a mutant that makes the tests hang is listed too); the comment goes away once everything is caught. Each one is a lead for a follow-up PR, unless the change doesn't alter behavior. A pull request that changes the mutation setup also runs part 0 of 64, so a broken setup shows before the weekly run. Every Saturday the workflow runs a quarter of all mutants, 16 of the 64 parts, so four weeks cover everything, and lists what survived in the open "Missed mutants" issue (it opens one if something lived).

`scripts/tests/mutants.sh` checks the two helper scripts with a fake `cargo` and a fake `gh`. `CI / Required` runs it.

## Test layers

1. Unit and property tests cover parsing, formatting, routing, accounting, idempotency, and state machines.
2. External-format fixtures preserve provider payload and persisted task-record compatibility.
3. Adapter tests use fake HTTP transports and real Redis/PostgreSQL where persistence semantics matter.
4. Integration tests cover billing replay/refund policy, scheduler claiming/advancement, Redis state, PostgreSQL schema and transactions, and real FFmpeg normalization.
5. Lifecycle tests cover polling offsets, retries, worker failure reporting, graceful shutdown, cancellation, and restart-safe claims.
6. The end-to-end test (`crates/botd/tests/telegram_end_to_end.rs`) runs the real `botd` binary against fake Telegram and OpenRouter servers (through `TELEGRAM_API_URL` and `OPENROUTER_BASE_URL`) with real PostgreSQL and Redis. It checks the command menus, `/help`, a plain command, a billed `/ask` reply, a button press, and a clean shutdown. A dead proxy blocks every other outbound request.
7. Container verification builds the release image and confirms `botd` is executable.
8. Legacy-data migration tests cover dry-run behavior, idempotent application,
   record versioning, task reconstruction, and conditional Redis writes.

## Critical scenarios

- Telegram: malformed updates, unsupported events, callbacks, pre-checkout, payments, captioned and replied media commands, 429, conflict, timeouts, and file limits.
- AI: streaming fragmentation, partial tool calls, repeated rounds, provider fallback, malformed payloads, cancellation, usage reconciliation, and delivery failure.
- Billing: payer choice, concurrent reservations/transfers, replay, exact settlement, refunds, debt, interrupted work, and retention.
- Memory: ordering, TTL repair, search, compaction thresholds, lease contention, retries, dead letters, and stale markers.
- Tasks: delay/interval/cron recurrence, time zones, competing owners, occurrence replay, cancellation, restart reconstruction, AI billing, and delivery.
- External data: fresh/stale/missing cache, malformed and partial provider responses, SSRF/redirect rejection, and deterministic rendering.
