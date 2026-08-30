# Rust Migration Plan

## Status

- Branch: `codex/rust-migration`
- Current runtime: Python 3.14
- Current application size: approximately 34,770 lines under `api/`
- Baseline verification: 1,015 Python tests passing
- Final runtime: Rust only, with no Python interpreter, Python sidecar, or embedded Python

## Objectives

1. Preserve every existing bot feature throughout the migration.
2. Keep the production bot deployable and operational after every change.
3. Replace weakly typed runtime boundaries with explicit Rust types.
4. Produce a clean, simple, maintainable, and easy-to-read Rust codebase.
5. Establish meaningful unit, contract, integration, concurrency, failure-path,
   and end-to-end coverage.
6. Preserve existing Redis, PostgreSQL, Telegram, and scheduled-task data.
7. End with one native Rust application and remove all Python code and tooling.

## Working Documents

- [`docs/rust-migration/ARCHITECTURE.md`](docs/rust-migration/ARCHITECTURE.md)
  defines the final crate boundaries, type rules, concurrency ownership, and
  temporary bridge constraints.
- [`docs/rust-migration/FEATURE_MATRIX.md`](docs/rust-migration/FEATURE_MATRIX.md)
  tracks every user-visible and cross-cutting feature through parity and removal.
- [`docs/rust-migration/PERSISTENCE_CONTRACTS.md`](docs/rust-migration/PERSISTENCE_CONTRACTS.md)
  records PostgreSQL, Redis, task, configuration, and rollback contracts.
- [`docs/rust-migration/BILLING_CUTOVER.md`](docs/rust-migration/BILLING_CUTOVER.md)
  defines billing ownership, transaction invariants, and the staged writer cutover.
- [`docs/rust-migration/TEST_PLAN.md`](docs/rust-migration/TEST_PLAN.md) defines
  the contract runner, critical scenarios, coverage gates, and CI layers.

## Non-goals

- Do not redesign user-visible behavior during the compatibility migration.
- Do not remove, rename, or temporarily degrade features.
- Do not translate large Python modules line by line into equally large Rust modules.
- Do not introduce a permanent Python/Rust service boundary.
- Do not run multiple Telegram pollers, schedulers, or billing writers.
- Do not combine feature changes with migration changes.

Behavior improvements can follow in separate work after the relevant Rust path has
reached parity.

## Invariants

These conditions apply to every migration phase:

- Production has exactly one Telegram poller.
- Production has exactly one scheduler owner.
- Production has exactly one authoritative billing writer.
- Existing persistent data remains readable.
- Data migrations are additive and backward-compatible until rollback support ends.
- A Rust implementation is not authoritative until parity tests pass.
- An incomplete Rust path continues through the existing Python implementation.
- Every release has a tested rollback path.
- Synthetic, non-identifying data is used in tests and fixtures.

## Target Architecture

The final workspace will contain three crates:

```text
botd
  Application composition, configuration, startup, shutdown, and workers
    |
    v
bot-adapters
  Telegram, HTTP, Redis, PostgreSQL, providers, scheduling, and media
    |
    v
bot-core
  Domain types, routing, billing rules, AI state machines, and pure logic
```

Dependency direction must remain `botd -> bot-adapters -> bot-core`.

During the migration, a temporary `python-bridge` crate will expose Rust behavior
to the running Python application through PyO3. It is not part of the final
architecture and will be deleted with the Python runtime.

### Core design rules

- Deserialize untrusted payloads once at an adapter boundary.
- Do not pass `serde_json::Value` or untyped maps through domain logic.
- Use specific identifier types such as `ChatId`, `UserId`, `MessageId`, and
  `TaskId` instead of interchangeable strings and integers.
- Use enums to represent routing decisions, AI stream events, tool events,
  billing states, and failures.
- Keep I/O out of `bot-core`.
- Introduce traits only at real external boundaries.
- Avoid global mutable state and service locators.
- Return errors with actionable context; do not silently discard them.
- Avoid `unwrap` and `expect` in production request paths.
- Organize code by feature and responsibility, not in generic utility modules.
- Prefer direct, readable code over speculative abstractions.

## Testing Strategy

Tests are the primary migration control, not cleanup work performed afterward.

### Behavioral contracts

Language-neutral fixtures will describe:

- Incoming Telegram messages, callbacks, and payments.
- Expected outbound messages, edits, keyboards, and acknowledgements.
- Expected Redis and PostgreSQL state changes.
- Expected billing reservations, settlements, refunds, and ledger entries.
- Expected AI events, tools, provider fallback, and usage accounting.
- Expected scheduled-task creation, execution, recurrence, and cancellation.

During the hybrid period, the same fixtures will run against Python and Rust.
Rust must match the documented observable behavior before it becomes authoritative.

### Test layers

1. **Unit tests** for domain logic, parsing, routing, formatting, and state
   transitions.
2. **Property tests** for accounting invariants, parser robustness, idempotency,
   and state-machine properties.
3. **Contract tests** comparing Python and Rust behavior.
4. **Integration tests** using temporary Redis and PostgreSQL services and fake
   HTTP servers.
5. **End-to-end tests** that send Telegram update JSON into the application and
   inspect fake Telegram API requests.
6. **Concurrency tests** for simultaneous messages, duplicate callbacks,
   billing transactions, scheduler claims, and shutdown.
7. **Failure-path tests** for timeouts, malformed responses, service outages,
   retries, partial streams, and restarts.
8. **Fuzz tests** for commands, callback payloads, URLs, provider events, and
   Telegram payload decoding.
9. **Mutation tests** for critical routing, billing, and scheduling rules.

### Coverage policy

- `bot-core`: enforce at least 95% line coverage and target 90% branch coverage.
- Adapters: enforce at least 85% line coverage.
- Add branch-coverage enforcement when the stable Rust coverage toolchain reports
  branches; current `cargo llvm-cov` output exposes no branch counters.
- Routing, billing, and scheduling: test every documented state transition.
- Coverage must not decrease without an explicit explanation.
- Coverage percentages do not replace assertions, failure tests, concurrency
  tests, or mutation testing.

### CI gates

Every change must run at least:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
uv run --locked ruff check api/ tests/
uv run --locked mypy api/
uv run --locked python -m pytest -q
```

The Python checks remain until the corresponding Python implementation and its
tests have been replaced. Extended fuzz, mutation, restart, and load suites can
run outside the fast pull-request path.

## Migration Phases

### Phase 0: Behavior and data inventory

Create a feature matrix covering:

- Every Telegram command and alias.
- Free-form message routing and random replies.
- AI chat, summaries, streaming, tools, and provider fallback.
- Text, image, audio, video, and transcription behavior.
- Link inspection and replacement.
- Inline callbacks and Telegram Stars payments.
- Billing, reconciliation, transfers, balances, and history.
- Scheduled tasks and maintenance jobs.
- Memory, summaries, compaction, caching, and RediSearch.
- Market, weather, Polymarket, Hacker News, and BCRA behavior.
- Configuration, localization, admin reporting, and error handling.

Document all Redis keys and value formats, PostgreSQL tables and transaction
invariants, scheduler records, environment variables, external API behavior,
background workers, and deployment entrypoints.

**Exit gate:** every public feature has a success contract and every critical
feature has relevant failure and recovery contracts.

### Phase 1: No-behavior-change Rust foundation

- Add the Cargo workspace and crate boundaries.
- Pin the Rust toolchain.
- Add formatting, Clippy, tests, and coverage to CI.
- Add a Rust builder stage to the container.
- Add architecture rules and a pull-request migration checklist.
- Add the Python/Rust contract-test harness.
- Add telemetry that identifies whether Python or Rust handled a path.

**Exit gate:** the existing Python image behaves identically and all Python and
Rust checks pass.

### Phase 2: Typed boundary and pure logic

Define Rust types for Telegram input, commands, identifiers, credit units,
scheduled-task triggers, AI events, tool events, bot actions, and errors.

Migrate pure logic in this order:

1. Credit-unit parsing, scaling, and formatting.
2. Command parsing and aliases.
3. Scheduled-task trigger parsing and validation.
4. Price-query parsing.
5. Market normalization and formatting.
6. Routing decisions.

Python remains the poller and side-effect executor. Rust produces typed actions,
for example:

```rust
enum BotAction {
    Ignore,
    SendMessage {
        chat_id: ChatId,
        text: String,
    },
    RunAi(AiRequest),
    ExecuteCommand(Command),
    ProcessMedia(MediaRequest),
}
```

Pure functions can run in shadow mode. Python remains authoritative until results
match and the Rust path completes a production observation window.

**Exit gate:** migrated paths have parity, property, and fuzz coverage and can be
disabled independently.

### Phase 3: Command vertical slices

Move complete command behavior from low to high risk:

1. Stateless commands and calculations.
2. Market, crypto, dollar, stock, BCRA, and arbitrage commands.
3. Weather, Hacker News, and Polymarket.
4. Chat configuration and localization.
5. Link inspection and replacement.
6. Media inspection and transcription orchestration.
7. Admin and reporting commands.

Each command family is released separately. External calls are represented by
adapter interfaces so `bot-core` tests remain deterministic.

**Exit gate:** each migrated feature passes unit, contract, integration, and
failure-path tests and retains an immediate Python rollback flag.

### Phase 4: Redis state and background processing

Migrate:

- Cache helpers and stale-cache behavior.
- Chat configuration.
- Message history and reply metadata.
- RediSearch indexing and retrieval.
- Memory summaries and compaction.
- Background queues, retries, and dead-letter handling.
- Media caches.

Version stored payloads before Rust writes them. Test mixed-version reads,
restarts, duplicate processing, worker crashes, queue recovery, and index repair.

**Exit gate:** Rust can read all existing data, Python can read Rust-written data
during the rollback period, and recovery tests pass.

### Phase 5: PostgreSQL billing

Migrate billing as one audited subsystem rather than porting isolated SQL helpers.

The Rust implementation must cover:

- Accounts and balances.
- Reservations and provider usage segments.
- Settlement, reconciliation, refunds, and debt.
- Transfers and onboarding grants.
- Telegram Stars payments.
- Ledger history, reporting, and maintenance.

Rollout order:

1. Rust performs shadow reads.
2. Compare balances and derived results.
3. Move read-only operations.
4. Switch all billing writes atomically to Rust.
5. Keep the last Python image as deployment rollback only.

Never execute the same financial write through both implementations.

**Exit gate:** transaction, locking, concurrency, retry, idempotency, interruption,
and reconciliation tests pass, and production has exactly one writer.

### Phase 6: AI orchestration and tools

Model AI execution as explicit state transitions:

```text
authorize -> reserve -> request -> stream/tools -> settle
                              \-> failure -> refund
```

Migrate:

- Provider clients and selection.
- Streaming event decoding.
- Partial and repeated tool calls.
- Tool registry and execution.
- Provider fallback and backoff.
- Usage accounting and reconciliation events.
- Image context and summary generation.
- Response cleanup and Telegram edit planning.

Test completed and interrupted streams, malformed events, cancellation, tool
loops, duplicate usage reports, provider fallback, settlement failure, and
restarts.

**Exit gate:** AI output, tools, fallback, and billing behavior match the Python
contracts under both normal and failure conditions.

### Phase 7: Scheduled tasks

APScheduler's Redis job store is Python-specific. The existing JSON task records
must first become the canonical language-neutral task schema.

Migration order:

1. Version the JSON task format.
2. Backfill and validate every existing task.
3. Make Python schedule exclusively from canonical records.
4. Implement Rust task claiming, leases, recurrence, and recovery.
5. Run the Rust scheduler in verification mode without executing tasks.
6. Switch scheduler ownership atomically to Rust.
7. Remove APScheduler records after rollback support ends.

Test time zones, clock changes, missed runs, coalescing, recurrence, duplicate
workers, cancellation, restarts, AI execution, and billing idempotency.

**Exit gate:** no existing task is lost or duplicated and Rust is the sole
scheduler owner.

### Phase 8: Rust Telegram runtime

- Move polling, Telegram requests, callbacks, payments, and file downloads to
  `botd`.
- Move price refresh, summary workers, reconciliation, and maintenance entrypoints.
- Verify all flows using a separate Telegram test bot.
- Perform an atomic production cutover: stop Python, then start Rust.
- Never run both pollers with the same token.
- Keep the previous Python image and deployment configuration for rollback.

**Exit gate:** the Rust binary is the only application process and all smoke,
end-to-end, restart, and rollback tests pass.

### Phase 9: Remove Python

After a defined Rust-only production observation period:

- Port every remaining meaningful Python test.
- Delete Python feature flags and fallback paths.
- Delete `python-bridge` and PyO3/Maturin integration.
- Delete `api/`, Python entrypoints, `pyproject.toml`, and `uv.lock`.
- Remove Python, uv, APScheduler, and all Python dependencies from CI and images.
- Verify that credits, history, configuration, media state, and tasks remain intact.
- Verify that the production image contains no Python interpreter.

**Exit gate:** one Rust binary supplies every bot feature and no Python code,
runtime, sidecar, bridge, or fallback remains.

## Per-change Definition of Done

A migration change is complete only when:

- It is a small, focused feature or boundary change.
- Observable behavior is unchanged.
- Existing tests and new Rust tests pass.
- Python/Rust contract fixtures agree.
- Relevant failure and concurrency paths are tested.
- Persistent data remains compatible.
- The implementation emits useful structured diagnostics.
- Rust and Python ownership cannot overlap for side effects.
- The Rust path has a tested rollback mechanism.
- Documentation and the feature matrix are updated.
- Python code is removed only after the Rust path completes its observation period.

## Initial Implementation Batch

The first batch should contain three independently reviewable changes:

1. **Architecture and behavior documentation**
   - Complete the feature and persistence inventory.
   - Record architecture decisions and migration invariants.

2. **Rust and CI scaffold**
   - Add the workspace and temporary bridge.
   - Add Rust checks, tests, coverage, and the container build stage.
   - Do not route production behavior through Rust yet.

3. **First contract-protected Rust slice**
   - Port credit-unit parsing, scaling, and formatting to `bot-core`.
   - Run Python and Rust against the same fixtures.
   - Add property and boundary tests.
   - Enable the Rust path behind a disabled-by-default feature flag.

No later phase begins by deleting its Python implementation. Deletion is always
the final step after parity, production observation, and rollback verification.
