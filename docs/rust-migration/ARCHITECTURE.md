# Rust Architecture

## Purpose

This document defines the architecture that the migration must converge on. It
is a compatibility migration: internal structure may change, but observable bot
behavior and stored data must remain compatible until a separately approved
behavior change is made.

## Current runtime boundaries

The Python application currently has these major runtime owners:

- `run_polling.py` owns startup, the price refresh thread, summary worker,
  billing reconciler, APScheduler, Telegram command publication, and polling.
- `api.bot.ptb` owns the `python-telegram-bot` poller and dispatches synchronous
  handlers through a thread pool.
- `api.index` is the composition root and compatibility export surface.
- `ApplicationRuntime` groups Telegram, admin, cache, markets, providers, AI,
  memory, billing, media, and top-level handlers.
- Redis stores caches, conversation state, compaction jobs, callback state, and
  scheduled-task records.
- PostgreSQL stores chat configuration, credit accounts, payments, and the
  billing ledger.
- APScheduler stores executable job state in Redis database 1.

The service boundaries in `ApplicationRuntime` are useful migration seams, but
the final application must not preserve Python's `Any`-heavy callback wiring.

## Final workspace

```text
Cargo.toml
crates/
  bot-core/
  bot-adapters/
  botd/
```

During the hybrid period only:

```text
crates/
  python-bridge/
```

### `bot-core`

`bot-core` contains deterministic application and domain behavior:

- Strong identifier and money types.
- Telegram input normalization after raw decoding.
- Command parsing and routing decisions.
- Chat settings and localization selection.
- Market query parsing, normalization, and formatting.
- Billing authorization and settlement state machines.
- AI request, streaming, tool, and fallback state machines.
- Typed AI prompt, assistant tool-call, and tool-result messages; provider JSON
  conversion stays at the runtime/adapter boundary.
- Scheduled-task trigger and recurrence rules.
- Memory compaction planning.
- Typed actions and typed subsystem errors.

It must not depend on Telegram clients, HTTP clients, Redis, PostgreSQL,
filesystems, environment variables, wall-clock globals, randomness globals,
threads, or Python.

### `bot-adapters`

`bot-adapters` implements external boundaries:

- Telegram polling, requests, downloads, callbacks, and payments.
- OpenRouter, Groq, Firecrawl, market, weather, BCRA, Giphy, and link HTTP APIs.
- Redis caches, conversation state, RediSearch, queues, leases, and task records.
- PostgreSQL chat configuration, billing transactions, and migrations.
- Media inspection, image handling, and FFmpeg process execution.
- Clock, randomness, workspace files, and environment configuration.

Adapters deserialize untrusted data and convert it to `bot-core` types. Untyped
JSON must not escape an adapter without validation.

### `botd`

`botd` is the only final executable. It owns:

- Configuration validation.
- Dependency construction.
- Telegram polling.
- Background worker lifecycle.
- Scheduler lifecycle.
- Graceful shutdown and cancellation.
- Health and readiness state.
- Structured logging and process exit codes.

Business rules do not belong in `botd`.

## Dependency rule

The compile-time dependency direction is:

```text
botd -> bot-adapters -> bot-core
```

`bot-core` defines ports when application logic needs an external capability.
Adapters implement those ports. Traits are introduced only for real external
boundaries or for deterministic test control.

## Typed request flow

```text
raw Telegram update
  -> adapter validation
  -> IncomingUpdate
  -> routing/use case
  -> BotAction or ActionBatch
  -> adapter execution
  -> typed result/event
  -> state transition and persistence
```

The core action model will be developed incrementally. The expected shape is:

```rust
enum BotAction {
    Ignore,
    SendMessage(SendMessage),
    EditMessage(EditMessage),
    AnswerCallback(AnswerCallback),
    RunAi(AiRequest),
    ExecuteCommand(CommandRequest),
    ProcessMedia(MediaRequest),
    ApplyBilling(BillingOperation),
    ScheduleTask(ScheduledTaskRequest),
}
```

Actions must carry typed identifiers and validated payloads. Invalid states
should be unrepresentable where practical.

## Type rules

- Define newtypes for `ChatId`, `UserId`, `MessageId`, `TaskId`, `OperationId`,
  `SettlementId`, `ProviderGenerationId`, and credit units.
- Credit units are integer hundredths. Floating-point values must not represent
  credit balances or ledger mutations.
- Use enums for chat type, locale, command, callback action, task trigger,
  provider, AI event, billing state, and cache status.
- Preserve unknown Telegram fields at the raw adapter boundary only when needed
  for forward compatibility.
- Accept unknown provider fields while rejecting malformed required fields.
- Validate string lengths, URL schemes, identifiers, and numeric ranges at the
  boundary.

## Error rules

- Each subsystem has a concrete error enum with source context.
- Retry classification is explicit; it is not inferred from formatted strings
  outside the adapter that received the error.
- User-facing localized failures are selected separately from diagnostic errors.
- Errors may be deliberately downgraded only at a documented product boundary.
- Production request paths must not use `unwrap`, `expect`, or discarded
  `Result` values.
- Logs must not contain Telegram tokens, API keys, database URLs, private media,
  or full user prompts unless explicitly safe.

## Concurrency and ownership

- There is one Telegram poller per token.
- There is one authoritative scheduler owner.
- There is one authoritative billing writer.
- Background work uses cancellation-aware tasks and bounded concurrency.
- Redis leases use unique owner tokens and compare-and-delete release semantics.
- Billing transactions use PostgreSQL transactions and preserve current lock and
  idempotency behavior.
- Duplicate Telegram callbacks, AI usage events, payments, task claims, and
  reconciliation attempts must be safe.

## Temporary Python bridge

The bridge exists only to let the Python poller call typed Rust functionality
while features move incrementally.

- The public bridge surface is small and feature-oriented.
- Bridge DTOs are validated immediately into `bot-core` types.
- Python exceptions are translations of typed Rust errors.
- Pure functions can run in shadow mode.
- Side-effecting paths never execute in both languages.
- Every bridge entry has a removal condition in the feature matrix.
- New product behavior must not be implemented only in the bridge.

## Module quality rules

- Prefer cohesive feature modules over large orchestration files.
- Avoid generic `utils`, `helpers`, or `common` dumping grounds.
- Keep public APIs smaller than their implementation modules.
- Prefer composition over macros for ordinary application behavior.
- Add an abstraction after a real boundary or repeated stable pattern is known,
  not in anticipation of possible reuse.
- Comments explain invariants and reasons, not line-by-line mechanics.
- Tests assert behavior and state transitions, not source layout.

## Migration decision record

Decisions that change persistence, external ownership, or crate boundaries must
be recorded before implementation. At minimum, later work must record:

1. Python bridge packaging and Python 3.14 ABI strategy.
2. Telegram framework versus direct Bot API adapter.
3. Canonical scheduled-task schema and claim protocol.
4. Redis payload versioning strategy.
5. Billing repository transaction and idempotency mapping.
6. AI streaming event model.
7. Final container base and system dependency strategy.

Recorded decisions:

- [ADR 0003: Canonical Scheduled Tasks and Single-Owner Claims](decisions/0003-canonical-scheduled-tasks.md)
