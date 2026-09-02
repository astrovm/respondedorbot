# Architecture

## Runtime

Production runs one `botd` binary. It owns Telegram long polling, command publication, graceful shutdown, price refresh, memory compaction, billing reconciliation, and scheduled-task execution. There is one poller, one scheduler owner, and one authoritative billing writer.

```text
botd
  application composition, configuration, polling, workers, orchestration
    |
    v
bot-adapters
  Telegram, HTTP, Redis, PostgreSQL, providers, scheduling, media
    |
    v
bot-core
  typed domain rules, routing, billing, AI state machines, pure formatting
```

The dependency direction is `botd -> bot-adapters -> bot-core`.

## Boundary rules

- Decode untrusted JSON, HTTP responses, database rows, and Redis values at adapter boundaries.
- Use identifier, credit, command, event, action, and error types inside domain logic.
- Keep I/O, environment access, wall-clock access, and global randomness out of `bot-core`.
- Put external behavior behind narrow traits only where tests or composition need a boundary.
- Return classified errors with safe diagnostics. Do not silently continue to another runtime.
- Avoid global mutable state, service locators, `unwrap`, and `expect` in request paths.
- Keep feature modules focused and favor direct code over speculative abstractions.

## State ownership

Redis stores caches, conversation history and metadata, compaction jobs, callback state, and canonical task records. PostgreSQL stores chat configuration, credit accounts, payments, provider usage, and the billing ledger. Stored formats remain backward compatible.

Scheduled tasks use canonical JSON records plus atomic Redis ownership and occurrence leases. Billing mutations use PostgreSQL transactions, stable operation identifiers, idempotent provider segments, and exact-once settlement guards.

## Process lifecycle

Startup validates all required configuration before contacting Telegram. The runtime constructs adapters, starts supervised background workers, publishes the command menu, recovers durable updates, and then polls. Each new update is stored in Redis before its Telegram offset advances. Eight workers process updates independently while polling continues; failures are retried from the durable queue and terminal failures are quarantined. Polling transport and API failures retain the offset and use bounded retry delays. Shutdown stops polling and drains the accepted update backlog before joining background workers.

Maintenance and task verification use the same binary:

```bash
botd --maintenance
botd --verify-tasks
botd --check-config
```

## Deployment safety

The container includes only the release binary, native shared libraries, CA certificates, and FFmpeg. Persistent Redis and PostgreSQL data are external to the image. Every CI release also has an immutable SHA tag, which is the rollback unit.
