# ADR 0003: Canonical Scheduled Tasks and Single-Owner Claims

## Status

Accepted and implemented.

## Context

APScheduler stores executable Python objects in Redis database 1. Rust cannot
safely decode that state. The existing `task:data:*` JSON record is readable by
both runtimes, but recurring records do not contain their next run. Reading the
APScheduler job store is therefore still required to list or reconstruct them.

The migration must preserve existing jobs, coalescing, the five-minute misfire
grace period, cancellation, AI billing, and the invariant that only one
scheduler executes tasks.

## Decision

`task:data:{task_id}` becomes the canonical, language-neutral record. Version 1
is additive and keeps every legacy field so the current Python image can read a
Rust-compatible record:

```json
{
  "schema_version": 1,
  "id": "abc12345",
  "chat_id": "-100123",
  "text": "synthetic task",
  "user_name": "synthetic-user",
  "user_id": 42,
  "interval_seconds": 3600,
  "run_date": null,
  "trigger_config": null,
  "timezone_offset": -3,
  "locale": "es",
  "schedule_anchor_at": "2026-08-30T12:00:00+00:00",
  "next_run_at": "2026-08-30T13:00:00+00:00",
  "last_execution_id": null
}
```

`run_date`, `interval_seconds`, and `trigger_config` remain the rollback trigger
representation. `next_run_at` is authoritative for all trigger kinds.
`schedule_anchor_at` preserves interval alignment across restarts.
`last_execution_id` records the most recently completed occurrence. Unknown
fields remain permitted during the compatibility period.

During cutover, version 1 records were backfilled from the former scheduler's
next-run state. The native scheduler now treats those records as the only source
of truth and reports malformed records rather than inventing recurrence.

Rust uses these additional Redis keys:

| Key | Purpose |
| --- | --- |
| `task:due` | Global sorted set of task IDs scored by `next_run_at` |
| `task:claim:{task_id}:{execution_id}` | Bounded lease acquired with `SET NX` |
| `task:scheduler:owner` | Renewable unique-token lease for the active owner |

An execution ID is deterministically derived from the task ID and scheduled UTC
instant. Claim completion is a Lua compare-and-update operation: it validates
the lease token and expected execution ID, records completion, advances or
deletes the canonical record, updates indexes, and releases the claim. A stale
worker cannot advance a newer occurrence.

The Rust scheduler preserves APScheduler policy:

- coalesce multiple due occurrences into one execution;
- allow an occurrence up to 300 seconds late;
- skip occurrences beyond that grace period and advance to the first future
  occurrence;
- allow at most one in-flight execution per task;
- keep recurring tasks after AI or credit failures, matching the current task
  executor;
- remove successful one-shot tasks.

The native engine has two explicit modes. Verification mode reads and evaluates
due tasks without leases, claims, execution, or writes. Authoritative mode must
renew `task:scheduler:owner`, claims every occurrence before execution or skip,
repairs stale due-index entries, releases claims for retryable execution, and
advances state only through the compare-and-update Lua operation.

The ownership switch was atomic at deployment level. Verification mode still
computes decisions without acquiring claims, executing tasks, or writing state.

## Consequences

Task listing and execution depend only on language-neutral records in Redis
database 0. Older fields remain readable for stored-data compatibility.
The claim protocol prevents concurrent workers from executing the same live
occurrence. As with the existing implementation, a process crash after an
external Telegram send but before durable completion cannot provide strict
exactly-once delivery; stable execution and billing identifiers make retries
detectable and financial mutations idempotent.

Redis database 1 is not used by the native runtime.
