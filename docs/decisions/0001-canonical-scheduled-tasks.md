# ADR 0001: Canonical Scheduled Tasks and Single-Owner Claims

## Status

Accepted and implemented.

## Context

Scheduled tasks must be reconstructable from application data rather than
runtime-specific executable objects. Recurring records therefore need durable
next-run state in addition to their trigger definition. The scheduler must also
preserve coalescing, the five-minute misfire grace period, cancellation, AI
billing, and single-owner execution.

## Decision

`task:data:{task_id}` is the canonical record. Version 1 retains the established
trigger fields for stored-data compatibility:

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

`run_date`, `interval_seconds`, and `trigger_config` remain the stored trigger
representation. `next_run_at` is authoritative for all trigger kinds.
`schedule_anchor_at` preserves interval alignment across restarts.
`last_execution_id` records the most recently completed occurrence. Unknown
fields remain permitted for forward compatibility. The scheduler treats these
records as the only source of truth and reports malformed records rather than
inventing recurrence.

The scheduler uses these additional Redis keys:

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

The scheduler follows these execution rules:

- coalesce multiple due occurrences into one execution;
- allow an occurrence up to 300 seconds late;
- skip occurrences beyond that grace period and advance to the first future
  occurrence;
- allow at most one in-flight execution per task;
- keep recurring tasks after AI or credit failures, matching the current task
  executor;
- remove successful one-shot tasks.

The engine has two explicit modes. Verification mode reads and evaluates
due tasks without leases, claims, execution, or writes. Authoritative mode must
renew `task:scheduler:owner`, claims every occurrence before execution or skip,
repairs stale due-index entries, releases claims for retryable execution, and
advances state only through the compare-and-update Lua operation.

Verification mode computes decisions without acquiring claims, executing tasks,
or writing state.

## Consequences

Task listing and execution depend only on canonical records in Redis database 0.
Older fields remain readable for stored-data compatibility.
The claim protocol prevents concurrent workers from executing the same live
occurrence. As with the existing implementation, a process crash after an
external Telegram send but before durable completion cannot provide strict
exactly-once delivery; stable execution and billing identifiers make retries
detectable and financial mutations idempotent.

Redis database 1 is not used by the native runtime.
