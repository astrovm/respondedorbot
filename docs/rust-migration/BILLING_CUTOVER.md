# Billing Cutover

## Purpose

Billing moves only through checkpoints that keep one authoritative writer and
preserve every balance, ledger, payment, and idempotency contract. Integer
credit units remain hundredths of one displayed credit. Floating-point values
must never enter balance mutations.

## Ownership

| Checkpoint | Reads | Writes | Rollback |
| --- | --- | --- | --- |
| Current shadow | Python is authoritative; Rust compares `get_balance` with a read-only `SELECT` | Python only | Disable `RUST_BILLING_READ_SHADOW_ENABLED` |
| Rust reads | Rust is authoritative for proven read operations | Python only | Disable the per-operation Rust read flag |
| Shadow transaction decisions | Python commits; Rust evaluates the same synthetic transaction inputs without I/O | Python only | Disable the decision shadow flag |
| Rust writer canary | Rust owns one proven mutation family | Rust for that family; Python for all others | Disable that family's writer flag before another owner starts |
| Rust billing | Rust owns all reads, mutations, migrations, and reconciliation | Rust only | Deploy the last backward-compatible Python image |

No checkpoint may run Python and Rust writers for the same operation family.

## Schema contracts

Rust must preserve the tables and indexes documented in
`PERSISTENCE_CONTRACTS.md`, including:

- `(scope_type, scope_id)` account identity, with only `user` and `chat` scopes.
- The onboarding grant primary key and its hourly and daily abuse limits.
- Telegram payment-charge idempotency.
- Settlement, provider-segment, reservation, refund, and compaction-repair
  identifiers stored in ledger metadata.
- Transactional, idempotent schema migrations protected by their existing
  advisory-lock keys.

The application must accept existing rows before it writes any new format.

## Transaction invariants

- Lock a user account before its related chat account whenever both participate.
- Use `SELECT ... FOR UPDATE` for balance-changing account reads.
- Commit the balance update and its ledger evidence in one transaction.
- Retry only serialization failures and deadlocks, at most three attempts.
- Return an already-recorded result for an idempotent replay.
- Never create two settlement results for one `(user_id, settlement_id)`.
- Never create two provider-usage rows for one `(operation_id, segment_id)`.
- Never grant credits twice for one Telegram payment charge.
- Preserve reservation, extension, settlement, refund, debt, and reconciliation
  outcomes during failures and restarts.

## Required verification per mutation family

Before Rust becomes authoritative for a mutation family, tests must cover:

1. Existing and missing accounts.
2. Exact integer balance and ledger changes.
3. User-first and explicit-chat payer selection where applicable.
4. Duplicate requests and concurrent duplicate requests.
5. Deadlock or serialization retry and final failure.
6. Connection loss before commit and after an uncertain commit.
7. Rust-written data read by the rollback Python implementation.
8. A real temporary PostgreSQL instance using synthetic identifiers.

The canary must expose mismatch and failure telemetry without logging database
credentials, provider secrets, prompts, or payment payloads.

## Current checkpoint

`RUST_BILLING_READ_SHADOW_ENABLED=1` compares Python's authoritative
`get_balance` result with Rust's read-only query. A mismatch or Rust failure is
logged, but the Python result is returned. Python still creates missing accounts
and remains the only billing writer.
