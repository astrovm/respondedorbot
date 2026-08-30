# Billing Cutover

## Purpose

Billing moves only through checkpoints that keep one authoritative writer and
preserve every balance, ledger, payment, and idempotency contract. Integer
credit units remain hundredths of one displayed credit. Floating-point values
must never enter balance mutations.

## Ownership

| Checkpoint | Reads | Writes | Rollback |
| --- | --- | --- | --- |
| Balance shadow | Python is authoritative; Rust compares `get_balance` with a read-only `SELECT` | Python only | Disable `RUST_BILLING_READ_SHADOW_ENABLED` |
| Current balance I/O | Rust reads balances and creates missing zero-balance accounts | Rust only for the idempotent account insert; Python owns all other writes | Disable `RUST_BILLING_BALANCE_IO_ENABLED` |
| Current onboarding writer | Rust owns idempotent onboarding grants, overflow decisions, balance updates, and their ledger rows | Rust only for onboarding; Python owns all other mutations | Disable `RUST_BILLING_ONBOARDING_ENABLED` |
| Current Stars writer | Rust owns payment insertion, duplicate handling, user balance updates, and top-up ledger rows | Rust only for Stars payments; Python owns all other mutations | Disable `RUST_BILLING_STAR_PAYMENTS_ENABLED` |
| Current manual-credit writer | Rust owns administrator mint and user-to-chat transfer transactions | Rust only for these command mutations; Python owns all other mutations | Disable `RUST_BILLING_MANUAL_CREDITS_ENABLED` |
| Current chat-owned AI writer | Rust owns chat-only AI reserves, refunds, and debts | Rust only for these automation mutations; Python owns all other AI billing | Disable `RUST_BILLING_CHAT_AI_CREDITS_ENABLED` |
| Current general AI debt writer | Rust owns user-or-chat settlement debt | Rust only for this debt mutation; Python owns all other interaction billing | Disable `RUST_BILLING_AI_DEBT_ENABLED` |
| Current AI refund writer | Rust owns user-or-chat refunds and their replay/settlement guards | Rust only for this refund mutation; Python owns all other interaction billing | Disable `RUST_BILLING_AI_REFUNDS_ENABLED` |
| Current AI charge writer | Rust owns user-first payer selection, explicit payer charges, reserves, and replay guards | Rust only for these charge mutations; Python owns settlement and provider usage | Disable `RUST_BILLING_AI_CHARGES_ENABLED` |
| Current provider-usage writer | Rust owns idempotent provider segment insertion | Rust only for provider segment records; Python owns settlement | Disable `RUST_BILLING_PROVIDER_USAGE_ENABLED` |
| Current AI settlement writer | Rust owns exact-once operation settlement, refunds, and settlement debt | Rust only for current operation settlement; Python owns legacy usage-tag settlement | Disable `RUST_BILLING_AI_SETTLEMENTS_ENABLED` |
| Legacy AI settlement writer | Rust owns exact-once memory-compaction settlement by usage tag | Rust only for legacy usage-tag settlement; Python owns audit and reporting paths | Disable `RUST_BILLING_LEGACY_SETTLEMENTS_ENABLED` |
| AI settlement audit writer | Rust owns idempotent non-monetary audit insertion | Rust only for settlement result audit writes; Python owns reporting reads | Disable `RUST_BILLING_AUDIT_WRITES_ENABLED` |
| AI settlement audit reader | Rust reads recent settlement results newest first | Rust with safe Python read fallback; Python owns charge history | Disable `RUST_BILLING_AUDIT_READS_ENABLED` |
| AI reconciliation reader | Rust reads unsettled holds, activity, and provider segments | Rust with safe Python read fallback; Python owns reconciliation decisions | Disable `RUST_BILLING_RECONCILIATION_READS_ENABLED` |
| AI ledger maintenance | Rust deletes expired AI ledger event families in one transaction | Rust only for retention purge; Python owns schema migration | Disable `RUST_BILLING_MAINTENANCE_ENABLED` |
| User AI charge history | Rust selects finalized and pending charge groups with cursor pagination | Rust query with safe Python read fallback; Python groups and formats rows | Disable `RUST_BILLING_CHARGE_HISTORY_ENABLED` |
| Rust reads | Rust is authoritative for proven read operations | Python owns non-balance writes | Disable the per-operation Rust read flag |
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

`RUST_BILLING_BALANCE_IO_ENABLED=1` makes Rust authoritative for `get_balance`
and preserves its existing missing-account insert. A Rust failure uses the
idempotent Python fallback. All balance mutations, ledger writes, payments,
migrations, and reconciliation remain Python-owned.

When authoritative balance I/O is disabled,
`RUST_BILLING_READ_SHADOW_ENABLED=1` compares Python's result with Rust's
read-only query. A mismatch or Rust failure is logged, but Python's result is
returned.

`RUST_BILLING_ONBOARDING_ENABLED=1` makes Rust the only onboarding transaction
writer. It preserves the advisory lock, account row lock, hourly/daily limits,
three-attempt concurrency retry, grant idempotency, and ledger metadata. An
uncertain failure is returned to the caller and never retried through Python.

`RUST_BILLING_STAR_PAYMENTS_ENABLED=1` makes Rust the only Telegram Stars
payment writer. The Telegram charge ID remains the idempotency key. Duplicate
deliveries return the locked current balance without awarding credits or adding
another ledger row. Uncertain failures do not start the Python writer.

`RUST_BILLING_MANUAL_CREDITS_ENABLED=1` makes Rust authoritative for
administrator minting and user-to-chat transfers. Transfers lock the user
account before the chat account and commit the two balance updates with both
ledger rows. Insufficient transfers do not mutate state. These operations have
no replay key, so uncertain failures fail closed without invoking Python.

`RUST_BILLING_CHAT_AI_CREDITS_ENABLED=1` makes Rust authoritative for
chat-owned automation reserves, refunds, and debt. Each transaction locks only
the chat account and commits the balance mutation with its ledger row. An
insufficient reserve does not write a ledger row. These operations have no
replay key, so uncertain failures fail closed without invoking Python.

`RUST_BILLING_AI_DEBT_ENABLED=1` makes Rust authoritative for general
user-or-chat settlement debt. It locks the user before the optional chat,
allows the selected balance to become negative, and preserves the legacy user
fallback when chat is not a valid payer. This operation has no replay key, so
uncertain failures fail closed without invoking Python.

`RUST_BILLING_AI_REFUNDS_ENABLED=1` makes Rust authoritative for general AI
refunds. It preserves optional idempotency keys, refuses refunds after an
operation settlement, locks user before chat, and records the chosen payer in
ledger metadata. An uncertain failure never starts the Python writer.

`RUST_BILLING_AI_CHARGES_ENABLED=1` makes Rust authoritative for AI reserves
and charges. It locks user before chat, uses user-first funding unless a payer
is explicit, returns prior idempotent results, and rejects refunded or settled
reservations. An uncertain failure never starts the Python writer.

`RUST_BILLING_PROVIDER_USAGE_ENABLED=1` makes Rust authoritative for durable
provider-usage segment insertion, ordered reads, and reconciled segment
replacement. The `(operation_id, segment_id)` database index is the idempotency
boundary, so retry and concurrent duplicates add only one row. An uncertain
failure never starts the Python writer.

`RUST_BILLING_AI_SETTLEMENTS_ENABLED=1` makes Rust authoritative for current
operation-scoped settlement. It locks user before chat, rejects mixed payers,
applies refund or debt with the settlement result in one transaction, and
returns current balances on replay. Uncertain failures never start Python.

`RUST_BILLING_LEGACY_SETTLEMENTS_ENABLED=1` makes Rust authoritative for the
legacy usage-tag settlement used by memory compaction. It preserves the
user-before-chat lock order, legacy audit payload merge order, and fail-closed
retries.

`RUST_BILLING_AUDIT_WRITES_ENABLED=1` makes Rust authoritative for non-monetary
AI settlement result records. The partial unique settlement index remains the
restart-safe idempotency boundary, and uncertain failures never start Python.

`RUST_BILLING_AUDIT_READS_ENABLED=1` makes Rust authoritative for the recent
settlement audit list used by administrators. A Rust read failure may safely use
the Python query because neither path changes financial state.

`RUST_BILLING_RECONCILIATION_READS_ENABLED=1` makes Rust authoritative for the
unsettled-operation snapshot used after interruption or restart. It preserves
legacy-compaction exclusions, provider call order, activity timestamps, and a
safe Python fallback because the operation is read-only. Chat-only automation
rows without a user payer remain outside this user-scoped reconciler.

`RUST_BILLING_MAINTENANCE_ENABLED=1` makes Rust authoritative for AI ledger
retention. The event allowlist and minimum one-day retention remain compatible,
and an uncertain failure never starts the Python delete transaction.

`RUST_BILLING_CHARGE_HISTORY_ENABLED=1` makes Rust authoritative for selecting
the grouped charge-history rows and page cursors. Python temporarily keeps the
presentation grouping, and a Rust failure may safely use the read-only query.
