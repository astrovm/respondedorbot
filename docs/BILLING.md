# Billing Ownership

## Ownership

Rust is the sole billing reader and writer. PostgreSQL remains the system of record for accounts, Stars payments, onboarding grants, ledger entries, AI operations, provider usage segments, settlements, refunds, debts, transfers, and audit history.

## Transaction invariants

- Credit values use integer credit units and checked conversions at storage boundaries.
- Account rows are locked in deterministic order for multi-account mutations.
- A reservation records its payer and operation identity before provider work.
- Provider usage segments use stable identifiers and idempotent insertion.
- Settlement is exact-once. It charges actual incurred usage, refunds unused reservation, and records debt when required.
- Failed delivery still settles incurred provider usage; a call with no usage refunds its hold.
- Successful uncached YouTube transcripts cost 0.60 displayed credits for either Supadata or Apify; cached transcripts and unsuccessful retrievals cost nothing.
- Telegram Stars payment payloads and provider charge identifiers are replay protected.
- Onboarding grants, administrator credits, transfers, and maintenance writes are transactional.
- `/transfer` and `/printcredits` carry an operation id from their Telegram message, so a retried update applies them once. A `/transfer` that replies to a person moves personal credits to that person instead of the group.
- Reconciliation reads durable operation and segment state and cannot duplicate completed settlement.

## Adding personal credits

- `/topup` opens the full amount and payment-method menu. `/topup 50` starts at 50 credits.
- Group and supergroup menus, their controls, Lightning invoices, and payment notices are ephemeral and visible only to the requester. Private chats retain the same controls and typed amount replies.
- Telegram Stars checkout uses `createInvoiceLink` and a native checkout URL button. It never publishes an invoice to the group or redirects the buyer to a private chat.
- New Stars payloads bind the amount and intended buyer. Checkout and settlement reject another payer or a changed amount. Exact legacy pack-only payloads credit the actual payer; malformed user bindings are rejected.
- Exported-invoice bot-origin receipts identify the buyer through the positive private chat id, never through the bot sender id or the payload alone. Charge-id ledger writes remain idempotent.
- Plain `/start` includes **Cargar créditos** and **Ayuda**, or **Add credits** and **Help**, alongside the existing help controls.

## Scheduled tasks

- A task run is paid like an AI message from the member who created it. Private chats use personal credits only.
- In a group, the chat setting "Tareas con créditos del grupo" (`group_pays_tasks`, default on) decides whether the group's credits can pay. When on, the run follows `group_pays_first`, counts toward the member's hourly allowance (`creditless_user_hourly_limit`, or their `/limit` override), and goes member then group, or group then member.
- The setting is read each time a task runs, so turning it off also changes existing tasks. A run already reserved still settles against the payer that held it.
- `task_set` checks credits when the task is created with the same rules: personal plus group credits when on, personal only when off. The message says which case applies.

## Compatibility

Schema creation and upgrades are additive and idempotent. Existing account scopes, ledger metadata, timestamps, usage tags, and payment identifiers remain readable. Rollbacks use an immutable previously verified container image and never permit two billing writers at once.

## Verification

Rust unit, property, failure-path, and PostgreSQL integration tests cover account creation, payer selection, insufficient credit, concurrent mutations, duplicate payments and segments, reserve/settle/refund/debt paths, interrupted operations, reconciliation, history, and retention. Synthetic test identities use reserved ranges and are removed before each integration run.
