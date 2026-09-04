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
- Reconciliation reads durable operation and segment state and cannot duplicate completed settlement.

## Compatibility

Schema creation and upgrades are additive and idempotent. Existing account scopes, ledger metadata, timestamps, usage tags, and payment identifiers remain readable. Rollbacks use an immutable previously verified container image and never permit two billing writers at once.

## Verification

Rust unit, property, failure-path, and PostgreSQL integration tests cover account creation, payer selection, insufficient credit, concurrent mutations, duplicate payments and segments, reserve/settle/refund/debt paths, interrupted operations, reconciliation, history, and retention. Synthetic test identities use reserved ranges and are removed before each integration run.
