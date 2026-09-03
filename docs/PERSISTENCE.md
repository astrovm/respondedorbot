# Persistence and Runtime Contracts

## Scope

These formats are compatibility boundaries. The application reads existing
records and writes the documented stable formats.

## PostgreSQL

### `credit_accounts`

Primary key: `(scope_type, scope_id)`.

| Column | Contract |
| --- | --- |
| `scope_type` | `user` or `chat` |
| `scope_id` | Telegram identifier stored as `BIGINT` |
| `balance` | Signed integer credit units; 100 units equal one displayed credit |
| `updated_at` | Updated when the balance changes |

Balance-changing operations lock the relevant rows with `FOR UPDATE`. Rust must
preserve lock ordering when user and chat accounts participate in one operation.

### `onboarding_grants`

`user_id` is the primary key, making onboarding grants idempotent. `credits` is
stored in hundredths-of-credit units. Existing hourly/daily abuse limits and
denied-overflow ledger behavior must remain intact.

### `star_payments`

`telegram_payment_charge_id` is the primary key and is the payment idempotency
boundary. Records also contain user, pack, XTR amount, awarded credit units,
payload, and creation time. A duplicate Telegram payment must not award credits
twice.

### `credit_ledger`

The append-oriented ledger stores event type, actor/user/chat identifiers,
signed integer amount, JSON metadata, and creation time.

Known event types that must remain readable include:

- `onboarding_grant`
- `onboarding_denied_overflow`
- `ai_charge`
- `ai_reserve`
- `ai_refund`
- `ai_settlement_charge`
- `ai_settlement_debt`
- `ai_settlement_result`
- `ai_provider_usage`
- `ai_reconciliation_correction`
- `memory_compaction_settlement`
- `transfer_user_to_chat`
- `admin_command`

Important metadata identifiers include `operation_id`, `settlement_id`,
`segment_id`, `usage_tag`, provider generation/request identifiers,
`credit_scale`, reserved units, actual units, and tool/provider usage details.

Current uniqueness contracts include:

- One AI settlement result per `(user_id, settlement_id)` when the identifier is
  present.
- One provider usage row per `(operation_id, segment_id)`.
- Idempotent reservation/refund lookup through event type and metadata keys.
- Special compaction settlement/refund repair behavior.

Rust must preserve transaction boundaries, advisory locks used by schema
migrations, retry classification, and integer scaling. It must not use floating
point for balances or mutations.

### `credit_schema_migrations`

Migration `name` is the primary key. Existing scale migrations and repair
migrations must remain recognized. New migrations must be transactional,
idempotent, safe under concurrent startup, and reversible at the application
compatibility level.

### `chat_configs`

Primary key: textual `chat_id`. `config` is JSONB.

Recognized settings and defaults:

| Key | Default | Validation contract |
| --- | --- | --- |
| `language` | `auto` | `auto`, `es`, or `en` |
| `link_mode` | `reply` | Existing supported link modes |
| `ai_command_followups` | `true` | Boolean |
| `ignore_link_fix_followups` | `true` | Boolean |
| `timezone_offset` | `-3` | Integer from -12 through 14 |
| `ai_random_replies` | `true` | Boolean; group setting |
| `creditless_user_hourly_limit` | `5` | Existing bounded integer choices |

Unknown stored fields must not cause existing recognized settings to be lost
during a read-modify-write operation.

## Redis database 0

Unless explicitly noted, application Redis state is in the configured default
database and values are decoded as UTF-8 strings.

### Conversation state

| Key pattern | Type and value | Lifetime |
| --- | --- | --- |
| `chat_history:{chat_id}` | List of JSON message records, newest first | 30 days |
| `chat_message_order:{chat_id}` | Sorted set of message IDs by per-chat sequence | 30 days |
| `chat_message_sequence:{chat_id}` | Integer sequence counter | 30 days |
| `chat_message_ids:{chat_id}` | Legacy deduplication set retained for stored-data compatibility | maintenance repairs TTL |
| `chatmsg:{chat_id}:{message_id}` | RediSearch hash with chat, role, user, reply, mention, text, and timestamp fields | 30 days |
| `chat_summary:{chat_id}` | Summary text | 30 days |
| `chat_user_summary:{chat_id}` | User-specific summary text | 30 days |
| `chat_compacted_until:{chat_id}` | Compaction marker | 30 days |
| `chat_user_compacted_until:{chat_id}` | User compaction marker | 30 days |
| `bot_message_meta:{chat_id}:{message_id}` | Version 1 JSON metadata for bot replies | 3 days |
| `chat_members:{chat_id}` | Hash of user ID to JSON member data | 30 days |

The RediSearch index name is `idx:chat_messages`, with prefix `chatmsg:`. Its
schema includes TAG fields for chat/role/user/reply/mention, TEXT fields for
username/text, and a sortable numeric timestamp. Redis Stack and `FT.CREATE` /
`FT.SEARCH` support remain deployment requirements until intentionally changed.

Message save is atomic through a Lua script. It deduplicates, assigns a
sequence, trims history and the sorted index, writes the search hash, and sets
TTLs. The application must preserve the atomic outcome.

### Memory compaction

| Key | Type and value |
| --- | --- |
| `memory:compaction:jobs` | Hash from chat ID to JSON `CompactionJob` |
| `memory:compaction:dead_jobs` | Hash of failed job records |
| `memory:compaction:lock:{chat_id}` | Lease token with one-hour TTL |

`CompactionJob` includes messages, prior summary, expected and target markers,
billing reservation, user/message identifiers, locale, attempts, retry time,
and optional result/usage information. A job is coupled to a billing reservation;
enqueue failure, final failure, and successful completion must settle correctly.

### Telegram update queue

| Key | Type and value |
| --- | --- |
| `telegram:updates:pending` | Hash from update ID to a versioned JSON update record |
| `telegram:updates:dead` | Hash of updates quarantined after repeated failures |

The polling runtime writes each decoded update to the pending hash before it
advances the Telegram offset. Parallel workers mark successful updates as
completed, persist retry counts, and atomically move terminal failures to the
dead hash. A completed record is deleted only after a successful Telegram poll
confirms its offset. Pending and completed records are recovered before polling
starts after a process restart. Redis must use `noeviction` or a `volatile-*`
maxmemory policy so the non-expiring pending hash cannot be evicted after
Telegram acknowledges it. The production Redis service enables AOF with
`appendfsync always`; an accepted update must reach durable storage before the
bot advances its polling offset.

### Scheduled tasks

| Key pattern | Type and value | Lifetime |
| --- | --- | --- |
| `task:data:{task_id}` | JSON task record | 10 years |
| `task:chat:{chat_id}` | Sorted set of task IDs scored by run date | 10 years |
| `task:chat:{chat_id}:indexed` | Index-complete marker | 10 years |

Canonical task fields currently are:

- `id`
- `chat_id`
- `text`
- `user_name`
- `user_id`
- `interval_seconds`
- `run_date` as an ISO timestamp or null
- `trigger_config`
- `timezone_offset`
- `locale`

Trigger variants are delay seconds, interval seconds, interval days, and cron
with hour/minute plus optional weekday list or day of month.

The authoritative version 1 schema adds next-run state, schedule anchors, and
execution idempotency. The claim protocol is defined in
[ADR 0001](decisions/0001-canonical-scheduled-tasks.md). Older trigger fields
remain readable as stored-data compatibility fields.

### Caches and callback state

Known compatibility key families include:

- `request_cache:{sha256}`
- `request_cache_history:{hour}:{sha256}`
- `giphy_pool:{category}`
- `giphy_pool_stale:{category}`
- `market:dolar:formatted:{hours_ago}`
- `market:stock_screener:mega_cap`
- `context:hacker_news:best`
- `bcra_mayorista:{date}`
- `token_signal:{signal_id}` and `token_signal:*` enrichment caches
- `chat_admin:{chat_id}:{user_id}`
- `creditless_cap:{chat_id}:{user_id}`

Generic JSON cache records and stale-cache records preserve their established
timestamp/value shapes and TTL semantics.

## Redis database 1

The native application does not use Redis database 1. Historical scheduler
objects in that database are not executable state and may be removed after the
deployment owner confirms that no rollback image needs them. Canonical tasks in
database 0 are the only scheduling source of truth.

## Legacy-data migration

`botd --migrate-legacy` inspects database 0 and PostgreSQL and prints a JSON
report without changing data. Add `--apply` to perform the migration after a
backup. Repeated runs are safe.

The migration:

- adds schema version 1 to unversioned conversation history, chat members, and
  bot-message metadata;
- rewrites scheduled tasks to the canonical version 1 record, calculates their
  next future occurrence, and rebuilds the per-chat and global due indexes;
- removes the obsolete `world_cup_goal_alerts` chat configuration field.

The task migration does not execute occurrences missed before migration. Redis
writes compare the value read during inspection before replacing it, so a
concurrent change stops the command instead of overwriting newer data.
Malformed or unsupported conversation records are counted in the report and
left unchanged. An invalid scheduled task stops the migration because silently
skipping executable state could hide a user-visible task.

## Files and environment

The application reads `workspace/SOUL.md` and `workspace/RULES.md`, or uses
`BOT_SYSTEM_PROMPT`. The deployed workspace mount is read-only and must remain so.

Configuration includes:

- `TELEGRAM_TOKEN`, `TELEGRAM_USERNAME`
- `BOT_SYSTEM_PROMPT`, `BOT_TRIGGER_WORDS`, `BOT_INSTANCE_NAME`
- `TELEGRAM_LONG_POLL_SECONDS`
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
- `REDIS_MAXMEMORY`, `REDIS_MAXMEMORY_POLICY`
- `SUPABASE_POSTGRES_URL`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`, `GROQ_FREE_API_KEY`
- `FIRECRAWL_API_KEY`
- `COINMARKETCAP_KEY`
- `GIPHY_API_KEY`
- `ADMIN_CHAT_ID`, `FRIENDLY_INSTANCE_NAME`
- `AI_RECONCILIATION_INTERVAL_SECONDS`, `AI_RECONCILIATION_RETRY_SECONDS`
- `AI_RECONCILIATION_SAFETY_CREDIT_UNITS`, `AI_RECONCILIATION_STALE_SECONDS`
- `AI_LEDGER_RETENTION_DAYS`

Configuration must distinguish required, optional, and feature-gating
variables and must redact secret values from errors and logs.

## External ownership and rollback

The native process is the only Telegram poller, scheduler, and billing writer.
Rollback stops the current container before starting one immutable, previously
verified image. Redis and PostgreSQL stay external and are never duplicated or
rewound as part of an application rollback.
