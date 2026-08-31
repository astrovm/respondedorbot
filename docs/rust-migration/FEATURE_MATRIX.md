# Final Feature Matrix

Status for every row is **Rust authoritative**. The compatibility baseline passed before the retired implementation was deleted.

| Feature family | Preserved behavior | Primary Rust ownership and evidence |
| --- | --- | --- |
| Telegram runtime | Polling, update decoding, offsets, retries, command publication, files, actions | `bot-adapters::telegram_*`, `botd::runtime`, `botd::application`; unit and fake-transport tests |
| Routing and commands | All commands, aliases, bot suffixes, localized validation, Unicode command conversion | `bot-core::command_*`, `botd::dispatcher`; parser properties and dispatcher matrices |
| AI chat | Private/group routing, trigger words, streaming, cleanup, tool rounds, provider fallback | `botd::conversation`, `native_ai`, `telegram_stream`; success, interruption, cancellation, and failure tests |
| AI tools | Calculator, markets, weather, web fetch, Hacker News, random, tasks, capabilities | `bot-core::native_tools`, `botd::tool_requests`; schema, bounds, missing-service, and adapter tests |
| Media | Image context, audio transcription, Groq/OpenRouter fallback, FFmpeg, media cache | `botd::media*`, `bot-adapters::media_provider`; real FFmpeg and provider-failure tests |
| Links | URL validation, SSRF protection, page previews, social replacement, video upload limits | `bot-core::links`, `bot-adapters::link_preview`, `web_fetch`; redirect and private-network tests |
| Markets | CoinMarketCap, Yahoo Finance, CriptoYa, BCRA, oil, Devo, rulo, token signals, Polymarket | Market modules in all three crates; cache, malformed payload, stale fallback, and rendering tests |
| Chat settings | PostgreSQL settings, locale changes, group-admin authorization, callbacks | `chat_config`, `config_command`, dispatcher callbacks; database round trips and authorization tests |
| Billing | Balances, onboarding, Stars, transfers, reservations, usage, settlement, refund, debt, history | `bot-adapters::billing_read`, billing state machines and dispatcher; transaction and replay integration tests |
| Memory | Redis history, reply metadata, RediSearch retrieval, summaries, compaction queue and leases | Redis adapters plus `conversation_adapters` and compaction workers; ordering, TTL, crash, and queue tests |
| Scheduled tasks | Create/list/cancel, canonical records, recurrence, leases, verification, delivery and billing | `task_*`, `scheduler`, Redis task store; competing-owner, retry, replay, and integration tests |
| Background work | Price refresh, compaction, reconciliation, scheduler, maintenance, graceful stop | `botd::background`, `price_refresh`, `reconciliation`; supervised lifecycle and failure tests |
| Administration | Help, instance, reporting, credit mint/log, authorization and diagnostics | `admin_*` and dispatcher; locale, bounds, authorization, and persistence tests |
| Deployment | Rust-only binary/container, Quadlet, maintenance timer, immutable rollback image | Container smoke inspection, release build, CI checks, systemd/Quadlet configuration |

## Command coverage

The authoritative command catalog includes AI and summaries; media commands; prices, crypto, dollar, oil, stocks, elections, Devo, rulo, Bitcoin models, BCRA, weather, random and base conversion; settings and language; top-up, balance, history and transfer; tasks; greetings, help, time, instance; and administrative credit/reporting commands. Alias publication and dispatch use the same typed catalog.

## Final invariants

- No route delegates to a second implementation.
- Missing native services and unsupported inputs are explicit outcomes.
- Existing Redis, PostgreSQL, and canonical task formats remain readable.
- Financial and scheduled side effects have one owner and replay protection.
- Tests use synthetic identities and isolate their reserved database ranges.
