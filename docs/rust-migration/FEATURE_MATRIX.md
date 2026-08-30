# Feature Compatibility Matrix

## Use

This matrix is the migration checklist for observable behavior. A feature may
become Rust-authoritative only after its contracts pass in both implementations.
Its Python implementation may be removed only after the Rust path completes the
agreed production observation period and rollback has been verified.

Status values used during implementation:

- `python`: Python is authoritative.
- `shadow`: Rust runs only for side-effect-free comparison.
- `rust-flagged`: Rust is authoritative behind a rollback flag.
- `rust`: Rust is authoritative and the Python fallback has completed burn-in.
- `removed`: the Python implementation and bridge entry are gone.

Rows not listed under migrated components remain Python-authoritative.

## Migrated components

| Component | Status | Rollback control | Shared contract |
| --- | --- | --- | --- |
| Credit-unit parsing, scaling, and formatting | `rust-flagged` | `RUST_CREDIT_UNITS_ENABLED=0` or missing bridge | `contracts/credit_units.json` |
| Command parsing | `rust-flagged` | `RUST_COMMAND_PARSING_ENABLED=0` or missing bridge | `contracts/command_parsing.json` |
| Command text normalization | `rust-flagged` | `RUST_COMMAND_NORMALIZATION_ENABLED=0` or missing bridge | `contracts/command_normalization.json` |
| Scheduled-task trigger parsing | `rust-flagged` | `RUST_TASK_TRIGGERS_ENABLED=0` or missing bridge | `contracts/task_triggers.json` |
| Unified price-query parsing | `rust-flagged` | `RUST_PRICE_QUERY_PARSING_ENABLED=0` or missing bridge | `contracts/price_queries.json` |
| AI market-context normalization and formatting | `rust-flagged` | `RUST_MARKET_CONTEXT_ENABLED=0` or missing bridge | `contracts/market_context.json` |
| Bitcoin power-law and rainbow model calculations | `rust-flagged` | `RUST_MARKET_MODELS_ENABLED=0` or missing bridge | `contracts/market_models.json` |
| Satoshi quote calculation and formatting | `rust-flagged` | `RUST_SATOSHI_ENABLED=0` or missing bridge | `contracts/satoshi.json` |
| Devo arbitrage parsing and calculation | `rust-flagged` | `RUST_DEVO_ENABLED=0` or missing bridge | `contracts/devo.json` |
| Rulo route selection and calculation | `rust-flagged` | `RUST_RULO_ENABLED=0` or missing bridge | `contracts/rulo.json` |
| Weather location and forecast-row selection | `rust-flagged` | `RUST_WEATHER_SELECTION_ENABLED=0` or missing bridge | `contracts/weather_selection.json` |
| Polymarket live-price reconciliation and ranking | `rust-flagged` | `RUST_POLYMARKET_RANKING_ENABLED=0` or missing bridge | `contracts/polymarket_ranking.json` |
| Hacker News item normalization and formatting | `rust-flagged` | `RUST_HACKER_NEWS_ENABLED=0` or missing bridge | `contracts/hacker_news.json` |
| Chat-configuration callback transitions | `rust-flagged` | `RUST_CONFIG_CALLBACKS_ENABLED=0` or missing bridge | `contracts/config_callbacks.json` |
| Telegram link entity parsing and URL selection | `rust-flagged` | `RUST_LINK_PARSING_ENABLED=0` or missing bridge | `contracts/link_parsing.json` |
| Admin credit-log limit parsing and truncation | `rust-flagged` | `RUST_ADMIN_REPORTS_ENABLED=0` or missing bridge | `contracts/admin_reports.json` |
| Redis cache keys, TTLs, and stale-while-refresh decision | `rust-flagged` | `RUST_CACHE_POLICY_ENABLED=0` or missing bridge | `contracts/cache_policy.json` |
| Redis conversation-message keys, versioned writes/member records, and search ranking | `rust-flagged` | `RUST_MESSAGE_STATE_ENABLED=0` or missing bridge | `contracts/message_state.json` |
| Durable memory-compaction due, recovery, obsolescence, and retry transitions | `rust-flagged` | `RUST_COMPACTION_POLICY_ENABLED=0` or missing bridge | `contracts/compaction_policy.json` |
| Versioned durable memory-compaction Redis job payload | `rust-flagged` | `RUST_COMPACTION_JOBS_ENABLED=0` or missing bridge | `contracts/compaction_jobs.json` |
| Durable memory-compaction Redis queue, atomic leases, and quarantine | `rust-flagged` | `RUST_COMPACTION_QUEUE_ENABLED=0` or missing bridge | `contracts/compaction_queue.json` |
| Redis summaries, compaction markers, bot metadata, and chat-member I/O | `rust-flagged` | `RUST_MESSAGE_AUX_IO_ENABLED=0` or missing bridge | `contracts/message_aux_io.json` |
| Redis conversation Lua writes, ordered history reads, index lifecycle, and RediSearch queries | `rust-flagged` | `RUST_MESSAGE_HISTORY_IO_ENABLED=0` or missing bridge | `contracts/message_history_io.json` |
| Redis media transcription/description cache I/O | `rust-flagged` | `RUST_MEDIA_CACHE_ENABLED=0` or missing bridge | `contracts/media_cache.json` |
| Redis chat-administrator authorization cache I/O | `rust-flagged` | `RUST_CHAT_ADMIN_CACHE_ENABLED=0` or missing bridge | `contracts/chat_admin_cache.json` |
| Redis external-request cache and stale-history I/O | `rust-flagged` | `RUST_REQUEST_CACHE_IO_ENABLED=0` or missing bridge | `contracts/request_cache_io.json` |
| Automatic media-routing decision | `rust-flagged` | `RUST_MEDIA_ROUTING_ENABLED=0` or missing bridge | `contracts/media_routing.json` |
| General response-routing state machine | `rust-flagged` | `RUST_RESPONSE_ROUTING_ENABLED=0` or missing bridge | `contracts/response_routing.json` |
| Base-conversion command | `rust-flagged` | `RUST_BASE_CONVERSION_ENABLED=0` or missing bridge | `contracts/base_conversion.json` |
| Random-selection command parsing | `rust-flagged` | `RUST_RANDOM_SELECTION_ENABLED=0` or missing bridge | `contracts/random_selection.json` |
| Spontaneous random-reply outcome | `rust-flagged` | `RUST_RANDOM_REPLY_ENABLED=0` or missing bridge | `contracts/random_reply.json` |

## Commands and user-visible features

| Feature | Entrypoints | Critical observable contract | Current test evidence |
| --- | --- | --- | --- |
| AI chat | `/ask`, `/pregunta`, `/che`, `/gordo`, private chat, mentions, replies, random group replies | Routing, typing/stream edits, tools, memory, billing, fallback, localized errors | `test_message_routing.py`, `test_ai_service.py`, `test_ai_requests.py`, `test_streaming*.py`, `test_message_billing.py` |
| Summary | `/resumen`, `/summary`, `/tldr` | Summary source selection, streaming, compaction markers, billing, insufficient-credit response | `test_memory_compaction.py`, `test_background_compaction.py`, `test_response_pipeline.py` |
| Media | `/transcribe`, `/describe`, automatic private/mention/reply processing | Telegram download, size/duration rules, image description, transcription fallback, reservation/settlement | `test_media_runtime.py`, `test_message_transcription.py`, `test_message_ai_media.py`, `test_media_cache.py` |
| Prices | `/prices` and all aliases, `/crypto`, `/criptos` | Provider scope, conversions, CoinMarketCap/Yahoo fallback, formatting, token-card precedence | `test_market_commands.py`, `test_market_formatting.py`, `test_crypto_selection.py` |
| Token cards | address and `$ticker` messages, `sig:*` callbacks | Chain detection, cached enrichment, keyboard URLs, refresh/delete authorization | `test_token_signals.py`, `test_command_routing.py` |
| Weather | `/clima`, `/weather`, AI tool | Geocoding, forecast mapping, default location, localization, upstream errors | `test_tools.py`, `test_market_commands.py` |
| Dollar | `/dolar`, `/dollar`, `/usd` | Current and historical rates, stale cache, ordering, formatting | `test_market_commands.py`, `test_market_formatting.py`, `test_stale_cache.py` |
| Oil and stocks | `/petroleo`, `/oil`, `/acciones`, `/stocks` | Yahoo search/chart behavior, Finviz list, unresolved symbols, formatting | `test_market_commands.py`, `test_market_formatting.py` |
| Elections | `/eleccion` and aliases | Liquidity ordering, midpoint batch/fallback, country flags, keyboard/link output | `test_polymarket_elections.py` |
| BCRA | `/bcra`, `/variables` | API series, ITCRM spreadsheet, historical lookup, stale cache, Spanish formatting | `test_bcra.py`, `test_bcra_itcrm_series.py`, `test_market_formatting.py` |
| Arbitrage and charts | `/rulo`, `/devo`, `/powerlaw`, `/rainbow`, `/satoshi` | Calculations, thresholds, missing prices, formatting, chart/photo behavior | `test_market_commands.py`, `test_market_formatting.py` |
| Random | `/random`, AI tool | Choice versus numeric range semantics and localized validation | `test_random_replies_internal.py`, `test_tools.py` |
| Utilities | `/convertbase`, `/comando`, `/command`, `/time`, `/instance` | Parsing, bounds, exact command/text format, configured instance fallback | `test_command_routing.py` |
| Giphy | `/gm`, `/gn` | API query, pool/stale pool behavior, media send fallback | `test_command_routing.py`, `test_redis_helpers.py` |
| Help and command publication | `/help`, startup `setMyCommands` | Complete visible catalog, hidden admin commands, Spanish and English descriptions | `test_feature_catalog.py`, `test_telegram_bot_commands.py`, `test_i18n.py` |
| Chat configuration | `/config`, `/configs`, `/settings`, `cfg:*` callbacks | Admin authorization in groups, defaults, keyboard state, PostgreSQL persistence, Redis admin cache | `test_chat_settings.py`, `test_chat_config_service.py`, `test_command_routing.py` |
| Language | `/language`, `/idioma`, `cfg:language:*` | Auto/es/en selection, persistence, command menu and response localization | `test_i18n.py`, `test_chat_settings.py` |
| Credits | `/balance`, `/charges`, `/history`, `/gastos`, `/transfer` | Scope selection, pagination, integer hundredths, authorization, idempotent mutations | `test_credits_db.py`, `test_message_billing.py`, `test_ai_billing_internal.py` |
| Top-up and payments | `/topup`, `topup:*`, pre-checkout, successful payment | Pack keyboard, invoice payload, ownership validation, payment idempotency, ledger mutation | `test_message_billing.py`, `test_ai_billing_internal.py`, `test_bot_ptb.py` |
| Admin credits | `/printcredits`, `/creditlog` | Admin-only authorization, parsing, ledger reports, redaction | `test_admin_reporting.py`, `test_command_routing.py` |
| Tasks | `/tarea`, `/tareas`, `/task`, `/tasks`, AI tools, `task:*` callbacks | Trigger validation, ownership, list/delete keyboards, persistence, execution, AI billing | `test_task_executor.py`, `test_task_scheduler_fixes.py`, `test_tools.py`, `test_command_routing.py` |
| Links | implicit supported-link handling and AI web fetch | URL safety, redirects, metadata, replacement modes, media handling, suppression rules | `test_links.py`, `test_link_pipeline.py`, `test_agent_tools.py`, `test_web_search.py` |

## Cross-cutting runtime features

| Feature | Critical observable contract | Current test evidence |
| --- | --- | --- |
| Telegram polling | Allowed update types, concurrent handling, conflict/network reporting, clean shutdown | `test_run_polling.py`, `test_bot_ptb.py`, `test_telegram_gateway.py` |
| Telegram gateway | Token redaction, retry-after handling, text limits, parse mode fallback, edit/send/media/file behavior | `test_telegram_gateway.py`, `test_streaming.py` |
| Command routing | Aliases, bot suffix removal, private/group rules, mentions, replies, random trigger, follow-up settings | `test_command_registry_internal.py`, `test_routing_policy.py`, `test_command_routing.py`, `test_message_routing.py` |
| AI providers | Availability, provider ordering, backoff, usage extraction, tool calls, stream conversion | `test_providers.py`, `test_provider_runtime.py`, `test_provider_configuration.py` |
| AI billing | Authorize, reserve, extend, record segments, settle, refund, debt, reconcile | `test_ai_billing_internal.py`, `test_incremental_ai_billing.py`, `test_message_billing.py`, `test_credits_db.py` |
| Tool registry | Availability by environment/context, schemas, argument parsing, errors, task-safe filtering | `test_tool_registry.py`, `test_tools.py`, `test_provider_runtime.py` |
| Conversation state | Ordered history, deduplication, TTL, search index, relevant retrieval, bot metadata, members | `test_message_state.py`, `test_message_state_internal.py`, `test_prompt_context.py` |
| Memory compaction | Planning, durable job, lease, retry/dead letter, summary/marker atomicity, billing | `test_memory_compaction.py`, `test_background_compaction.py` |
| Cache behavior | JSON encoding, request hashing/history, stale-while-refresh, TTL and maintenance | `test_redis_helpers.py`, `test_http_client.py`, `test_stale_cache.py`, `test_maintenance_internal.py` |
| Background reconciliation | Active-operation exclusion, provider generation lookup, correction, interval and shutdown | `test_incremental_ai_billing.py`, `test_ai_billing_internal.py` |
| Maintenance | Redis policy, legacy cache cleanup, TTL repair, ledger retention | `test_maintenance_internal.py` |
| Deployment | Quadlet settings, Redis Stack/RediSearch requirement, image startup, maintenance timers | `test_quadlets.py`, CI container build |

## Required contract shape

Each feature fixture must specify the relevant subset of:

```json
{
  "name": "stable scenario identifier",
  "input": {},
  "environment": {},
  "clock": "fixed UTC instant",
  "random": [],
  "http": [],
  "initial_redis": {},
  "initial_postgres": {},
  "expected_actions": [],
  "expected_redis": {},
  "expected_postgres": {},
  "expected_diagnostics": []
}
```

Secrets and real user content must never appear in fixtures.

## Removal rule

A row can move to `removed` only when:

1. Success, validation, upstream failure, retry, and restart scenarios relevant
   to the feature pass in Rust.
2. Stored-data and side-effect contracts match.
3. Production metrics distinguish and validate the Rust path.
4. Rollback was tested before the Python fallback was retired.
5. All meaningful Python tests were ported or replaced by stronger tests.
