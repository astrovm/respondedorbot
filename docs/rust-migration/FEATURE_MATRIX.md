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
| Durable provider segment identity and unresolved-usage policy | `rust-flagged` | `RUST_AI_USAGE_POLICY_ENABLED=0` or missing bridge | `contracts/ai_usage_policy.json` |
| AI token and preflight credit reserve estimates | `rust-flagged` | `RUST_AI_RESERVE_ESTIMATES_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_reserve_estimates.json` |
| Exact AI provider, model, transcription, cache, and tool pricing | `rust-flagged` | `RUST_AI_PRICING_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_pricing.json` |
| AI response persona, context, identity, duplicate, and Markdown cleanup | `rust-flagged` | `RUST_AI_RESPONSE_CLEANUP_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_response_cleanup.json` |
| Telegram streamed response send, edit, and finalize planning | `rust-flagged` | `RUST_TELEGRAM_STREAM_PLANNING_ENABLED=0`, missing bridge, or bridge error | `contracts/telegram_stream_planning.json` |
| AI tool argument parsing and environment/context/task availability | `rust-flagged` | `RUST_TOOL_REGISTRY_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/tool_registry_policy.json` |
| AI provider availability ordering and completion fallback outcome | `rust-flagged` | `RUST_PROVIDER_CHAIN_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_chain_policy.json` |
| Provider credentials, account ordering, backoff keys, scope availability, and Firecrawl tool configuration | `rust-flagged` | `RUST_PROVIDER_CONFIG_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_config_policy.json` |
| Provider response model, upstream, service-tier, and source normalization | `rust-flagged` | `RUST_PROVIDER_USAGE_NORMALIZATION_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_usage_normalization.json` |
| Stored assistant-text normalization before AI requests | `rust-flagged` | `RUST_AI_REQUEST_SANITIZATION_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_request_sanitization.json` |
| AI conversation, summary, media, fallback, and delivery-failure settlement decisions | `rust-flagged` | `RUST_AI_SETTLEMENT_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_settlement_policy.json` |
| Vision description billing and prompt-context append planning | `rust-flagged` | `RUST_AI_IMAGE_CONTEXT_PLANNING_ENABLED=0`, missing bridge, or bridge error | `contracts/ai_image_context_planning.json` |
| Provider tool-call missing, unregistered, and execute dispatch decisions | `rust-flagged` | `RUST_TOOL_EXECUTION_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/tool_execution_policy.json` |
| Firecrawl web-search HTTP requests, retries, response normalization, and accounting metadata | `rust-flagged` | `RUST_FIRECRAWL_ADAPTER_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/firecrawl_adapter.json` |
| OpenRouter finalized-generation lookup for interrupted AI billing reconciliation | `rust-flagged` | `RUST_OPENROUTER_GENERATION_ADAPTER_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/openrouter_generation_adapter.json` |
| Native typed OpenRouter chat completions and incremental SSE streams, split UTF-8/SSE frames, text and tool-call deltas, usage metadata, completion markers, rate limits, interrupted streams, provider errors, consumer cancellation, malformed responses, and transport failures | `shadow` | Python provider orchestration remains authoritative until the complete native AI state machine, tools, streaming, and billing path passes parity | Rust adapter request/response, incremental stream, interruption, and failure-path tests |
| Native typed bilingual AI system/conversation prompts and bounded provider/tool-round orchestration, including typed assistant/tool messages, known-call filtering, malformed-argument normalization, per-round billing segments, partial failure preservation, and a hard round limit | `shadow` | Python remains the deployed free-form AI orchestrator until concrete native tools, reservation/settlement, Telegram delivery, and provider fallback are composed and pass parity | Rust prompt-order, Unicode-boundary, request-conversion, multi-round, malformed-call, usage, interruption, and limit tests |
| Native scheduled-task AI prompt, bounded OpenRouter web search, stable personal-credit reservation, durable provider segments, exact-once settlement/refund, response cleanup, and Telegram delivery | `shadow` | Python remains the deployed task executor; the native executor is composed but cannot own occurrences until the atomic scheduler cutover | Native provider-request, billing-idempotency, executor-state, delivery, and failure-path tests |
| Telegram Bot API HTTP requests and file downloads for messages, media uploads, edits, callbacks, payments, admin checks, typing, deletion, and command publication | `rust-flagged` | `RUST_TELEGRAM_HTTP_ADAPTER_ENABLED=0`, unsupported multipart shape, missing bridge, bridge error, or invalid bridge result | `contracts/telegram_http_adapter.json` |
| Giphy greeting search requests and original GIF URL extraction | `rust-flagged` | `RUST_GIPHY_ADAPTER_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/giphy_adapter.json` |
| Stock query planning, Yahoo quote and symbol parsing, and Finviz mega-cap requests | `rust-flagged` | `RUST_STOCK_MARKET_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/stock_market.json` |
| Typed Telegram message text, poll, media identifier, chat type, user identity, and numeric ID parsing | `rust-flagged` | `RUST_TELEGRAM_INPUT_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/telegram_input.json` |
| Typed Telegram callback identity, chat, sender, message, and feature-route parsing | `rust-flagged` | `RUST_TELEGRAM_CALLBACKS_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/telegram_callbacks.json` |
| Telegram Stars pre-checkout and successful-payment identity, pack, amount, currency, invoice-owner, and persistence-input validation | `rust-flagged` | `RUST_TELEGRAM_PAYMENTS_ENABLED=0`, missing bridge, bridge error, or invalid bridge result | `contracts/telegram_payments.json` |
| Telegram long-poll request construction, supported-update decoding, offset advancement, and retry classification | `shadow` | Python PTB remains the sole production poller until the atomic runtime cutover | Rust adapter tests |
| Native process configuration, polling offset ownership, ordered dispatch, and retry lifecycle | `shadow` | `botd` refuses to poll until the native dispatcher is complete; Python PTB remains authoritative | `botd` unit tests |
| Typed outbound send, edit, delete, typing, callback-answer, and pre-checkout-answer actions | `shadow` | Python Telegram gateway remains authoritative until native dispatcher cutover | Rust core and adapter tests |
| Native localized `/convertbase`, `/time`, and `/instance` command-to-action vertical slices | `shadow` | Python command dispatcher remains authoritative until native message-state storage and full dispatch are connected | Rust core and dispatcher tests |
| PostgreSQL chat-configuration schema, typed reads, and typed upserts | `rust-flagged` | `RUST_CHAT_CONFIG_IO_ENABLED=0`, missing bridge, or bridge error | Rust core, adapter integration, and Python service tests |
| Native typed message-envelope dispatch with chat configuration and action execution ports | `shadow` | `botd` still refuses to poll; unsupported or incomplete paths remain legacy-owned | `botd` dispatcher tests |
| Concrete native Telegram polling, PostgreSQL config, and confirmed-delivery action composition | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner | `botd` composition tests |
| Native command user history, group member, assistant history, and reply-metadata writes | `shadow` | `botd` still refuses to start polling; Redis failures remain non-fatal and observable like the Python path | Rust core, adapter, and dispatcher tests |
| Native `/random` choice, arbitrary-precision inclusive range, and localized validation replies | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner | Rust parser contracts plus deterministic dispatcher and system-random tests |
| Native exact-parity Spanish and English `/help` catalog replies | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner | `contracts/stateless_commands.json`, core and dispatcher tests |
| Native sorted bilingual Telegram command publication with hidden-command filtering | `shadow` | Python remains the startup publisher until the atomic native runtime cutover | `contracts/telegram_commands.json`, core, action-adapter, and composition tests |
| Native `/language` and `/idioma` reads, validation, PostgreSQL updates, inline keyboard, group-admin authorization, cache, and audit diagnostics | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner | Core, dispatcher, Telegram/Redis adapter, and PostgreSQL tests |
| Native `/config`, `/configs`, and `/settings` bilingual summaries, exact inline keyboards, private/group visibility, group-admin authorization, and command-state writes | `shadow` | `botd` still refuses to start polling; `cfg:*` callbacks and Python remain authoritative until the complete settings slice is cut over | Python parity fingerprints plus core and dispatcher tests |
| Native `cfg:*` callback parsing, typed transitions, PostgreSQL persistence, group-admin authorization, message editing, fallback delivery, acknowledgement, and audit diagnostics | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all update routes are native | Shared callback contracts plus core, dispatcher, adapter, and PostgreSQL tests |
| Native Telegram Stars pre-checkout pack, owner, currency, amount, localization, and typed answer dispatch | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all payment routes are native | Shared payment contracts plus core, dispatcher, and Telegram action-adapter tests |
| Native Telegram Stars successful-payment decoding, validation, exact-once PostgreSQL ledger write, duplicate handling, localization, and diagnostics | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all payment routes are native | Shared payment contracts plus core, polling, dispatcher, PostgreSQL transaction, and Telegram action-adapter tests |
| Native `/topup` pack catalog, private-chat redirect, `topup:*` callback guards, localized Stars invoice construction, delivery feedback, and acknowledgement | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all update routes are native | Python message/callback parity plus core, dispatcher, Telegram action-adapter, and composition tests |
| Native `/balance` onboarding grant, personal/group PostgreSQL reads, exact bilingual formatting, non-fatal onboarding diagnostics, and command-state writes | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all update routes are native | Python formatting parity plus core, dispatcher, PostgreSQL integration, and composition tests |
| Native `/transfer` fixed-point parsing, guard order, bilingual replies, atomic user-to-group PostgreSQL transfer, insufficient-balance handling, and command-state writes | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all update routes are native | Python command parity plus core, dispatcher, concurrent PostgreSQL transaction, and composition tests |
| Native `/charges`, `/history`, and `/gastos`: bounded arguments, per-user PostgreSQL grouping, itemized model/tool costs, payer attribution, configured timezone, bilingual formatting, owned `chg:*` pagination callbacks, edit feedback, and page markup | `shadow` | `botd` still refuses to start polling; Python remains the only side-effect owner until all update routes are native | Python command/callback examples plus core formatter/parser, dispatcher, PostgreSQL query, and page-grouping tests |
| Native `/command` and `/comando` ASCII normalization, bilingual empty/invalid replies, bot mentions, reply actions, and command-state writes | `shadow` | Non-ASCII input and replied-message extraction remain on Python until native emoji/Japanese preprocessing and reply envelopes are complete; `botd` still refuses to poll | Shared normalization contracts plus core and dispatcher tests |
| Native admin-only `/printcredits` authorization, exact fixed-point parsing, atomic PostgreSQL mint and ledger write, bilingual guards/results, failure diagnostics, and command-state writes | `shadow` | `botd` still refuses to poll; Python remains the only side-effect owner until all update routes are native | Credit-unit contracts plus core, dispatcher, and PostgreSQL transaction tests |
| Native admin-only `/creditlog` authorization, bounded parsing, recent PostgreSQL settlement reads, bilingual itemized model/tool/cache/payer reports, Unicode-safe truncation, safe failures, and command-state writes | `shadow` | Unicode integer parsing remains on Python; `botd` still refuses to poll until every route is native | Admin-report contracts plus Python report examples, core formatter, dispatcher, and PostgreSQL query tests |
| Native `/satoshi`, `/sat`, `/sats`, `/powerlaw`, and `/rainbow` typed CoinMarketCap quote reads, exact calculations, bilingual valuation/error replies, sequential currency guards, diagnostics, and command-state writes | `shadow` | Python retains production polling and shared HTTP-cache ownership; missing native credentials/source stays legacy-owned | CoinMarketCap adapter tests, market-model and satoshi contracts, core renderers, and dispatcher tests |
| Native `/prices`, `/price`, `/precios`, `/precio`, `/presios`, `/presio`, `/bresio`, `/bresios`, `/brecio`, `/brecios`, `/c`, `/crypto`, and `/criptos` CoinMarketCap listings/quote reads, Python-compatible Redis request cache and stale fallback, amount conversions, supported currencies, `1h`/`24h`/`7d`/`30d`, stablecoin expansion, symbol/slug recovery, stock-only and mixed Yahoo fallback, bilingual failures, diagnostics, and command-state writes | `shadow` | Python retains production polling and a missing CoinMarketCap credential/source stays legacy-owned until the atomic runtime cutover | Price-query contracts, core selection/conversion/formatting tests, exact request-cache adapter tests, provider failure tests, dispatcher tests, and composition wiring |
| Native implicit Twitter/X, Bluesky, Instagram, and Reddit link replacement with profile exclusions, query cleanup, Telegram-compatible live preview probes, eeInstagram retries and fallback frontends, hourly media cache buckets, original-link buttons, localized sender attribution, reply/delete modes, failed-preview suppression, compatible context state, and bounded oversized-video multipart fallback | `shadow` | Python retains production polling until every implicit and AI route is native; Rust never sends or deletes links while the native poller is disabled | Pure rewrite/action tests, live-preview transport failure tests, oversized-video bounds, multipart action tests, polling-envelope tests, dispatcher tests, and composition wiring |
| Native `/devo` ASCII parsing, typed CriptoYa dollar quote read, exact arbitrage projection, bilingual validation/results/failures, diagnostics, and command-state writes | `shadow` | Python retains production polling and shared request-cache ownership; Unicode-number parsing and a missing native source stay legacy-owned | Devo contracts plus core parser/calculator/renderer, CriptoYa adapter, dispatcher, and composition tests |
| Native `/rulo` typed CriptoYa dollar and exchange-book reads, provider exclusions and precedence, exact route selection/formatting, bilingual results/failures, partial-book diagnostics, and command-state writes | `shadow` | Python retains production polling and shared request-cache ownership; a missing native source stays legacy-owned | Rulo contracts plus core evaluator/renderer, ordered CriptoYa adapter, orchestration, and dispatcher tests |
| Native `/gm` and `/gn` Redis-compatible fresh/stale Giphy pools, four-term typed searches, random selection, bilingual text fallback, `sendAnimation`, silent media-delivery failure, diagnostics, and command-state writes | `shadow` | Python retains production polling; a missing native source stays legacy-owned | Greeting core contracts, Giphy pool/cache failure tests, Telegram action tests, dispatcher tests, and composition wiring |
| Native `/clima` and `/weather` Open-Meteo geocoding and forecasts, Python-compatible Redis request-cache keys and stale fallback, qualified/default location selection, bilingual descriptions/results/errors, diagnostics, and command-state writes | `shadow` | Python retains production polling; replied commands and native AI-tool execution stay legacy-owned until the complete message and AI routes migrate | Weather contracts plus core renderer, typed Open-Meteo/cache adapter failure tests, dispatcher tests, and composition wiring |
| Native `/petroleo` and `/oil` typed Yahoo Finance chart reads, shared Python-compatible Redis request-cache and stale fallback, partial Brent/WTI results, bilingual failures, diagnostics, and command-state writes | `shadow` | Python retains production polling; the scheduled price-cache refresh stays Python-owned until background workers migrate | Oil formatter tests, shared request-cache recovery tests, typed Yahoo adapter tests, dispatcher tests, and composition wiring |
| Native `/acciones` and `/stocks` typed Yahoo chart/search reads, exact multi-word fallback order, Redis-cached Finviz mega-cap defaults, Python-compatible request-cache and stale fallback, bilingual results/failures, diagnostics, and command-state writes | `shadow` | Python retains production polling; market context and scheduled cache refresh stay Python-owned until AI and background workers migrate | Stock planning/formatter tests, Yahoo search/chart and Finviz cache failure tests, dispatcher tests, and composition orchestration tests |
| Native `/dolar`, `/dollar`, and `/usd` CriptoYa reads, exact request/hourly/formatted Redis cache compatibility, `1h`/`6h`/`12h`/`24h`/`48h` history, nine dollar rates, optional TCRM 100 and exchange-rate bands, bilingual validation/history/failures, diagnostics, and command-state writes | `shadow` | Python retains production polling; Rust BCRA writes remain rollback-readable and missing native sources stay legacy-owned until the atomic cutover | Timeframe and formatter unit tests, exact cache-key/history/stale adapter tests, provider failure tests, dispatcher localization/state tests, and composition wiring |
| Native `/bcra` and `/variables` BCRA v4 variables and exchange-rate bands, BondTerminal country risk, pure-Rust official ITCRM workbook parsing, TCRM 100 calculation, compatible fresh/last-success/mayorista/hourly Redis writes, bilingual formatting/staleness/failures, diagnostics, and command-state writes | `shadow` | Python retains production polling; all Rust cache payloads remain readable by Python during rollback and a missing native source remains legacy-owned | Complete indicator formatter tests, exact request-key and timezone tests, typed API/band/risk/ITCRM parsing, stale-provider failure tests, dispatcher tests, and composition wiring |
| Native `/eleccion`, `/elecciones`, `/election`, and `/elections` typed Gamma event reads, Python-compatible Redis request cache, deduplicated CLOB midpoint batch, liquidity ordering, complete ISO/regional flags, escaped HTML links, bilingual details/errors, diagnostics, and command-state writes | `shadow` | Python retains production polling; a missing native source stays legacy-owned until the atomic runtime cutover | Polymarket normalization/ranking/rendering tests, generated ISO lookup parity, adapter failure tests, dispatcher HTML-action tests, and composition wiring |
| Provider rate-limit and Groq account-fallback classification | `rust-flagged` | `RUST_PROVIDER_ERROR_POLICY_ENABLED=0` or missing bridge | `contracts/provider_error_policy.json` |
| Provider retry-window parsing and rate-limit header precedence | `rust-flagged` | `RUST_PROVIDER_RETRY_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_retry_policy.json` |
| Provider exception, usage, finish-response, and retry-delay policy | `rust-flagged` | `RUST_PROVIDER_RUNTIME_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_runtime_policy.json` |
| Provider pseudo web-fetch tool-call parsing and authorization | `rust-flagged` | `RUST_PROVIDER_TOOL_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_tool_policy.json` |
| Provider web-search limits, accounting, source extraction, and grounding | `rust-flagged` | `RUST_PROVIDER_WEB_SEARCH_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_web_search_policy.json` |
| Provider streaming text hold-and-release state machine | `rust-flagged` | `RUST_PROVIDER_STREAM_POLICY_ENABLED=0`, missing bridge, or bridge error | `contracts/provider_stream_policy.json` |
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
| Redis stale-while-refresh values and atomic refresh locks | `rust-flagged` | `RUST_STALE_CACHE_IO_ENABLED=0` or missing bridge | `contracts/stale_cache_io.json` |
| Redis memory policy, TTL repair, and legacy request-cache cleanup | `rust-flagged` | `RUST_REDIS_MAINTENANCE_ENABLED=0` or missing bridge | `contracts/redis_maintenance.json` |
| Scheduled-task payload and per-chat Redis index I/O | `rust-flagged` | `RUST_TASK_STORE_IO_ENABLED=0` or missing bridge | `contracts/task_store_io.json` |
| Canonical scheduled-task records, recurrence, coalescing, and misfire decisions | `shadow` | Python APScheduler remains the sole executor | `contracts/task_records.json` |
| Scheduled-task list commands and deletion callbacks | `shadow` | Python remains the Telegram poller and callback owner | `contracts/task_records.json` |
| Canonical APScheduler reconstruction and native verify/authoritative scheduler engine | `shadow` | Python remains the deployed scheduler owner; Rust verification mode performs no writes or execution until the atomic owner cutover | Python rebuild tests, native state-machine tests, Redis protocol tests, and real-Redis scheduler integration |
| Read-only `botd --verify-tasks` production-state observation entrypoint | `shadow` | Verification mode never acquires ownership, claims occurrences, executes AI, sends Telegram messages, or writes task state | CLI configuration tests plus a real-Redis empty-state verification run |
| PostgreSQL billing balance read parity | `shadow` | `RUST_BILLING_READ_SHADOW_ENABLED=0` or missing bridge | `contracts/billing_reads.json` |
| PostgreSQL billing balance account I/O | `rust-flagged` | `RUST_BILLING_BALANCE_IO_ENABLED=0` or missing bridge | `contracts/billing_reads.json` |
| PostgreSQL idempotent onboarding grants and overflow ledger | `rust-flagged` | `RUST_BILLING_ONBOARDING_ENABLED=0` or missing bridge | `contracts/billing_onboarding.json` |
| PostgreSQL idempotent Telegram Stars payments, balances, and ledger | `rust-flagged` | `RUST_BILLING_STAR_PAYMENTS_ENABLED=0` or missing bridge | `contracts/billing_star_payments.json` |
| PostgreSQL administrator mint and user-to-chat transfer transactions | `rust-flagged` | `RUST_BILLING_MANUAL_CREDITS_ENABLED=0` or missing bridge | `contracts/billing_manual_credits.json` |
| PostgreSQL chat-owned AI reserve, refund, and debt transactions | `rust-flagged` | `RUST_BILLING_CHAT_AI_CREDITS_ENABLED=0` or missing bridge | `contracts/billing_chat_ai_credits.json` |
| PostgreSQL user-or-chat AI debt transactions | `rust-flagged` | `RUST_BILLING_AI_DEBT_ENABLED=0` or missing bridge | `contracts/billing_ai_debt.json` |
| PostgreSQL idempotent user-or-chat AI refunds | `rust-flagged` | `RUST_BILLING_AI_REFUNDS_ENABLED=0` or missing bridge | `contracts/billing_ai_refunds.json` |
| PostgreSQL AI payer selection, reserve, charge, and replay guards | `rust-flagged` | `RUST_BILLING_AI_CHARGES_ENABLED=0` or missing bridge | `contracts/billing_ai_charges.json` |
| PostgreSQL idempotent AI provider-usage segment record, read, and reconciliation update | `rust-flagged` | `RUST_BILLING_PROVIDER_USAGE_ENABLED=0` or missing bridge | `contracts/billing_provider_usage.json` |
| PostgreSQL exact-once AI operation settlement | `rust-flagged` | `RUST_BILLING_AI_SETTLEMENTS_ENABLED=0` or missing bridge | `contracts/billing_ai_settlements.json` |
| PostgreSQL exact-once legacy usage-tag settlement | `rust-flagged` | `RUST_BILLING_LEGACY_SETTLEMENTS_ENABLED=0` or missing bridge | `contracts/billing_legacy_settlements.json` |
| PostgreSQL idempotent AI settlement audit writes | `rust-flagged` | `RUST_BILLING_AUDIT_WRITES_ENABLED=0` or missing bridge | `contracts/billing_audit_writes.json` |
| PostgreSQL recent AI settlement audit reads | `rust-flagged` | `RUST_BILLING_AUDIT_READS_ENABLED=0` or missing bridge | `contracts/billing_audit_reads.json` |
| PostgreSQL unsettled AI operation reconciliation reads | `rust-flagged` | `RUST_BILLING_RECONCILIATION_READS_ENABLED=0` or missing bridge | `contracts/billing_reconciliation_reads.json` |
| PostgreSQL AI ledger retention purge | `rust-flagged` | `RUST_BILLING_MAINTENANCE_ENABLED=0` or missing bridge | `contracts/billing_maintenance.json` |
| PostgreSQL paginated user AI charge-history query | `rust-flagged` | `RUST_BILLING_CHARGE_HISTORY_ENABLED=0` or missing bridge | `contracts/billing_charge_history.json` |
| PostgreSQL billing schema and historical data migrations | `rust-flagged` | `RUST_BILLING_SCHEMA_ENABLED=0` or missing bridge | `contracts/billing_schema.json` |
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
