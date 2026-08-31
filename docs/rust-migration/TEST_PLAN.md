# Compatibility and Test Plan

## Baseline

At migration start, `uv run --locked pytest -q` passes 1,015 tests. The suite is
fast and covers substantial internal behavior, but many tests use Python mocks.
The migration must retain those tests while adding language-neutral contracts,
real storage integration, concurrency, restart, and end-to-end verification.

## Fixture principles

- Use synthetic Telegram identities and content.
- Fix time and randomness explicitly.
- Record provider responses as minimal synthetic protocol fixtures, not copied
  user data or large third-party payloads.
- Compare structured actions and state before comparing logs.
- Normalize nondeterministic identifiers through named placeholders.
- Do not assert implementation details or source layout.
- Store secrets as symbolic fixture references only.

## Contract runner

The contract runner will accept one fixture and an implementation selector:

```text
contract-runner --implementation python fixture.json
contract-runner --implementation rust fixture.json
```

It will provide controlled adapters for:

- Clock and randomness.
- Telegram requests and file downloads.
- Provider and market HTTP requests.
- Redis reads, writes, scripts, search, and failures.
- PostgreSQL reads, transactions, locks, and failures.
- Filesystem bootstrap files.
- FFmpeg/media probe results.

It will capture typed actions, outbound requests, state mutations, billing
events, and classified diagnostics. During the hybrid phase, CI compares the
normalized Python and Rust results.

## Critical scenario catalog

### Routing and Telegram

- Private text, group mention, reply to bot, random trigger, and ignored group text.
- Known and unknown commands, aliases, bot username suffix, empty text, and
  malformed messages.
- Link-fix reply suppression and command-follow-up settings.
- Callback acknowledgement on success, rejection, and internal failure.
- Pre-checkout ownership and payload validation.
- Telegram 429 retry-after, message-not-modified, parse failure, network timeout,
  and token redaction.
- Polling conflict throttling and graceful shutdown.

### Billing

- User-funded and chat-funded reservation.
- Insufficient balance and creditless allowance.
- Reservation extension.
- Exact settlement, partial refund, extra settlement charge, and debt.
- Provider segment duplication.
- Settlement duplication and reconciliation retry.
- Concurrent reservations and transfers with deterministic lock order.
- Telegram payment replay.
- Transaction failure before and after mutation.
- Process termination followed by reconciliation.
- Credit scale and legacy migration behavior.

### AI and tools

- Complete non-streaming and streaming responses.
- Empty response and fallback provider.
- Partial tool arguments across stream chunks.
- Multiple tools and repeated tool rounds.
- Unknown tool and invalid arguments.
- Exact tool names, JSON schemas, runtime availability, and task-mode filtering.
- Calculator precedence, exact integers, float rounding, floor/modulo semantics,
  forbidden syntax, zero division, and bounded complexity.
- Typed assistant tool-call and tool-result messages across provider rounds.
- Hard tool-round termination, with completed provider/tool billing segments
  retained for settlement.
- Partial text and reconcilable provider usage retained when a later stream or
  response consumer fails.
- Firecrawl authorization and usage accounting.
- Provider rate limit and backoff expiration.
- Cancellation, timeout, malformed event, and provider disconnect.
- Usage received before/after text completion and duplicate usage events.
- Image context, transcription primary/fallback, and summary calls.

### Memory

- Atomic deduplication and ordering.
- History trimming and TTL repair.
- Search index creation and relevant-message retrieval.
- Summary planning thresholds and marker selection.
- Compaction enqueue race, lease contention, retry, dead letter, restart, and
  settlement.
- Atomic summary/marker update and stale expected marker.

### Scheduled tasks

- Delay, second interval, day interval, daily, weekly, and monthly cron.
- Bounds and malformed trigger configurations.
- Time-zone offsets and date formatting.
- Missed execution/coalescing behavior.
- Restart reconstruction.
- Two scheduler workers competing for one task.
- Native scheduled-task provider requests, personal-credit reservation replay,
  durable usage segments, exact-once settlement/refund, and non-repeated side
  effects after a settled execution.
- `botd --verify-tasks` against canonical Redis state; verification must perform
  no ownership, occurrence claim, execution, delivery, or state write.
- OpenRouter SSE boundaries split across arbitrary transport chunks and UTF-8
  code points, including tool-call fragment accumulation inputs, final usage,
  explicit provider errors, missing completion markers, transport interruption,
  and downstream consumer cancellation.
- Cancellation during claim and execution.
- AI fallback/retry and billing idempotency.
- Existing APScheduler-to-canonical-record migration.

### External features

- Fresh, stale, and missing cache behavior.
- Each market/provider success, empty, malformed, timeout, and retry path.
- CoinMarketCap/Yahoo resolution and symbol ambiguity.
- BCRA API and spreadsheet fallback.
- Link redirects, blocked/private destinations, oversized bodies, media links,
  and unsupported hosts.
- Media size/duration limits, download failure, FFmpeg failure, and cache hit.

## Integration environments

Pull-request integration tests will use disposable services:

- Redis Stack with `FT.CREATE` and `FT.SEARCH` enabled.
- PostgreSQL with the production schema migrations.
- A local fake HTTP server for Telegram and providers.
- Temporary directories for workspace and media files.

All state is created from synthetic fixtures and discarded after the run.
The CI checks job currently provides PostgreSQL 17 to Rust tests through
`TEST_POSTGRES_URL`; migrated billing transactions therefore contribute real
database execution to the adapter coverage gate.
Coverage targets are cleaned between crate gates so instrumented binaries from
an earlier crate cannot dilute or inflate a later crate's report. CI enforces
the documented line thresholds independently for `bot-core`, `bot-adapters`,
and `botd`.

## Coverage and mutation gates

- `bot-core`: enforce at least 95% line coverage and target 90% branch coverage.
- Adapters: enforce at least 85% line coverage.
- Enforce branch thresholds when the stable Rust coverage toolchain exposes
  branch counters; the current report does not.
- No reduction from the established Rust baseline without explanation.
- Routing, billing, and scheduling state-transition matrices must be complete.
- Mutation testing must cover critical pure rules before their Python versions
  are removed.
- Fuzz targets must complete a defined CI smoke duration and a longer scheduled
  run without panics, hangs, or uncontrolled allocation.

## CI stages

### Fast pull-request checks

- Python Ruff, strict mypy, and existing tests.
- Rust formatting, `cargo check`, Clippy, and unit tests.
- Contract parity for migrated features.
- Coverage ratchet.

### Integration checks

- Redis Stack and PostgreSQL tests.
- Fake Telegram/provider end-to-end flows.
- Restart and limited concurrency scenarios.
- Container build and startup smoke test.

### Extended checks

- Fuzzing.
- Mutation testing.
- High-contention billing and scheduler tests.
- Long streaming and cancellation tests.
- Resource and load tests.

## Per-feature release evidence

Every feature status change in `FEATURE_MATRIX.md` requires:

1. Links to its unit, contract, integration, and relevant failure tests.
2. Coverage report for the changed Rust modules.
3. Confirmation that side-effect ownership cannot overlap.
4. Feature-flag rollback instructions.
5. Production observation results with secrets and user data excluded.
6. Confirmation that meaningful Python tests were ported before deletion.
