#![allow(clippy::panic)]

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use bot_adapters::bcra::ReqwestBcraTransport;
use bot_adapters::billing_read::BillingRepository;
use bot_adapters::billing_schema::BillingSchemaRepository;
use bot_adapters::criptoya::ReqwestCriptoYaTransport;
use bot_adapters::dollar::ReqwestDollarTransport;
use bot_adapters::finviz::ReqwestFinvizTransport;
use bot_adapters::firecrawl::ReqwestFirecrawlTransport;
use bot_adapters::giphy::ReqwestGiphyTransport;
use bot_adapters::giphy_pool::GiphyPoolCache;
use bot_adapters::hacker_news::ReqwestHackerNewsTransport;
use bot_adapters::link_preview;
use bot_adapters::link_preview::ReqwestLinkPreviewTransport;
use bot_adapters::openrouter_chat::{
    self, ChatCompletionRequest, OpenRouterChatError, ReqwestOpenRouterTransport,
};
use bot_adapters::openrouter_generation::ReqwestGenerationTransport;
use bot_adapters::polymarket::ReqwestPolymarketTransport;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_json_cache::RedisJsonCache;
use bot_adapters::redis_message_state::RedisMessageState;
use bot_adapters::request_cache::RequestCache as JsonRequestCache;
use bot_adapters::stock_pool::StockPoolCache;
use bot_adapters::task_record::{TaskRecordError, decode_task_record, encode_task_record};
use bot_adapters::telegram_http::{
    self, MultipartUpload, ReqwestTelegramTransport, TelegramHttpError,
};
use bot_adapters::token_signal::{
    BinaryResponse, JsonResponse, ReqwestTokenSignalTransport, TokenSignalAdapter,
    TokenSignalCache, TokenSignalTransport, render_signal_chart,
};
use bot_adapters::weather::ReqwestWeatherTransport;
use bot_adapters::web_fetch::{HostResolver, ReqwestWebFetchTransport, SystemHostResolver};
use bot_adapters::yahoo_finance::ReqwestYahooFinanceTransport;
use bot_core::message_state::prepare_message_write;
use bot_core::token_signals::{
    PairToken, SignalQuery, SignalState, TokenAddress, TokenPair, TokenSignal,
};
use serde_json::json;
use std::time::Duration;

#[test]
fn production_http_transports_construct_without_contacting_providers() {
    assert!(ReqwestBcraTransport::new().is_ok());
    assert!(ReqwestCriptoYaTransport::new().is_ok());
    assert!(ReqwestDollarTransport::new().is_ok());
    assert!(ReqwestFinvizTransport::new().is_ok());
    assert!(ReqwestFirecrawlTransport::new().is_ok());
    assert!(ReqwestGiphyTransport::new().is_ok());
    assert!(ReqwestHackerNewsTransport::new().is_ok());
    assert!(ReqwestLinkPreviewTransport::new().is_ok());
    assert!(ReqwestOpenRouterTransport::new().is_ok());
    assert!(ReqwestGenerationTransport::new().is_ok());
    assert!(ReqwestPolymarketTransport::new().is_ok());
    assert!(ReqwestTelegramTransport::new().is_ok());
    assert!(ReqwestTokenSignalTransport::new().is_ok());
    assert!(ReqwestWeatherTransport::new().is_ok());
    assert!(ReqwestWebFetchTransport::new().is_ok());
    assert!(ReqwestYahooFinanceTransport::new().is_ok());
}

#[test]
fn shared_redis_cache_implements_every_provider_cache_port() {
    let Some(port) = std::env::var("TEST_REDIS_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
    else {
        return;
    };
    let endpoint = RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    };
    let mut cache = RedisJsonCache::new(&endpoint).unwrap_or_else(|_| unreachable!());

    assert!(JsonRequestCache::set(&mut cache, "coverage95:request", "1", 60).is_ok());
    assert!(matches!(
        JsonRequestCache::get(&mut cache, "coverage95:request"),
        Ok(Some(value)) if value == "1"
    ));
    assert!(GiphyPoolCache::set(&mut cache, "coverage95:giphy", "[]", 60).is_ok());
    assert!(matches!(
        GiphyPoolCache::get(&mut cache, "coverage95:giphy"),
        Ok(Some(value)) if value == "[]"
    ));
    assert!(StockPoolCache::set(&mut cache, "coverage95:stocks", "[]", 60).is_ok());
    assert!(matches!(
        StockPoolCache::get(&mut cache, "coverage95:stocks"),
        Ok(Some(value)) if value == "[]"
    ));
}

#[test]
fn message_state_write_refreshes_every_canonical_ttl() {
    let Some(port) = std::env::var("TEST_REDIS_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
    else {
        return;
    };
    let endpoint = RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    };
    let state = RedisMessageState::new(&endpoint).unwrap_or_else(|_| unreachable!());
    let plan = prepare_message_write(
        "coverage95-ttl",
        "1",
        "synthetic message",
        1,
        None,
        Some("7"),
        Some("user"),
        None,
        false,
    )
    .unwrap_or_else(|_| unreachable!());

    assert!(state.save_message(&plan, 60, 10).unwrap_or(false));

    let client = redis::Client::open(format!("redis://{}:{}/", endpoint.host, endpoint.port))
        .unwrap_or_else(|_| unreachable!());
    let mut connection = client.get_connection().unwrap_or_else(|_| unreachable!());
    for key in [
        &plan.keys.history,
        &plan.keys.order,
        &plan.keys.sequence,
        &plan.keys.search_document,
    ] {
        let ttl: i64 = redis::cmd("TTL")
            .arg(key)
            .query(&mut connection)
            .unwrap_or(-1);
        assert!((1..=60).contains(&ttl), "missing TTL for {key}");
    }
    let _: usize = redis::cmd("DEL")
        .arg(&plan.keys.history)
        .arg(&plan.keys.order)
        .arg(&plan.keys.sequence)
        .arg(&plan.keys.search_document)
        .query(&mut connection)
        .unwrap_or(0);
}

#[test]
fn billing_operations_report_a_missing_schema_at_each_public_boundary() {
    let Some(database_url) = std::env::var("TEST_POSTGRES_URL").ok() else {
        return;
    };
    let separator = if database_url.contains('?') { '&' } else { '?' };
    let database_url =
        format!("{database_url}{separator}options=-csearch_path%3Dcoverage95_missing");
    let repository = BillingRepository::new(&database_url);
    let metadata = serde_json::Map::new();

    assert!(repository.get_or_create_balance("user", 1).is_err());
    assert!(
        repository
            .record_star_payment("synthetic-charge", 1, "synthetic-pack", 1, 1, None)
            .is_err()
    );
    assert!(repository.mint_user_credits(1, 1, None).is_err());
    assert!(repository.transfer_user_to_chat(1, -1, 1).is_err());
    assert!(
        repository
            .record_ai_provider_usage(1, None, &json!({"operation_id":"synthetic"}))
            .is_err()
    );
    assert!(
        repository
            .list_ai_provider_segments(1, "synthetic")
            .is_err()
    );
    assert!(
        repository
            .update_ai_provider_usage("synthetic", "segment", &json!({}))
            .is_err()
    );
    assert!(
        repository
            .compaction_reservation_settled(1, "synthetic", "synthetic")
            .is_err()
    );
    assert!(
        repository
            .record_ai_settlement_result(1, None, 1, "synthetic", &metadata)
            .is_err()
    );
    assert!(repository.list_recent_ai_settlement_results(1).is_err());
    assert!(repository.list_unsettled_ai_operations(1).is_err());
    assert!(repository.purge_expired_ai_ledger_events(30).is_err());
    assert!(
        repository
            .list_user_ai_charge_rows(1, None, "older", 1)
            .is_err()
    );
}

#[derive(Default)]
struct CacheState {
    values: BTreeMap<String, String>,
    writes: Vec<(String, String, i64)>,
    fail_get: bool,
    fail_set: bool,
}

#[derive(Clone, Default)]
struct Cache(Arc<Mutex<CacheState>>);

impl TokenSignalCache for Cache {
    type Error = &'static str;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        let state = self.0.lock().map_err(|_| "synthetic cache lock failure")?;
        if state.fail_get {
            Err("synthetic cache read failure")
        } else {
            Ok(state.values.get(key).cloned())
        }
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        let mut state = self.0.lock().map_err(|_| "synthetic cache lock failure")?;
        if state.fail_set {
            Err("synthetic cache write failure")
        } else {
            state
                .writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            state.values.insert(key.to_owned(), value.to_owned());
            Ok(())
        }
    }
}

#[derive(Default)]
struct TransportState {
    json: VecDeque<Result<JsonResponse, String>>,
    post: VecDeque<Result<JsonResponse, String>>,
    binary: VecDeque<Result<BinaryResponse, String>>,
}

#[derive(Clone, Default)]
struct Transport(Arc<Mutex<TransportState>>);

impl TokenSignalTransport for Transport {
    fn get_json(&self, _url: &str, _query: &[(&str, String)]) -> Result<JsonResponse, String> {
        self.0
            .lock()
            .map_err(|_| "synthetic transport lock failure".to_owned())?
            .json
            .pop_front()
            .unwrap_or_else(|| Err("synthetic GET failure".to_owned()))
    }

    fn post_json(&self, _url: &str, _body: &serde_json::Value) -> Result<JsonResponse, String> {
        self.0
            .lock()
            .map_err(|_| "synthetic transport lock failure".to_owned())?
            .post
            .pop_front()
            .unwrap_or_else(|| Err("synthetic POST failure".to_owned()))
    }

    fn get_binary(&self, _url: &str) -> Result<BinaryResponse, String> {
        self.0
            .lock()
            .map_err(|_| "synthetic transport lock failure".to_owned())?
            .binary
            .pop_front()
            .unwrap_or_else(|| Err("synthetic image failure".to_owned()))
    }
}

fn token(chain_id: &str) -> TokenAddress {
    TokenAddress {
        chain_id: chain_id.to_owned(),
        network: if chain_id == "solana" {
            "solana"
        } else {
            "eth"
        }
        .to_owned(),
        tag: if chain_id == "solana" { "SOL" } else { "ETH" }.to_owned(),
        address: if chain_id == "solana" {
            "SyntheticAddress111111111111111111111pump"
        } else {
            "0x0000000000000000000000000000000000000001"
        }
        .to_owned(),
    }
}

fn pair(chain_id: &str, address: &str, symbol: &str, pair_address: &str) -> serde_json::Value {
    json!({
        "chainId": chain_id,
        "pairAddress": pair_address,
        "baseToken": {"address": address, "symbol": symbol},
        "liquidity": {"usd": 100},
        "volume": {"h24": 10}
    })
}

fn signal() -> TokenSignal {
    TokenSignal {
        token: token("ethereum"),
        pair: TokenPair {
            base_token: PairToken {
                symbol: "SYN".to_owned(),
                ..PairToken::default()
            },
            price_usd: json!(2.5),
            market_cap: json!(1_000_000),
            ..TokenPair::default()
        },
        candles: Vec::new(),
        supply: None,
        token_image_url: None,
        socials: BTreeMap::new(),
        pump: None,
    }
}

#[test]
fn token_signal_cache_and_provider_failures_remain_diagnostic() {
    let cases = [
        (
            CacheState {
                fail_get: true,
                ..CacheState::default()
            },
            Err("synthetic provider failure".to_owned()),
            "could not read DexScreener pairs cache",
        ),
        (
            CacheState::default(),
            Ok(JsonResponse {
                status_code: 503,
                body: String::new(),
            }),
            "DexScreener pairs HTTP 503",
        ),
        (
            CacheState::default(),
            Ok(JsonResponse {
                status_code: 200,
                body: "not-json".to_owned(),
            }),
            "invalid DexScreener pairs response",
        ),
    ];
    for (cache_state, response, expected) in cases {
        let cache = Cache(Arc::new(Mutex::new(cache_state)));
        let transport = Transport::default();
        transport
            .0
            .lock()
            .unwrap_or_else(|_| unreachable!())
            .json
            .push_back(response);
        let mut adapter = TokenSignalAdapter::new(transport, cache);
        let load = adapter.load_token(&token("ethereum"));
        assert!(load.signal.is_none());
        assert!(
            load.diagnostics
                .iter()
                .any(|entry| entry.contains(expected))
        );
    }

    let cache = Cache::default();
    cache
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .values
        .insert(
            "token_signal:pairs:ethereum:0x0000000000000000000000000000000000000001".to_owned(),
            "not-json".to_owned(),
        );
    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .json
        .push_back(Ok(JsonResponse {
            status_code: 200,
            body: "[]".to_owned(),
        }));
    let mut adapter = TokenSignalAdapter::new(transport, cache);
    let load = adapter.load_token(&token("ethereum"));
    assert!(load.diagnostics[0].contains("invalid DexScreener pairs cache"));
}

#[test]
fn token_signal_fallbacks_cover_cached_pairs_symbols_and_image_failures() {
    let cache = Cache::default();
    cache
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .values
        .insert(
            "token_signal:pairs:ethereum:0x0000000000000000000000000000000000000001".to_owned(),
            json!([pair(
                "ethereum",
                "0x0000000000000000000000000000000000000001",
                "SYN",
                ""
            )])
            .to_string(),
        );
    let mut adapter = TokenSignalAdapter::new(Transport::default(), cache);
    assert!(adapter.load_token(&token("ethereum")).signal.is_some());

    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .json
        .push_back(Ok(JsonResponse {
            status_code: 200,
            body: json!({"pairs": [
                pair("unsupported", "synthetic", "SYN", "ignored"),
                pair("ethereum", "0x0000000000000000000000000000000000000001", "OTHER", "")
            ]})
            .to_string(),
        }));
    let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
    let load = adapter.load_query(&SignalQuery::Symbol("SYN".to_owned()));
    assert_eq!(
        load.signal
            .as_ref()
            .map(|value| value.pair.base_token.symbol.as_str()),
        None
    );

    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .binary
        .push_back(Ok(BinaryResponse {
            status_code: 200,
            content_type: "text/plain".to_owned(),
            body: b"not an image".to_vec(),
        }));
    let adapter = TokenSignalAdapter::new(transport, Cache::default());
    let mut without_chart = signal();
    without_chart.token_image_url = Some("https://example.test/image".to_owned());
    assert!(
        adapter
            .render_photo(&without_chart)
            .is_ok_and(|image| image.starts_with(b"\x89PNG"))
    );
}

#[test]
fn token_signal_state_round_trips_and_reports_storage_failures() {
    let state = SignalState {
        chart_period: None,
        chat_id: "synthetic-chat".to_owned(),
        message_id: 2,
        source_message_id: 1,
        requester_id: "synthetic-user".to_owned(),
        chain_id: "ethereum".to_owned(),
        network: "eth".to_owned(),
        tag: "ETH".to_owned(),
        address: "0x0000000000000000000000000000000000000001".to_owned(),
        last_refresh_at: Some(10),
    };
    let cache = Cache::default();
    let mut adapter = TokenSignalAdapter::new(Transport::default(), cache.clone());
    assert_eq!(adapter.load_state("missing"), Ok(None));
    assert_eq!(adapter.save_state("state", &state), Ok(()));
    assert_eq!(adapter.load_state("state"), Ok(Some(state.clone())));

    cache
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .values
        .insert("token_signal:invalid".to_owned(), "bad-json".to_owned());
    assert!(adapter.load_state("invalid").is_err());

    cache.0.lock().unwrap_or_else(|_| unreachable!()).fail_get = true;
    assert!(adapter.load_state("state").is_err());
    {
        let mut cache_state = cache.0.lock().unwrap_or_else(|_| unreachable!());
        cache_state.fail_get = false;
        cache_state.fail_set = true;
    }
    assert!(adapter.save_state("state", &state).is_err());
}

#[test]
fn token_signal_chart_handles_boundaries_and_candle_variants() {
    assert_eq!(
        render_signal_chart(&signal(), 119, 200),
        Err("token chart dimensions are too small".to_owned())
    );

    let mut chart = signal();
    chart.pair.base_token.symbol.clear();
    chart.pair.price_usd = json!(0);
    chart.pair.market_cap = json!(0);
    chart.pair.fdv = json!(25_000);
    chart.candles = vec![
        vec![3.0, 2.0, 2.0, 2.0, 2.0],
        vec![2.0, 2.0, 2.0, 2.0, 2.0],
        vec![1.0, 2.0, 2.0, 2.0, 2.0],
    ];
    assert!(render_signal_chart(&chart, 320, 240).is_ok_and(|image| image.starts_with(b"\x89PNG")));
    assert!(
        TokenSignalAdapter::new(Transport::default(), Cache::default())
            .render_photo(&chart)
            .is_ok_and(|image| image.starts_with(b"\x89PNG"))
    );
}

#[test]
fn token_symbol_selection_tries_supported_fallback_pairs_and_cache_errors() {
    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .json
        .extend([
        Ok(JsonResponse {
            status_code: 200,
            body: json!({"pairs": [
                pair("unsupported", "synthetic", "SYN", "ignored"),
                pair("ethereum", "0x0000000000000000000000000000000000000001", "OTHER", "fallback"),
                pair("ethereum", "0x0000000000000000000000000000000000000002", "SYN", "")
            ]})
            .to_string(),
        }),
        Ok(JsonResponse {
            status_code: 200,
            body: json!({"data":{"attributes":{"ohlcv_list":[[1,1,2,0.5,1.5]]}}}).to_string(),
        }),
    ]);
    let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
    let load = adapter.load_symbol("$SYN");
    assert_eq!(
        load.signal
            .as_ref()
            .map(|value| value.pair.pair_address.as_str()),
        Some("")
    );

    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .json
        .push_back(Ok(JsonResponse {
            status_code: 200,
            body: "{}".to_owned(),
        }));
    let mut adapter = TokenSignalAdapter::new(transport, Cache::default());
    let missing = adapter.load_symbol("SYN");
    assert!(missing.signal.is_none());
    assert!(
        missing
            .diagnostics
            .iter()
            .any(|entry| entry.contains("expected value"))
    );

    let cache = Cache::default();
    cache.0.lock().unwrap_or_else(|_| unreachable!()).fail_set = true;
    let transport = Transport::default();
    transport
        .0
        .lock()
        .unwrap_or_else(|_| unreachable!())
        .json
        .push_back(Ok(JsonResponse {
            status_code: 200,
            body: json!({"pairs": []}).to_string(),
        }));
    let mut adapter = TokenSignalAdapter::new(transport, cache);
    let load = adapter.load_symbol("SYN");
    assert!(
        load.diagnostics
            .iter()
            .any(|entry| entry.contains("could not write DexScreener search cache"))
    );
}

fn task_payload(trigger: serde_json::Value) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "id": "synthetic1",
        "chat_id": "synthetic-chat",
        "text": "synthetic task",
        "user_name": "synthetic-user",
        "trigger_config": trigger,
        "timezone_offset": -3,
        "locale": "en"
    })
}

#[test]
fn task_records_cover_interval_cron_and_validation_boundaries() {
    let interval =
        decode_task_record(&task_payload(json!({"type":"interval", "days":30})).to_string())
            .unwrap_or_else(|_| unreachable!());
    let encoded = encode_task_record(&interval).unwrap_or_else(|_| unreachable!());
    assert!(encoded.contains(r#""days":30"#));

    let cron = decode_task_record(
        &task_payload(json!({
            "type":"cron", "hour":23, "minute":59, "day":31, "day_of_week":"mon,fri"
        }))
        .to_string(),
    )
    .unwrap_or_else(|_| unreachable!());
    let encoded = encode_task_record(&cron).unwrap_or_else(|_| unreachable!());
    assert!(encoded.contains(r#""day":31"#));
    assert!(encoded.contains("mon,fri"));

    for payload in [
        task_payload(json!({"type":"interval", "days":0})),
        task_payload(json!({"type":"interval"})),
        task_payload(json!({"type":"cron", "hour":24, "minute":0})),
        task_payload(json!({"type":"cron", "hour":0, "minute":60})),
        task_payload(json!({"type":"cron", "hour":0, "minute":0, "day":0})),
        task_payload(json!({"type":"unknown"})),
    ] {
        assert!(matches!(
            decode_task_record(&payload.to_string()),
            Err(TaskRecordError::InvalidTrigger)
        ));
    }

    for (field, value) in [
        ("id", json!(null)),
        ("chat_id", json!("")),
        ("text", json!([])),
        ("user_name", json!({})),
        ("timezone_offset", json!(15)),
    ] {
        let mut payload = task_payload(json!({"type":"interval", "days":1}));
        payload[field] = value;
        assert!(decode_task_record(&payload.to_string()).is_err());
    }
}

#[test]
fn billing_schema_repairs_user_and_chat_funded_duplicate_refunds()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
        return Ok(());
    };
    BillingSchemaRepository::new(&database_url).ensure_schema()?;
    let mut client = postgres::Client::connect(&database_url, postgres::NoTls)?;
    client.batch_execute(
        "DELETE FROM credit_ledger WHERE user_id IN (8300000000001, 8300000000002); \
         DELETE FROM credit_accounts WHERE scope_id IN \
            (8300000000001, 8300000000002, -8300000000002); \
         DELETE FROM credit_schema_migrations \
            WHERE name = 'repair_duplicate_compaction_refunds_v1'; \
         INSERT INTO credit_accounts (scope_type, scope_id, balance) VALUES \
            ('user', 8300000000001, 500), \
            ('user', 8300000000002, 500), \
            ('chat', -8300000000002, 500); \
         INSERT INTO credit_ledger \
            (event_type, user_id, chat_id, amount, metadata) VALUES \
            ('memory_compaction_settlement', 8300000000001, NULL, 0, \
             '{\"usage_tag\":\"memory_compaction:synthetic:user\"}'), \
            ('ai_settlement_result', 8300000000001, NULL, 0, \
             '{\"operation_id\":\"synthetic-user-operation\",\"settled_credit_units\":0}'), \
            ('ai_refund', 8300000000001, NULL, 25, \
             '{\"source\":\"user\",\"operation_id\":\"synthetic-user-operation\",\"usage_tag\":\"memory_compaction:synthetic:user\",\"reason\":\"unused_stale_reservation\"}'), \
            ('memory_compaction_settlement', 8300000000002, -8300000000002, 0, \
             '{\"usage_tag\":\"memory_compaction:synthetic:chat\"}'), \
            ('ai_settlement_result', 8300000000002, -8300000000002, 0, \
             '{\"operation_id\":\"synthetic-chat-operation\",\"settled_credit_units\":0}'), \
            ('ai_refund', 8300000000002, -8300000000002, 30, \
             '{\"source\":\"chat\",\"operation_id\":\"synthetic-chat-operation\",\"usage_tag\":\"memory_compaction:synthetic:chat\",\"reason\":\"unused_stale_reservation\"}');",
    )?;

    let result = BillingSchemaRepository::new(&database_url).ensure_schema()?;
    assert_eq!(result.repaired_compaction_refunds, 2);
    let balances = client.query(
        "SELECT scope_type, scope_id, balance FROM credit_accounts \
         WHERE scope_id IN (8300000000001, -8300000000002) ORDER BY scope_type",
        &[],
    )?;
    assert_eq!(balances.len(), 2);
    assert_eq!(balances[0].get::<_, i32>(2), 470);
    assert_eq!(balances[1].get::<_, i32>(2), 475);
    Ok(())
}

#[test]
fn public_network_entrypoints_validate_requests_before_provider_io() {
    let request = ChatCompletionRequest::new("synthetic-model", Vec::new());
    assert_eq!(
        openrouter_chat::complete("", &request),
        Err(OpenRouterChatError::MissingApiKey)
    );
    let mut streaming = request.clone();
    streaming.stream = true;
    assert_eq!(
        openrouter_chat::stream("", &streaming, |_| Ok(())),
        Err(OpenRouterChatError::MissingApiKey)
    );

    assert_eq!(
        telegram_http::request("synthetic-token", "sendMessage", "POST", None, None, 0),
        Err(TelegramHttpError::InvalidTimeout)
    );
    assert_eq!(
        telegram_http::multipart_request(MultipartUpload {
            token: "synthetic-token".to_owned(),
            endpoint: "sendPhoto".to_owned(),
            data_payload: json!({}),
            file_field: "photo".to_owned(),
            file_name: "synthetic.png".to_owned(),
            file_bytes: vec![1, 2, 3],
            content_type: "image/png".to_owned(),
            timeout: Duration::ZERO,
        }),
        Err(TelegramHttpError::InvalidTimeout)
    );
    assert_eq!(
        telegram_http::download_file("synthetic-token", "synthetic-file", 0),
        Err(TelegramHttpError::InvalidTimeout)
    );

    let inspection = link_preview::inspect("not a URL");
    assert!(!inspection.embeddable);
    assert!(inspection.failure.is_some());

    let addresses = SystemHostResolver.addresses("localhost", 80);
    assert!(addresses.is_ok_and(|values| !values.is_empty()));
}
