//! Typed Polymarket global-election, market-search and live-midpoint adapter.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Duration;

use bot_core::polymarket::{
    ElectionEvent, SearchEvent, parse_election_events, parse_search_events, provider_number,
};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_request_cache_key,
};

const EVENTS_URL: &str = "https://gamma-api.polymarket.com/events";
const MIDPOINTS_URL: &str = "https://clob.polymarket.com/midpoints";
const SEARCH_URL: &str = "https://gamma-api.polymarket.com/public-search";
/// Events returned per search. Each one can carry hundreds of markets.
const SEARCH_EVENT_LIMIT: &str = "10";
const CACHE_TTL_SECONDS: i64 = 60;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MidpointsRequest {
    pub token_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

pub use crate::http_client::TransportFailureKind;
use crate::http_client::classify_error;

pub trait PolymarketTransport {
    fn events(&self) -> Result<HttpResponse, TransportFailureKind>;

    fn midpoints(&self, request: &MidpointsRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub trait PolymarketSearchTransport {
    fn search(&self, query: &str) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub struct ReqwestPolymarketTransport {
    client: Client,
    events_url: String,
    midpoints_url: String,
    search_url: String,
}

impl ReqwestPolymarketTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(REQUEST_TIMEOUT).build()
        })
        .map(|client| Self {
            client,
            events_url: EVENTS_URL.to_owned(),
            midpoints_url: MIDPOINTS_URL.to_owned(),
            search_url: SEARCH_URL.to_owned(),
        })
        .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_urls(events_url: &str, midpoints_url: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.events_url = events_url.to_owned();
            transport.midpoints_url = midpoints_url.to_owned();
            transport
        })
    }

    #[cfg(test)]
    fn with_search_url(search_url: &str) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.search_url = search_url.to_owned();
            transport
        })
    }
}

impl PolymarketTransport for ReqwestPolymarketTransport {
    fn events(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(&self.events_url)
            .query(&[
                ("limit", "10"),
                ("active", "true"),
                ("closed", "false"),
                ("tag_slug", "global-elections"),
                ("order", "liquidity"),
                ("ascending", "false"),
            ])
            .send()
            .map_err(classify_error)?;
        response_body(response)
    }

    fn midpoints(&self, request: &MidpointsRequest) -> Result<HttpResponse, TransportFailureKind> {
        let payload = request
            .token_ids
            .iter()
            .map(|token_id| json!({"token_id": token_id}))
            .collect::<Vec<_>>();
        let response = self
            .client
            .post(&self.midpoints_url)
            .json(&payload)
            .send()
            .map_err(classify_error)?;
        response_body(response)
    }
}

impl PolymarketSearchTransport for ReqwestPolymarketTransport {
    fn search(&self, query: &str) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(&self.search_url)
            .query(&[
                ("q", query),
                ("limit_per_type", SEARCH_EVENT_LIMIT),
                ("events_status", "active"),
            ])
            .send()
            .map_err(classify_error)?;
        response_body(response)
    }
}

fn response_body(
    response: reqwest::blocking::Response,
) -> Result<HttpResponse, TransportFailureKind> {
    let status_code = response.status().as_u16();
    response
        .text()
        .map(|body| HttpResponse { status_code, body })
        .map_err(classify_error)
}

fn events_cache_key() -> String {
    python_request_cache_key(concat!(
        "{\"api_url\": \"https://gamma-api.polymarket.com/events\", ",
        "\"headers\": null, \"parameters\": {\"active\": \"true\", ",
        "\"ascending\": \"false\", \"closed\": \"false\", \"limit\": 10, ",
        "\"order\": \"liquidity\", \"tag_slug\": \"global-elections\"}}"
    ))
}

fn search_cache_key(query: &str) -> String {
    python_request_cache_key(&json!({"api_url": SEARCH_URL, "q": query}).to_string())
}

/// Search responses run to megabytes because sports events carry hundreds of
/// props. Cache the parsed events instead of the raw body.
fn slim_search_body(body: String) -> String {
    serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| serde_json::to_string(&parse_search_events(&value)).ok())
        .unwrap_or(body)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarketSearchLoad {
    /// `None` when Polymarket could not be reached and nothing was cached.
    pub events: Option<Vec<SearchEvent>>,
    /// How old `events` are when Polymarket failed and an expired cache
    /// entry stood in. `None` for fresh data.
    pub stale_seconds: Option<i64>,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn search_markets<T: PolymarketSearchTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    query: &str,
    now_unix: i64,
) -> MarketSearchLoad {
    let load = load_cached_json(
        cache,
        &search_cache_key(query),
        CACHE_TTL_SECONDS,
        now_unix,
        "Polymarket market search request",
        || {
            transport
                .search(query)
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: if response.status_code < 400 {
                        slim_search_body(response.body)
                    } else {
                        response.body
                    },
                })
                .map_err(|error| format!("transport {error:?}"))
        },
        || transport.before_retry(),
    );
    let mut diagnostics = load.diagnostics;
    let stale_seconds = load
        .fetched_at
        .map(|fetched_at| now_unix.saturating_sub(fetched_at))
        .filter(|age| *age > CACHE_TTL_SECONDS);
    let events = load.data.and_then(|data| {
        serde_json::from_value::<Vec<SearchEvent>>(data)
            .map_err(|error| {
                diagnostics.push(format!(
                    "Polymarket search returned an unusable payload: {error}"
                ));
            })
            .ok()
    });
    MarketSearchLoad {
        events,
        stale_seconds,
        diagnostics,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectionsLoad {
    pub events: Vec<ElectionEvent>,
    pub live_prices: HashMap<String, f64>,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn load_elections<T: PolymarketTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
) -> ElectionsLoad {
    let load = load_cached_json(
        cache,
        &events_cache_key(),
        CACHE_TTL_SECONDS,
        now_unix,
        "Polymarket global elections request",
        || {
            transport
                .events()
                .map(|response| JsonHttpResponse {
                    status_code: response.status_code,
                    body: response.body,
                })
                .map_err(|error| format!("transport {error:?}"))
        },
        || transport.before_retry(),
    );
    let mut diagnostics = load.diagnostics;
    let events = load
        .data
        .as_ref()
        .map(parse_election_events)
        .unwrap_or_default();
    if events.is_empty() {
        diagnostics.push("Polymarket returned no usable global elections".to_owned());
        return ElectionsLoad {
            events,
            live_prices: HashMap::new(),
            diagnostics,
        };
    }
    let mut seen = HashSet::new();
    let token_ids = events
        .iter()
        .flat_map(|event| &event.quotes)
        .filter_map(|quote| quote.token_id.clone())
        .filter(|token_id| seen.insert(token_id.clone()))
        .collect::<Vec<_>>();
    if token_ids.is_empty() {
        return ElectionsLoad {
            events,
            live_prices: HashMap::new(),
            diagnostics,
        };
    }
    let request = MidpointsRequest { token_ids };
    let live_prices = match transport.midpoints(&request) {
        Ok(response) if response.status_code < 400 => {
            match serde_json::from_str::<Value>(&response.body) {
                Ok(Value::Object(values)) => values
                    .into_iter()
                    .filter_map(|(token_id, value)| {
                        provider_number(&value).map(|price| (token_id, price))
                    })
                    .collect(),
                Ok(_) => {
                    diagnostics
                        .push("Polymarket midpoints returned a non-object payload".to_owned());
                    HashMap::new()
                }
                Err(error) => {
                    diagnostics.push(format!(
                        "Polymarket midpoints returned invalid JSON: {error}"
                    ));
                    HashMap::new()
                }
            }
        }
        Ok(response) => {
            diagnostics.push(format!(
                "Polymarket midpoints returned HTTP {}",
                response.status_code
            ));
            HashMap::new()
        }
        Err(error) => {
            diagnostics.push(format!("Polymarket midpoints transport failed: {error:?}"));
            HashMap::new()
        }
    };
    ElectionsLoad {
        events,
        live_prices,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        HttpResponse, MidpointsRequest, PolymarketSearchTransport, PolymarketTransport,
        ReqwestPolymarketTransport, TransportFailureKind, events_cache_key, load_elections,
        search_cache_key, search_markets,
    };
    use crate::request_cache::RequestCache;

    struct Transport {
        events: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        midpoints: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<MidpointsRequest>>,
    }

    impl PolymarketTransport for Transport {
        fn events(&self) -> Result<HttpResponse, TransportFailureKind> {
            self.events
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }

        fn midpoints(
            &self,
            request: &MidpointsRequest,
        ) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.midpoints
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[derive(Default)]
    struct Cache;

    impl RequestCache for Cache {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn set(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
            Ok(())
        }

        fn take(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn claim(
            &mut self,
            _key: &str,
            _value: &str,
            _ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            Ok(true)
        }
    }

    fn response(body: &str) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: body.to_owned(),
        })
    }

    #[test]
    fn cache_take_matches_empty_stub_contract() {
        let mut cache = Cache;
        assert_eq!(cache.take("synthetic"), Ok(None));
        assert_eq!(cache.claim("synthetic", "1", 60), Ok(true));
    }

    fn events() -> &'static str {
        r#"[{"title":"Election","slug":"election","liquidity":1000,"markets":[{"groupItemTitle":"A","outcomes":["Yes","No"],"outcomePrices":[0.4,0.6],"clobTokenIds":["a","a-no"]},{"groupItemTitle":"B","outcomes":["Yes","No"],"outcomePrices":[0.6,0.4],"clobTokenIds":["b","b-no"]}]}]"#
    }

    #[test]
    fn event_request_key_matches_python_and_batch_midpoints_are_deduplicated() {
        assert_eq!(
            events_cache_key(),
            "request_cache:1a9eb5fc646dc3a204dd7f5fc897d5046a45819cf985471a850e4cc50d028dd4"
        );
        let transport = Transport {
            events: RefCell::new(VecDeque::from([response(events())])),
            midpoints: RefCell::new(VecDeque::from([response(
                r#"{"a":"0.72","b":"invalid","a-no":"NaN"}"#,
            )])),
            requests: RefCell::default(),
        };
        let load = load_elections(&transport, &mut Cache, 100);
        assert_eq!(load.events.len(), 1);
        assert_eq!(load.live_prices.get("a"), Some(&0.72));
        assert_eq!(load.live_prices.len(), 1);
        assert_eq!(transport.requests.borrow()[0].token_ids, ["a", "b"]);
        assert!(load.diagnostics.is_empty());
    }

    #[test]
    fn event_and_midpoint_failures_fall_back_without_panics() {
        let transport = Transport {
            events: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            midpoints: RefCell::default(),
            requests: RefCell::default(),
        };
        let load = load_elections(&transport, &mut Cache, 100);
        assert!(load.events.is_empty());
        assert_eq!(load.diagnostics.len(), 3);

        let transport = Transport {
            events: RefCell::new(VecDeque::from([response(events())])),
            midpoints: RefCell::new(VecDeque::from([Err(TransportFailureKind::Timeout)])),
            requests: RefCell::default(),
        };
        let load = load_elections(&transport, &mut Cache, 100);
        assert_eq!(load.events.len(), 1);
        assert!(load.live_prices.is_empty());
        assert_eq!(load.diagnostics.len(), 1);
    }

    #[test]
    fn midpoint_payload_variants_and_tokenless_events_are_explicit() {
        let tokenless = r#"[{"title":"Election","slug":"election","markets":[{"groupItemTitle":"A","outcomes":["Yes","No"],"outcomePrices":[0.4,0.6]}]}]"#;
        let transport = Transport {
            events: RefCell::new(VecDeque::from([response(tokenless)])),
            midpoints: RefCell::default(),
            requests: RefCell::default(),
        };
        let load = load_elections(&transport, &mut Cache, 100);
        assert_eq!(load.events.len(), 1);
        assert!(load.live_prices.is_empty());
        assert!(transport.requests.borrow().is_empty());
        assert_eq!(
            bot_core::polymarket::provider_number(&serde_json::json!(0.5)),
            Some(0.5)
        );
        assert_eq!(
            bot_core::polymarket::provider_number(&serde_json::json!(true)),
            None
        );

        for (midpoint, expected) in [
            (response("[]"), "non-object payload"),
            (response("not-json"), "invalid JSON"),
            (
                Ok(HttpResponse {
                    status_code: 503,
                    body: String::new(),
                }),
                "HTTP 503",
            ),
        ] {
            let transport = Transport {
                events: RefCell::new(VecDeque::from([response(events())])),
                midpoints: RefCell::new(VecDeque::from([midpoint])),
                requests: RefCell::default(),
            };
            let load = load_elections(&transport, &mut Cache, 100);
            assert_eq!(load.events.len(), 1);
            assert!(load.live_prices.is_empty());
            assert!(
                load.diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.contains(expected))
            );
        }
    }

    #[test]
    fn reqwest_transport_preserves_event_query_and_midpoint_batch() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            for method in ["GET", "POST"] {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 8_192];
                let bytes = stream.read(&mut request).unwrap_or_default();
                let request = String::from_utf8_lossy(&request[..bytes]);
                if method == "GET" {
                    assert!(request.starts_with("GET /events?"), "{request}");
                    for parameter in [
                        "limit=10",
                        "active=true",
                        "closed=false",
                        "tag_slug=global-elections",
                        "order=liquidity",
                        "ascending=false",
                    ] {
                        assert!(request.contains(parameter), "{request}");
                    }
                } else {
                    assert!(request.starts_with("POST /midpoints HTTP/1.1"), "{request}");
                    assert!(
                        request
                            .contains(r#"[{"token_id":"synthetic-a"},{"token_id":"synthetic-b"}]"#)
                    );
                }
                let body = r#"{"synthetic":true}"#;
                let written = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                written?;
            }
            Ok(())
        });
        let base = format!("http://{address}");
        let transport = ReqwestPolymarketTransport::with_urls(
            &format!("{base}/events"),
            &format!("{base}/midpoints"),
        )
        .ok()
        .ok_or("unexpected error")?;
        assert!(transport.events().is_ok());
        assert!(
            transport
                .midpoints(&MidpointsRequest {
                    token_ids: vec!["synthetic-a".to_owned(), "synthetic-b".to_owned()],
                })
                .is_ok()
        );
        assert!(matches!(server.join(), Ok(Ok(()))));
        let unavailable = ReqwestPolymarketTransport::with_urls(
            "http://127.0.0.1:1/events",
            "http://127.0.0.1:1/midpoints",
        )
        .ok()
        .ok_or("unexpected error")?;
        assert!(unavailable.events().is_err());
        Ok(())
    }

    struct SearchTransport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        queries: RefCell<Vec<String>>,
        retries: RefCell<usize>,
    }

    impl SearchTransport {
        fn new(responses: Vec<Result<HttpResponse, TransportFailureKind>>) -> Self {
            Self {
                responses: RefCell::new(VecDeque::from(responses)),
                queries: RefCell::default(),
                retries: RefCell::default(),
            }
        }
    }

    impl PolymarketSearchTransport for SearchTransport {
        fn search(&self, query: &str) -> Result<HttpResponse, TransportFailureKind> {
            self.queries.borrow_mut().push(query.to_owned());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }

        fn before_retry(&self) {
            *self.retries.borrow_mut() += 1;
        }
    }

    #[derive(Default)]
    struct SearchCache {
        stored: Option<String>,
        writes: Vec<(String, String, i64)>,
    }

    impl RequestCache for SearchCache {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.stored.clone())
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            self.stored = Some(value.to_owned());
            Ok(())
        }

        fn take(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(None)
        }

        fn claim(
            &mut self,
            _key: &str,
            _value: &str,
            _ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            Ok(true)
        }
    }

    fn search_body() -> String {
        serde_json::json!({"events": [{
            "title": "Synthetic vote", "slug": "synthetic-vote", "negRisk": true, "liquidity": 1000,
            "description": "long rules text that should not reach the cache",
            "markets": [
                {"groupItemTitle": "A", "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": "[\"0.3\", \"0.7\"]", "description": "more rules"},
                {"groupItemTitle": "B", "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": "[\"0.6\", \"0.4\"]"}
            ]
        }], "pagination": {"hasMore": false}})
        .to_string()
    }

    #[test]
    fn search_caches_slim_events_and_serves_them_until_they_expire() {
        let transport = SearchTransport::new(vec![response(&search_body())]);
        let mut cache = SearchCache::default();
        let load = search_markets(&transport, &mut cache, "synthetic vote", 100);
        let events = load.events.unwrap_or_default();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].markets[0].title, "B");
        assert!(load.diagnostics.is_empty());
        let (key, value, ttl) = &cache.writes[0];
        assert_eq!(key, &search_cache_key("synthetic vote"));
        assert_ne!(key, &search_cache_key("Synthetic vote"));
        assert_eq!(*ttl, 60);
        assert!(!value.contains("rules"), "{value}");
        assert!(!value.contains("pagination"), "{value}");

        assert_eq!(load.stale_seconds, None);

        let cached = search_markets(&transport, &mut cache, "synthetic vote", 160);
        assert_eq!(cached.events, Some(events.clone()));
        assert_eq!(cached.stale_seconds, None);
        assert_eq!(transport.queries.borrow().len(), 1);

        // Polymarket is down once the entry expires: the old events stand in,
        // labeled with their age.
        let stale = search_markets(&transport, &mut cache, "synthetic vote", 161);
        assert_eq!(stale.events, Some(events));
        assert_eq!(stale.stale_seconds, Some(61));
    }

    #[test]
    fn search_separates_no_results_from_unreachable_polymarket() {
        let transport = SearchTransport::new(vec![response(r#"{"events": []}"#)]);
        let load = search_markets(&transport, &mut SearchCache::default(), "nothing", 100);
        assert_eq!(load.events, Some(Vec::new()));

        let transport = SearchTransport::new(vec![
            Err(TransportFailureKind::Timeout),
            Ok(HttpResponse {
                status_code: 503,
                body: "<html>unavailable</html>".to_owned(),
            }),
        ]);
        let load = search_markets(&transport, &mut SearchCache::default(), "down", 100);
        assert_eq!(load.events, None);
        assert_eq!(*transport.retries.borrow(), 1);
        assert_eq!(load.diagnostics.len(), 2);
        assert!(
            load.diagnostics[1].contains("HTTP 503"),
            "{:?}",
            load.diagnostics
        );

        let transport = SearchTransport::new(vec![response("not json"), response("[1")]);
        let load = search_markets(&transport, &mut SearchCache::default(), "garbled", 100);
        assert_eq!(load.events, None);
        assert!(
            load.diagnostics[0].contains("invalid JSON"),
            "{:?}",
            load.diagnostics
        );
    }

    #[test]
    fn search_reports_cached_payloads_it_cannot_read() {
        let mut cache = SearchCache {
            stored: Some(r#"{"timestamp": 100, "data": {"events": []}}"#.to_owned()),
            ..SearchCache::default()
        };
        let transport = SearchTransport::new(Vec::new());
        let load = search_markets(&transport, &mut cache, "old shape", 120);
        assert_eq!(load.events, None);
        assert!(
            load.diagnostics[0].contains("unusable payload"),
            "{:?}",
            load.diagnostics
        );
        assert!(transport.queries.borrow().is_empty());
        assert_eq!(cache.take("synthetic"), Ok(None));
        assert_eq!(cache.claim("synthetic", "1", 60), Ok(true));
    }

    #[test]
    fn reqwest_search_sends_the_query_and_active_event_filters() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 8_192];
            let bytes = stream.read(&mut request).unwrap_or_default();
            let request = String::from_utf8_lossy(&request[..bytes]);
            assert!(request.starts_with("GET /public-search?"), "{request}");
            for parameter in [
                "q=fed+rates+%26+cuts",
                "limit_per_type=10",
                "events_status=active",
            ] {
                assert!(request.contains(parameter), "{request}");
            }
            let body = r#"{"events":[]}"#;
            let written = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            written?;
            Ok(())
        });
        let transport =
            ReqwestPolymarketTransport::with_search_url(&format!("http://{address}/public-search"))
                .ok()
                .ok_or("unexpected error")?;
        let response = transport
            .search("fed rates & cuts")
            .ok()
            .ok_or("search failed")?;
        assert_eq!(response.body, r#"{"events":[]}"#);
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }
}
