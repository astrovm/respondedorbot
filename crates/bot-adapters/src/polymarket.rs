//! Typed Polymarket global-election and live-midpoint adapter.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use bot_core::polymarket::{ElectionEvent, parse_election_events};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_request_cache_key,
};

const EVENTS_URL: &str = "https://gamma-api.polymarket.com/events";
const MIDPOINTS_URL: &str = "https://clob.polymarket.com/midpoints";
const CACHE_TTL_SECONDS: i64 = 5;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

pub trait PolymarketTransport {
    fn events(&self) -> Result<HttpResponse, TransportFailureKind>;

    fn midpoints(&self, request: &MidpointsRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub struct ReqwestPolymarketTransport {
    client: Client,
}

impl ReqwestPolymarketTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl PolymarketTransport for ReqwestPolymarketTransport {
    fn events(&self) -> Result<HttpResponse, TransportFailureKind> {
        let response = self
            .client
            .get(EVENTS_URL)
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
            .post(MIDPOINTS_URL)
            .json(&payload)
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

fn classify_error(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

fn events_cache_key() -> String {
    python_request_cache_key(concat!(
        "{\"api_url\": \"https://gamma-api.polymarket.com/events\", ",
        "\"headers\": null, \"parameters\": {\"active\": \"true\", ",
        "\"ascending\": \"false\", \"closed\": \"false\", \"limit\": 10, ",
        "\"order\": \"liquidity\", \"tag_slug\": \"global-elections\"}}"
    ))
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElectionsLoad {
    pub events: Vec<ElectionEvent>,
    pub live_prices: HashMap<String, f64>,
    pub diagnostics: Vec<String>,
}

fn midpoint_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
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
                        midpoint_number(&value).map(|price| (token_id, price))
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
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use super::{
        HttpResponse, MidpointsRequest, PolymarketTransport, TransportFailureKind,
        events_cache_key, load_elections,
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
    }

    fn response(body: &str) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: body.to_owned(),
        })
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
            midpoints: RefCell::new(VecDeque::from([response(r#"{"a":"0.72","b":"invalid"}"#)])),
            requests: RefCell::default(),
        };
        let load = load_elections(&transport, &mut Cache, 100);
        assert_eq!(load.events.len(), 1);
        assert_eq!(load.live_prices.get("a"), Some(&0.72));
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
}
