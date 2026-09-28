//! Blocking Hacker News RSS adapter with Python-compatible stale caching.

use std::time::Duration;

use bot_core::hacker_news::{HackerNewsItem, normalize_feed_item};
use quick_xml::Reader;
use quick_xml::events::Event;
use reqwest::blocking::Client;
use std::sync::OnceLock;
use thiserror::Error;

use crate::redis_json_cache::RedisJsonCache;

pub const PRIMARY_URL: &str = "https://hnrss.org/best";
pub const FALLBACK_URL: &str = "https://news.ycombinator.com/rss";
pub const CACHE_KEY: &str = "context:hacker_news:best";
pub const CACHE_TTL_SECONDS: i64 = 600;
pub const MAX_ITEMS: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HackerNewsResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HackerNewsTransportError {
    #[error("Hacker News request timed out")]
    Timeout,
    #[error("Hacker News connection failed")]
    Connection,
    #[error("Hacker News request failed: {0}")]
    Other(String),
}

pub trait HackerNewsTransport {
    fn get(&self, url: &str) -> Result<HackerNewsResponse, HackerNewsTransportError>;
}

pub trait HackerNewsCache {
    fn get(&mut self, key: &str) -> Result<Option<String>, String>;
    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), String>;
}

impl HackerNewsCache for RedisJsonCache {
    fn get(&mut self, key: &str) -> Result<Option<String>, String> {
        RedisJsonCache::get(self, key).map_err(|error| error.to_string())
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), String> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds))
            .map(|_written| ())
            .map_err(|error| error.to_string())
    }
}

pub struct ReqwestHackerNewsTransport {
    client: Client,
}

impl ReqwestHackerNewsTransport {
    pub fn new() -> Result<Self, HackerNewsTransportError> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(Duration::from_secs(10)).build()
        })
        .map(|client| Self { client })
        .map_err(classify_error)
    }
}

impl HackerNewsTransport for ReqwestHackerNewsTransport {
    fn get(&self, url: &str) -> Result<HackerNewsResponse, HackerNewsTransportError> {
        let response = self.client.get(url).send().map_err(classify_error)?;
        Ok(HackerNewsResponse {
            status_code: response.status().as_u16(),
            body: response
                .text()
                .map_err(|error| HackerNewsTransportError::Other(error.to_string()))?,
        })
    }
}

fn classify_error(error: reqwest::Error) -> HackerNewsTransportError {
    if error.is_timeout() {
        HackerNewsTransportError::Timeout
    } else if error.is_connect() {
        HackerNewsTransportError::Connection
    } else {
        HackerNewsTransportError::Other(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HackerNewsLoad {
    pub items: Vec<HackerNewsItem>,
    pub diagnostics: Vec<String>,
}

pub fn load_hacker_news<T: HackerNewsTransport, C: HackerNewsCache>(
    transport: &T,
    cache: &mut C,
    limit: usize,
) -> HackerNewsLoad {
    let limit = limit.clamp(1, MAX_ITEMS);
    let mut diagnostics = Vec::new();
    let cached = read_cache(cache, &mut diagnostics);
    if cached.as_ref().is_some_and(|items| !items.is_empty()) {
        return HackerNewsLoad {
            items: cached.unwrap_or_default().into_iter().take(limit).collect(),
            diagnostics,
        };
    }

    let response = fetch(transport, &mut diagnostics);
    let Some(response) = response else {
        return HackerNewsLoad {
            items: cached.unwrap_or_default().into_iter().take(limit).collect(),
            diagnostics,
        };
    };
    match parse_feed(&response.body, MAX_ITEMS) {
        Ok(items) if !items.is_empty() => {
            // Plain string/number items always serialize.
            if let Ok(payload) = serde_json::to_string(&items)
                && let Err(error) = cache.set(CACHE_KEY, &payload, CACHE_TTL_SECONDS)
            {
                diagnostics.push(format!("could not write Hacker News cache: {error}"));
            }
            HackerNewsLoad {
                items: items.into_iter().take(limit).collect(),
                diagnostics,
            }
        }
        Ok(_) => HackerNewsLoad {
            items: cached.unwrap_or_default().into_iter().take(limit).collect(),
            diagnostics,
        },
        Err(error) => {
            diagnostics.push(format!("could not parse Hacker News RSS: {error}"));
            HackerNewsLoad {
                items: cached.unwrap_or_default().into_iter().take(limit).collect(),
                diagnostics,
            }
        }
    }
}

fn read_cache<C: HackerNewsCache>(
    cache: &mut C,
    diagnostics: &mut Vec<String>,
) -> Option<Vec<HackerNewsItem>> {
    match cache.get(CACHE_KEY) {
        Ok(Some(payload)) => match serde_json::from_str(&payload) {
            Ok(items) => Some(items),
            Err(error) => {
                diagnostics.push(format!("invalid Hacker News cache: {error}"));
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            diagnostics.push(format!("could not read Hacker News cache: {error}"));
            None
        }
    }
}

fn fetch<T: HackerNewsTransport>(
    transport: &T,
    diagnostics: &mut Vec<String>,
) -> Option<HackerNewsResponse> {
    for url in [PRIMARY_URL, FALLBACK_URL] {
        match transport.get(url) {
            Ok(response) if (200..300).contains(&response.status_code) => return Some(response),
            Ok(response) => diagnostics.push(format!(
                "Hacker News RSS {url} returned HTTP {}",
                response.status_code
            )),
            Err(error) => diagnostics.push(format!("Hacker News RSS {url}: {error}")),
        }
    }
    None
}

pub fn parse_feed(body: &str, max_items: usize) -> Result<Vec<HackerNewsItem>, String> {
    let mut reader = Reader::from_str(body);
    reader.config_mut().trim_text(false);
    let mut items = Vec::new();
    let mut in_item = false;
    let mut field = None;
    let mut title = String::new();
    let mut link = String::new();
    let mut description = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(start)) => match start.local_name().as_ref() {
                "item" => {
                    in_item = true;
                    title.clear();
                    link.clear();
                    description.clear();
                }
                "title" | "link" | "description" if in_item => {
                    field = Some(start.local_name().as_ref().to_owned());
                }
                _ => {}
            },
            Ok(Event::Text(text)) if in_item => {
                // The reader reports entity and character references as separate
                // `GeneralRef` events, so text events carry no escapes.
                let decoded: &str = &text;
                append_field(
                    field.as_deref(),
                    decoded,
                    &mut title,
                    &mut link,
                    &mut description,
                );
            }
            Ok(Event::CData(text)) if in_item => {
                append_field(
                    field.as_deref(),
                    &text,
                    &mut title,
                    &mut link,
                    &mut description,
                );
            }
            Ok(Event::GeneralRef(reference)) if in_item => {
                let value = if let Some(character) = reference
                    .resolve_char_ref()
                    .map_err(|error| error.to_string())?
                {
                    character.to_string()
                } else {
                    let name = reference.as_ref();
                    quick_xml::escape::resolve_predefined_entity(name)
                        .ok_or_else(|| format!("unknown XML entity: &{name};"))?
                        .to_owned()
                };
                append_field(
                    field.as_deref(),
                    &value,
                    &mut title,
                    &mut link,
                    &mut description,
                );
            }
            Ok(Event::End(end)) => match end.local_name().as_ref() {
                "item" if in_item => {
                    if let Some(item) = normalize_feed_item(&title, &link, &description)
                        .map_err(|error| error.to_string())?
                    {
                        items.push(item);
                    }
                    in_item = false;
                    field = None;
                    if items.len() >= max_items {
                        break;
                    }
                }
                "title" | "link" | "description" => field = None,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(items)
}

fn append_field(
    field: Option<&str>,
    value: &str,
    title: &mut String,
    link: &mut String,
    description: &mut String,
) {
    match field {
        Some("title") => title.push_str(value),
        Some("link") => link.push_str(value),
        Some("description") => description.push_str(value),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    struct Transport {
        responses: RefCell<Vec<Result<HackerNewsResponse, HackerNewsTransportError>>>,
        urls: RefCell<Vec<String>>,
    }

    impl HackerNewsTransport for Transport {
        fn get(&self, url: &str) -> Result<HackerNewsResponse, HackerNewsTransportError> {
            self.urls.borrow_mut().push(url.to_owned());
            self.responses.borrow_mut().remove(0)
        }
    }

    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
        writes: Vec<(String, String, i64)>,
        read_error: Option<String>,
        write_error: Option<String>,
    }

    impl HackerNewsCache for Cache {
        fn get(&mut self, key: &str) -> Result<Option<String>, String> {
            if let Some(error) = &self.read_error {
                return Err(error.clone());
            }
            Ok(self.values.get(key).cloned())
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), String> {
            if let Some(error) = &self.write_error {
                return Err(error.clone());
            }
            self.writes
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    const FEED: &str = r#"<?xml version="1.0"?><rss><channel>
      <item><title>Synthetic &amp; safe</title><link>https://example.test/story</link>
      <description><![CDATA[Points: 12<br># Comments: 3<br>Comments URL: <a href="https://news.ycombinator.com/item?id=1">comments</a>]]></description></item>
      <item><title>Second</title><link>https://example.test/second</link><description>Points: 2</description></item>
    </channel></rss>"#;

    #[test]
    fn parses_entities_cdata_metadata_and_bounds_items() {
        let items = parse_feed(FEED, 1).unwrap_or_default();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "Synthetic & safe");
        assert_eq!(items[0].points, Some(12));
        assert_eq!(items[0].comments, Some(3));
        assert_eq!(
            items[0].comments_url,
            "https://news.ycombinator.com/item?id=1"
        );
    }

    #[test]
    fn reads_compatible_cache_without_http_and_honors_limit() {
        let cached = parse_feed(FEED, 10).unwrap_or_default();
        let mut cache = Cache::default();
        cache.values.insert(
            CACHE_KEY.to_owned(),
            serde_json::to_string(&cached).unwrap_or_default(),
        );
        let transport = Transport {
            responses: RefCell::new(Vec::new()),
            urls: RefCell::new(Vec::new()),
        };
        let load = load_hacker_news(&transport, &mut cache, 1);
        assert_eq!(load.items.len(), 1);
        assert!(transport.urls.borrow().is_empty());
    }

    #[test]
    fn reqwest_transport_reads_status_and_body_from_an_injected_url() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 1_024];
            let bytes = stream.read(&mut request).unwrap_or_default();
            assert!(String::from_utf8_lossy(&request[..bytes]).starts_with("GET /feed HTTP/1.1"));
            let body = "synthetic feed";
            let written = write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            written?;
            Ok(())
        });
        let transport = ReqwestHackerNewsTransport::new()?;
        let response = transport.get(&format!("http://{address}/feed"))?;
        assert_eq!(response.status_code, 206);
        assert_eq!(response.body, "synthetic feed");
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn falls_back_then_caches_with_the_existing_key_and_ttl() {
        let transport = Transport {
            responses: RefCell::new(vec![
                Ok(HackerNewsResponse {
                    status_code: 503,
                    body: String::new(),
                }),
                Ok(HackerNewsResponse {
                    status_code: 200,
                    body: FEED.to_owned(),
                }),
            ]),
            urls: RefCell::new(Vec::new()),
        };
        let mut cache = Cache::default();
        let load = load_hacker_news(&transport, &mut cache, 10);
        assert_eq!(load.items.len(), 2);
        assert_eq!(*transport.urls.borrow(), [PRIMARY_URL, FALLBACK_URL]);
        assert_eq!(cache.writes.len(), 1);
        assert_eq!(cache.writes[0].0, CACHE_KEY);
        assert_eq!(cache.writes[0].2, CACHE_TTL_SECONDS);
        assert!(load.diagnostics[0].contains("HTTP 503"));
    }

    #[test]
    fn invalid_cache_and_network_or_xml_failures_are_diagnostic_and_safe() {
        let transport = Transport {
            responses: RefCell::new(vec![
                Err(HackerNewsTransportError::Connection),
                Ok(HackerNewsResponse {
                    status_code: 200,
                    body: "<broken".to_owned(),
                }),
            ]),
            urls: RefCell::new(Vec::new()),
        };
        let mut cache = Cache::default();
        cache
            .values
            .insert(CACHE_KEY.to_owned(), "not json".to_owned());
        let load = load_hacker_news(&transport, &mut cache, 5);
        assert!(load.items.is_empty());
        assert_eq!(load.diagnostics.len(), 3);
        assert!(load.diagnostics[0].contains("invalid Hacker News cache"));
        assert!(load.diagnostics[1].contains("connection failed"));
        assert!(load.diagnostics[2].contains("could not parse"));
    }

    #[test]
    fn cache_io_and_empty_feed_results_remain_nonfatal() {
        let transport = Transport {
            responses: RefCell::new(vec![Ok(HackerNewsResponse {
                status_code: 200,
                body: "<rss><channel></channel></rss>".to_owned(),
            })]),
            urls: RefCell::new(Vec::new()),
        };
        let mut cache = Cache {
            read_error: Some("synthetic cache read failure".to_owned()),
            ..Cache::default()
        };
        let load = load_hacker_news(&transport, &mut cache, 0);
        assert!(load.items.is_empty());
        assert!(load.diagnostics[0].contains("synthetic cache read failure"));

        let transport = Transport {
            responses: RefCell::new(vec![Ok(HackerNewsResponse {
                status_code: 200,
                body: FEED.to_owned(),
            })]),
            urls: RefCell::new(Vec::new()),
        };
        let mut cache = Cache {
            write_error: Some("synthetic cache write failure".to_owned()),
            ..Cache::default()
        };
        let load = load_hacker_news(&transport, &mut cache, MAX_ITEMS + 1);
        assert_eq!(load.items.len(), 2);
        assert!(
            load.diagnostics
                .iter()
                .any(|item| item.contains("synthetic cache write failure"))
        );

        assert!(parse_feed("<rss><item><title>&unknown;</title></item></rss>", 1).is_err());
        assert!(parse_feed("<rss><item><title>&#xZZ;</title></item></rss>", 1).is_err());
        assert_eq!(
            parse_feed(
                "<rss><item><title>t</title><link>https://example.test/a</link><description>Points: 99999999999999999999</description></item></rss>",
                1
            ),
            Err("feed metadata integer is outside the supported range".to_owned())
        );
        assert_eq!(
            parse_feed(
                "<rss><item><title>Caf&#233; &#x41;PI</title><link>https://example.test/a</link></item></rss>",
                1
            )
            .map(|items| items.into_iter().map(|item| item.title).collect::<Vec<_>>()),
            Ok(vec!["Caf\u{e9} API".to_owned()])
        );
        assert!(
            parse_feed("<rss><item><title></title></item></rss>", 1)
                .is_ok_and(|items| items.is_empty())
        );
    }

    #[test]
    fn exhausted_feeds_keep_the_previous_cache_and_report_each_url() {
        let transport = Transport {
            responses: RefCell::new(vec![
                Ok(HackerNewsResponse {
                    status_code: 503,
                    body: String::new(),
                }),
                Err(HackerNewsTransportError::Timeout),
            ]),
            urls: RefCell::new(Vec::new()),
        };
        let mut cache = Cache::default();
        cache.values.insert(CACHE_KEY.to_owned(), "[]".to_owned());
        let load = load_hacker_news(&transport, &mut cache, 5);
        assert!(load.items.is_empty());
        assert_eq!(load.diagnostics.len(), 2);
        assert!(load.diagnostics[0].contains("returned HTTP 503"));
        assert!(load.diagnostics[1].contains("timed out"));
        assert_eq!(*transport.urls.borrow(), [PRIMARY_URL, FALLBACK_URL]);
        assert!(cache.writes.is_empty());
    }

    #[test]
    fn reqwest_transport_reports_truncated_feed_bodies()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(
            move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 1_024];
                let _ = stream.read(&mut request)?;
                let truncated =
                    b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n<rss>";
                stream.write_all(truncated)?;
                Ok(())
            },
        );
        let transport = ReqwestHackerNewsTransport::new()?;
        assert!(matches!(
            transport.get(&format!("http://{address}/rss")),
            Err(HackerNewsTransportError::Other(detail)) if !detail.is_empty()
        ));
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn redis_json_cache_implements_the_hacker_news_cache_contract()
    -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::redis_connection::{RedisEndpoint, test_support::read_command};
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let server = thread::spawn(
            move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
                for (expected, response) in [
                    (vec!["GET", CACHE_KEY], b"$2\r\n[]\r\n".as_slice()),
                    (vec!["SETEX", CACHE_KEY, "60", "[1]"], b"+OK\r\n".as_slice()),
                ] {
                    assert_eq!(read_command(&mut stream)?, expected);
                    stream.write_all(response)?;
                }
                Ok(())
            },
        );
        let endpoint = |port| RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            password: None,
        };
        let mut cache = RedisJsonCache::new(&endpoint(port))?;
        assert_eq!(
            HackerNewsCache::get(&mut cache, CACHE_KEY),
            Ok(Some("[]".to_owned()))
        );
        assert_eq!(
            HackerNewsCache::set(&mut cache, CACHE_KEY, "[1]", 60),
            Ok(())
        );
        assert!(
            HackerNewsCache::set(&mut cache, CACHE_KEY, "[1]", -1)
                .is_err_and(|error| error.contains("non-negative"))
        );
        assert!(server.join().is_ok_and(|result| result.is_ok()));

        let closed_port = TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port();
        let mut offline = RedisJsonCache::new(&endpoint(closed_port))?;
        assert!(
            HackerNewsCache::get(&mut offline, CACHE_KEY)
                .is_err_and(|error| error.contains("Redis JSON-cache operation failed"))
        );
        Ok(())
    }

    #[test]
    fn reqwest_failures_are_classified_by_cause() {
        use crate::web_fetch::reqwest_error_fixtures as fixtures;
        assert_eq!(
            fixtures::timeout().map(super::classify_error),
            Some(super::HackerNewsTransportError::Timeout)
        );
        assert_eq!(
            fixtures::connection().map(super::classify_error),
            Some(super::HackerNewsTransportError::Connection)
        );
        assert!(matches!(
            fixtures::request().map(super::classify_error),
            Some(super::HackerNewsTransportError::Other(detail)) if detail.contains("builder error")
        ));
    }
}
