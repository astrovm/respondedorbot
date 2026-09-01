//! Bounded public-web fetcher for AI tools with redirect-level SSRF checks.

use std::io::Read;
use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;

use reqwest::blocking::Client;
use reqwest::header::{CONTENT_TYPE, LOCATION, USER_AGENT};
use reqwest::redirect::Policy;
use serde_json::Value;
use thiserror::Error;
use url::{Host, Url};

pub const FETCH_MAX_BYTES: usize = 262_144;
pub const FETCH_MAX_CHARS: usize = 12_000;
pub const FETCH_MAX_REDIRECTS: usize = 5;

const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebFetchResponse {
    pub status_code: u16,
    pub content_type: String,
    pub location: Option<String>,
    pub body: Vec<u8>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WebFetchTransportError {
    #[error("request timed out")]
    Timeout,
    #[error("connection failed")]
    Connection,
    #[error("request failed: {0}")]
    Other(String),
}

pub trait WebFetchTransport {
    fn get(&self, url: &str) -> Result<WebFetchResponse, WebFetchTransportError>;
}

pub trait HostResolver {
    fn addresses(&self, hostname: &str, port: u16) -> Result<Vec<IpAddr>, String>;
}

pub struct SystemHostResolver;

impl HostResolver for SystemHostResolver {
    fn addresses(&self, hostname: &str, port: u16) -> Result<Vec<IpAddr>, String> {
        (hostname, port)
            .to_socket_addrs()
            .map(|addresses| addresses.map(|address| address.ip()).collect())
            .map_err(|error| error.to_string())
    }
}

pub struct ReqwestWebFetchTransport {
    client: Client,
}

impl ReqwestWebFetchTransport {
    pub fn new() -> Result<Self, WebFetchTransportError> {
        Client::builder()
            .timeout(Duration::from_secs(8))
            .redirect(Policy::none())
            .build()
            .map(|client| Self { client })
            .map_err(|error| WebFetchTransportError::Other(error.to_string()))
    }
}

impl WebFetchTransport for ReqwestWebFetchTransport {
    fn get(&self, url: &str) -> Result<WebFetchResponse, WebFetchTransportError> {
        let mut response = self
            .client
            .get(url)
            .header(USER_AGENT, BROWSER_USER_AGENT)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let location = response
            .headers()
            .get(LOCATION)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        let mut body = Vec::new();
        response
            .by_ref()
            .take(u64::try_from(FETCH_MAX_BYTES).unwrap_or(u64::MAX) + 1)
            .read_to_end(&mut body)
            .map_err(|error| WebFetchTransportError::Other(error.to_string()))?;
        let truncated = body.len() > FETCH_MAX_BYTES;
        body.truncate(FETCH_MAX_BYTES);
        Ok(WebFetchResponse {
            status_code,
            content_type,
            location,
            body,
            truncated,
        })
    }
}

fn classify_error(error: reqwest::Error) -> WebFetchTransportError {
    if error.is_timeout() {
        WebFetchTransportError::Timeout
    } else if error.is_connect() {
        WebFetchTransportError::Connection
    } else {
        WebFetchTransportError::Other(error.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebFetchContent {
    pub url: String,
    pub status_code: u16,
    pub content_type: String,
    pub title: Option<String>,
    pub canonical_url: Option<String>,
    pub content: String,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PublicFetchError {
    #[error("url no permitida")]
    Blocked { url: String },
    #[error("no se pudo obtener la url")]
    Request { url: String, detail: String },
}

impl PublicFetchError {
    #[must_use]
    pub fn url(&self) -> &str {
        match self {
            Self::Blocked { url } | Self::Request { url, .. } => url,
        }
    }

    #[must_use]
    pub const fn public_message(&self) -> &'static str {
        match self {
            Self::Blocked { .. } => "url no permitida",
            Self::Request { .. } => "no se pudo obtener la url",
        }
    }
}

pub fn fetch_public_url<T: WebFetchTransport, R: HostResolver>(
    transport: &T,
    resolver: &R,
    raw_url: &str,
) -> Result<WebFetchContent, PublicFetchError> {
    let mut url = normalize_http_url(raw_url).ok_or_else(|| PublicFetchError::Blocked {
        url: raw_url.trim().to_owned(),
    })?;
    for redirect_count in 0..=FETCH_MAX_REDIRECTS {
        ensure_public(&url, resolver)?;
        let response = transport
            .get(url.as_str())
            .map_err(|error| PublicFetchError::Request {
                url: url.to_string(),
                detail: error.to_string(),
            })?;
        if (300..400).contains(&response.status_code) {
            let Some(location) = response.location.as_deref() else {
                return Err(PublicFetchError::Request {
                    url: url.to_string(),
                    detail: "redirect response omitted Location".to_owned(),
                });
            };
            if redirect_count == FETCH_MAX_REDIRECTS {
                return Err(PublicFetchError::Request {
                    url: url.to_string(),
                    detail: "too many redirects".to_owned(),
                });
            }
            url = url
                .join(location)
                .ok()
                .and_then(|value| normalize_http_url(value.as_str()))
                .ok_or_else(|| PublicFetchError::Blocked {
                    url: location.to_owned(),
                })?;
            continue;
        }
        if !(200..300).contains(&response.status_code) {
            return Err(PublicFetchError::Request {
                url: url.to_string(),
                detail: format!("HTTP {}", response.status_code),
            });
        }
        let decoded = String::from_utf8_lossy(&response.body);
        let is_html = response.content_type.to_ascii_lowercase().contains("html")
            || response
                .body
                .get(..response.body.len().min(400))
                .is_some_and(|prefix| {
                    String::from_utf8_lossy(prefix)
                        .to_ascii_lowercase()
                        .contains("<html")
                });
        let (title, canonical_url, content) = if is_html {
            let (title, content) = extract_text_from_html(&decoded);
            (
                title,
                extract_meta_content(&decoded, "og:url")
                    .and_then(|value| url.join(&value).ok())
                    .map(|value| value.to_string()),
                content,
            )
        } else {
            (
                None,
                None,
                collapse_whitespace(&decode_html_entities(&decoded)),
            )
        };
        let (content, text_truncated) = truncate_chars(&content, FETCH_MAX_CHARS);
        return Ok(WebFetchContent {
            url: url.to_string(),
            status_code: response.status_code,
            content_type: response.content_type,
            title,
            canonical_url,
            content,
            truncated: response.truncated || text_truncated,
        });
    }
    Err(PublicFetchError::Request {
        url: url.to_string(),
        detail: "too many redirects".to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TweetContent {
    pub url: String,
    pub author: String,
    pub date: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiFetchOutcome {
    Page(WebFetchContent),
    Tweet(TweetContent),
    TweetError { url: String },
}

pub fn fetch_ai_url<T: WebFetchTransport, R: HostResolver>(
    transport: &T,
    resolver: &R,
    raw_url: &str,
) -> Result<AiFetchOutcome, PublicFetchError> {
    if let Some(canonical_url) = canonical_tweet_url(raw_url) {
        return Ok(
            fetch_tweet_oembed(transport, resolver, &canonical_url).map_or_else(
                || AiFetchOutcome::TweetError { url: canonical_url },
                AiFetchOutcome::Tweet,
            ),
        );
    }
    let page = fetch_public_url(transport, resolver, raw_url)?;
    if is_twitter_frontend(raw_url)
        && let Some(canonical_url) = page.canonical_url.as_deref().and_then(canonical_tweet_url)
    {
        return Ok(
            fetch_tweet_oembed(transport, resolver, &canonical_url).map_or_else(
                || AiFetchOutcome::TweetError { url: canonical_url },
                AiFetchOutcome::Tweet,
            ),
        );
    }
    Ok(AiFetchOutcome::Page(page))
}

fn canonical_tweet_url(raw_url: &str) -> Option<String> {
    let url = normalize_http_url(raw_url)?;
    if !is_twitter_frontend(url.as_str()) {
        return None;
    }
    let segments = url.path_segments()?.collect::<Vec<_>>();
    let (username, status_id) = match segments.as_slice() {
        [username, "status", status_id, ..] => (*username, *status_id),
        _ => return None,
    };
    if username.is_empty()
        || !username
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        || status_id.is_empty()
        || !status_id
            .bytes()
            .all(|character| character.is_ascii_digit())
    {
        return None;
    }
    Some(format!("https://x.com/{username}/status/{status_id}"))
}

fn is_twitter_frontend(raw_url: &str) -> bool {
    normalize_http_url(raw_url)
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            matches!(
                host.trim_start_matches("www."),
                "twitter.com" | "x.com" | "fixupx.com" | "fxtwitter.com" | "xcancel.com"
            )
        })
}

fn fetch_tweet_oembed<T: WebFetchTransport, R: HostResolver>(
    transport: &T,
    resolver: &R,
    canonical_url: &str,
) -> Option<TweetContent> {
    let mut endpoint = Url::parse("https://publish.twitter.com/oembed").ok()?;
    endpoint
        .query_pairs_mut()
        .append_pair("url", canonical_url)
        .append_pair("omit_script", "true");
    let response = fetch_public_url(transport, resolver, endpoint.as_str()).ok()?;
    let payload: Value = serde_json::from_str(&response.content).ok()?;
    let html = payload
        .get("html")
        .and_then(Value::as_str)
        .unwrap_or_default();
    Some(TweetContent {
        url: canonical_url.to_owned(),
        author: payload
            .get("author_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        date: extract_tweet_date(html),
        text: extract_first_element_text(html, "p"),
    })
}

fn extract_first_element_text(html: &str, element: &str) -> String {
    let lowercase = html.to_ascii_lowercase();
    let Some(start) = lowercase.find(&format!("<{element}")) else {
        return String::new();
    };
    let Some(content_start) = lowercase[start..].find('>') else {
        return String::new();
    };
    let content_start = start + content_start + 1;
    let Some(relative_end) = lowercase[content_start..].find(&format!("</{element}>")) else {
        return String::new();
    };
    extract_text_from_html(&html[content_start..content_start + relative_end]).1
}

fn extract_tweet_date(html: &str) -> String {
    let mut remaining = html;
    while let Some(start) = remaining.find('>') {
        remaining = &remaining[start + 1..];
        let Some(end) = remaining.find('<') else {
            break;
        };
        let candidate = collapse_whitespace(&decode_html_entities(&remaining[..end]));
        if candidate.contains(',')
            && candidate
                .chars()
                .any(|character| character.is_ascii_digit())
            && candidate.split_whitespace().count() == 3
        {
            return candidate;
        }
        remaining = &remaining[end..];
    }
    String::new()
}

fn extract_meta_content(html: &str, requested_key: &str) -> Option<String> {
    for tag in html
        .split('<')
        .filter_map(|part| part.split_once('>').map(|(tag, _)| tag))
    {
        let tag = tag.trim();
        if !tag
            .get(..4)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("meta"))
        {
            continue;
        }
        let attributes = parse_attributes(tag.get(4..).unwrap_or_default());
        let key = attributes
            .iter()
            .find(|(name, _)| name == "property" || name == "name")
            .map(|(_, value)| value.to_ascii_lowercase());
        if key.as_deref() == Some(requested_key) {
            return attributes
                .iter()
                .find(|(name, _)| name == "content")
                .map(|(_, value)| decode_html_entities(value));
        }
    }
    None
}

fn parse_attributes(input: &str) -> Vec<(String, String)> {
    let mut attributes = Vec::new();
    let mut remaining = input.trim();
    while !remaining.is_empty() {
        remaining = remaining.trim_start();
        let name_end = remaining
            .find(|character: char| character.is_whitespace() || character == '=')
            .unwrap_or(remaining.len());
        let name = remaining[..name_end].trim_matches('/').to_ascii_lowercase();
        remaining = &remaining[name_end..];
        remaining = remaining.trim_start();
        if !remaining.starts_with('=') {
            if !name.is_empty() {
                attributes.push((name, String::new()));
            }
            continue;
        }
        remaining = remaining[1..].trim_start();
        let (value, rest) = if let Some(quote @ ('\'' | '"')) = remaining.chars().next() {
            let after_quote = &remaining[quote.len_utf8()..];
            let end = after_quote.find(quote).unwrap_or(after_quote.len());
            (&after_quote[..end], &after_quote[end.saturating_add(1)..])
        } else {
            let end = remaining
                .find(char::is_whitespace)
                .unwrap_or(remaining.len());
            (&remaining[..end], &remaining[end..])
        };
        if !name.is_empty() {
            attributes.push((name, value.to_owned()));
        }
        remaining = rest;
    }
    attributes
}

pub fn normalize_http_url(raw_url: &str) -> Option<Url> {
    let candidate = raw_url.trim();
    if candidate.is_empty() {
        return None;
    }
    let candidate = if candidate.contains("://") {
        candidate.to_owned()
    } else {
        format!("https://{candidate}")
    };
    let mut url = Url::parse(&candidate).ok()?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str()?.chars().any(char::is_whitespace)
    {
        return None;
    }
    url.set_fragment(None);
    Some(url)
}

fn ensure_public<R: HostResolver>(url: &Url, resolver: &R) -> Result<(), PublicFetchError> {
    let hostname = url.host_str().unwrap_or_default();
    let blocked_name = hostname.eq_ignore_ascii_case("localhost")
        || hostname.to_ascii_lowercase().ends_with(".localhost");
    if blocked_name {
        return Err(PublicFetchError::Blocked {
            url: url.to_string(),
        });
    }
    let literal_address = match url.host() {
        Some(Host::Ipv4(address)) => Some(IpAddr::V4(address)),
        Some(Host::Ipv6(address)) => Some(IpAddr::V6(address)),
        _ => None,
    };
    if let Some(address) = literal_address {
        if !is_public_ip(address) {
            return Err(PublicFetchError::Blocked {
                url: url.to_string(),
            });
        }
        return Ok(());
    }
    let port = url.port_or_known_default().unwrap_or(443);
    if resolver
        .addresses(hostname, port)
        .unwrap_or_default()
        .into_iter()
        .any(|address| !is_public_ip(address))
    {
        return Err(PublicFetchError::Blocked {
            url: url.to_string(),
        });
    }
    Ok(())
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            let [a, b, c, d] = address.octets();
            !(a == 0
                || a == 10
                || a == 127
                || (a == 100 && (64..=127).contains(&b))
                || (a == 169 && b == 254)
                || (a == 172 && (16..=31).contains(&b))
                || (a == 192 && b == 0 && c == 0)
                || (a == 192 && b == 0 && c == 2)
                || (a == 192 && b == 168)
                || (a == 198 && (b == 18 || b == 19))
                || (a == 198 && b == 51 && c == 100)
                || (a == 203 && b == 0 && c == 113)
                || a >= 224
                || (a == 255 && b == 255 && c == 255 && d == 255))
        }
        IpAddr::V6(address) => {
            let segments = address.segments();
            !(address.is_unspecified()
                || address.is_loopback()
                || address.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
                || address
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| !is_public_ip(IpAddr::V4(mapped))))
        }
    }
}

pub fn extract_text_from_html(html: &str) -> (Option<String>, String) {
    let mut parser = VisibleTextParser::default();
    parser.feed(html);
    parser.finish()
}

#[derive(Default)]
struct VisibleTextParser {
    lines: Vec<String>,
    current: String,
    title: String,
    skip_depth: usize,
    in_title: bool,
}

impl VisibleTextParser {
    fn feed(&mut self, html: &str) {
        let mut remaining = html;
        while let Some(start) = remaining.find('<') {
            self.text(&remaining[..start]);
            remaining = &remaining[start..];
            if remaining.starts_with("<!--") {
                if let Some(end) = remaining.find("-->") {
                    remaining = &remaining[end + 3..];
                    continue;
                }
                break;
            }
            let Some(end) = remaining.find('>') else {
                self.text(remaining);
                return;
            };
            self.tag(&remaining[1..end]);
            remaining = &remaining[end + 1..];
        }
        self.text(remaining);
    }

    fn tag(&mut self, raw: &str) {
        let raw = raw.trim();
        let closing = raw.starts_with('/');
        let name = raw
            .trim_start_matches('/')
            .split(|character: char| character.is_whitespace() || character == '/')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if matches!(name.as_str(), "script" | "style" | "noscript") {
            if closing {
                self.skip_depth = self.skip_depth.saturating_sub(1);
            } else {
                self.skip_depth += 1;
            }
            return;
        }
        if name == "title" {
            self.in_title = !closing;
            return;
        }
        if is_block_tag(&name) && self.skip_depth == 0 {
            self.flush();
        }
    }

    fn text(&mut self, raw: &str) {
        if self.skip_depth > 0 {
            return;
        }
        let decoded = decode_html_entities(raw);
        let cleaned = collapse_whitespace(&decoded);
        if cleaned.is_empty() {
            return;
        }
        if self.in_title {
            if !self.title.is_empty() {
                self.title.push(' ');
            }
            self.title.push_str(&cleaned);
            return;
        }
        if !self.current.is_empty() {
            self.current.push(' ');
        }
        self.current.push_str(&cleaned);
    }

    fn flush(&mut self) {
        if !self.current.is_empty() {
            self.lines.push(std::mem::take(&mut self.current));
        }
    }

    fn finish(mut self) -> (Option<String>, String) {
        self.flush();
        let title = (!self.title.is_empty()).then_some(self.title);
        (title, self.lines.join("\n"))
    }
}

fn is_block_tag(name: &str) -> bool {
    matches!(
        name,
        "p" | "div"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "li"
            | "br"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
    )
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn decode_html_entities(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut cursor = 0;
    while cursor < value.len() {
        if value.as_bytes()[cursor] == b'&'
            && let Some(relative_end) = value[cursor + 1..].find(';')
            && relative_end <= 12
        {
            let end = cursor + 1 + relative_end;
            let entity = &value[cursor + 1..end];
            if let Some(decoded) = decode_entity(entity) {
                output.push(decoded);
                cursor = end + 1;
                continue;
            }
        }
        let Some(character) = value[cursor..].chars().next() else {
            break;
        };
        output.push(character);
        cursor += character.len_utf8();
    }
    output
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        value if value.starts_with("#x") || value.starts_with("#X") => {
            char::from_u32(u32::from_str_radix(&value[2..], 16).ok()?)
        }
        value if value.starts_with('#') => char::from_u32(value[1..].parse().ok()?),
        _ => None,
    }
}

fn truncate_chars(value: &str, limit: usize) -> (String, bool) {
    let mut characters = value.chars();
    let truncated = characters.clone().nth(limit).is_some();
    let output = characters.by_ref().take(limit).collect::<String>();
    (output.trim_end().to_owned(), truncated)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::net::{Ipv4Addr, Ipv6Addr};

    use super::*;

    struct Resolver(Vec<IpAddr>);

    impl HostResolver for Resolver {
        fn addresses(&self, _hostname: &str, _port: u16) -> Result<Vec<IpAddr>, String> {
            Ok(self.0.clone())
        }
    }

    struct Transport {
        responses: RefCell<Vec<Result<WebFetchResponse, WebFetchTransportError>>>,
        urls: RefCell<Vec<String>>,
    }

    impl WebFetchTransport for Transport {
        fn get(&self, url: &str) -> Result<WebFetchResponse, WebFetchTransportError> {
            self.urls.borrow_mut().push(url.to_owned());
            self.responses.borrow_mut().remove(0)
        }
    }

    fn response(status_code: u16, body: &str) -> WebFetchResponse {
        WebFetchResponse {
            status_code,
            content_type: "text/html; charset=utf-8".to_owned(),
            location: None,
            body: body.as_bytes().to_vec(),
            truncated: false,
        }
    }

    fn json_response(body: &str) -> WebFetchResponse {
        WebFetchResponse {
            status_code: 200,
            content_type: "application/json".to_owned(),
            location: None,
            body: body.as_bytes().to_vec(),
            truncated: false,
        }
    }

    #[test]
    fn normalizes_urls_and_rejects_local_or_non_http_targets_before_io() {
        assert_eq!(
            normalize_http_url(" example.com/path#fragment ")
                .map(|url| url.to_string())
                .as_deref(),
            Some("https://example.com/path")
        );
        for url in [
            "file:///etc/passwd",
            "http://127.0.0.1/secret",
            "http://[::1]/secret",
            "http://localhost/secret",
        ] {
            let transport = Transport {
                responses: RefCell::new(Vec::new()),
                urls: RefCell::new(Vec::new()),
            };
            assert!(matches!(
                fetch_public_url(&transport, &Resolver(Vec::new()), url),
                Err(PublicFetchError::Blocked { .. })
            ));
            assert!(transport.urls.borrow().is_empty());
        }
    }

    #[test]
    fn rejects_dns_private_addresses_and_every_private_redirect_target() {
        let transport = Transport {
            responses: RefCell::new(Vec::new()),
            urls: RefCell::new(Vec::new()),
        };
        assert!(matches!(
            fetch_public_url(
                &transport,
                &Resolver(vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))]),
                "https://internal.example.test"
            ),
            Err(PublicFetchError::Blocked { .. })
        ));

        let mut redirect = response(302, "");
        redirect.location = Some("http://169.254.169.254/latest/meta-data".to_owned());
        let transport = Transport {
            responses: RefCell::new(vec![Ok(redirect)]),
            urls: RefCell::new(Vec::new()),
        };
        assert!(matches!(
            fetch_public_url(
                &transport,
                &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
                "https://example.com/start"
            ),
            Err(PublicFetchError::Blocked { .. })
        ));
        assert_eq!(transport.urls.borrow().len(), 1);
    }

    #[test]
    fn follows_relative_public_redirects_and_extracts_visible_html() {
        let mut redirect = response(301, "");
        redirect.location = Some("/final".to_owned());
        let html = r#"<html><head><title> Example &amp; Title </title><style>.x{}</style><script>bad()</script></head><body><main><h1>Hola</h1><p>Este es el contenido&nbsp;principal.</p></main></body></html>"#;
        let transport = Transport {
            responses: RefCell::new(vec![Ok(redirect), Ok(response(200, html))]),
            urls: RefCell::new(Vec::new()),
        };
        let result = fetch_public_url(
            &transport,
            &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
            "https://example.com/start",
        )
        .ok();
        assert_eq!(
            result.as_ref().and_then(|value| value.title.as_deref()),
            Some("Example & Title")
        );
        assert_eq!(
            result.as_ref().map(|value| value.content.as_str()),
            Some("Hola\nEste es el contenido principal.")
        );
        assert_eq!(
            *transport.urls.borrow(),
            ["https://example.com/start", "https://example.com/final"]
        );
    }

    #[test]
    fn bounds_unicode_text_and_preserves_transport_diagnostics_in_errors() {
        let body = format!(
            "<html><body><p>{}</p></body></html>",
            "á".repeat(FETCH_MAX_CHARS + 10)
        );
        let transport = Transport {
            responses: RefCell::new(vec![Ok(response(200, &body))]),
            urls: RefCell::new(Vec::new()),
        };
        let result = fetch_public_url(
            &transport,
            &Resolver(vec![IpAddr::V6(Ipv6Addr::new(
                0x2606, 0x2800, 0x220, 1, 0, 0, 0, 1,
            ))]),
            "https://example.com",
        )
        .ok();
        assert_eq!(
            result.as_ref().map(|value| value.content.chars().count()),
            Some(FETCH_MAX_CHARS)
        );
        assert_eq!(result.as_ref().map(|value| value.truncated), Some(true));

        let transport = Transport {
            responses: RefCell::new(vec![Err(WebFetchTransportError::Timeout)]),
            urls: RefCell::new(Vec::new()),
        };
        let result = fetch_public_url(
            &transport,
            &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
            "https://example.com",
        );
        if let Err(error) = result {
            assert_eq!(error.public_message(), "no se pudo obtener la url");
            assert!(error.to_string().contains("no se pudo"));
            assert!(
                matches!(error, PublicFetchError::Request { detail, .. } if detail.contains("timed out"))
            );
        } else {
            assert!(result.is_err(), "synthetic timeout unexpectedly succeeded");
        }
    }

    #[test]
    fn private_ip_classification_covers_special_ipv4_and_ipv6_ranges() {
        for address in [
            IpAddr::V4(Ipv4Addr::new(100, 64, 0, 1)),
            IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            IpAddr::V6(Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1)),
            IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1)),
        ] {
            assert!(!is_public_ip(address), "{address}");
        }
        assert!(is_public_ip(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
    }

    #[test]
    fn fetches_direct_tweets_through_oembed_and_preserves_author_date_and_text() {
        let payload = serde_json::json!({
            "author_name": "Example User",
            "html": "<blockquote><p>This is an example &amp; safe status.</p><a href='https://x.com/user/status/123'>Jan 1, 2020</a></blockquote>"
        })
        .to_string();
        let transport = Transport {
            responses: RefCell::new(vec![Ok(json_response(&payload))]),
            urls: RefCell::new(Vec::new()),
        };
        let result = fetch_ai_url(
            &transport,
            &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
            "https://twitter.com/user/status/123",
        );
        assert_eq!(
            result,
            Ok(AiFetchOutcome::Tweet(TweetContent {
                url: "https://x.com/user/status/123".to_owned(),
                author: "Example User".to_owned(),
                date: "Jan 1, 2020".to_owned(),
                text: "This is an example & safe status.".to_owned(),
            }))
        );
        assert!(transport.urls.borrow()[0].starts_with("https://publish.twitter.com/oembed?"));
    }

    #[test]
    fn resolves_id_only_frontend_urls_from_og_metadata_and_reports_oembed_failure() {
        let page = r#"<html><head><meta content='https://x.com/example_user/status/456' property='og:url'></head><body>preview</body></html>"#;
        let transport = Transport {
            responses: RefCell::new(vec![
                Ok(response(200, page)),
                Ok(WebFetchResponse {
                    status_code: 503,
                    content_type: "application/json".to_owned(),
                    location: None,
                    body: Vec::new(),
                    truncated: false,
                }),
            ]),
            urls: RefCell::new(Vec::new()),
        };
        assert_eq!(
            fetch_ai_url(
                &transport,
                &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
                "https://fixupx.com/status/456",
            ),
            Ok(AiFetchOutcome::TweetError {
                url: "https://x.com/example_user/status/456".to_owned(),
            })
        );
        assert_eq!(transport.urls.borrow().len(), 2);
    }

    #[test]
    fn embedded_twitter_text_on_an_unrelated_host_remains_a_regular_page() {
        let transport = Transport {
            responses: RefCell::new(vec![Ok(response(
                200,
                "<html><body>read fixupx.com/status/123 here</body></html>",
            ))]),
            urls: RefCell::new(Vec::new()),
        };
        let result = fetch_ai_url(
            &transport,
            &Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
            "https://example.com/read/fixupx.com/status/123",
        );
        assert!(matches!(result, Ok(AiFetchOutcome::Page(_))));
        assert_eq!(transport.urls.borrow().len(), 1);
    }
}
