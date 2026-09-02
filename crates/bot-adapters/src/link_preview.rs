//! Telegram-compatible social-link preview probes.

use std::collections::HashMap;
use std::io::Read;
use std::thread;
use std::time::Duration;

use regex::Regex;
use reqwest::Method;
use reqwest::blocking::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, LOCATION, USER_AGENT};
use reqwest::redirect::Policy;

const TELEGRAM_PREVIEW_USER_AGENT: &str = "TelegramBot (like TwitterBot)";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const RETRY_DELAYS: [Duration; 2] = [Duration::from_millis(250), Duration::from_millis(500)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMethod {
    Get,
    Head,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewRequest {
    pub url: String,
    pub method: PreviewMethod,
    pub follow_redirects: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewResponse {
    pub status_code: u16,
    pub final_url: String,
    pub content_type: String,
    pub content_length: Option<u64>,
    pub location: Option<String>,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewFailure {
    Timeout,
    Connection,
    Request,
}

pub trait LinkPreviewTransport {
    fn request(&self, request: &PreviewRequest) -> Result<PreviewResponse, PreviewFailure>;

    fn download_video(&self, _url: &str, _max_bytes: u64) -> Result<Vec<u8>, PreviewFailure> {
        Err(PreviewFailure::Request)
    }
}

pub struct ReqwestLinkPreviewTransport {
    following: Client,
    no_redirect: Client,
}

impl ReqwestLinkPreviewTransport {
    pub fn new() -> Result<Self, PreviewFailure> {
        let following = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(classify_error)?;
        let no_redirect = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .redirect(Policy::none())
            .build()
            .map_err(classify_error)?;
        Ok(Self {
            following,
            no_redirect,
        })
    }
}

fn classify_error(error: reqwest::Error) -> PreviewFailure {
    if error.is_timeout() {
        PreviewFailure::Timeout
    } else if error.is_connect() {
        PreviewFailure::Connection
    } else {
        PreviewFailure::Request
    }
}

impl LinkPreviewTransport for ReqwestLinkPreviewTransport {
    fn request(&self, request: &PreviewRequest) -> Result<PreviewResponse, PreviewFailure> {
        let client = if request.follow_redirects {
            &self.following
        } else {
            &self.no_redirect
        };
        let method = match request.method {
            PreviewMethod::Get => Method::GET,
            PreviewMethod::Head => Method::HEAD,
        };
        let response = client
            .request(method, &request.url)
            .header(USER_AGENT, TELEGRAM_PREVIEW_USER_AGENT)
            .send()
            .map_err(classify_error)?;
        let status_code = response.status().as_u16();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let content_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse().ok());
        let location = response
            .headers()
            .get(LOCATION)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        let body = if request.method == PreviewMethod::Get {
            response.text().map_err(classify_error)?
        } else {
            String::new()
        };
        Ok(PreviewResponse {
            status_code,
            final_url,
            content_type,
            content_length,
            location,
            body,
        })
    }

    fn download_video(&self, url: &str, max_bytes: u64) -> Result<Vec<u8>, PreviewFailure> {
        let mut response = self
            .following
            .get(url)
            .header(USER_AGENT, TELEGRAM_PREVIEW_USER_AGENT)
            .timeout(Duration::from_secs(30))
            .send()
            .map_err(classify_error)?;
        if response.status().as_u16() >= 400
            || !response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.to_ascii_lowercase().starts_with("video/"))
        {
            return Err(PreviewFailure::Request);
        }
        let mut bytes = Vec::new();
        response
            .by_ref()
            .take(max_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|_| PreviewFailure::Request)?;
        if bytes.len() as u64 > max_bytes {
            return Err(PreviewFailure::Request);
        }
        Ok(bytes)
    }
}

pub const TELEGRAM_REMOTE_VIDEO_MAX_BYTES: u64 = 20_000_000;
pub const TELEGRAM_MULTIPART_VIDEO_MAX_BYTES: u64 = 50_000_000;

pub fn download_oversized_video<T: LinkPreviewTransport>(
    transport: &T,
    inspection: &PreviewInspection,
) -> Option<Vec<u8>> {
    let metadata = &inspection.metadata;
    let size = metadata.media_size?;
    if !inspection.embeddable
        || size <= TELEGRAM_REMOTE_VIDEO_MAX_BYTES
        || size > TELEGRAM_MULTIPART_VIDEO_MAX_BYTES
        || !metadata
            .media_content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("video/"))
    {
        return None;
    }
    transport
        .download_video(
            metadata.media_url.as_deref()?,
            TELEGRAM_MULTIPART_VIDEO_MAX_BYTES,
        )
        .ok()
        .filter(|bytes| !bytes.is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub canonical_url: Option<String>,
    pub media_url: Option<String>,
    pub media_content_type: Option<String>,
    pub media_size: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewInspection {
    pub embeddable: bool,
    pub status_code: Option<u16>,
    pub final_url: String,
    pub metadata: PreviewMetadata,
    pub failure: Option<PreviewFailure>,
}

fn empty_metadata() -> PreviewMetadata {
    PreviewMetadata {
        title: None,
        description: None,
        canonical_url: None,
        media_url: None,
        media_content_type: None,
        media_size: None,
    }
}

fn is_instagram_frontend(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| {
            ["eeinstagram.com", "vxinstagram.com", "kkinstagram.com"]
                .iter()
                .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
        })
}

fn is_eeinstagram_post(url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(url) else {
        return false;
    };
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    let first_segment = url
        .path_segments()
        .into_iter()
        .flatten()
        .find(|segment| !segment.is_empty());
    (host == "eeinstagram.com" || host.ends_with(".eeinstagram.com"))
        && matches!(first_segment, Some("p" | "reel" | "reels"))
}

fn transient(status_code: u16) -> bool {
    status_code == 429 || status_code >= 500
}

fn is_media_content_type(content_type: &str) -> bool {
    ["image/", "video/", "audio/"]
        .iter()
        .any(|prefix| content_type.starts_with(prefix))
}

fn request_retry<T: LinkPreviewTransport>(
    transport: &T,
    request: &PreviewRequest,
    retry: bool,
) -> Result<PreviewResponse, PreviewFailure> {
    let attempts = if retry { 3 } else { 1 };
    let mut last_failure = PreviewFailure::Request;
    for attempt in 0..attempts {
        match transport.request(request) {
            Ok(response) if !transient(response.status_code) || attempt + 1 == attempts => {
                return Ok(response);
            }
            Ok(_) => {}
            Err(failure) if attempt + 1 == attempts => return Err(failure),
            Err(failure) => last_failure = failure,
        }
        if let Some(delay) = RETRY_DELAYS.get(attempt) {
            thread::sleep(*delay);
        }
    }
    Err(last_failure)
}

fn meta_tags(html: &str) -> HashMap<String, String> {
    let Ok(tag_pattern) = Regex::new(r"(?is)<meta\s+[^>]*>") else {
        return HashMap::new();
    };
    let Ok(attribute_pattern) =
        Regex::new(r#"(?is)([a-zA-Z_:][-a-zA-Z0-9_:]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))"#)
    else {
        return HashMap::new();
    };
    let mut tags = HashMap::new();
    for tag in tag_pattern.find_iter(&html[..html.len().min(20_000)]) {
        let attributes = attribute_pattern
            .captures_iter(tag.as_str())
            .filter_map(|captures| {
                let name = captures.get(1)?.as_str().to_ascii_lowercase();
                let value = captures
                    .get(2)
                    .or_else(|| captures.get(3))
                    .or_else(|| captures.get(4))?
                    .as_str()
                    .trim()
                    .to_owned();
                Some((name, value))
            })
            .collect::<HashMap<_, _>>();
        let key = attributes
            .get("property")
            .or_else(|| attributes.get("name"))
            .map(|value| value.to_ascii_lowercase());
        if let (Some(key), Some(content)) = (key, attributes.get("content"))
            && (key.starts_with("og:") || key.starts_with("twitter:"))
            && !content.is_empty()
        {
            tags.insert(key, content.clone());
        }
    }
    tags
}

fn resolve_url(base: &str, reference: &str) -> String {
    reqwest::Url::parse(base)
        .ok()
        .and_then(|base| base.join(reference).ok())
        .map_or_else(|| reference.to_owned(), |url| url.to_string())
}

fn probe_media<T: LinkPreviewTransport>(transport: &T, url: &str) -> Option<PreviewResponse> {
    request_retry(
        transport,
        &PreviewRequest {
            url: url.to_owned(),
            method: PreviewMethod::Get,
            follow_redirects: true,
        },
        true,
    )
    .ok()
    .filter(|response| response.status_code < 400 && is_media_content_type(&response.content_type))
}

pub fn inspect_with<T: LinkPreviewTransport>(transport: &T, url: &str) -> PreviewInspection {
    if is_eeinstagram_post(url) {
        let head = request_retry(
            transport,
            &PreviewRequest {
                url: url.to_owned(),
                method: PreviewMethod::Head,
                follow_redirects: false,
            },
            true,
        );
        if let Ok(head) = head
            && head.status_code != 405
            && !transient(head.status_code)
        {
            if head.status_code >= 400 {
                return PreviewInspection {
                    embeddable: false,
                    status_code: Some(head.status_code),
                    final_url: url.to_owned(),
                    metadata: empty_metadata(),
                    failure: None,
                };
            }
            if (300..400).contains(&head.status_code) {
                let media = head
                    .location
                    .as_deref()
                    .map(|location| resolve_url(url, location))
                    .and_then(|media_url| probe_media(transport, &media_url));
                return PreviewInspection {
                    embeddable: media.is_some(),
                    status_code: Some(head.status_code),
                    final_url: url.to_owned(),
                    metadata: empty_metadata(),
                    failure: None,
                };
            }
        }
    }
    let response = request_retry(
        transport,
        &PreviewRequest {
            url: url.to_owned(),
            method: PreviewMethod::Get,
            follow_redirects: true,
        },
        is_eeinstagram_post(url),
    );
    let response = match response {
        Ok(response) => response,
        Err(failure) => {
            return PreviewInspection {
                embeddable: false,
                status_code: None,
                final_url: url.to_owned(),
                metadata: empty_metadata(),
                failure: Some(failure),
            };
        }
    };
    if response.status_code >= 400 {
        return PreviewInspection {
            embeddable: false,
            status_code: Some(response.status_code),
            final_url: response.final_url,
            metadata: empty_metadata(),
            failure: None,
        };
    }
    if is_media_content_type(&response.content_type) {
        let final_url = response.final_url.clone();
        return PreviewInspection {
            embeddable: true,
            status_code: Some(response.status_code),
            final_url: final_url.clone(),
            metadata: PreviewMetadata {
                media_url: Some(final_url),
                media_content_type: Some(response.content_type),
                media_size: response.content_length,
                ..empty_metadata()
            },
            failure: None,
        };
    }
    if !response.content_type.contains("text/html") {
        return PreviewInspection {
            embeddable: false,
            status_code: Some(response.status_code),
            final_url: response.final_url,
            metadata: empty_metadata(),
            failure: None,
        };
    }
    let tags = meta_tags(&response.body);
    let title = tags
        .get("og:title")
        .or_else(|| tags.get("twitter:title"))
        .cloned();
    let description = tags
        .get("og:description")
        .or_else(|| tags.get("twitter:description"))
        .cloned();
    let canonical_url = tags.get("og:url").cloned();
    let media_reference = tags
        .get("og:video")
        .or_else(|| tags.get("twitter:player:stream"))
        .or_else(|| tags.get("og:image"))
        .or_else(|| tags.get("twitter:image"));
    let has_media = media_reference.is_some()
        || tags.contains_key("twitter:player")
        || tags.contains_key("twitter:card");
    let instagram = is_instagram_frontend(url);
    let has_text = title.is_some() || description.is_some();
    let mut media = None;
    if instagram {
        media = media_reference
            .map(|reference| resolve_url(&response.final_url, reference))
            .and_then(|media_url| probe_media(transport, &media_url));
    }
    let media_ready = !instagram || media.is_some();
    let embeddable =
        (has_text && has_media || (instagram && media_reference.is_some())) && media_ready;
    PreviewInspection {
        embeddable,
        status_code: Some(response.status_code),
        final_url: response.final_url,
        metadata: PreviewMetadata {
            title,
            description,
            canonical_url,
            media_url: media.as_ref().map(|value| value.final_url.clone()),
            media_content_type: media.as_ref().map(|value| value.content_type.clone()),
            media_size: media.and_then(|value| value.content_length),
        },
        failure: None,
    }
}

#[must_use]
pub fn inspect(url: &str) -> PreviewInspection {
    match ReqwestLinkPreviewTransport::new() {
        Ok(transport) => inspect_with(&transport, url),
        Err(failure) => PreviewInspection {
            embeddable: false,
            status_code: None,
            final_url: url.to_owned(),
            metadata: empty_metadata(),
            failure: Some(failure),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    use super::{
        LinkPreviewTransport, PreviewFailure, PreviewInspection, PreviewMetadata, PreviewMethod,
        PreviewRequest, PreviewResponse, ReqwestLinkPreviewTransport, download_oversized_video,
        inspect_with,
    };

    #[test]
    fn reqwest_preview_transport_handles_redirect_policy_head_get_and_video_bounds() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap_or_else(|_| unreachable!());
        let address = listener.local_addr().unwrap_or_else(|_| unreachable!());
        let server = thread::spawn(move || {
            for body in [b"preview".as_slice(), b"".as_slice(), b"video".as_slice()] {
                let (mut stream, _) = listener.accept().unwrap_or_else(|_| unreachable!());
                let mut request = [0_u8; 4_096];
                let _ = stream.read(&mut request);
                let content_type = if body == b"video" {
                    "video/mp4"
                } else {
                    "text/html"
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                stream
                    .write_all(response.as_bytes())
                    .unwrap_or_else(|_| unreachable!());
                stream.write_all(body).unwrap_or_else(|_| unreachable!());
            }
        });
        let transport = ReqwestLinkPreviewTransport::new().unwrap_or_else(|_| unreachable!());
        let url = format!("http://{address}/resource");
        let get = transport
            .request(&PreviewRequest {
                url: url.clone(),
                method: PreviewMethod::Get,
                follow_redirects: true,
            })
            .unwrap_or_else(|_| unreachable!());
        assert_eq!(get.body, "preview");
        let head = transport
            .request(&PreviewRequest {
                url: url.clone(),
                method: PreviewMethod::Head,
                follow_redirects: false,
            })
            .unwrap_or_else(|_| unreachable!());
        assert!(head.body.is_empty());
        assert_eq!(transport.download_video(&url, 5), Ok(b"video".to_vec()));
        assert!(server.join().is_ok());
    }

    struct FakeTransport {
        responses: RefCell<Vec<Result<PreviewResponse, PreviewFailure>>>,
        requests: RefCell<Vec<PreviewRequest>>,
    }

    impl LinkPreviewTransport for FakeTransport {
        fn request(&self, request: &PreviewRequest) -> Result<PreviewResponse, PreviewFailure> {
            self.requests.borrow_mut().push(request.clone());
            self.responses.borrow_mut().remove(0)
        }
    }

    fn response(status_code: u16, content_type: &str, body: &str) -> PreviewResponse {
        PreviewResponse {
            status_code,
            final_url: "https://example.test/final".to_owned(),
            content_type: content_type.to_owned(),
            content_length: None,
            location: None,
            body: body.to_owned(),
        }
    }

    #[test]
    fn accepts_html_cards_regardless_of_meta_attribute_order() {
        let transport = FakeTransport {
            responses: RefCell::new(vec![Ok(response(
                200,
                "text/html; charset=utf-8",
                "<meta content='A title' property='og:title'><meta name=twitter:card content=summary>",
            ))]),
            requests: RefCell::new(Vec::new()),
        };
        let result = inspect_with(&transport, "https://fixupx.com/a/status/1");
        assert!(result.embeddable);
        assert_eq!(result.metadata.title.as_deref(), Some("A title"));
        assert_eq!(transport.requests.borrow()[0].method, PreviewMethod::Get);
    }

    #[test]
    fn rejects_http_non_html_and_incomplete_metadata() {
        for response in [
            response(404, "text/html", ""),
            response(200, "application/json", "{}"),
            response(200, "text/html", "<meta property=og:title content=Only>"),
        ] {
            let transport = FakeTransport {
                responses: RefCell::new(vec![Ok(response)]),
                requests: RefCell::new(Vec::new()),
            };
            assert!(!inspect_with(&transport, "https://fixupx.com/a").embeddable);
        }
    }

    #[test]
    fn direct_media_and_transport_failures_are_explicit() {
        let mut media = response(200, "video/mp4", "ignored");
        media.content_length = Some(123);
        let transport = FakeTransport {
            responses: RefCell::new(vec![Ok(media)]),
            requests: RefCell::new(Vec::new()),
        };
        let result = inspect_with(&transport, "https://example.test/video");
        assert!(result.embeddable);
        assert_eq!(result.metadata.media_size, Some(123));

        let transport = FakeTransport {
            responses: RefCell::new(vec![Err(PreviewFailure::Timeout)]),
            requests: RefCell::new(Vec::new()),
        };
        assert_eq!(
            inspect_with(&transport, "https://example.test").failure,
            Some(PreviewFailure::Timeout)
        );
    }

    #[test]
    fn instagram_head_redirect_requires_ready_media() {
        let mut head = response(302, "", "");
        head.location = Some("/video.mp4".to_owned());
        let transport = FakeTransport {
            responses: RefCell::new(vec![Ok(head), Ok(response(200, "video/mp4", ""))]),
            requests: RefCell::new(Vec::new()),
        };
        assert!(inspect_with(&transport, "https://eeinstagram.com/reel/abc").embeddable);
        let requests = transport.requests.borrow();
        assert_eq!(requests[0].method, PreviewMethod::Head);
        assert!(!requests[0].follow_redirects);
        assert!(requests[1].follow_redirects);
    }

    #[test]
    fn instagram_html_requires_live_media_probe_and_retries_transient_responses() {
        let html = response(
            200,
            "text/html",
            "<meta property=og:image content='/image.jpg'>",
        );
        let transport = FakeTransport {
            responses: RefCell::new(vec![
                Ok(response(405, "", "")),
                Ok(html),
                Ok(response(503, "text/plain", "")),
                Ok(response(200, "image/jpeg", "")),
            ]),
            requests: RefCell::new(Vec::new()),
        };
        assert!(inspect_with(&transport, "https://eeinstagram.com/p/abc").embeddable);
        assert_eq!(transport.requests.borrow().len(), 4);
    }

    #[test]
    fn downloads_only_videos_in_the_telegram_multipart_window() {
        struct Downloader {
            calls: RefCell<Vec<(String, u64)>>,
        }
        impl LinkPreviewTransport for Downloader {
            fn request(
                &self,
                _request: &PreviewRequest,
            ) -> Result<PreviewResponse, PreviewFailure> {
                Err(PreviewFailure::Request)
            }

            fn download_video(&self, url: &str, max_bytes: u64) -> Result<Vec<u8>, PreviewFailure> {
                self.calls.borrow_mut().push((url.to_owned(), max_bytes));
                Ok(vec![1, 2, 3])
            }
        }
        let transport = Downloader {
            calls: RefCell::new(Vec::new()),
        };
        let inspection = PreviewInspection {
            embeddable: true,
            status_code: Some(200),
            final_url: "https://eeinstagram.com/reel/a".to_owned(),
            metadata: PreviewMetadata {
                title: None,
                description: None,
                canonical_url: None,
                media_url: Some("https://cdn.test/video.mp4".to_owned()),
                media_content_type: Some("video/mp4".to_owned()),
                media_size: Some(20_000_001),
            },
            failure: None,
        };
        assert_eq!(
            download_oversized_video(&transport, &inspection),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            transport.calls.borrow().as_slice(),
            &[("https://cdn.test/video.mp4".to_owned(), 50_000_000)]
        );
        let mut too_large = inspection.clone();
        too_large.metadata.media_size = Some(50_000_001);
        assert_eq!(download_oversized_video(&transport, &too_large), None);
        assert_eq!(transport.calls.borrow().len(), 1);
    }
}
