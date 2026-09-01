//! Native bounded and SSRF-safe `web_fetch` AI tool.

use bot_adapters::web_fetch::{
    AiFetchOutcome, HostResolver, PublicFetchError, WebFetchTransport, fetch_ai_url,
};
use bot_core::locale::Locale;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct WebFetchTool<Transport, Resolver> {
    transport: Transport,
    resolver: Resolver,
    locale: Locale,
}

impl<Transport, Resolver> WebFetchTool<Transport, Resolver> {
    #[must_use]
    pub const fn new(transport: Transport, resolver: Resolver, locale: Locale) -> Self {
        Self {
            transport,
            resolver,
            locale,
        }
    }
}

impl<Transport, Resolver> ExternalToolExecutor for WebFetchTool<Transport, Resolver>
where
    Transport: WebFetchTransport,
    Resolver: HostResolver,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::WebFetch { url } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "web_fetch",
            ));
        };
        match fetch_ai_url(&self.transport, &self.resolver, &url) {
            Ok(AiFetchOutcome::Tweet(tweet)) => {
                let mut parts = Vec::new();
                if !tweet.author.is_empty() || !tweet.date.is_empty() {
                    let mut heading = match (self.locale, tweet.author.is_empty()) {
                        (_, true) => "Tweet".to_owned(),
                        (Locale::Es, false) => format!("Tweet de {}", tweet.author),
                        (Locale::En, false) => format!("Tweet by {}", tweet.author),
                    };
                    if !tweet.date.is_empty() {
                        heading.push_str(" · ");
                        heading.push_str(&tweet.date);
                    }
                    parts.push(heading);
                }
                if !tweet.text.is_empty() {
                    parts.push(tweet.text);
                }
                ToolExecutionResult::output(if parts.is_empty() {
                    localized(
                        self.locale,
                        "tweet sin texto legible",
                        "tweet has no readable text",
                    )
                } else {
                    parts.join("\n")
                })
            }
            Ok(AiFetchOutcome::TweetError { url }) => ToolExecutionResult::with_diagnostics(
                localized(
                    self.locale,
                    "no se pudo leer el tweet",
                    "could not read the tweet",
                ),
                vec![format!("Twitter oEmbed failed for {url}")],
            ),
            Ok(AiFetchOutcome::Page(page)) => {
                if page.content.contains("Something went wrong")
                    && page.content.contains("Try again")
                {
                    return ToolExecutionResult::output(localized(
                        self.locale,
                        "error obteniendo la página: X devolvió una página de error",
                        "error fetching the page: X returned an error page",
                    ));
                }
                let output = page.title.map_or(page.content.clone(), |title| {
                    format!("{}\n{}", title_label(self.locale, &title), page.content)
                });
                let mut diagnostics = Vec::new();
                if page.truncated {
                    diagnostics.push(format!("web_fetch truncated response from {}", page.url));
                }
                ToolExecutionResult::with_diagnostics(output, diagnostics)
            }
            Err(error) => error_result(&url, error, self.locale),
        }
    }
}

fn error_result(url: &str, error: PublicFetchError, locale: Locale) -> ToolExecutionResult {
    let output = match locale {
        Locale::Es => format!("error obteniendo {url}: {}", error.public_message(locale)),
        Locale::En => format!("error fetching {url}: {}", error.public_message(locale)),
    };
    let diagnostic = match &error {
        PublicFetchError::Blocked { url } => format!("web_fetch blocked URL {url}"),
        PublicFetchError::Request { url, detail } => {
            format!("web_fetch request failed for {url}: {detail}")
        }
    };
    ToolExecutionResult::with_diagnostics(output, vec![diagnostic])
}

fn title_label(locale: Locale, title: &str) -> String {
    match locale {
        Locale::Es => format!("Título: {title}"),
        Locale::En => format!("Title: {title}"),
    }
}

fn localized(locale: Locale, spanish: &str, english: &str) -> String {
    match locale {
        Locale::Es => spanish.to_owned(),
        Locale::En => english.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::net::{IpAddr, Ipv4Addr};

    use bot_adapters::web_fetch::{WebFetchResponse, WebFetchTransportError};

    use super::*;

    struct Resolver(Vec<IpAddr>);

    impl HostResolver for Resolver {
        fn addresses(&self, _hostname: &str, _port: u16) -> Result<Vec<IpAddr>, String> {
            Ok(self.0.clone())
        }
    }

    struct Transport(RefCell<Vec<Result<WebFetchResponse, WebFetchTransportError>>>);

    impl WebFetchTransport for Transport {
        fn get(&self, _url: &str) -> Result<WebFetchResponse, WebFetchTransportError> {
            self.0.borrow_mut().remove(0)
        }
    }

    fn make_tool(
        responses: Vec<Result<WebFetchResponse, WebFetchTransportError>>,
        locale: Locale,
    ) -> WebFetchTool<Transport, Resolver> {
        WebFetchTool::new(
            Transport(RefCell::new(responses)),
            Resolver(vec![IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34))]),
            locale,
        )
    }

    fn response(content_type: &str, body: &str) -> WebFetchResponse {
        WebFetchResponse {
            status_code: 200,
            content_type: content_type.to_owned(),
            location: None,
            body: body.as_bytes().to_vec(),
            truncated: false,
        }
    }

    fn request(url: &str) -> ExternalToolRequest {
        ExternalToolRequest::WebFetch {
            url: url.to_owned(),
        }
    }

    #[test]
    fn renders_regular_pages_titles_and_truncation_diagnostics() {
        let mut page = response(
            "text/html",
            "<html><head><title>Example</title></head><body><p>Hello world</p></body></html>",
        );
        page.truncated = true;
        let mut tool = make_tool(vec![Ok(page)], Locale::En);
        let result = tool.execute(request("https://example.com"), "call");
        assert_eq!(result.output, "Title: Example\nHello world");
        assert_eq!(result.diagnostics.len(), 1);
    }

    #[test]
    fn renders_direct_tweets_and_localizes_empty_oembed_content() {
        let payload = serde_json::json!({
            "author_name": "Example User",
            "html": "<blockquote><p>A status update.</p><a>Jan 1, 2020</a></blockquote>"
        })
        .to_string();
        let mut tool = make_tool(vec![Ok(response("application/json", &payload))], Locale::En);
        assert_eq!(
            tool.execute(request("https://x.com/user/status/123"), "call")
                .output,
            "Tweet by Example User · Jan 1, 2020\nA status update."
        );

        let empty = serde_json::json!({"html": "<blockquote></blockquote>"}).to_string();
        let mut tool = make_tool(vec![Ok(response("application/json", &empty))], Locale::Es);
        assert_eq!(
            tool.execute(request("https://x.com/user/status/123"), "call")
                .output,
            "tweet sin texto legible"
        );
    }

    #[test]
    fn rejects_private_urls_reports_safe_errors_and_detects_x_error_pages() {
        let mut tool = make_tool(Vec::new(), Locale::En);
        let result = tool.execute(request("http://127.0.0.1/secret"), "call");
        assert_eq!(
            result.output,
            "error fetching http://127.0.0.1/secret: URL is not allowed"
        );
        assert!(result.diagnostics[0].contains("blocked"));

        let mut tool = make_tool(
            vec![Ok(response(
                "text/html",
                "<html><body>Something went wrong Try again</body></html>",
            ))],
            Locale::Es,
        );
        assert_eq!(
            tool.execute(request("https://example.com"), "call").output,
            "error obteniendo la página: X devolvió una página de error"
        );
    }

    #[test]
    fn transport_tweet_and_incompatible_failures_are_explicit() {
        let mut tool = make_tool(vec![Err(WebFetchTransportError::Timeout)], Locale::En);
        let result = tool.execute(request("https://example.com"), "call");
        assert!(result.output.contains("error fetching"));
        assert!(result.diagnostics[0].contains("timed out"));

        let mut tool = make_tool(
            vec![Ok(WebFetchResponse {
                status_code: 503,
                content_type: "application/json".to_owned(),
                location: None,
                body: Vec::new(),
                truncated: false,
            })],
            Locale::Es,
        );
        assert_eq!(
            tool.execute(request("https://x.com/user/status/123"), "call")
                .output,
            "no se pudo leer el tweet"
        );
        assert_eq!(
            tool.execute(ExternalToolRequest::TaskList, "call").output,
            "la herramienta 'web_fetch' recibió una solicitud incompatible"
        );
    }
}
