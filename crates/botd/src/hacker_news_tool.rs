//! Native Hacker News tool backed by the cached RSS adapter.

use bot_adapters::hacker_news::{HackerNewsCache, HackerNewsTransport, load_hacker_news};
use bot_core::hacker_news::{HackerNewsRenderItem, format_items};
use bot_core::locale::Locale;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct HackerNewsTool<Transport, Cache> {
    transport: Transport,
    cache: Cache,
    locale: Locale,
}

impl<Transport, Cache> HackerNewsTool<Transport, Cache> {
    #[must_use]
    pub const fn new(transport: Transport, cache: Cache, locale: Locale) -> Self {
        Self {
            transport,
            cache,
            locale,
        }
    }
}

impl<Transport, Cache> ExternalToolExecutor for HackerNewsTool<Transport, Cache>
where
    Transport: HackerNewsTransport,
    Cache: HackerNewsCache,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::HackerNews { limit } = request else {
            return ToolExecutionResult::output("hacker_news received an incompatible request");
        };
        let load = load_hacker_news(&self.transport, &mut self.cache, usize::from(limit));
        let items = load
            .items
            .into_iter()
            .map(|item| HackerNewsRenderItem {
                title: item.title,
                url: item.url,
                points: item.points,
                comments: item.comments,
                comments_url: item.comments_url,
            })
            .collect::<Vec<_>>();
        let (no_data, comments) = match self.locale {
            Locale::Es => ("sin datos por ahora", "coms"),
            Locale::En => ("no data yet", "comments"),
        };
        ToolExecutionResult::with_diagnostics(
            format_items(&items, true, no_data, comments),
            load.diagnostics,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_adapters::hacker_news::{CACHE_KEY, HackerNewsResponse, HackerNewsTransportError};

    use super::*;

    struct Transport(RefCell<Vec<Result<HackerNewsResponse, HackerNewsTransportError>>>);

    impl HackerNewsTransport for Transport {
        fn get(&self, _url: &str) -> Result<HackerNewsResponse, HackerNewsTransportError> {
            self.0.borrow_mut().remove(0)
        }
    }

    #[derive(Default)]
    struct Cache {
        payload: Option<String>,
    }

    impl HackerNewsCache for Cache {
        fn get(&mut self, key: &str) -> Result<Option<String>, String> {
            assert_eq!(key, CACHE_KEY);
            Ok(self.payload.clone())
        }

        fn set(&mut self, _key: &str, value: &str, _ttl_seconds: i64) -> Result<(), String> {
            self.payload = Some(value.to_owned());
            Ok(())
        }
    }

    #[test]
    fn renders_bounded_localized_rss_items_for_the_model() {
        let feed = r#"<rss><channel><item><title>Story</title><link>https://example.test</link><description><![CDATA[Points: 9<br># Comments: 2<br>Comments URL: <a href="https://news.ycombinator.com/item?id=1">x</a>]]></description></item></channel></rss>"#;
        let mut tool = HackerNewsTool::new(
            Transport(RefCell::new(vec![Ok(HackerNewsResponse {
                status_code: 200,
                body: feed.to_owned(),
            })])),
            Cache::default(),
            Locale::En,
        );
        let result = tool.execute(ExternalToolRequest::HackerNews { limit: 5 }, "call");
        assert_eq!(
            result.output,
            "- Story (9 pts, 2 comments) → https://example.test (HN: https://news.ycombinator.com/item?id=1)"
        );
        assert!(result.diagnostics.is_empty());
    }

    #[test]
    fn returns_localized_empty_output_and_diagnostics_after_both_feeds_fail() {
        let mut tool = HackerNewsTool::new(
            Transport(RefCell::new(vec![
                Err(HackerNewsTransportError::Timeout),
                Err(HackerNewsTransportError::Connection),
            ])),
            Cache::default(),
            Locale::Es,
        );
        let result = tool.execute(ExternalToolRequest::HackerNews { limit: 5 }, "call");
        assert_eq!(result.output, "- sin datos por ahora");
        assert_eq!(result.diagnostics.len(), 2);
    }

    #[test]
    fn incompatible_requests_are_explicit() {
        let mut tool = HackerNewsTool::new(
            Transport(RefCell::new(Vec::new())),
            Cache::default(),
            Locale::En,
        );
        assert_eq!(
            tool.execute(ExternalToolRequest::TaskList, "call").output,
            "hacker_news received an incompatible request"
        );
    }
}
