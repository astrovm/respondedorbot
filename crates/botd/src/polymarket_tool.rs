//! Native Polymarket market-search tool backed by the cached Gamma adapter.

use bot_adapters::polymarket::{PolymarketSearchTransport, search_markets};
use bot_adapters::request_cache::RequestCache;
use bot_core::locale::Locale;
use bot_core::polymarket::{market_search_error, render_market_search};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct PolymarketTool<Transport, Cache, Now> {
    transport: Transport,
    cache: Cache,
    now: Now,
    locale: Locale,
}

impl<Transport, Cache, Now> PolymarketTool<Transport, Cache, Now> {
    #[must_use]
    pub const fn new(transport: Transport, cache: Cache, now: Now, locale: Locale) -> Self {
        Self {
            transport,
            cache,
            now,
            locale,
        }
    }
}

impl<Transport, Cache, Now> ExternalToolExecutor for PolymarketTool<Transport, Cache, Now>
where
    Transport: PolymarketSearchTransport,
    Cache: RequestCache,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::Polymarket { query } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "polymarket",
            ));
        };
        let load = search_markets(&self.transport, &mut self.cache, &query, (self.now)());
        let mut output = load.events.map_or_else(
            || market_search_error(self.locale).to_owned(),
            |events| render_market_search(&query, &events, self.locale),
        );
        if let Some(age) = load.stale_seconds {
            output = format!("{}\n\n{output}", stale_warning(age, self.locale));
        }
        ToolExecutionResult::with_diagnostics(output, load.diagnostics)
    }
}

/// Old prices read like live ones, so say plainly when Polymarket failed and
/// an expired cache entry stood in.
fn stale_warning(age_seconds: i64, locale: Locale) -> String {
    let minutes = age_seconds / 60;
    match locale {
        Locale::Es => format!(
            "Ojo: Polymarket no respondió. Estos datos son de hace {minutes} min, no los presentes como actuales"
        ),
        Locale::En => format!(
            "Warning: Polymarket did not respond. This data is {minutes} min old; do not present it as current"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use bot_adapters::polymarket::{HttpResponse, TransportFailureKind};

    use super::*;

    struct Transport {
        response: Result<HttpResponse, TransportFailureKind>,
        queries: RefCell<Vec<String>>,
    }

    impl PolymarketSearchTransport for Transport {
        fn search(&self, query: &str) -> Result<HttpResponse, TransportFailureKind> {
            self.queries.borrow_mut().push(query.to_owned());
            self.response.clone()
        }
    }

    /// Holds at most one cached entry, returned for every key.
    struct Cache(Option<String>);

    impl RequestCache for Cache {
        type Error = std::convert::Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.0.clone())
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

    fn tool(
        response: Result<HttpResponse, TransportFailureKind>,
        locale: Locale,
    ) -> PolymarketTool<Transport, Cache, impl FnMut() -> i64> {
        PolymarketTool::new(
            Transport {
                response,
                queries: RefCell::default(),
            },
            Cache(None),
            || 1_700_000_000,
            locale,
        )
    }

    fn request(query: &str) -> ExternalToolRequest {
        ExternalToolRequest::Polymarket {
            query: query.to_owned(),
        }
    }

    #[test]
    fn renders_searched_markets_for_the_model() {
        let body = r#"{"events":[{"title":"Synthetic vote","slug":"synthetic-vote","liquidity":1500,"volume":2000000,"endDate":"2027-01-01T00:00:00Z","markets":[{"groupItemTitle":"","question":"Synthetic vote","outcomes":"[\"Yes\", \"No\"]","outcomePrices":"[\"0.125\", \"0.875\"]"}]}]}"#;
        let mut tool = tool(
            Ok(HttpResponse {
                status_code: 200,
                body: body.to_owned(),
            }),
            Locale::En,
        );
        let result = tool.execute(request("synthetic vote"), "call");
        assert_eq!(
            result.output,
            concat!(
                "Polymarket markets for \"synthetic vote\" (%: implied probability)\n\n",
                "1. Synthetic vote\n",
                "https://polymarket.com/event/synthetic-vote\n",
                "Liquidity US$1.5K | Volume US$2M | Closes 2027-01-01\n",
                "- Yes 12.5%"
            )
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(*tool.transport.queries.borrow(), ["synthetic vote"]);
    }

    #[test]
    fn reports_empty_results_and_outages_differently() {
        let mut empty = tool(
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"events":[]}"#.to_owned(),
            }),
            Locale::Es,
        );
        assert_eq!(
            empty.execute(request("nada"), "call").output,
            "No encontré mercados abiertos en Polymarket para \"nada\""
        );

        let mut down = tool(Err(TransportFailureKind::Timeout), Locale::Es);
        let result = down.execute(request("caído"), "call");
        assert_eq!(
            result.output,
            "No pude buscar en Polymarket. Probá más tarde"
        );
        assert_eq!(result.diagnostics.len(), 2);
    }

    #[test]
    fn stale_cached_markets_come_with_their_age() {
        let events = r#"[{"title":"Vote","slug":"vote","liquidity":0.0,"volume":0.0,"end_date":"","hidden_markets":0,"markets":[{"title":"Vote","outcomes":[{"title":"Yes","probability":0.77}]}]}]"#;
        for (fetched_at, locale, warning) in [
            (
                1_699_999_000,
                Locale::Es,
                "Ojo: Polymarket no respondió. Estos datos son de hace 16 min, no los presentes como actuales\n\n",
            ),
            (
                1_699_999_930,
                Locale::En,
                "Warning: Polymarket did not respond. This data is 1 min old; do not present it as current\n\n",
            ),
        ] {
            let mut stale = tool(Err(TransportFailureKind::Timeout), locale);
            stale.cache = Cache(Some(format!(
                r#"{{"timestamp":{fetched_at},"data":{events}}}"#
            )));
            let output = stale.execute(request("vote"), "call").output;
            assert!(output.starts_with(warning), "{output}");
            assert!(output.ends_with("- Yes 77%"), "{output}");
        }
    }

    #[test]
    fn incompatible_requests_are_explicit() {
        let mut tool = tool(Err(TransportFailureKind::Request), Locale::En);
        assert_eq!(
            tool.execute(ExternalToolRequest::TaskList, "call").output,
            "tool 'polymarket' received an incompatible request"
        );
        assert!(tool.transport.queries.borrow().is_empty());
    }

    #[test]
    fn cache_stub_keeps_the_empty_contract() {
        assert_eq!(Cache(None).set("synthetic", "1", 60), Ok(()));
        assert_eq!(Cache(None).take("synthetic"), Ok(None));
        assert_eq!(Cache(None).claim("synthetic", "1", 60), Ok(true));
    }
}
