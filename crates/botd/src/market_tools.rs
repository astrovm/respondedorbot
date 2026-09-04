//! Native AI tools backed by the command market and weather sources.

use bot_core::dollar::{DollarCommandPlan, invalid_timeframe_message, plan_dollar_command};
use bot_core::locale::Locale;
use bot_core::market_prices::MarketPriceCommand;
use bot_core::stocks::render_stock_quotes;
use bot_core::weather::{render_weather, weather_load_error};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::dispatcher::{DollarMarketSource, MarketPriceSource, StockPriceSource, WeatherSource};
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

pub struct CryptoPricesTool<Source, Now> {
    source: Source,
    now: Now,
    locale: Locale,
}

impl<Source, Now> CryptoPricesTool<Source, Now> {
    #[must_use]
    pub const fn new(source: Source, now: Now, locale: Locale) -> Self {
        Self {
            source,
            now,
            locale,
        }
    }
}

impl<Source, Now> ExternalToolExecutor for CryptoPricesTool<Source, Now>
where
    Source: MarketPriceSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::CryptoPrices { query } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "crypto_prices",
            ));
        };
        let load = self.source.load(
            &query,
            MarketPriceCommand::CryptoOnly,
            self.locale,
            (self.now)(),
        );
        ToolExecutionResult::with_diagnostics(load.text, load.diagnostics)
    }
}

pub struct StockPricesTool<Source, Now> {
    source: Source,
    now: Now,
    locale: Locale,
}

impl<Source, Now> StockPricesTool<Source, Now> {
    #[must_use]
    pub const fn new(source: Source, now: Now, locale: Locale) -> Self {
        Self {
            source,
            now,
            locale,
        }
    }
}

impl<Source, Now> ExternalToolExecutor for StockPricesTool<Source, Now>
where
    Source: StockPriceSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::StockPrices { query } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "stock_prices",
            ));
        };
        let load = self.source.load(&query, (self.now)());
        let output = render_stock_quotes(load.quotes.as_deref(), self.locale);
        ToolExecutionResult::with_diagnostics(output, load.diagnostics)
    }
}

pub struct DollarRatesTool<Source, Now> {
    source: Source,
    now: Now,
    locale: Locale,
}

impl<Source, Now> DollarRatesTool<Source, Now> {
    #[must_use]
    pub const fn new(source: Source, now: Now, locale: Locale) -> Self {
        Self {
            source,
            now,
            locale,
        }
    }
}

impl<Source, Now> ExternalToolExecutor for DollarRatesTool<Source, Now>
where
    Source: DollarMarketSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::DollarRates { timeframe } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "dollar_rates",
            ));
        };
        let hours_ago = match plan_dollar_command(&timeframe) {
            DollarCommandPlan::Load { hours_ago } => hours_ago,
            DollarCommandPlan::InvalidTimeframe => {
                return ToolExecutionResult::output(invalid_timeframe_message(
                    &timeframe,
                    self.locale,
                ));
            }
        };
        let load = self.source.load(hours_ago, self.locale, (self.now)());
        let output = load.text.unwrap_or_else(|| dollar_load_error(self.locale));
        ToolExecutionResult::with_diagnostics(output, load.diagnostics)
    }
}

pub struct WeatherTool<Source, Now> {
    source: Source,
    now: Now,
    locale: Locale,
}

impl<Source, Now> WeatherTool<Source, Now> {
    #[must_use]
    pub const fn new(source: Source, now: Now, locale: Locale) -> Self {
        Self {
            source,
            now,
            locale,
        }
    }
}

impl<Source, Now> ExternalToolExecutor for WeatherTool<Source, Now>
where
    Source: WeatherSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::Weather { location } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(self.locale, "weather"));
        };
        let load = self.source.load(&location, (self.now)());
        let output = load.observation.as_ref().map_or_else(
            || weather_load_error(&location, self.locale),
            |observation| render_weather(observation, self.locale),
        );
        ToolExecutionResult::with_diagnostics(output, load.diagnostics)
    }
}

fn dollar_load_error(locale: Locale) -> String {
    match locale {
        Locale::Es => "no se pudieron obtener las cotizaciones del dólar".to_owned(),
        Locale::En => "I could not load the dollar rates".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use bot_core::stocks::StockQuote;
    use bot_core::weather::WeatherObservation;

    use super::*;
    use crate::dispatcher::{
        DollarMarketLoad, MarketPriceLoad, StockQuotesLoad, WeatherObservationLoad,
    };

    type MarketCall = (String, MarketPriceCommand, Locale, i64);

    struct MarketSource {
        load: MarketPriceLoad,
        calls: Rc<RefCell<Vec<MarketCall>>>,
    }

    impl MarketPriceSource for MarketSource {
        fn load(
            &mut self,
            query: &str,
            command: MarketPriceCommand,
            locale: Locale,
            now_unix: i64,
        ) -> MarketPriceLoad {
            self.calls
                .borrow_mut()
                .push((query.to_owned(), command, locale, now_unix));
            self.load.clone()
        }
    }

    #[test]
    fn crypto_uses_the_crypto_only_source_and_preserves_diagnostics() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut tool = CryptoPricesTool::new(
            MarketSource {
                load: MarketPriceLoad {
                    chart: None,
                    no_assets_found: false,
                    text: "BTC: 70000 USD".to_owned(),
                    diagnostics: vec!["synthetic provider note".to_owned()],
                },
                calls: Rc::clone(&calls),
            },
            || 1_700_000_000,
            Locale::En,
        );
        let result = tool.execute(
            ExternalToolRequest::CryptoPrices {
                query: "BTC in USD 24h".to_owned(),
            },
            "call",
        );
        assert_eq!(result.output, "BTC: 70000 USD");
        assert_eq!(result.diagnostics, ["synthetic provider note"]);
        assert_eq!(
            *calls.borrow(),
            [(
                "BTC in USD 24h".to_owned(),
                MarketPriceCommand::CryptoOnly,
                Locale::En,
                1_700_000_000
            )]
        );
    }

    struct StockSource(StockQuotesLoad);

    impl StockPriceSource for StockSource {
        fn load(&mut self, query: &str, now_unix: i64) -> StockQuotesLoad {
            assert_eq!(query, "AAPL,missing");
            assert_eq!(now_unix, 42);
            self.0.clone()
        }
    }

    #[test]
    fn stocks_render_success_missing_quotes_and_provider_failure() {
        let quote = StockQuote {
            symbol: "AAPL".to_owned(),
            name: "Apple".to_owned(),
            price: 200.0,
            currency: "USD".to_owned(),
            exchange: "NMS".to_owned(),
            variation: 1.5,
        };
        let mut tool = StockPricesTool::new(
            StockSource(StockQuotesLoad {
                quotes: Some(vec![
                    ("AAPL".to_owned(), Some(quote)),
                    ("missing".to_owned(), None),
                ]),
                diagnostics: Vec::new(),
            }),
            || 42,
            Locale::En,
        );
        let request = ExternalToolRequest::StockPrices {
            query: "AAPL,missing".to_owned(),
        };
        assert_eq!(
            tool.execute(request.clone(), "call").output,
            "AAPL: 200.00 USD (+1.50% 24h)\nmissing: not found"
        );
        tool.source.0.quotes = None;
        assert_eq!(
            tool.execute(request, "call").output,
            "I could not load the top stocks, try again"
        );
    }

    struct DollarSource {
        calls: Rc<RefCell<Vec<(i64, Locale, i64)>>>,
        load: DollarMarketLoad,
    }

    impl DollarMarketSource for DollarSource {
        fn load(&mut self, hours_ago: i64, locale: Locale, now_unix: i64) -> DollarMarketLoad {
            self.calls.borrow_mut().push((hours_ago, locale, now_unix));
            self.load.clone()
        }
    }

    #[test]
    fn dollar_defaults_to_24h_and_localizes_empty_or_invalid_results() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut tool = DollarRatesTool::new(
            DollarSource {
                calls: Rc::clone(&calls),
                load: DollarMarketLoad {
                    text: None,
                    diagnostics: vec!["synthetic dollar failure".to_owned()],
                },
            },
            || 99,
            Locale::Es,
        );
        let result = tool.execute(
            ExternalToolRequest::DollarRates {
                timeframe: String::new(),
            },
            "call",
        );
        assert_eq!(
            result.output,
            "no se pudieron obtener las cotizaciones del dólar"
        );
        assert_eq!(result.diagnostics, ["synthetic dollar failure"]);
        assert_eq!(*calls.borrow(), [(24, Locale::Es, 99)]);
        assert_eq!(
            tool.execute(
                ExternalToolRequest::DollarRates {
                    timeframe: "7h".to_owned(),
                },
                "call"
            )
            .output,
            "timeframe '7h' no soportado, uso: 1h, 6h, 12h, 24h, 48h"
        );
    }

    struct WeatherSourceImpl(WeatherObservationLoad);

    impl WeatherSource for WeatherSourceImpl {
        fn load(&mut self, location: &str, now_unix: i64) -> WeatherObservationLoad {
            assert_eq!(location, "Synthetic City");
            assert_eq!(now_unix, 123);
            self.0.clone()
        }
    }

    #[test]
    fn weather_renders_the_existing_contract_and_localizes_missing_data() {
        let mut tool = WeatherTool::new(
            WeatherSourceImpl(WeatherObservationLoad {
                observation: Some(WeatherObservation {
                    location: "Synthetic City".to_owned(),
                    apparent_temperature: "20".to_owned(),
                    precipitation_probability: "15".to_owned(),
                    weather_code: 1,
                    cloud_cover: "30".to_owned(),
                    visibility_meters: 12_500.0,
                }),
                diagnostics: Vec::new(),
            }),
            || 123,
            Locale::En,
        );
        let request = ExternalToolRequest::Weather {
            location: "Synthetic City".to_owned(),
        };
        assert_eq!(
            tool.execute(request.clone(), "call").output,
            "- Location: Synthetic City\n- Feels like: 20°C\n- Chance of rain: 15%\n- Condition: mostly clear\n- Cloud cover: 30%\n- Visibility: 12.5km"
        );
        tool.source.0.observation = None;
        assert_eq!(
            tool.execute(request, "call").output,
            "I could not load the weather for Synthetic City"
        );
    }

    #[test]
    fn incompatible_requests_are_explicit() {
        let mut crypto = CryptoPricesTool::new(
            MarketSource {
                load: MarketPriceLoad {
                    chart: None,
                    no_assets_found: false,
                    text: String::new(),
                    diagnostics: Vec::new(),
                },
                calls: Rc::new(RefCell::new(Vec::new())),
            },
            || 0,
            Locale::En,
        );
        assert_eq!(
            crypto.execute(ExternalToolRequest::TaskList, "call").output,
            "tool 'crypto_prices' received an incompatible request"
        );
    }
}
