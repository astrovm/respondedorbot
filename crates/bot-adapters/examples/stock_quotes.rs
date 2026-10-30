//! Loads live Yahoo quotes the way `/stocks` does, without a cache, and prints
//! the lines the bot would send, to check the 24-hour change by hand:
//!
//! ```bash
//! cargo run -p bot-adapters --example stock_quotes -- AAPL BZ=F
//! ```

use std::convert::Infallible;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::request_cache::RequestCache;
use bot_adapters::yahoo_finance::{ReqwestYahooFinanceTransport, load_quote};
use bot_core::stocks::render_stock_quote;

struct NoCache;

impl RequestCache for NoCache {
    type Error = Infallible;

    fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    fn set(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<(), Self::Error> {
        Ok(())
    }

    fn take(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    fn claim(&mut self, _key: &str, _value: &str, _ttl_seconds: i64) -> Result<bool, Self::Error> {
        Ok(true)
    }
}

fn main() -> ExitCode {
    let Ok(transport) = ReqwestYahooFinanceTransport::new() else {
        eprintln!("could not build the HTTP client");
        return ExitCode::FAILURE;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX)
        });
    let mut symbols = std::env::args().skip(1).collect::<Vec<_>>();
    if symbols.is_empty() {
        symbols = ["AAPL", "NVDA", "BZ=F"].map(str::to_owned).to_vec();
    }
    let mut status = ExitCode::SUCCESS;
    for symbol in symbols {
        let load = load_quote(&transport, &mut NoCache, &symbol, now);
        match load.quote {
            Some(quote) => println!("{}", render_stock_quote(&quote.symbol, &quote, None)),
            None => status = ExitCode::FAILURE,
        }
        for diagnostic in load.diagnostics {
            eprintln!("{symbol}: {diagnostic}");
        }
    }
    status
}
