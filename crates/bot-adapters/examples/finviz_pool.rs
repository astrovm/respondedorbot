//! Fetches the live Finviz mega-cap screener the way `/stocks` does and
//! prints the symbols it finds, to check the parser after Finviz changes its
//! page. Run it by hand:
//!
//! ```bash
//! cargo run -p bot-adapters --example finviz_pool
//! ```

use std::process::ExitCode;

use bot_adapters::finviz::{ReqwestFinvizTransport, ScreenerOutcome, fetch_with};

fn main() -> ExitCode {
    let Ok(transport) = ReqwestFinvizTransport::new() else {
        eprintln!("could not build the HTTP client");
        return ExitCode::FAILURE;
    };
    match fetch_with(&transport) {
        ScreenerOutcome::Success { symbols } if !symbols.is_empty() => {
            println!("{} symbols: {}", symbols.len(), symbols.join(", "));
            ExitCode::SUCCESS
        }
        outcome => {
            eprintln!("no symbols: {outcome:?}");
            ExitCode::FAILURE
        }
    }
}
