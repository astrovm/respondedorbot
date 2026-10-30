//! Reads a page the way the bot's `web_fetch` tool does and prints what the
//! model would see, so rendering can be checked against real sites.
//!
//! It needs `chromium` on `PATH` to render; without it, pages that need
//! rendering come back as static text with the failure shown. Run it by hand:
//!
//! ```bash
//! cargo run -p bot-adapters --example read_page -- https://example.com [render]
//! ```

use std::process::ExitCode;
use std::time::Instant;

use bot_adapters::browser::{PageRenderer, shared_renderer};
use bot_adapters::web_fetch::{
    AiFetchOutcome, ReqwestWebFetchTransport, SystemHostResolver, fetch_ai_url,
};

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(url) = arguments.next() else {
        eprintln!("usage: read_page <url> [render]");
        return ExitCode::FAILURE;
    };
    let render = arguments.next().as_deref() == Some("render");
    let transport = match ReqwestWebFetchTransport::new() {
        Ok(transport) => transport,
        Err(error) => {
            eprintln!("could not build the HTTP client: {error}");
            return ExitCode::FAILURE;
        }
    };
    let renderer = shared_renderer();
    let started = Instant::now();
    let outcome = fetch_ai_url(
        &transport,
        &SystemHostResolver,
        renderer
            .as_deref()
            .map(|renderer| renderer as &dyn PageRenderer),
        &url,
        render,
    );
    let elapsed = started.elapsed();
    match outcome {
        Ok(AiFetchOutcome::Page(page)) => {
            println!("url: {}", page.url);
            println!("rendering: {:?} in {elapsed:.1?}", page.rendering);
            println!("title: {}", page.title.unwrap_or_default());
            println!("chars: {}\n", page.content.chars().count());
            println!("{}", page.content);
            ExitCode::SUCCESS
        }
        Ok(other) => {
            println!("{other:?}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("error: {error:?}");
            ExitCode::FAILURE
        }
    }
}
