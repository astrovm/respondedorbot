//! Gets a file the way the bot's `send_media` and `page_screenshot` tools do
//! and saves it, so downloads and screenshots can be checked against real
//! sites without Telegram.
//!
//! Screenshots need `chromium` on `PATH`. Run it by hand:
//!
//! ```bash
//! cargo run -p bot-adapters --example share_media -- link https://example.com/cat.jpg /tmp/cat
//! cargo run -p bot-adapters --example share_media -- screenshot https://example.com /tmp/page.jpg [full]
//! ```

use std::process::ExitCode;
use std::time::Instant;

use bot_adapters::browser::{PageCapturer, shared_renderer};
use bot_adapters::web_fetch::{ReqwestWebFetchTransport, SystemHostResolver, fetch_public_media};
use bot_core::media_reply::classify_media;

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let [mode, url, output, rest @ ..] = arguments.as_slice() else {
        eprintln!("usage: share_media <link|screenshot> <url> <output file> [full]");
        return ExitCode::FAILURE;
    };
    let started = Instant::now();
    let fetched = match mode.as_str() {
        "link" => ReqwestWebFetchTransport::for_media()
            .map_err(|error| format!("{error:?}"))
            .and_then(|transport| {
                fetch_public_media(&transport, &SystemHostResolver, url)
                    .map_err(|error| format!("{error:?}"))
            })
            .map(|file| (file.url, file.content_type, file.body)),
        "screenshot" => shared_renderer()
            .ok_or_else(|| "the browser proxy did not start".to_owned())
            .and_then(|renderer| {
                renderer
                    .screenshot(url, rest.first().map(String::as_str) == Some("full"))
                    .map_err(|error| error.to_string())
            })
            .map(|image| (url.clone(), "image/jpeg".to_owned(), image)),
        _ => Err(format!("unknown mode {mode}")),
    };
    let (source, content_type, body) = match fetched {
        Ok(fetched) => fetched,
        Err(error) => {
            eprintln!("error after {:.1?}: {error}", started.elapsed());
            return ExitCode::FAILURE;
        }
    };
    let path = url::Url::parse(&source)
        .map(|url| url.path().to_owned())
        .unwrap_or_default();
    println!("source: {source}");
    println!("served as: {content_type}, {} bytes", body.len());
    println!("sent as: {:?}", classify_media(&content_type, &body, &path));
    println!("took: {:.1?}", started.elapsed());
    match std::fs::write(output, &body) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("could not save {output}: {error}");
            ExitCode::FAILURE
        }
    }
}
