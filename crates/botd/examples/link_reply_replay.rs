//! Replays "someone replies to a link and asks the bot about it" against the
//! production model, with the real link reader.
//!
//! This calls OpenRouter and costs money (under a cent per run), so it never
//! runs in CI. Pass the tweet on the command line, so no real post lands in
//! the repo. Chat history is synthetic, and it mentions an unrelated topic to
//! tempt the model off the link:
//!
//! ```bash
//! OPENROUTER_API_KEY=... cargo run -p botd --example link_reply_replay -- \
//!     --url https://fixupx.com/user/status/123 --expect word --runs 5
//! ```
//!
//! `--mode old` shows the model what it used to get: no link context, no photo,
//! and a text-only `web_fetch`. A run passes when no tool call or answer goes
//! off on the distraction and, with `--expect`, the answer has that word. Use
//! a word only the photo shows. Web search is stubbed to find nothing, so it
//! can't come from there. Images go as downloaded, not through ffmpeg.

use std::process::ExitCode;

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, ChatMessage, ChatRole, ToolCall, complete,
};
use bot_adapters::web_fetch::{ReqwestWebFetchTransport, SystemHostResolver, fetch_tweet};
use bot_core::ai_prompt::{
    ConversationPromptInput, HistoryMessage, PromptContent, PromptRole, build_conversation_prompt,
    build_system_prompt,
};
use bot_core::locale::Locale;
use botd::ai_dispatch::LinkImage;
use botd::composition::read_live_link_context;
use botd::conversation::link_image_note;
use botd::dispatcher::LinkContextRequest;
use botd::native_ai::PRIMARY_CHAT_MODEL;
use botd::native_tools::catalog_schemas;
use serde_json::{Value, json};

const DISTRACTION_MESSAGE: &str =
    "Ana: che salió el powerball, un solo ticket en new jersey se llevó 847 palos";
const DISTRACTION_WORDS: [&str; 5] = ["powerball", "847", "new jersey", "lotería", "loteria"];
const MAX_ROUNDS: usize = 5;

struct Options {
    url: String,
    question: String,
    expect: Option<String>,
    runs: usize,
    old: bool,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        url: String::new(),
        question: "real?".to_owned(),
        expect: None,
        runs: 5,
        old: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--url" => options.url = value,
            "--question" => options.question = value,
            "--expect" => options.expect = Some(value.to_lowercase()),
            "--runs" => {
                options.runs = value
                    .parse()
                    .map_err(|_| "--runs needs a number".to_owned())?;
            }
            "--mode" => options.old = value == "old",
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if options.url.is_empty() {
        return Err("--url is required".to_owned());
    }
    Ok(options)
}

fn user_message(options: &Options, reply: &str) -> (String, Option<Vec<u8>>) {
    let mut context = None;
    let mut image = None;
    if !options.old {
        let read = read_live_link_context(LinkContextRequest {
            text: &options.question,
            replied_text: Some(reply),
            own_links_known: false,
            own_image_url: None,
            wants_image: true,
        });
        let attached = matches!(read.image, LinkImage::Attached { .. });
        let note = link_image_note(&read.image, attached, Locale::Es);
        context = Some(
            [read.text, note]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
        if let LinkImage::Attached { bytes, .. } = read.image {
            image = Some(bytes.to_vec());
        }
        for diagnostic in read.diagnostics {
            println!("diagnostic: {diagnostic}");
        }
    }
    let conversation = build_conversation_prompt(&ConversationPromptInput {
        locale: Locale::Es,
        chat_type: "supergroup".to_owned(),
        chat_title: "Eval".to_owned(),
        first_name: "Eval".to_owned(),
        username: String::new(),
        formatted_time: "00:48".to_owned(),
        timezone_offset_hours: -3,
        message_text: options.question.clone(),
        reply_context: Some(format!("Gordo (@gordo): {reply}")),
        reply_target_is_latest: false,
        link_context: context,
        enable_web_search: true,
        summary: None,
        history: vec![HistoryMessage {
            role: PromptRole::User,
            text: DISTRACTION_MESSAGE.to_owned(),
            tool_notes: Vec::new(),
        }],
        retrieved: Vec::new(),
    });
    let text = conversation
        .iter()
        .map(|message| match &message.content {
            PromptContent::Text(text) => text.clone(),
            PromptContent::TextParts(parts) => parts.join("\n"),
            _ => String::new(),
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    (text, image)
}

/// The turn's first user message, with the picture if there is one.
fn first_message(text: String, image: Option<&[u8]>) -> ChatMessage {
    let Some(image) = image else {
        return ChatMessage::text(ChatRole::User, text);
    };
    let data_uri = format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(image));
    let mut message = ChatMessage::text(ChatRole::User, "");
    message.content = Some(json!([
        {"type": "text", "text": text},
        {"type": "text", "text": "[Imagen adjunta]"},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]));
    message
}

/// What the tool would have said. In old mode `web_fetch` gives the tweet's
/// text and its dead media link, which is all oEmbed had.
fn tool_output(call: &ToolCall, options: &Options) -> String {
    if call.function.name != "web_fetch" {
        return "sin resultados".to_owned();
    }
    let tweet = ReqwestWebFetchTransport::new()
        .ok()
        .and_then(|transport| fetch_tweet(&transport, &SystemHostResolver, &options.url));
    tweet.map_or_else(
        || "no se pudo leer el tweet".to_owned(),
        |tweet| {
            let media = if options.old {
                " pic.twitter.com/media".to_owned()
            } else {
                format!("\nTiene {} foto(s).", tweet.photos.len())
            };
            format!(
                "Tweet de {}, {}\n{}{media}",
                tweet.author, tweet.date, tweet.text
            )
        },
    )
}

fn distraction_in(text: &str) -> bool {
    let text = text.to_lowercase();
    DISTRACTION_WORDS.iter().any(|word| text.contains(word))
}

fn run_once(
    api_key: &str,
    options: &Options,
    system: &str,
    first: &ChatMessage,
    cost: &mut f64,
) -> Result<(bool, String), String> {
    let mut messages = vec![ChatMessage::text(ChatRole::System, system), first.clone()];
    let mut clean = true;
    let mut trace = Vec::new();
    for _ in 0..MAX_ROUNDS {
        let mut request = ChatCompletionRequest::new(PRIMARY_CHAT_MODEL, messages.clone());
        request.tools = catalog_schemas();
        let completion = complete(api_key, &request).map_err(|error| error.to_string())?;
        *cost += completion
            .usage
            .get("cost")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if completion.tool_calls.is_empty() {
            clean &= !distraction_in(&completion.text);
            if let Some(word) = &options.expect {
                clean &= completion.text.to_lowercase().contains(word);
            }
            trace.push(format!("answer: {}", completion.text));
            return Ok((clean, trace.join("\n      ")));
        }
        let mut assistant = ChatMessage::text(ChatRole::Assistant, completion.text.clone());
        assistant.tool_calls.clone_from(&completion.tool_calls);
        messages.push(assistant);
        for call in &completion.tool_calls {
            clean &= !distraction_in(&call.function.arguments);
            trace.push(format!(
                "{} {}",
                call.function.name, call.function.arguments
            ));
            messages.push(ChatMessage::tool_result(
                call.id.clone(),
                tool_output(call, options),
            ));
        }
    }
    trace.push("no answer within the round limit".to_owned());
    Ok((false, trace.join("\n      ")))
}

fn main() -> ExitCode {
    let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") else {
        eprintln!("set OPENROUTER_API_KEY to run the replay");
        return ExitCode::from(2);
    };
    let options = match options() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let reply = format!("{}\n\ncompartido por @ana", options.url);
    let (text, image) = user_message(&options, &reply);
    println!(
        "mode: {}, image: {} bytes",
        if options.old { "old" } else { "new" },
        image.as_ref().map_or(0, Vec::len)
    );
    let system = build_system_prompt(
        "You are a friendly Argentine Telegram bot. Keep answers short.",
        Locale::Es,
        "2026-11-06",
        true,
        false,
    );
    let first = first_message(text, image.as_deref());
    let (mut passed, mut cost) = (0, 0.0);
    for run in 1..=options.runs {
        match run_once(&api_key, &options, &system, &first, &mut cost) {
            Ok((clean, trace)) => {
                passed += usize::from(clean);
                println!(
                    "{} run {run}\n      {trace}",
                    if clean { "PASS" } else { "FAIL" }
                );
            }
            Err(error) => println!("FAIL run {run}: {error}"),
        }
    }
    println!("\n{passed}/{} passed, cost US${cost:.4}", options.runs);
    if passed == options.runs {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
