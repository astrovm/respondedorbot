//! Checks which tool the production model picks for real prompts.
//!
//! This calls OpenRouter and costs money (a few US cents per run), so it
//! never runs in CI. Run it by hand after changing tool descriptions or the
//! system prompt:
//!
//! ```bash
//! OPENROUTER_API_KEY=... cargo run -p botd --example tool_choice_evals
//! ```
//!
//! It uses `workspace/SOUL.md` and `workspace/RULES.md` as the persona when
//! they exist, like the bot does, and a neutral stand-in otherwise. Each case
//! sends one message with every tool offered and checks the first round's
//! tool calls. Cases live in `tool_choice_cases.json`.

use std::process::ExitCode;

use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, ChatMessage, ChatRole, ToolCall, complete,
};
use bot_core::ai_prompt::{
    ConversationPromptInput, PromptContent, build_conversation_prompt, build_system_prompt,
};
use bot_core::locale::Locale;
use botd::native_ai::PRIMARY_CHAT_MODEL;
use botd::native_tools::catalog_schemas;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    prompt: String,
    #[serde(default)]
    locale: Option<String>,
    expect: Vec<Expectation>,
}

/// The tool must be called, and its arguments must contain at least one of
/// `args_any` (case-insensitive). `""` accepts any arguments.
#[derive(Deserialize)]
struct Expectation {
    tool: String,
    args_any: Vec<String>,
}

fn persona() -> String {
    let parts = ["workspace/SOUL.md", "workspace/RULES.md"]
        .iter()
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .collect::<Vec<_>>();
    if parts.len() == 2 {
        parts.join("\n\n")
    } else {
        "You are a friendly Argentine Telegram bot. Keep answers short.".to_owned()
    }
}

fn user_message(case: &Case, locale: Locale) -> String {
    let conversation = build_conversation_prompt(&ConversationPromptInput {
        locale,
        chat_type: "private".to_owned(),
        chat_title: String::new(),
        first_name: "Eval".to_owned(),
        username: String::new(),
        formatted_time: "12:00".to_owned(),
        timezone_offset_hours: -3,
        message_text: case.prompt.clone(),
        reply_context: None,
        reply_target_is_latest: false,
        link_context: None,
        enable_web_search: true,
        summary: None,
        history: Vec::new(),
        retrieved: Vec::new(),
    });
    conversation
        .last()
        .map(|message| match &message.content {
            PromptContent::Text(text) => text.clone(),
            PromptContent::TextParts(parts) => parts.join("\n"),
            _ => case.prompt.clone(),
        })
        .unwrap_or_else(|| case.prompt.clone())
}

fn grade(case: &Case, calls: &[ToolCall]) -> Result<(), String> {
    if case.expect.is_empty() {
        return match calls.first() {
            Some(call) => Err(format!("expected no tool, got {}", call.function.name)),
            None => Ok(()),
        };
    }
    for expected in &case.expect {
        let matching = calls
            .iter()
            .filter(|call| call.function.name == expected.tool)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            let called = calls
                .iter()
                .map(|call| call.function.name.as_str())
                .collect::<Vec<_>>();
            return Err(format!("expected {}, got {called:?}", expected.tool));
        }
        let fits = matching.iter().any(|call| {
            let arguments = call.function.arguments.to_lowercase();
            expected
                .args_any
                .iter()
                .any(|needle| arguments.contains(&needle.to_lowercase()))
        });
        if !fits {
            return Err(format!(
                "{} arguments {} miss all of {:?}",
                expected.tool, matching[0].function.arguments, expected.args_any
            ));
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") else {
        eprintln!("set OPENROUTER_API_KEY to run the tool-choice evals");
        return ExitCode::from(2);
    };
    let cases: Vec<Case> = match serde_json::from_str(include_str!("tool_choice_cases.json")) {
        Ok(cases) => cases,
        Err(error) => {
            eprintln!("invalid tool_choice_cases.json: {error}");
            return ExitCode::from(2);
        }
    };
    let persona = persona();
    let tools = catalog_schemas();
    let (mut failures, mut cost) = (0, 0.0);
    for case in &cases {
        let locale = if case.locale.as_deref() == Some("en") {
            Locale::En
        } else {
            Locale::Es
        };
        let system = build_system_prompt(&persona, locale, "2026-10-25", true, false);
        let mut request = ChatCompletionRequest::new(
            PRIMARY_CHAT_MODEL,
            vec![
                ChatMessage::text(ChatRole::System, system),
                ChatMessage::text(ChatRole::User, user_message(case, locale)),
            ],
        );
        request.tools.clone_from(&tools);
        let verdict = match complete(&api_key, &request) {
            Ok(completion) => {
                cost += completion
                    .usage
                    .get("cost")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                grade(case, &completion.tool_calls)
            }
            Err(error) => Err(format!("request failed: {error}")),
        };
        match verdict {
            Ok(()) => println!("PASS  {}", case.prompt),
            Err(reason) => {
                failures += 1;
                println!("FAIL  {}\n      {reason}", case.prompt);
            }
        }
    }
    println!(
        "\n{}/{} passed, cost US${cost:.4}",
        cases.len() - failures,
        cases.len()
    );
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
