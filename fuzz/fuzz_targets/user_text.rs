//! Text users type: command parsing and every parser that reads command
//! arguments must accept anything without panicking.

#![no_main]

use bot_core::ai_calculator::calculate_expression;
use bot_core::base_conversion::convert_base;
use bot_core::command_parsing::{bang_command_as_slash, parse_command};
use bot_core::locale::Locale;
use bot_core::price_queries::parse_price_query;
use bot_core::stateless_commands::plan_stateless_command;
use bot_core::telegram_input::{ChatId, MessageId};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &str| {
    let parsed = parse_command(text, "@synthetic_bot");
    // The arguments are always a piece of the original message.
    assert!(text.contains(parsed.message_text.as_str()));
    if let Some(slashed) = bang_command_as_slash(text, "@synthetic_bot") {
        assert!(slashed.starts_with('/'));
        let _ = parse_command(&slashed, "@synthetic_bot");
    }
    let timeframes = ["1h".to_owned(), "24h".to_owned(), "7d".to_owned()];
    let _ = parse_price_query(text, &timeframes);
    let _ = convert_base(text);
    for locale in [Locale::Es, Locale::En] {
        let _ = calculate_expression(text, locale);
        let _ = plan_stateless_command(ChatId(1), MessageId(1), text, "@synthetic_bot", locale);
    }
});
