//! Telegram getUpdates bodies: decoding never panics, and the next offset
//! follows the newest update. It may move backwards on purpose: Telegram picks
//! a random new ID after a week without updates.

#![no_main]

use bot_adapters::telegram_polling::{PollOutcome, next_offset, parse_response};
use bot_core::telegram_input::extract_message_content;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|body: &str| {
    if let Ok(PollOutcome::Updates(updates)) = parse_response(200, body) {
        let newest = updates.iter().map(|update| update.update_id).max();
        let expected = newest
            .and_then(|update_id| update_id.checked_add(1))
            .or(Some(1));
        assert_eq!(next_offset(&updates, Some(1)), expected);
    }
    let _ = parse_response(429, body);
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        let _ = extract_message_content(&value);
        let _ = bot_core::telegram_callbacks::parse_callback_context(&value);
        let _ = bot_core::telegram_payments::evaluate_default_successful_payment(&value, true);
        let _ = bot_core::telegram_payments::plan_pre_checkout(
            &value,
            true,
            bot_core::locale::Locale::En,
        );
        if let Some(updates) = value.get("result").and_then(serde_json::Value::as_array) {
            for update in updates {
                if let Some(callback) = update.get("callback_query") {
                    let _ = bot_core::telegram_callbacks::parse_callback_context(callback);
                }
                if let Some(message) = update.get("message") {
                    let _ = bot_core::telegram_payments::evaluate_default_successful_payment(
                        message, true,
                    );
                }
            }
        }
    }
});
