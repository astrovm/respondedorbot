//! Deterministic message-routing decisions.

use crate::telegram_input::without_poll_notes;

/// Normalized facts needed to decide automatic media processing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MediaRoutingInput {
    pub chat_type: String,
    pub known_command: bool,
    pub message_text: String,
    pub bot_username: Option<String>,
    pub reply_username: Option<String>,
}

/// Decide whether media should be transcribed or described automatically.
#[must_use]
pub fn should_auto_process_media(input: &MediaRoutingInput) -> bool {
    if input.chat_type == "private" || input.known_command {
        return true;
    }

    let Some(bot_username) = input
        .bot_username
        .as_deref()
        .map(str::trim)
        .filter(|username| !username.is_empty())
    else {
        return false;
    };
    let bot_name = format!("@{bot_username}").to_lowercase();
    let is_mention = input.message_text.to_lowercase().contains(&bot_name);
    let is_reply = input.reply_username.as_deref() == Some(bot_username);
    is_mention || is_reply
}

/// Normalized facts for deciding whether the bot should respond.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseRoutingInput {
    pub known_command: bool,
    pub command_starts_with_slash: bool,
    pub message_text: String,
    pub is_private: bool,
    pub is_mention: bool,
    pub is_reply: bool,
    pub reply_text: String,
    pub ignore_link_fix_followups: bool,
    pub is_non_ai_command_followup: bool,
    pub ai_command_followups: bool,
    pub random_replies_enabled: bool,
    pub trigger_words: Option<Vec<String>>,
    pub random_sample: Option<f64>,
}

/// An evaluation may request one external value before reaching a decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseRoutingEvaluation {
    Ignore,
    Respond,
    NeedsTriggerWords,
    NeedsRandomSample,
}

const LINK_REPLACEMENT_DOMAINS: [&str; 8] = [
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "eeinstagram.com",
    "vxinstagram.com",
    "kkinstagram.com",
    "rxddit.com",
    "vxtiktok.com", // Still present in messages sent before TikTok rewrites were disabled.
];

/// Whether `text` is one of the bot's link-fix messages, which followups to
/// are ignored when `ignore_link_fix_followups` is on.
#[must_use]
pub fn is_link_fix_text(text: &str) -> bool {
    crate::links::HTTP_URL.as_ref().is_some_and(|regex| {
        regex.find_iter(text).any(|found| {
            url::Url::parse(&crate::links::trim_detected_url(found.as_str()))
                .ok()
                .and_then(|url| url.host_str().map(str::to_owned))
                .is_some_and(|host| {
                    LINK_REPLACEMENT_DOMAINS.iter().any(|domain| {
                        host == *domain
                            || host
                                .strip_suffix(domain)
                                .is_some_and(|prefix| prefix.ends_with('.'))
                    })
                })
        })
    })
}

/// Evaluate response routing while preserving explicit config and RNG effects.
#[must_use]
pub fn evaluate_response_routing(input: &ResponseRoutingInput) -> ResponseRoutingEvaluation {
    if !input.known_command
        && input.is_reply
        && input.ignore_link_fix_followups
        && is_link_fix_text(&input.reply_text)
    {
        return ResponseRoutingEvaluation::Ignore;
    }
    if !input.known_command
        && input.is_reply
        && input.is_non_ai_command_followup
        && !input.ai_command_followups
    {
        return ResponseRoutingEvaluation::Ignore;
    }

    let Some(trigger_words) = input.trigger_words.as_ref() else {
        return ResponseRoutingEvaluation::NeedsTriggerWords;
    };
    let message_lower = without_poll_notes(&input.message_text).to_lowercase();
    let matches_trigger = input.random_replies_enabled
        && trigger_words
            .iter()
            .any(|word| mentions_trigger_word(&message_lower, &word.to_lowercase()));
    if matches_trigger && input.random_sample.is_none() {
        return ResponseRoutingEvaluation::NeedsRandomSample;
    }
    let random_trigger = matches_trigger && input.random_sample.is_some_and(|sample| sample < 0.1);
    let should_respond = input.known_command
        || (!input.command_starts_with_slash
            && (random_trigger || input.is_private || input.is_mention || input.is_reply));
    if should_respond {
        ResponseRoutingEvaluation::Respond
    } else {
        ResponseRoutingEvaluation::Ignore
    }
}

/// Whether `text` uses `word` as a word of its own. Plurals ("bots") and
/// stretched endings ("gordooo") still count; "robot" and "botón" don't.
fn mentions_trigger_word(text: &str, word: &str) -> bool {
    let Some(last) = word.chars().last() else {
        return false;
    };
    text.match_indices(word).any(|(start, _)| {
        let starts_word = text[..start]
            .chars()
            .next_back()
            .is_none_or(|before| !before.is_alphanumeric());
        let tail = text[start + word.len()..]
            .chars()
            .take_while(|after| after.is_alphanumeric())
            .collect::<String>();
        starts_word && (tail == "s" || tail == "es" || tail.chars().all(|after| after == last))
    })
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        MediaRoutingInput, ResponseRoutingEvaluation, ResponseRoutingInput,
        evaluate_response_routing, mentions_trigger_word, should_auto_process_media,
    };

    fn input() -> MediaRoutingInput {
        MediaRoutingInput {
            chat_type: "group".to_owned(),
            known_command: false,
            message_text: "ordinary message".to_owned(),
            bot_username: Some("testbot".to_owned()),
            reply_username: None,
        }
    }

    fn response_input() -> ResponseRoutingInput {
        ResponseRoutingInput {
            known_command: false,
            command_starts_with_slash: false,
            message_text: "hola".to_owned(),
            is_private: false,
            is_mention: false,
            is_reply: false,
            reply_text: String::new(),
            ignore_link_fix_followups: true,
            is_non_ai_command_followup: false,
            ai_command_followups: true,
            random_replies_enabled: true,
            trigger_words: None,
            random_sample: None,
        }
    }

    #[test]
    fn identifies_legacy_link_replies_by_url_host_not_substrings() {
        for (text, ignored) in [
            ("https://vxtiktok.com/video/123", true),
            ("https://www.fxtwitter.com/a/status/1", true),
            ("https://en.fxtwitter.com/a/status/1", true),
            ("https://FXTWITTER.COM/a/status/1", true),
            ("about fxtwitter.com in prose", false),
            ("https://fxtwitter.com.evil.test/a", false),
            ("https://notfxtwitter.com/a", false),
            ("https://example.test/fxtwitter.com", false),
            ("https://fxtwitter.com@example.test/a", false),
        ] {
            let mut value = response_input();
            value.is_reply = true;
            value.reply_text = text.to_owned();
            value.trigger_words = Some(Vec::new());
            assert_eq!(
                evaluate_response_routing(&value),
                if ignored {
                    ResponseRoutingEvaluation::Ignore
                } else {
                    ResponseRoutingEvaluation::Respond
                },
                "{text}"
            );
        }
    }

    #[test]
    fn private_chats_and_known_commands_always_process_media() {
        let mut private = input();
        private.chat_type = "private".to_owned();
        private.bot_username = None;
        assert!(should_auto_process_media(&private));

        let mut command = input();
        command.known_command = true;
        command.bot_username = None;
        assert!(should_auto_process_media(&command));
    }

    #[test]
    fn groups_require_a_mention_or_exact_reply_username() {
        let mut mention = input();
        mention.message_text = "hola @TESTBOT".to_owned();
        assert!(should_auto_process_media(&mention));

        let mut reply = input();
        reply.reply_username = Some("testbot".to_owned());
        assert!(should_auto_process_media(&reply));

        let mut different_case_reply = input();
        different_case_reply.reply_username = Some("TestBot".to_owned());
        assert!(!should_auto_process_media(&different_case_reply));
    }

    #[test]
    fn missing_or_blank_username_disables_group_matching() {
        for username in [None, Some(String::new()), Some("  ".to_owned())] {
            let mut value = input();
            value.bot_username = username;
            value.message_text = "@testbot".to_owned();
            value.reply_username = Some("testbot".to_owned());
            assert!(!should_auto_process_media(&value));
        }
    }

    #[test]
    fn response_routing_requests_external_values_in_order() {
        let mut value = response_input();
        value.message_text = "hola bot".to_owned();
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::NeedsTriggerWords
        );

        value.trigger_words = Some(vec!["bot".to_owned()]);
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::NeedsRandomSample
        );

        value.random_sample = Some(0.05);
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::Respond
        );
        value.random_sample = Some(0.5);
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::Ignore
        );
    }

    fn triggers(text: &str, words: &[&str]) -> bool {
        let mut value = response_input();
        value.message_text = text.to_owned();
        value.trigger_words = Some(words.iter().map(|word| (*word).to_owned()).collect());
        evaluate_response_routing(&value) == ResponseRoutingEvaluation::NeedsRandomSample
    }

    #[test]
    fn notes_the_bot_adds_to_polls_are_not_trigger_words() {
        // A non-anonymous poll as the bot renders it for the model.
        let poll = "Sabes claramente de donde viene esta referencia?\n\
            Opciones:\n- Si\n- Me suena\n- No tengo idea\n\
            Telegram no le muestra al bot quién votó";
        assert!(!triggers(poll, &["bot"]));
        let anonymous = "Qué comemos?\nOpciones:\n- Pizza\n\
            Encuesta cerrada\nEncuesta anónima: nadie ve quién votó";
        assert!(!triggers(anonymous, &["encuesta", "opciones", "anónima"]));
        assert!(triggers(
            "Le preguntamos al bot?\nOpciones:\n- Si",
            &["bot"]
        ));
        assert!(triggers(
            "Qué comemos?\nOpciones:\n- Lo que diga el bot",
            &["bot"]
        ));
    }

    #[test]
    fn trigger_words_match_whole_words_plurals_and_stretched_endings() {
        for text in [
            "bot",
            "hola bot",
            "Bot, vení",
            "che gordo!",
            "GORDOOO",
            "los bots",
            "qué hacés (gordo)",
            "gordoo\nbien?",
        ] {
            assert!(triggers(text, &["bot", "gordo"]), "{text}");
        }
        for text in [
            "robot",
            "el botón",
            "bototo",
            "abot",
            "gordos2",
            "gordura",
            "",
        ] {
            assert!(!triggers(text, &["bot", "gordo"]), "{text}");
        }
        assert!(triggers("ya va el gordito", &["Gordito"]));
        assert!(triggers("che dogores", &["dogor"]));
        assert!(triggers("robot y bot", &["bot"]));
        assert!(!mentions_trigger_word("bot", ""));
    }

    #[test]
    fn deterministic_routes_still_preserve_legacy_random_sampling() {
        let mut value = response_input();
        value.known_command = true;
        value.message_text = "bot".to_owned();
        value.trigger_words = Some(vec!["bot".to_owned()]);
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::NeedsRandomSample
        );
        value.random_sample = Some(0.9);
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::Respond
        );
    }

    #[test]
    fn early_followup_suppression_needs_no_config_or_randomness() {
        let mut link = response_input();
        link.is_reply = true;
        link.reply_text = "https://fxtwitter.com/example/status/1".to_owned();
        assert_eq!(
            evaluate_response_routing(&link),
            ResponseRoutingEvaluation::Ignore
        );

        let mut command = response_input();
        command.is_reply = true;
        command.is_non_ai_command_followup = true;
        command.ai_command_followups = false;
        assert_eq!(
            evaluate_response_routing(&command),
            ResponseRoutingEvaluation::Ignore
        );
    }

    #[test]
    fn private_mentions_replies_and_commands_respond_after_config() {
        for configure in [
            |value: &mut ResponseRoutingInput| value.is_private = true,
            |value: &mut ResponseRoutingInput| value.is_mention = true,
            |value: &mut ResponseRoutingInput| value.is_reply = true,
            |value: &mut ResponseRoutingInput| value.known_command = true,
        ] {
            let mut value = response_input();
            value.random_replies_enabled = false;
            value.trigger_words = Some(Vec::new());
            configure(&mut value);
            assert_eq!(
                evaluate_response_routing(&value),
                ResponseRoutingEvaluation::Respond
            );
        }
    }

    #[test]
    fn unknown_slash_commands_never_use_incidental_routes() {
        let mut value = response_input();
        value.command_starts_with_slash = true;
        value.is_private = true;
        value.is_mention = true;
        value.is_reply = true;
        value.trigger_words = Some(Vec::new());
        assert_eq!(
            evaluate_response_routing(&value),
            ResponseRoutingEvaluation::Ignore
        );
    }

    proptest! {
        #[test]
        fn arbitrary_text_and_usernames_never_panic(
            text in ".{0,512}",
            bot_username in ".{0,64}",
            reply_username in proptest::option::of(".{0,64}")
        ) {
            let value = MediaRoutingInput {
                chat_type: "group".to_owned(),
                known_command: false,
                message_text: text,
                bot_username: Some(bot_username),
                reply_username,
            };
            let _decision = should_auto_process_media(&value);
        }
    }
}
