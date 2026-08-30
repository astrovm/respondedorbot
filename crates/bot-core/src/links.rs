//! Pure Telegram link parsing, social-front-end rewriting, and action planning.

use std::collections::HashSet;

use regex::Regex;
use url::Url;

use crate::locale::Locale;
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, SendMessage, TelegramAction,
};
use crate::telegram_input::{ChatId, MessageId};

const REPLACEABLE_HOSTS: [&str; 6] = [
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
];

const SOCIAL_HOSTS: [&str; 15] = [
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "eeinstagram.com",
    "vxinstagram.com",
    "kkinstagram.com",
    "rxddit.com",
    "www.reddit.com",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkReplacement {
    pub text: String,
    pub changed: bool,
    pub original_links: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMode {
    Reply,
    Delete,
    Off,
}

impl LinkMode {
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value {
            "off" => Self::Off,
            "delete" => Self::Delete,
            _ => Self::Reply,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkActionPlan {
    pub send: TelegramAction,
    pub delete_original: Option<TelegramAction>,
    pub stored_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkActionContext<'a> {
    pub chat_id: ChatId,
    pub incoming_message_id: MessageId,
    pub replied_message_id: Option<MessageId>,
    pub shared_by: Option<&'a str>,
    pub locale: Locale,
    pub link_context: Option<&'a str>,
}

fn normalized_host(url: &Url) -> String {
    let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
    host.strip_prefix("www.").unwrap_or(&host).to_owned()
}

fn host_matches(host: &str, domain: &str) -> bool {
    host == domain || host.ends_with(&format!(".{domain}"))
}

#[must_use]
pub fn is_social_frontend(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    SOCIAL_HOSTS
        .iter()
        .any(|domain| host_matches(&host, domain))
}

#[must_use]
pub fn has_replaceable_link(text: &str) -> bool {
    let Ok(pattern) = Regex::new(r#"https?://[^\s]+"#) else {
        return false;
    };
    pattern.find_iter(text).any(|matched| {
        let raw = matched.as_str().trim_matches(
            &[
                '(', ')', '[', ']', '{', '}', '<', '>', '"', '\'', '.', ',', ';', '!', '?',
            ][..],
        );
        Url::parse(raw).is_ok_and(|url| {
            let host = normalized_host(&url);
            REPLACEABLE_HOSTS
                .iter()
                .any(|domain| host_matches(&host, domain))
        })
    })
}

fn is_twitter_profile(url: &Url) -> bool {
    let host = normalized_host(url);
    if !matches!(host.as_str(), "twitter.com" | "x.com" | "xcancel.com") {
        return false;
    }
    let segments = url
        .path_segments()
        .into_iter()
        .flatten()
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.trim_start_matches('@').to_ascii_lowercase())
        .collect::<Vec<_>>();
    if segments.is_empty() || segments.iter().any(|segment| segment == "status") {
        return false;
    }
    if matches!(
        segments[0].as_str(),
        "home"
            | "share"
            | "intent"
            | "i"
            | "search"
            | "explore"
            | "notifications"
            | "messages"
            | "settings"
            | "compose"
            | "privacy"
            | "tos"
    ) {
        return false;
    }
    segments.len() == 1
        || (segments.len() == 2
            && matches!(segments[1].as_str(), "with_replies" | "media" | "likes"))
}

fn replacement_hosts(host: &str) -> Option<&'static [&'static str]> {
    match host {
        "twitter.com" => Some(&["fxtwitter.com"]),
        "x.com" | "xcancel.com" => Some(&["fixupx.com"]),
        "bsky.app" => Some(&["fxbsky.app"]),
        "instagram.com" => Some(&["eeinstagram.com", "vxinstagram.com", "kkinstagram.com"]),
        "reddit.com" => Some(&["rxddit.com"]),
        _ if host.ends_with(".reddit.com") => Some(&["rxddit.com"]),
        _ => None,
    }
}

fn clean_original(mut url: Url) -> String {
    url.set_query(None);
    url.set_fragment(None);
    url.to_string()
}

fn candidate_urls(url: &Url) -> Vec<Url> {
    let host = normalized_host(url);
    let Some(hosts) = replacement_hosts(&host) else {
        return Vec::new();
    };
    hosts
        .iter()
        .filter_map(|replacement_host| {
            let mut candidate = url.clone();
            let candidate_host = host.strip_suffix(".reddit.com").map_or_else(
                || (*replacement_host).to_owned(),
                |prefix| format!("{prefix}.{replacement_host}"),
            );
            candidate.set_host(Some(&candidate_host)).ok()?;
            if matches!(host.as_str(), "x.com" | "xcancel.com" | "twitter.com") {
                let normalized = Regex::new(r"(?i)^/i/(?:web/)?status/")
                    .ok()?
                    .replace(candidate.path(), "/status/")
                    .into_owned();
                candidate.set_path(&normalized);
            }
            candidate.set_query(None);
            candidate.set_fragment(None);
            Some(candidate)
        })
        .collect()
}

/// Replace supported social URLs only when the supplied live-preview checker
/// confirms that Telegram can render the alternative front end.
#[must_use]
pub fn replace_social_links(
    text: &str,
    unix_timestamp: i64,
    mut can_embed: impl FnMut(&str) -> bool,
) -> LinkReplacement {
    let Ok(pattern) = Regex::new(r"(?i)https?://[^\s]+") else {
        return LinkReplacement {
            text: text.to_owned(),
            changed: false,
            original_links: Vec::new(),
        };
    };
    let mut changed = false;
    let mut originals = Vec::new();
    let rewritten = pattern.replace_all(text, |captures: &regex::Captures<'_>| {
        let original = captures.get(0).map_or("", |value| value.as_str());
        let Ok(url) = Url::parse(original) else {
            return original.to_owned();
        };
        let host = normalized_host(&url);
        if is_twitter_profile(&url) || replacement_hosts(&host).is_none() {
            return clean_social_tracking(original);
        }
        for mut candidate in candidate_urls(&url) {
            let probe = candidate.to_string();
            if !can_embed(&probe) {
                continue;
            }
            changed = true;
            originals.push(clean_original(url.clone()));
            if matches!(
                normalized_host(&candidate).as_str(),
                "eeinstagram.com" | "vxinstagram.com" | "kkinstagram.com"
            ) {
                candidate.set_query(Some(&format!("tg={}", unix_timestamp.div_euclid(3600))));
            }
            return candidate.to_string();
        }
        original.to_owned()
    });
    LinkReplacement {
        text: rewritten.into_owned(),
        changed,
        original_links: originals,
    }
}

fn clean_social_tracking(raw: &str) -> String {
    let Ok(mut url) = Url::parse(raw) else {
        return raw.to_owned();
    };
    if !is_social_frontend(url.host_str().unwrap_or_default()) {
        return raw.to_owned();
    }
    let keep_instagram_bucket = matches!(
        normalized_host(&url).as_str(),
        "eeinstagram.com" | "vxinstagram.com" | "kkinstagram.com"
    ) && url
        .query()
        .is_some_and(|query| Regex::new(r"^tg=\d+$").is_ok_and(|pattern| pattern.is_match(query)));
    if !keep_instagram_bucket {
        url.set_query(None);
    }
    url.set_fragment(None);
    url.to_string()
}

#[must_use]
pub fn plan_link_actions(
    replacement: &LinkReplacement,
    mode: LinkMode,
    context: LinkActionContext<'_>,
) -> Option<LinkActionPlan> {
    if mode == LinkMode::Off || !replacement.changed {
        return None;
    }
    let mut text = replacement.text.clone();
    if let Some(shared_by) = context.shared_by.filter(|value| !value.is_empty()) {
        let label = match context.locale {
            Locale::Es => "compartido por",
            Locale::En => "shared by",
        };
        text.push_str(&format!("\n\n{label} {shared_by}"));
    }
    let stored_text = context
        .link_context
        .filter(|value| !value.is_empty())
        .map_or_else(|| text.clone(), |context| format!("{text}\n\n{context}"));
    let buttons = replacement
        .original_links
        .iter()
        .map(|url| InlineKeyboardButton {
            text: match context.locale {
                Locale::Es => "abrir en la app".to_owned(),
                Locale::En => "open in app".to_owned(),
            },
            url: Some(url.clone()),
            callback_data: None,
        })
        .map(|button| vec![button])
        .collect::<Vec<_>>();
    let mut message = SendMessage::new(context.chat_id, &text);
    message.reply_to_message_id = context
        .replied_message_id
        .or((mode == LinkMode::Reply).then_some(context.incoming_message_id));
    if !buttons.is_empty() {
        message.reply_markup = Some(InlineKeyboardMarkup {
            inline_keyboard: buttons,
        });
    }
    Some(LinkActionPlan {
        send: TelegramAction::SendMessage(message),
        delete_original: (mode == LinkMode::Delete).then_some(TelegramAction::DeleteMessage {
            chat_id: context.chat_id,
            message_id: context.incoming_message_id,
        }),
        stored_text,
    })
}

/// Slice text by Telegram's UTF-16 code-unit offsets, dropping incomplete
/// surrogate pairs in the same way as the legacy adapter's decoding policy.
#[must_use]
pub fn utf16_slice(text: &str, offset: i64, length: i64) -> String {
    if text.is_empty() || length <= 0 {
        return String::new();
    }
    let units = text.encode_utf16().collect::<Vec<_>>();
    let start = usize::try_from(offset.max(0))
        .unwrap_or(usize::MAX)
        .min(units.len());
    let requested_length = usize::try_from(length.max(0)).unwrap_or(usize::MAX);
    let end = start.saturating_add(requested_length).min(units.len());
    char::decode_utf16(units[start..end].iter().copied())
        .filter_map(Result::ok)
        .collect()
}

/// Remove message punctuation that Telegram's broad URL matcher may capture.
#[must_use]
pub fn trim_detected_url(raw_url: &str) -> String {
    raw_url
        .trim()
        .trim_end_matches(&['.', ',', ';', ':', '!', '?', ')', '"', ']', '}', '\''][..])
        .to_owned()
}

/// Keep the first occurrence of each URL and apply the configured message
/// limit. The legacy implementation returns one URL for a zero limit.
#[must_use]
pub fn select_unique_urls(candidates: &[String], max_links: usize) -> Vec<String> {
    let effective_limit = max_links.max(1);
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for candidate in candidates {
        if seen.insert(candidate.as_str()) {
            selected.push(candidate.clone());
            if selected.len() >= effective_limit {
                break;
            }
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::{
        LinkActionContext, LinkMode, LinkReplacement, has_replaceable_link, plan_link_actions,
        replace_social_links, select_unique_urls, trim_detected_url, utf16_slice,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    #[test]
    fn slices_telegram_utf16_offsets_across_emoji() {
        let text = "a😀 link";
        assert_eq!(utf16_slice(text, 3, 5), " link");
        assert_eq!(utf16_slice(text, 1, 1), "");
        assert_eq!(utf16_slice(text, -5, 1), "a");
        assert_eq!(utf16_slice(text, 0, 0), "");
        assert_eq!(utf16_slice("", 0, 2), "");
    }

    #[test]
    fn trims_only_the_legacy_url_suffix_characters() {
        assert_eq!(
            trim_detected_url("  https://example.test/path).  "),
            "https://example.test/path"
        );
        assert_eq!(
            trim_detected_url("https://example.test/path("),
            "https://example.test/path("
        );
    }

    #[test]
    fn deduplicates_stably_and_preserves_the_zero_limit_quirk() {
        let candidates = vec![
            "https://a.test".to_owned(),
            "https://a.test".to_owned(),
            "https://b.test".to_owned(),
            "https://c.test".to_owned(),
        ];
        assert_eq!(
            select_unique_urls(&candidates, 2),
            vec!["https://a.test".to_owned(), "https://b.test".to_owned()]
        );
        assert_eq!(
            select_unique_urls(&candidates, 0),
            vec!["https://a.test".to_owned()]
        );
        assert!(select_unique_urls(&[], 3).is_empty());
    }

    #[test]
    fn detects_supported_links_without_accepting_lookalikes() {
        assert!(has_replaceable_link("mirá https://www.x.com/a/status/1"));
        assert!(has_replaceable_link("https://old.reddit.com/r/rust"));
        assert!(!has_replaceable_link("https://notx.com/a/status/1"));
        assert!(!has_replaceable_link("https://fixupx.com/a/status/1"));
    }

    #[test]
    fn replaces_all_supported_frontends_and_strips_tracking() {
        let input = concat!(
            "https://twitter.com/a/status/1?utm=1 ",
            "https://x.com/i/status/2#x ",
            "https://xcancel.com/a/status/3?x=1 ",
            "https://bsky.app/profile/a/post/4?x=1 ",
            "https://instagram.com/reel/5?igsh=1 ",
            "https://old.reddit.com/r/rust/comments/6?x=1"
        );
        let mut probed = Vec::new();
        let result = replace_social_links(input, 7_200, |candidate| {
            probed.push(candidate.to_owned());
            true
        });
        assert!(result.changed);
        assert_eq!(
            result.text,
            concat!(
                "https://fxtwitter.com/a/status/1 ",
                "https://fixupx.com/status/2 ",
                "https://fixupx.com/a/status/3 ",
                "https://fxbsky.app/profile/a/post/4 ",
                "https://eeinstagram.com/reel/5?tg=2 ",
                "https://old.rxddit.com/r/rust/comments/6"
            )
        );
        assert_eq!(result.original_links.len(), 6);
        assert_eq!(probed.len(), 6);
    }

    #[test]
    fn falls_back_between_instagram_frontends_and_keeps_failed_links() {
        let mut probes = Vec::new();
        let result = replace_social_links(
            "https://instagram.com/p/one https://x.com/a/status/2",
            0,
            |candidate| {
                probes.push(candidate.to_owned());
                candidate.contains("kkinstagram")
            },
        );
        assert_eq!(
            result.text,
            "https://kkinstagram.com/p/one?tg=0 https://x.com/a/status/2"
        );
        assert_eq!(result.original_links, vec!["https://instagram.com/p/one"]);
        assert_eq!(probes.len(), 4);
    }

    #[test]
    fn skips_twitter_profiles_and_preserves_non_social_urls() {
        let mut calls = 0;
        let result = replace_social_links(
            "https://twitter.com/alice/media?x=1 https://example.com/?x=1",
            0,
            |_| {
                calls += 1;
                true
            },
        );
        assert_eq!(
            result.text,
            "https://twitter.com/alice/media https://example.com/?x=1"
        );
        assert!(!result.changed);
        assert_eq!(calls, 0);
    }

    #[test]
    fn plans_reply_and_delete_side_effects_with_localized_identity() {
        let replacement = LinkReplacement {
            text: "https://fixupx.com/a/status/1".to_owned(),
            changed: true,
            original_links: vec!["https://x.com/a/status/1".to_owned()],
        };
        let reply = plan_link_actions(
            &replacement,
            LinkMode::Reply,
            LinkActionContext {
                chat_id: ChatId(42),
                incoming_message_id: MessageId(7),
                replied_message_id: None,
                shared_by: Some("@ana"),
                locale: Locale::Es,
                link_context: Some("LINKS DEL MENSAJE"),
            },
        );
        let Some(reply) = reply else {
            return;
        };
        let TelegramAction::SendMessage(message) = reply.send else {
            return;
        };
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(
            message.text,
            "https://fixupx.com/a/status/1\n\ncompartido por @ana"
        );
        assert!(message.reply_markup.is_some());
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard[0][0].text.as_str()),
            Some("abrir en la app")
        );
        assert!(reply.delete_original.is_none());
        assert!(reply.stored_text.ends_with("LINKS DEL MENSAJE"));

        let delete = plan_link_actions(
            &replacement,
            LinkMode::Delete,
            LinkActionContext {
                chat_id: ChatId(42),
                incoming_message_id: MessageId(7),
                replied_message_id: Some(MessageId(3)),
                shared_by: Some("Ana"),
                locale: Locale::En,
                link_context: None,
            },
        );
        let Some(delete) = delete else {
            return;
        };
        let TelegramAction::SendMessage(message) = delete.send else {
            return;
        };
        assert_eq!(message.reply_to_message_id, Some(MessageId(3)));
        assert_eq!(
            message.text,
            "https://fixupx.com/a/status/1\n\nshared by Ana"
        );
        assert_eq!(
            message
                .reply_markup
                .as_ref()
                .map(|markup| markup.inline_keyboard[0][0].text.as_str()),
            Some("open in app")
        );
        assert_eq!(
            delete.delete_original,
            Some(TelegramAction::DeleteMessage {
                chat_id: ChatId(42),
                message_id: MessageId(7),
            })
        );
    }

    #[test]
    fn does_not_plan_off_or_unchanged_replacements() {
        let unchanged = LinkReplacement {
            text: "text".to_owned(),
            changed: false,
            original_links: Vec::new(),
        };
        assert!(
            plan_link_actions(
                &unchanged,
                LinkMode::Reply,
                LinkActionContext {
                    chat_id: ChatId(1),
                    incoming_message_id: MessageId(2),
                    replied_message_id: None,
                    shared_by: None,
                    locale: Locale::Es,
                    link_context: None,
                },
            )
            .is_none()
        );
        let changed = LinkReplacement {
            changed: true,
            ..unchanged
        };
        assert!(
            plan_link_actions(
                &changed,
                LinkMode::Off,
                LinkActionContext {
                    chat_id: ChatId(1),
                    incoming_message_id: MessageId(2),
                    replied_message_id: None,
                    shared_by: None,
                    locale: Locale::Es,
                    link_context: None,
                },
            )
            .is_none()
        );
    }
}
