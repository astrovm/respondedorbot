//! Pure Telegram link parsing, social-front-end rewriting, and action planning.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use url::Url;

use crate::locale::Locale;
use crate::telegram_actions::{
    InlineKeyboardButton, InlineKeyboardMarkup, SendMessage, TelegramAction,
};
use crate::telegram_input::{ChatId, MessageId};

pub static HTTP_URL: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"https?://[^\s]+").ok());
static HTTP_URL_CASELESS: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)https?://[^\s]+").ok());
static X_STATUS_PATH: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?i)^/i/(?:web/)?status/").ok());
static INSTAGRAM_BUCKET_QUERY: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"^tg=\d+$").ok());

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
    HTTP_URL.as_ref().is_some_and(|pattern| {
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

fn candidate_urls(url: &Url, host: &str, hosts: &[&str]) -> Vec<Url> {
    hosts
        .iter()
        .filter_map(|replacement_host| {
            let mut candidate = url.clone();
            let candidate_host = host.strip_suffix(".reddit.com").map_or_else(
                || (*replacement_host).to_owned(),
                |prefix| format!("{prefix}.{replacement_host}"),
            );
            candidate.set_host(Some(&candidate_host)).ok()?;
            if matches!(host, "x.com" | "xcancel.com" | "twitter.com") {
                let normalized = X_STATUS_PATH
                    .as_ref()?
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
    let mut changed = false;
    let mut originals = Vec::new();
    let mut rewrite = |captures: &regex::Captures<'_>| {
        let original = captures.get(0).map_or("", |value| value.as_str());
        let Ok(url) = Url::parse(original) else {
            return original.to_owned();
        };
        let host = normalized_host(&url);
        let Some(hosts) = replacement_hosts(&host).filter(|_| !is_twitter_profile(&url)) else {
            return clean_social_tracking(url, original);
        };
        for mut candidate in candidate_urls(&url, &host, hosts) {
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
    };
    let rewritten = HTTP_URL_CASELESS
        .as_ref()
        .map_or(std::borrow::Cow::Borrowed(text), |pattern| {
            pattern.replace_all(text, &mut rewrite)
        });
    LinkReplacement {
        text: rewritten.into_owned(),
        changed,
        original_links: originals,
    }
}

fn clean_social_tracking(mut url: Url, raw: &str) -> String {
    if !is_social_frontend(url.host_str().unwrap_or_default()) {
        return raw.to_owned();
    }
    let keep_instagram_bucket = matches!(
        normalized_host(&url).as_str(),
        "eeinstagram.com" | "vxinstagram.com" | "kkinstagram.com"
    ) && url.query().is_some_and(|query| {
        INSTAGRAM_BUCKET_QUERY
            .as_ref()
            .is_some_and(|pattern| pattern.is_match(query))
    });
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
        .enumerate()
        .map(|(index, url)| InlineKeyboardButton {
            text: {
                let host = Url::parse(url)
                    .ok()
                    .and_then(|url| url.host_str().map(ToOwned::to_owned))
                    .unwrap_or_default();
                let site = match host.trim_start_matches("www.") {
                    "x.com" | "twitter.com" => "X",
                    "bsky.app" => "Bluesky",
                    "instagram.com" => "Instagram",
                    "reddit.com" | "old.reddit.com" => "Reddit",
                    other => other,
                };
                let verb = crate::menu_ui::localized(context.locale, "Abrir", "Open");
                if replacement.original_links.len() > 1 {
                    format!("{verb} {site} ({})", index + 1)
                } else {
                    format!("{verb} {site}")
                }
            },
            url: Some(url.clone()),
            callback_data: None,
            copy_text: None,
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
    use crate::telegram_actions::{SendMessage, TelegramAction};
    use crate::telegram_input::{ChatId, MessageId};

    /// Every test probes through this one closure type, so `replace_social_links`
    /// has a single instantiation that the tests cover together.
    fn replace(
        text: &str,
        unix_timestamp: i64,
        accept: fn(&str) -> bool,
        probes: &mut Vec<String>,
    ) -> LinkReplacement {
        replace_social_links(text, unix_timestamp, |candidate| {
            probes.push(candidate.to_owned());
            accept(candidate)
        })
    }

    fn sent(action: &TelegramAction) -> Option<&SendMessage> {
        match action {
            TelegramAction::SendMessage(message) => Some(message),
            _ => None,
        }
    }

    fn context(shared_by: Option<&str>) -> LinkActionContext<'_> {
        LinkActionContext {
            chat_id: ChatId(42),
            incoming_message_id: MessageId(7),
            replied_message_id: None,
            shared_by,
            locale: Locale::En,
            link_context: Some(""),
        }
    }

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
        let result = replace(input, 7_200, |_| true, &mut probed);
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
        let result = replace(
            "https://instagram.com/p/one https://x.com/a/status/2",
            0,
            |candidate| candidate.contains("kkinstagram"),
            &mut probes,
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
        let mut probes = Vec::new();
        let result = replace(
            "https://twitter.com/alice/media?x=1 https://example.com/?x=1 https://bsky.app/profile/a/post/1",
            0,
            |_| false,
            &mut probes,
        );
        assert_eq!(
            result.text,
            "https://twitter.com/alice/media https://example.com/?x=1 https://bsky.app/profile/a/post/1"
        );
        assert!(!result.changed);
        // Only the post link is probed; the profile and the non-social URL are not.
        assert_eq!(probes, vec!["https://fxbsky.app/profile/a/post/1"]);
    }

    #[test]
    fn plans_reply_and_delete_side_effects_with_localized_identity() -> Result<(), String> {
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
        let reply = reply.ok_or("reply plan")?;
        let message = sent(&reply.send).ok_or("reply message")?;
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
            Some("Abrir X")
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
        let delete = delete.ok_or("delete plan")?;
        let message = sent(&delete.send).ok_or("delete message")?;
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
            Some("Open X")
        );
        assert_eq!(
            delete.delete_original,
            Some(TelegramAction::DeleteMessage {
                chat_id: ChatId(42),
                message_id: MessageId(7),
            })
        );
        Ok(())
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

    #[test]
    fn link_modes_parse_with_reply_as_the_default() {
        assert_eq!(LinkMode::parse("off"), LinkMode::Off);
        assert_eq!(LinkMode::parse("delete"), LinkMode::Delete);
        assert_eq!(LinkMode::parse("reply"), LinkMode::Reply);
        assert_eq!(LinkMode::parse("anything"), LinkMode::Reply);
    }

    #[test]
    fn reserved_x_paths_invalid_urls_and_frontend_buckets_are_handled() {
        let result = replace(
            "https://x.com/search?q=rust http://[broken https://kkinstagram.com/p/a?tg=5 https://vxinstagram.com/p/b?igsh=1#top",
            0,
            |_| true,
            &mut Vec::new(),
        );
        assert_eq!(
            result.text,
            "https://fixupx.com/search http://[broken https://kkinstagram.com/p/a?tg=5 https://vxinstagram.com/p/b"
        );
        assert_eq!(result.original_links, vec!["https://x.com/search"]);
    }

    #[test]
    fn buttons_name_each_site_and_number_multiple_links() -> Result<(), String> {
        let replacement = LinkReplacement {
            text: "rewritten".to_owned(),
            changed: true,
            original_links: vec![
                "https://bsky.app/profile/a/post/1".to_owned(),
                "https://www.instagram.com/p/2".to_owned(),
                "https://old.reddit.com/r/rust".to_owned(),
                "https://news.example/story".to_owned(),
            ],
        };
        for shared_by in [None, Some("")] {
            let plan = plan_link_actions(&replacement, LinkMode::Delete, context(shared_by))
                .ok_or("delete plan")?;
            let message = sent(&plan.send).ok_or("message")?;
            assert_eq!(message.text, "rewritten");
            assert_eq!(message.reply_to_message_id, None);
            assert_eq!(plan.stored_text, "rewritten");
            let labels = message
                .reply_markup
                .as_ref()
                .map(|markup| {
                    markup
                        .inline_keyboard
                        .iter()
                        .map(|row| row[0].text.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            assert_eq!(
                labels,
                vec![
                    "Open Bluesky (1)",
                    "Open Instagram (2)",
                    "Open Reddit (3)",
                    "Open news.example (4)"
                ]
            );
        }
        let no_buttons = LinkReplacement {
            original_links: Vec::new(),
            ..replacement
        };
        let plan =
            plan_link_actions(&no_buttons, LinkMode::Reply, context(None)).ok_or("reply plan")?;
        let message = sent(&plan.send).ok_or("message")?;
        assert!(message.reply_markup.is_none());
        assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
        assert_eq!(
            sent(&TelegramAction::DeleteMessage {
                chat_id: ChatId(1),
                message_id: MessageId(2),
            }),
            None
        );
        Ok(())
    }
}
