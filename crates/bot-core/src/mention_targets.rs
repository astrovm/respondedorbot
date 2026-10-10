//! The member a moderation command can name instead of replying to them:
//! either one picked from Telegram's mention list, which carries the user,
//! or an @username. Telegram doesn't let bots look a username up, so it is
//! matched against the members the bot has seen write in the chat.

use crate::chat_bans::BanTarget;
use crate::chat_members::KnownChatMember;
use crate::locale::Locale;
use crate::telegram_input::TextMention;

/// The stand-ins Telegram sends for anonymous admins and channel posts.
/// Members stored before the bot flag was saved can only be bots if they are
/// one of these, since bots never see other bots' messages.
const TELEGRAM_STAND_IN_BOTS: [&str; 2] = ["1087968824", "136817688"];

/// Who a command's leading word names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamedMember<'a> {
    /// Picked from the mention list, so Telegram sent who it is.
    Picked(&'a TextMention),
    /// Typed as `@username`, without the `@`.
    Username(&'a str),
}

/// Splits who a command names off its text, returning the rest. A member
/// picked from the mention list wins when the text starts with its name.
/// Otherwise any first word starting with `@` counts, even a mistyped one:
/// it then matches nobody, so a typo while replying to someone never falls
/// back to that member. Text that names nobody comes back whole.
#[must_use]
pub fn split_named_member<'a>(
    text: &'a str,
    mentions: &'a [TextMention],
) -> (Option<NamedMember<'a>>, &'a str) {
    let text = text.trim();
    let picked = mentions.iter().find_map(|mention| {
        let rest = text.strip_prefix(mention.text.as_str())?;
        let whole_name =
            !mention.text.trim().is_empty() && rest.chars().next().is_none_or(char::is_whitespace);
        whole_name.then_some((NamedMember::Picked(mention), rest.trim()))
    });
    if let Some((picked, rest)) = picked {
        return (Some(picked), rest);
    }
    let (first, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    match first.strip_prefix('@') {
        Some(username) => (Some(NamedMember::Username(username)), rest.trim()),
        None => (None, text),
    }
}

/// The member who used `username` most recently, ignoring case, so a
/// username someone gave up and another took points to the new owner.
#[must_use]
pub fn find_member_by_username<'a>(
    members: &'a [KnownChatMember],
    username: &str,
) -> Option<&'a KnownChatMember> {
    members
        .iter()
        .filter(|member| member.username.eq_ignore_ascii_case(username))
        .max_by_key(|member| member.last_seen)
}

/// A member found by @username, or `None` when the stored id isn't a number.
#[must_use]
pub fn known_member_target(member: &KnownChatMember) -> Option<BanTarget> {
    Some(BanTarget {
        user_id: member.user_id.parse().ok()?,
        is_bot: member.is_bot || TELEGRAM_STAND_IN_BOTS.contains(&member.user_id.as_str()),
        name: display_name(&member.first_name, &member.username, ""),
    })
}

#[must_use]
pub fn picked_member_target(mention: &TextMention) -> BanTarget {
    BanTarget {
        user_id: mention.user_id,
        is_bot: mention.is_bot,
        name: display_name(&mention.first_name, &mention.username, &mention.text),
    }
}

/// The first name, else the @username, else the text the mention covered.
fn display_name(first_name: &str, username: &str, text: &str) -> String {
    match (first_name.trim(), username.trim()) {
        ("", "") => text.trim().to_owned(),
        ("", username) => format!("@{username}"),
        (first_name, _) => first_name.to_owned(),
    }
}

#[must_use]
pub fn unknown_mention_reply(username: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!(
            "No sé quién es @{username}: tiene que haber escrito en el grupo, o respondé a un mensaje suyo"
        ),
        Locale::En => format!(
            "I don't know who @{username} is: they need to have written in the group, or reply to one of their messages"
        ),
    }
}

/// The members the bot has seen couldn't be read, so nobody can be named.
#[must_use]
pub fn mention_lookup_failed_reply(username: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => {
            format!("No pude buscar a @{username}, probá de nuevo o respondé a un mensaje suyo")
        }
        Locale::En => {
            format!("I couldn't look up @{username}. Try again, or reply to one of their messages")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NamedMember, find_member_by_username, known_member_target, mention_lookup_failed_reply,
        picked_member_target, split_named_member, unknown_mention_reply,
    };
    use crate::chat_bans::BanTarget;
    use crate::chat_members::KnownChatMember;
    use crate::locale::Locale;
    use crate::telegram_input::TextMention;

    fn member(user_id: &str, first_name: &str, username: &str, last_seen: i64) -> KnownChatMember {
        KnownChatMember {
            user_id: user_id.to_owned(),
            first_name: first_name.to_owned(),
            username: username.to_owned(),
            last_seen,
            is_bot: false,
        }
    }

    fn picked(text: &str, user_id: i64) -> TextMention {
        TextMention {
            text: text.to_owned(),
            user_id,
            first_name: text.to_owned(),
            username: String::new(),
            is_bot: false,
        }
    }

    #[test]
    fn splits_a_leading_username_from_the_rest() {
        for (text, expected) in [
            ("@lemon", (Some("lemon"), "")),
            (" @Lemon_2  0 ", (Some("Lemon_2"), "0")),
            ("@lemon off", (Some("lemon"), "off")),
            ("3", (None, "3")),
            ("3 @lemon", (None, "3 @lemon")),
            // Mistyped usernames still count, so they match nobody.
            ("@", (Some(""), "")),
            ("@juan. 3", (Some("juan."), "3")),
            ("@juán", (Some("juán"), "")),
            ("", (None, "")),
        ] {
            let (named, rest) = split_named_member(text, &[]);
            assert_eq!(
                (named, rest),
                (expected.0.map(NamedMember::Username), expected.1),
                "{text}"
            );
        }
    }

    #[test]
    fn a_picked_member_wins_when_the_text_starts_with_their_name() {
        let mentions = [picked("Lemon Pie", 77), picked("Ana", 78)];
        assert_eq!(
            split_named_member(" Lemon Pie 3 ", &mentions),
            (Some(NamedMember::Picked(&mentions[0])), "3")
        );
        assert_eq!(
            split_named_member("Ana", &mentions),
            (Some(NamedMember::Picked(&mentions[1])), "")
        );
        // Only a whole name at the start counts.
        assert_eq!(
            split_named_member("Anabel 3", &mentions),
            (None, "Anabel 3")
        );
        assert_eq!(
            split_named_member("3 Lemon Pie", &mentions),
            (None, "3 Lemon Pie")
        );
        assert_eq!(
            split_named_member("@lemon 3", &mentions),
            (Some(NamedMember::Username("lemon")), "3")
        );
        // An empty mention never swallows the text.
        let empty = [picked(" ", 79)];
        assert_eq!(split_named_member(" 3", &empty), (None, "3"));
    }

    #[test]
    fn finds_the_most_recent_owner_of_a_username_ignoring_case() {
        let members = [
            member("1", "Old", "lemon", 10),
            member("2", "New", "LEMON", 20),
            member("3", "Other", "lime", 30),
        ];
        assert_eq!(
            find_member_by_username(&members, "Lemon").map(|found| found.user_id.as_str()),
            Some("2")
        );
        assert_eq!(find_member_by_username(&members, "orange"), None);
    }

    #[test]
    fn known_members_are_bots_by_their_flag_or_as_telegram_stand_ins() {
        let target = |member: &KnownChatMember| known_member_target(member);
        assert_eq!(
            target(&member("7", " Ana ", "ana", 1)),
            Some(BanTarget {
                user_id: 7,
                is_bot: false,
                name: "Ana".to_owned(),
            })
        );
        assert_eq!(
            target(&member("7", " ", "ana", 1)).map(|target| target.name),
            Some("@ana".to_owned())
        );
        // A person whose username ends in "bot" is still a person.
        assert_eq!(
            target(&member("8", "Abbot", "the_abbot", 1)).map(|target| target.is_bot),
            Some(false)
        );
        let flagged = KnownChatMember {
            is_bot: true,
            ..member("9", "Helper", "helper", 1)
        };
        assert_eq!(target(&flagged).map(|target| target.is_bot), Some(true));
        for stand_in in ["1087968824", "136817688"] {
            assert_eq!(
                target(&member(stand_in, "Group", "GroupAnonymousBot", 1))
                    .map(|target| target.is_bot),
                Some(true),
                "{stand_in}"
            );
        }
        assert_eq!(target(&member("x", "Broken", "broken", 1)), None);
    }

    #[test]
    fn picked_members_keep_telegrams_bot_flag_and_best_name() {
        assert_eq!(
            picked_member_target(&picked("Lemon Pie", 77)),
            BanTarget {
                user_id: 77,
                is_bot: false,
                name: "Lemon Pie".to_owned(),
            }
        );
        let bot = TextMention {
            text: "Helper".to_owned(),
            user_id: 78,
            first_name: String::new(),
            username: "helper_bot".to_owned(),
            is_bot: true,
        };
        assert_eq!(
            picked_member_target(&bot),
            BanTarget {
                user_id: 78,
                is_bot: true,
                name: "@helper_bot".to_owned(),
            }
        );
        let nameless = TextMention {
            first_name: String::new(),
            ..picked(" Lemon ", 79)
        };
        assert_eq!(picked_member_target(&nameless).name, "Lemon");
    }

    #[test]
    fn replies_explain_how_to_reach_the_member() {
        assert_eq!(
            unknown_mention_reply("lemon", Locale::Es),
            "No sé quién es @lemon: tiene que haber escrito en el grupo, o respondé a un mensaje suyo"
        );
        assert_eq!(
            unknown_mention_reply("lemon", Locale::En),
            "I don't know who @lemon is: they need to have written in the group, or reply to one of their messages"
        );
        assert_eq!(
            mention_lookup_failed_reply("lemon", Locale::Es),
            "No pude buscar a @lemon, probá de nuevo o respondé a un mensaje suyo"
        );
        assert_eq!(
            mention_lookup_failed_reply("lemon", Locale::En),
            "I couldn't look up @lemon. Try again, or reply to one of their messages"
        );
    }
}
