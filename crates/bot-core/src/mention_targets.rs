//! The @username a moderation command can name instead of replying to the
//! member. Telegram doesn't let bots look a username up, so it is matched
//! against the members the bot has seen write in the chat.

use crate::chat_members::KnownChatMember;
use crate::locale::Locale;

/// Splits a leading `@username` off a command's text, returning it without
/// the `@` and the rest of the text. Any first word starting with `@` counts,
/// even a mistyped one: it then matches nobody, so a typo while replying to
/// someone never falls back to that member. Text that doesn't start with `@`
/// comes back whole.
#[must_use]
pub fn split_leading_mention(text: &str) -> (Option<&str>, &str) {
    let text = text.trim();
    let (first, rest) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
    match first.strip_prefix('@') {
        Some(username) => (Some(username), rest.trim()),
        None => (None, text),
    }
}

/// Telegram requires every bot's username to end in "bot", so a member
/// stored with one is a bot even though the stored data doesn't say so.
#[must_use]
pub fn is_bot_username(username: &str) -> bool {
    username.to_ascii_lowercase().ends_with("bot")
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

#[must_use]
pub fn mentioned_member_name(member: &KnownChatMember) -> String {
    match member.first_name.trim() {
        "" => format!("@{}", member.username),
        first_name => first_name.to_owned(),
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

#[cfg(test)]
mod tests {
    use super::{
        find_member_by_username, is_bot_username, mentioned_member_name, split_leading_mention,
        unknown_mention_reply,
    };
    use crate::chat_members::KnownChatMember;
    use crate::locale::Locale;

    fn member(user_id: &str, first_name: &str, username: &str, last_seen: i64) -> KnownChatMember {
        KnownChatMember {
            user_id: user_id.to_owned(),
            first_name: first_name.to_owned(),
            username: username.to_owned(),
            last_seen,
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
            assert_eq!(split_leading_mention(text), expected, "{text}");
        }
    }

    #[test]
    fn bot_usernames_end_in_bot() {
        for (username, expected) in [
            ("GroupAnonymousBot", true),
            ("channel_bot", true),
            ("robotic", false),
            ("lemon", false),
        ] {
            assert_eq!(is_bot_username(username), expected, "{username}");
        }
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
    fn names_prefer_the_first_name_then_the_username() {
        assert_eq!(
            mentioned_member_name(&member("1", " Ana ", "ana", 1)),
            "Ana"
        );
        assert_eq!(mentioned_member_name(&member("1", " ", "ana", 1)), "@ana");
    }

    #[test]
    fn unknown_usernames_explain_how_to_reach_the_member() {
        assert_eq!(
            unknown_mention_reply("lemon", Locale::Es),
            "No sé quién es @lemon: tiene que haber escrito en el grupo, o respondé a un mensaje suyo"
        );
        assert_eq!(
            unknown_mention_reply("lemon", Locale::En),
            "I don't know who @lemon is: they need to have written in the group, or reply to one of their messages"
        );
    }
}
