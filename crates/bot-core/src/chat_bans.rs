//! Plans and replies for the commands group admins use to ban members from
//! the bot in their group.

use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanCommand {
    Ban,
    Unban,
    List,
}

/// The member a ban command replies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BanTarget {
    pub user_id: i64,
    pub is_bot: bool,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BanCommandContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub sender_id: i64,
    pub locale: Locale,
    pub target: Option<BanTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BanCommandPlan {
    Reply(TelegramAction),
    Ban { user_id: i64, name: String },
    Unban { user_id: i64, name: String },
    List,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BannedUser {
    pub user_id: i64,
    pub name: String,
}

#[must_use]
pub fn classify_ban_command(message_text: &str, bot_name: &str) -> Option<BanCommand> {
    match parse_command(message_text, bot_name).command.as_str() {
        "/vetar" | "/ban" => Some(BanCommand::Ban),
        "/desvetar" | "/unban" => Some(BanCommand::Unban),
        "/vetados" | "/banned" => Some(BanCommand::List),
        _ => None,
    }
}

#[must_use]
pub fn ban_reply(chat_id: ChatId, message_id: MessageId, text: &str) -> TelegramAction {
    let mut message = SendMessage::new(chat_id, text);
    message.reply_to_message_id = Some(message_id);
    TelegramAction::SendMessage(message)
}

/// Admin rights are checked by the caller, which owns the Telegram lookup.
#[must_use]
pub fn plan_ban_command(command: BanCommand, context: BanCommandContext) -> BanCommandPlan {
    let reply =
        |text: &str| BanCommandPlan::Reply(ban_reply(context.chat_id, context.message_id, text));
    let locale = context.locale;
    if command == BanCommand::List {
        return BanCommandPlan::List;
    }
    let Some(target) = context.target else {
        return reply(match (command, locale) {
            (BanCommand::Ban, Locale::Es) => {
                "Respondé al mensaje de quien quieras vetar con /vetar"
            }
            (BanCommand::Ban, Locale::En) => "Reply to someone's message with /ban to ban them",
            (_, Locale::Es) => "Respondé al mensaje de quien quieras desvetar con /desvetar",
            (_, Locale::En) => "Reply to someone's message with /unban to unban them",
        });
    };
    if command == BanCommand::Unban {
        return BanCommandPlan::Unban {
            user_id: target.user_id,
            name: target.name,
        };
    }
    if target.user_id == context.sender_id {
        return reply(match locale {
            Locale::Es => "No podés vetarte",
            Locale::En => "You can't ban yourself",
        });
    }
    if target.is_bot {
        return reply(match locale {
            Locale::Es => "A los bots no los puedo vetar",
            Locale::En => "I can't ban bots",
        });
    }
    BanCommandPlan::Ban {
        user_id: target.user_id,
        name: target.name,
    }
}

#[must_use]
pub const fn bans_group_only(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "Esto funciona solo en grupos",
        Locale::En => "This only works in groups",
    }
}

#[must_use]
pub const fn ban_admin_target(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "A los admins no los puedo vetar",
        Locale::En => "I can't ban admins",
    }
}

#[must_use]
pub const fn ban_store_failed(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "No pude guardar el cambio, probá de nuevo",
        Locale::En => "I couldn't save that, try again",
    }
}

#[must_use]
pub const fn ban_list_failed(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "No pude cargar la lista, probá de nuevo",
        Locale::En => "I couldn't load the list, try again",
    }
}

#[must_use]
pub fn ban_result_reply(name: &str, inserted: bool, locale: Locale) -> String {
    match (inserted, locale) {
        (true, Locale::Es) => format!("Listo, {name} ya no puede usarme en este grupo"),
        (true, Locale::En) => format!("Done, {name} can't use me in this group anymore"),
        (false, Locale::Es) => format!("{name} ya tenía veto en este grupo"),
        (false, Locale::En) => format!("{name} was already banned in this group"),
    }
}

#[must_use]
pub fn unban_result_reply(name: &str, removed: bool, locale: Locale) -> String {
    match (removed, locale) {
        (true, Locale::Es) => format!("Listo, {name} puede volver a usarme"),
        (true, Locale::En) => format!("Done, {name} can use me again"),
        (false, Locale::Es) => format!("{name} no tenía veto"),
        (false, Locale::En) => format!("{name} wasn't banned"),
    }
}

#[must_use]
pub fn render_ban_list(users: &[BannedUser], locale: Locale) -> String {
    if users.is_empty() {
        return match locale {
            Locale::Es => "No hay nadie vetado en este grupo",
            Locale::En => "Nobody is banned in this group",
        }
        .to_owned();
    }
    let header = match locale {
        Locale::Es => "Vetados en este grupo",
        Locale::En => "Banned in this group",
    };
    let lines = users
        .iter()
        .map(|user| {
            let name = user.name.trim();
            if name.is_empty() {
                format!("- {}", user.user_id)
            } else {
                format!("- {name}")
            }
        })
        .collect::<Vec<_>>();
    format!("{header}\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{
        BanCommand, BanCommandContext, BanCommandPlan, BanTarget, BannedUser, ban_admin_target,
        ban_list_failed, ban_reply, ban_result_reply, ban_store_failed, bans_group_only,
        classify_ban_command, plan_ban_command, render_ban_list, unban_result_reply,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::{SendMessage, TelegramAction};
    use crate::telegram_input::{ChatId, MessageId};

    fn context(locale: Locale, target: Option<BanTarget>) -> BanCommandContext {
        BanCommandContext {
            chat_id: ChatId(-100),
            message_id: MessageId(7),
            sender_id: 1,
            locale,
            target,
        }
    }

    fn target(user_id: i64, is_bot: bool) -> Option<BanTarget> {
        Some(BanTarget {
            user_id,
            is_bot,
            name: "Ana".to_owned(),
        })
    }

    fn reply_text(plan: &BanCommandPlan) -> Option<&str> {
        match plan {
            BanCommandPlan::Reply(TelegramAction::SendMessage(message)) => {
                assert_eq!(message.chat_id, ChatId(-100));
                assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
                Some(message.text.as_str())
            }
            _ => None,
        }
    }

    #[test]
    fn classifies_spanish_english_and_addressed_aliases() {
        for (text, expected) in [
            ("/vetar", Some(BanCommand::Ban)),
            ("/ban@gordo_bot", Some(BanCommand::Ban)),
            ("/BAN", Some(BanCommand::Ban)),
            ("/desvetar", Some(BanCommand::Unban)),
            ("/unban", Some(BanCommand::Unban)),
            ("/vetados", Some(BanCommand::List)),
            ("/banned extra", Some(BanCommand::List)),
            ("/bann", None),
            ("ban", None),
            ("", None),
            ("/ban@other_bot", None),
        ] {
            assert_eq!(classify_ban_command(text, "gordo_bot"), expected, "{text}");
        }
    }

    #[test]
    fn list_needs_no_target() {
        assert_eq!(
            plan_ban_command(BanCommand::List, context(Locale::Es, None)),
            BanCommandPlan::List
        );
    }

    #[test]
    fn ban_and_unban_without_a_reply_explain_how() {
        for (command, locale, expected) in [
            (
                BanCommand::Ban,
                Locale::Es,
                "Respondé al mensaje de quien quieras vetar con /vetar",
            ),
            (
                BanCommand::Ban,
                Locale::En,
                "Reply to someone's message with /ban to ban them",
            ),
            (
                BanCommand::Unban,
                Locale::Es,
                "Respondé al mensaje de quien quieras desvetar con /desvetar",
            ),
            (
                BanCommand::Unban,
                Locale::En,
                "Reply to someone's message with /unban to unban them",
            ),
        ] {
            let plan = plan_ban_command(command, context(locale, None));
            assert_eq!(reply_text(&plan), Some(expected));
        }
    }

    #[test]
    fn ban_refuses_self_and_bots() {
        for (locale, expected) in [
            (Locale::Es, "No podés vetarte"),
            (Locale::En, "You can't ban yourself"),
        ] {
            let plan = plan_ban_command(BanCommand::Ban, context(locale, target(1, false)));
            assert_eq!(reply_text(&plan), Some(expected));
        }
        for (locale, expected) in [
            (Locale::Es, "A los bots no los puedo vetar"),
            (Locale::En, "I can't ban bots"),
        ] {
            let plan = plan_ban_command(BanCommand::Ban, context(locale, target(2, true)));
            assert_eq!(reply_text(&plan), Some(expected));
        }
    }

    #[test]
    fn ban_and_unban_carry_the_replied_member() {
        assert_eq!(
            plan_ban_command(BanCommand::Ban, context(Locale::Es, target(2, false))),
            BanCommandPlan::Ban {
                user_id: 2,
                name: "Ana".to_owned()
            }
        );
        // Unbanning never refuses: a bot or yourself simply has no ban.
        for target in [target(2, true), target(1, false)] {
            let user_id = target.as_ref().map_or(0, |target| target.user_id);
            assert_eq!(
                plan_ban_command(BanCommand::Unban, context(Locale::En, target)),
                BanCommandPlan::Unban {
                    user_id,
                    name: "Ana".to_owned()
                }
            );
        }
    }

    #[test]
    fn reply_quotes_the_command_message() {
        let mut expected = SendMessage::new(ChatId(5), "hola");
        expected.reply_to_message_id = Some(MessageId(9));
        assert_eq!(
            ban_reply(ChatId(5), MessageId(9), "hola"),
            TelegramAction::SendMessage(expected)
        );
    }

    #[test]
    fn localized_fixed_replies() {
        assert_eq!(bans_group_only(Locale::Es), "Esto funciona solo en grupos");
        assert_eq!(bans_group_only(Locale::En), "This only works in groups");
        assert_eq!(
            ban_admin_target(Locale::Es),
            "A los admins no los puedo vetar"
        );
        assert_eq!(ban_admin_target(Locale::En), "I can't ban admins");
        assert_eq!(
            ban_store_failed(Locale::Es),
            "No pude guardar el cambio, probá de nuevo"
        );
        assert_eq!(
            ban_store_failed(Locale::En),
            "I couldn't save that, try again"
        );
        assert_eq!(
            ban_list_failed(Locale::Es),
            "No pude cargar la lista, probá de nuevo"
        );
        assert_eq!(
            ban_list_failed(Locale::En),
            "I couldn't load the list, try again"
        );
    }

    #[test]
    fn result_replies_tell_new_changes_from_no_ops() {
        assert_eq!(
            ban_result_reply("Ana", true, Locale::Es),
            "Listo, Ana ya no puede usarme en este grupo"
        );
        assert_eq!(
            ban_result_reply("Ana", true, Locale::En),
            "Done, Ana can't use me in this group anymore"
        );
        assert_eq!(
            ban_result_reply("Ana", false, Locale::Es),
            "Ana ya tenía veto en este grupo"
        );
        assert_eq!(
            ban_result_reply("Ana", false, Locale::En),
            "Ana was already banned in this group"
        );
        assert_eq!(
            unban_result_reply("Ana", true, Locale::Es),
            "Listo, Ana puede volver a usarme"
        );
        assert_eq!(
            unban_result_reply("Ana", true, Locale::En),
            "Done, Ana can use me again"
        );
        assert_eq!(
            unban_result_reply("Ana", false, Locale::Es),
            "Ana no tenía veto"
        );
        assert_eq!(
            unban_result_reply("Ana", false, Locale::En),
            "Ana wasn't banned"
        );
    }

    #[test]
    fn ban_list_names_members_and_falls_back_to_ids() {
        assert_eq!(
            render_ban_list(&[], Locale::Es),
            "No hay nadie vetado en este grupo"
        );
        assert_eq!(
            render_ban_list(&[], Locale::En),
            "Nobody is banned in this group"
        );
        let users = [
            BannedUser {
                user_id: 2,
                name: "Ana ".to_owned(),
            },
            BannedUser {
                user_id: 3,
                name: "  ".to_owned(),
            },
        ];
        assert_eq!(
            render_ban_list(&users, Locale::Es),
            "Vetados en este grupo\n- Ana\n- 3"
        );
        assert_eq!(
            render_ban_list(&users, Locale::En),
            "Banned in this group\n- Ana\n- 3"
        );
    }
}
