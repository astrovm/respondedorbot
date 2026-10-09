//! Plans and replies for the commands group admins use to give one member
//! their own hourly limit of AI messages paid by the group.

use crate::chat_bans::{BanTarget, ban_reply};
use crate::command_parsing::parse_command;
use crate::locale::Locale;
use crate::telegram_actions::TelegramAction;
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommand {
    /// The text after `/limit`: a number of messages per hour, or `off`.
    Change(String),
    List,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitCommandContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub sender_id: i64,
    pub locale: Locale,
    /// The member the command replies to, bots included so the planner can
    /// refuse them.
    pub target: Option<BanTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommandPlan {
    Reply(TelegramAction),
    Set {
        user_id: i64,
        name: String,
        hourly_limit: i64,
    },
    Clear {
        user_id: i64,
        name: String,
    },
    List,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitedUser {
    pub user_id: i64,
    pub name: String,
    pub hourly_limit: i64,
}

#[must_use]
pub fn classify_limit_command(message_text: &str, bot_name: &str) -> Option<LimitCommand> {
    let parsed = parse_command(message_text, bot_name);
    match parsed.command.as_str() {
        "/limitar" | "/limit" => Some(LimitCommand::Change(parsed.message_text)),
        "/limitados" | "/limited" => Some(LimitCommand::List),
        _ => None,
    }
}

/// `Some(None)` removes the member's own limit; `Some(Some(n))` sets it.
fn parse_limit_argument(argument: &str) -> Option<Option<i64>> {
    let argument = argument.trim();
    if argument.eq_ignore_ascii_case("off") {
        return Some(None);
    }
    argument
        .parse::<u32>()
        .ok()
        .map(|limit| Some(i64::from(limit)))
}

/// Admin rights, of the sender and of the target, are checked by the caller,
/// which owns the Telegram lookup.
#[must_use]
pub fn plan_limit_command(command: LimitCommand, context: LimitCommandContext) -> LimitCommandPlan {
    let reply =
        |text: &str| LimitCommandPlan::Reply(ban_reply(context.chat_id, context.message_id, text));
    let locale = context.locale;
    let LimitCommand::Change(argument) = command else {
        return LimitCommandPlan::List;
    };
    let (Some(target), Some(argument)) = (context.target, parse_limit_argument(&argument)) else {
        return reply(match locale {
            Locale::Es => {
                "Respondé al mensaje de alguien con /limitar y cuántos mensajes por hora le paga el grupo, o con /limitar off para sacarle el límite"
            }
            Locale::En => {
                "Reply to someone's message with /limit and how many messages per hour the group pays for, or /limit off to remove it"
            }
        });
    };
    // Removing a limit never refuses: a bot or yourself simply has none.
    let Some(hourly_limit) = argument else {
        return LimitCommandPlan::Clear {
            user_id: target.user_id,
            name: target.name,
        };
    };
    if target.user_id == context.sender_id {
        return reply(match locale {
            Locale::Es => "No podés limitarte",
            Locale::En => "You can't limit yourself",
        });
    }
    if target.is_bot {
        return reply(match locale {
            Locale::Es => "A los bots no los puedo limitar",
            Locale::En => "I can't limit bots",
        });
    }
    LimitCommandPlan::Set {
        user_id: target.user_id,
        name: target.name,
        hourly_limit,
    }
}

#[must_use]
pub const fn limit_admin_target(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "A los admins no los puedo limitar",
        Locale::En => "I can't limit admins",
    }
}

#[must_use]
pub fn limit_result_reply(name: &str, hourly_limit: i64, locale: Locale) -> String {
    match (hourly_limit, locale) {
        (0, Locale::Es) => {
            format!("Listo, {name} ya no puede usar el saldo del grupo, solo el suyo")
        }
        (0, Locale::En) => {
            format!("Done, {name} can't use the group's balance anymore, only their own")
        }
        (1, Locale::Es) => format!("Listo, el grupo le paga a {name} hasta 1 mensaje por hora"),
        (1, Locale::En) => {
            format!("Done, the group pays for up to 1 message per hour from {name}")
        }
        (_, Locale::Es) => {
            format!("Listo, el grupo le paga a {name} hasta {hourly_limit} mensajes por hora")
        }
        (_, Locale::En) => {
            format!("Done, the group pays for up to {hourly_limit} messages per hour from {name}")
        }
    }
}

#[must_use]
pub fn unlimit_result_reply(name: &str, removed: bool, locale: Locale) -> String {
    match (removed, locale) {
        (true, Locale::Es) => format!("Listo, {name} vuelve al límite del grupo"),
        (true, Locale::En) => format!("Done, {name} is back to the group's limit"),
        (false, Locale::Es) => format!("{name} no tenía un límite propio"),
        (false, Locale::En) => format!("{name} had no limit of their own"),
    }
}

#[must_use]
pub fn render_limit_list(users: &[LimitedUser], locale: Locale) -> String {
    if users.is_empty() {
        return match locale {
            Locale::Es => "Nadie tiene un límite propio en este grupo",
            Locale::En => "Nobody has their own limit in this group",
        }
        .to_owned();
    }
    let header = match locale {
        Locale::Es => "Límites propios en este grupo",
        Locale::En => "Own limits in this group",
    };
    let lines = users
        .iter()
        .map(|user| {
            let name = user.name.trim();
            let name = if name.is_empty() {
                user.user_id.to_string()
            } else {
                name.to_owned()
            };
            let limit = match (user.hourly_limit, locale) {
                (0, Locale::Es) => "solo sus créditos".to_owned(),
                (0, Locale::En) => "own credits only".to_owned(),
                (limit, Locale::Es) => format!("{limit} por hora"),
                (limit, Locale::En) => format!("{limit} per hour"),
            };
            format!("- {name}: {limit}")
        })
        .collect::<Vec<_>>();
    format!("{header}\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{
        LimitCommand, LimitCommandContext, LimitCommandPlan, LimitedUser, classify_limit_command,
        limit_admin_target, limit_result_reply, plan_limit_command, render_limit_list,
        unlimit_result_reply,
    };
    use crate::chat_bans::BanTarget;
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn context(locale: Locale, target: Option<BanTarget>) -> LimitCommandContext {
        LimitCommandContext {
            chat_id: ChatId(-100),
            message_id: MessageId(7),
            sender_id: 1,
            locale,
            target,
        }
    }

    fn change(argument: &str) -> LimitCommand {
        LimitCommand::Change(argument.to_owned())
    }

    fn target(user_id: i64, is_bot: bool) -> Option<BanTarget> {
        Some(BanTarget {
            user_id,
            is_bot,
            name: "Ana".to_owned(),
        })
    }

    fn reply_text(plan: &LimitCommandPlan) -> Option<&str> {
        match plan {
            LimitCommandPlan::Reply(TelegramAction::SendMessage(message)) => {
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
            ("/limitar 3", Some(change("3"))),
            ("/limit@gordo_bot off", Some(change("off"))),
            ("/LIMIT", Some(change(""))),
            ("/limit  0", Some(change("0"))),
            ("/limitados", Some(LimitCommand::List)),
            ("/limited extra", Some(LimitCommand::List)),
            ("/limite 3", None),
            ("limit 3", None),
            ("", None),
            ("/limit@other_bot 3", None),
        ] {
            assert_eq!(
                classify_limit_command(text, "gordo_bot"),
                expected,
                "{text}"
            );
        }
    }

    #[test]
    fn list_needs_no_target() {
        assert_eq!(
            plan_limit_command(LimitCommand::List, context(Locale::Es, None)),
            LimitCommandPlan::List
        );
        assert_eq!(reply_text(&LimitCommandPlan::List), None);
    }

    #[test]
    fn missing_reply_or_bad_argument_explains_how() {
        for (locale, expected) in [
            (
                Locale::Es,
                "Respondé al mensaje de alguien con /limitar y cuántos mensajes por hora le paga el grupo, o con /limitar off para sacarle el límite",
            ),
            (
                Locale::En,
                "Reply to someone's message with /limit and how many messages per hour the group pays for, or /limit off to remove it",
            ),
        ] {
            for (target, argument) in [
                (None, "3"),
                (target(2, false), ""),
                (target(2, false), "-1"),
                (target(2, false), "tres"),
                (target(2, false), "3 por abuso"),
                (target(2, false), "99999999999"),
            ] {
                let plan = plan_limit_command(change(argument), context(locale, target));
                assert_eq!(reply_text(&plan), Some(expected), "{argument}");
            }
        }
    }

    #[test]
    fn setting_refuses_self_and_bots() {
        for (locale, expected) in [
            (Locale::Es, "No podés limitarte"),
            (Locale::En, "You can't limit yourself"),
        ] {
            let plan = plan_limit_command(change("3"), context(locale, target(1, false)));
            assert_eq!(reply_text(&plan), Some(expected));
        }
        for (locale, expected) in [
            (Locale::Es, "A los bots no los puedo limitar"),
            (Locale::En, "I can't limit bots"),
        ] {
            let plan = plan_limit_command(change("3"), context(locale, target(2, true)));
            assert_eq!(reply_text(&plan), Some(expected));
        }
    }

    #[test]
    fn set_and_clear_carry_the_replied_member() {
        for (argument, hourly_limit) in [("0", 0), (" 3 ", 3), ("+4", 4)] {
            assert_eq!(
                plan_limit_command(change(argument), context(Locale::Es, target(2, false))),
                LimitCommandPlan::Set {
                    user_id: 2,
                    name: "Ana".to_owned(),
                    hourly_limit
                }
            );
        }
        for target in [target(2, true), target(1, false)] {
            let user_id = target.as_ref().map_or(0, |target| target.user_id);
            assert_eq!(
                plan_limit_command(change("OFF"), context(Locale::En, target)),
                LimitCommandPlan::Clear {
                    user_id,
                    name: "Ana".to_owned()
                }
            );
        }
    }

    #[test]
    fn localized_replies() {
        assert_eq!(
            limit_admin_target(Locale::Es),
            "A los admins no los puedo limitar"
        );
        assert_eq!(limit_admin_target(Locale::En), "I can't limit admins");
        for (hourly_limit, locale, expected) in [
            (
                0,
                Locale::Es,
                "Listo, Ana ya no puede usar el saldo del grupo, solo el suyo",
            ),
            (
                0,
                Locale::En,
                "Done, Ana can't use the group's balance anymore, only their own",
            ),
            (
                1,
                Locale::Es,
                "Listo, el grupo le paga a Ana hasta 1 mensaje por hora",
            ),
            (
                1,
                Locale::En,
                "Done, the group pays for up to 1 message per hour from Ana",
            ),
            (
                3,
                Locale::Es,
                "Listo, el grupo le paga a Ana hasta 3 mensajes por hora",
            ),
            (
                3,
                Locale::En,
                "Done, the group pays for up to 3 messages per hour from Ana",
            ),
        ] {
            assert_eq!(limit_result_reply("Ana", hourly_limit, locale), expected);
        }
        assert_eq!(
            unlimit_result_reply("Ana", true, Locale::Es),
            "Listo, Ana vuelve al límite del grupo"
        );
        assert_eq!(
            unlimit_result_reply("Ana", true, Locale::En),
            "Done, Ana is back to the group's limit"
        );
        assert_eq!(
            unlimit_result_reply("Ana", false, Locale::Es),
            "Ana no tenía un límite propio"
        );
        assert_eq!(
            unlimit_result_reply("Ana", false, Locale::En),
            "Ana had no limit of their own"
        );
    }

    #[test]
    fn limit_list_names_members_and_falls_back_to_ids() {
        assert_eq!(
            render_limit_list(&[], Locale::Es),
            "Nadie tiene un límite propio en este grupo"
        );
        assert_eq!(
            render_limit_list(&[], Locale::En),
            "Nobody has their own limit in this group"
        );
        let users = [
            LimitedUser {
                user_id: 2,
                name: "Ana ".to_owned(),
                hourly_limit: 3,
            },
            LimitedUser {
                user_id: 3,
                name: "  ".to_owned(),
                hourly_limit: 0,
            },
        ];
        assert_eq!(
            render_limit_list(&users, Locale::Es),
            "Límites propios en este grupo\n- Ana: 3 por hora\n- 3: solo sus créditos"
        );
        assert_eq!(
            render_limit_list(&users, Locale::En),
            "Own limits in this group\n- Ana: 3 per hour\n- 3: own credits only"
        );
    }
}
