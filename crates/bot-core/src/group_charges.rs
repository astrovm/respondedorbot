//! Plans and replies for the command group admins use to see who spent the
//! group's credits.

use crate::chat_bans::ban_reply;
use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, display_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::TelegramAction;
use crate::telegram_input::{ChatId, MessageId};

/// The AI ledger keeps 30 days by default, so older spending is gone. A
/// shorter `AI_LEDGER_RETENTION_DAYS` lowers it further.
pub const MAX_GROUP_CHARGES_DAYS: i64 = 30;
pub const GROUP_CHARGES_LIMIT: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSpender {
    pub user_id: i64,
    pub name: String,
    pub credit_units: i64,
    pub messages: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupChargesPlan {
    Reply(TelegramAction),
    Load { days: i64 },
}

/// Returns the text after `/gastosgrupo` or `/groupcharges`, or `None` for
/// any other message.
#[must_use]
pub fn classify_group_charges_command(message_text: &str, bot_name: &str) -> Option<String> {
    let parsed = parse_command(message_text, bot_name);
    matches!(parsed.command.as_str(), "/gastosgrupo" | "/groupcharges")
        .then_some(parsed.message_text)
}

/// How far back the list can go: the ledger's retention, up to
/// [`MAX_GROUP_CHARGES_DAYS`].
#[must_use]
pub fn group_charges_max_days(ledger_retention_days: i64) -> i64 {
    ledger_retention_days.clamp(1, MAX_GROUP_CHARGES_DAYS)
}

/// No argument is the last day; otherwise a number of days up to
/// `max_days`.
#[must_use]
pub fn plan_group_charges_command(
    argument: &str,
    max_days: i64,
    chat_id: ChatId,
    message_id: MessageId,
    locale: Locale,
) -> GroupChargesPlan {
    let argument = argument.trim();
    if argument.is_empty() {
        return GroupChargesPlan::Load { days: 1 };
    }
    match argument.parse::<i64>() {
        Ok(days) if (1..=max_days).contains(&days) => GroupChargesPlan::Load { days },
        _ => GroupChargesPlan::Reply(ban_reply(
            chat_id,
            message_id,
            &match locale {
                Locale::Es => format!(
                    "Mandá /gastosgrupo para el último día, o /gastosgrupo y una cantidad de días, hasta {max_days}"
                ),
                Locale::En => format!(
                    "Send /groupcharges for the last day, or /groupcharges and a number of days, up to {max_days}"
                ),
            },
        )),
    }
}

/// The first name, or the @username without one, or nothing so the list
/// falls back to the id.
#[must_use]
pub fn spender_name(first_name: &str, username: &str) -> String {
    match (first_name.trim(), username.trim()) {
        ("", "") => String::new(),
        ("", username) => format!("@{username}"),
        (first_name, _) => first_name.to_owned(),
    }
}

#[must_use]
pub const fn group_charges_failed(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "No pude cargar los gastos, probá de nuevo",
        Locale::En => "I couldn't load the spending, try again",
    }
}

#[must_use]
pub fn render_group_charges(spenders: &[GroupSpender], days: i64, locale: Locale) -> String {
    let period = match (days, locale) {
        (1, Locale::Es) => "las últimas 24 horas".to_owned(),
        (1, Locale::En) => "the last 24 hours".to_owned(),
        (_, Locale::Es) => format!("los últimos {days} días"),
        (_, Locale::En) => format!("the last {days} days"),
    };
    if spenders.is_empty() {
        return match locale {
            Locale::Es => format!("Nadie gastó créditos del grupo en {period}"),
            Locale::En => format!("Nobody spent the group's credits in {period}"),
        };
    }
    let header = match locale {
        Locale::Es => format!("Créditos del grupo gastados en {period}"),
        Locale::En => format!("Group credits spent in {period}"),
    };
    let lines = spenders
        .iter()
        .map(|spender| {
            let name = spender.name.trim();
            let name = if name.is_empty() {
                spender.user_id.to_string()
            } else {
                name.to_owned()
            };
            let credits = display_credit_units(CreditUnits::new(spender.credit_units));
            let messages = match (spender.messages, locale) {
                (1, Locale::Es) => "1 mensaje".to_owned(),
                (1, Locale::En) => "1 message".to_owned(),
                (count, Locale::Es) => format!("{count} mensajes"),
                (count, Locale::En) => format!("{count} messages"),
            };
            format!("- {name}: {credits} ({messages})")
        })
        .collect::<Vec<_>>();
    format!("{header}\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{
        GroupChargesPlan, GroupSpender, classify_group_charges_command, group_charges_failed,
        group_charges_max_days, plan_group_charges_command, render_group_charges, spender_name,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn plan(argument: &str, locale: Locale) -> GroupChargesPlan {
        plan_group_charges_command(argument, 30, ChatId(-100), MessageId(7), locale)
    }

    fn reply_text(plan: &GroupChargesPlan) -> Option<&str> {
        match plan {
            GroupChargesPlan::Reply(TelegramAction::SendMessage(message)) => {
                assert_eq!(message.chat_id, ChatId(-100));
                assert_eq!(message.reply_to_message_id, Some(MessageId(7)));
                Some(message.text.as_str())
            }
            _ => None,
        }
    }

    #[test]
    fn classifies_the_command_with_its_argument() {
        for (text, expected) in [
            ("/groupcharges", Some("")),
            ("/groupcharges@gordo_bot 7", Some("7")),
            ("/GROUPCHARGES  30", Some("30")),
            ("/gastosgrupo 3", Some("3")),
            ("/gastos", None),
            ("groupcharges", None),
            ("/groupcharges@other_bot", None),
        ] {
            assert_eq!(
                classify_group_charges_command(text, "gordo_bot").as_deref(),
                expected,
                "{text}"
            );
        }
    }

    #[test]
    fn days_default_to_one_and_stay_within_the_ledger_window() {
        for (argument, days) in [("", 1), (" 7 ", 7), ("30", 30), ("1", 1)] {
            let plan = plan(argument, Locale::Es);
            assert_eq!(reply_text(&plan), None);
            assert_eq!(plan, GroupChargesPlan::Load { days }, "{argument}");
        }
        for (locale, expected) in [
            (
                Locale::Es,
                "Mandá /gastosgrupo para el último día, o /gastosgrupo y una cantidad de días, hasta 30",
            ),
            (
                Locale::En,
                "Send /groupcharges for the last day, or /groupcharges and a number of days, up to 30",
            ),
        ] {
            for argument in ["0", "31", "-1", "una semana"] {
                let plan = plan(argument, locale);
                assert_eq!(reply_text(&plan), Some(expected), "{argument}");
            }
        }
    }

    #[test]
    fn a_shorter_ledger_retention_lowers_the_maximum() {
        for (retention, max_days) in [(7, 7), (1, 1), (30, 30), (90, 30), (0, 1), (-5, 1)] {
            assert_eq!(group_charges_max_days(retention), max_days, "{retention}");
        }
        let short = |argument| {
            plan_group_charges_command(argument, 7, ChatId(-100), MessageId(7), Locale::En)
        };
        assert_eq!(short("7"), GroupChargesPlan::Load { days: 7 });
        assert_eq!(
            reply_text(&short("8")),
            Some(
                "Send /groupcharges for the last day, or /groupcharges and a number of days, up to 7"
            )
        );
    }

    #[test]
    fn names_prefer_the_first_name_then_the_username() {
        assert_eq!(spender_name(" Ana ", "ana"), "Ana");
        assert_eq!(spender_name(" ", "ana"), "@ana");
        assert_eq!(spender_name("", " "), "");
    }

    #[test]
    fn localized_failure() {
        assert_eq!(
            group_charges_failed(Locale::Es),
            "No pude cargar los gastos, probá de nuevo"
        );
        assert_eq!(
            group_charges_failed(Locale::En),
            "I couldn't load the spending, try again"
        );
    }

    #[test]
    fn renders_spenders_with_credits_messages_and_id_fallback() {
        assert_eq!(
            render_group_charges(&[], 1, Locale::Es),
            "Nadie gastó créditos del grupo en las últimas 24 horas"
        );
        assert_eq!(
            render_group_charges(&[], 7, Locale::En),
            "Nobody spent the group's credits in the last 7 days"
        );
        let spenders = [
            GroupSpender {
                user_id: 2,
                name: "Ana ".to_owned(),
                credit_units: 123_450,
                messages: 8,
            },
            GroupSpender {
                user_id: 3,
                name: " ".to_owned(),
                credit_units: 5,
                messages: 1,
            },
        ];
        assert_eq!(
            render_group_charges(&spenders, 7, Locale::Es),
            "Créditos del grupo gastados en los últimos 7 días\n- Ana: 1,234.50 (8 mensajes)\n- 3: 0.05 (1 mensaje)"
        );
        assert_eq!(
            render_group_charges(&spenders, 1, Locale::En),
            "Group credits spent in the last 24 hours\n- Ana: 1,234.50 (8 messages)\n- 3: 0.05 (1 message)"
        );
    }
}
