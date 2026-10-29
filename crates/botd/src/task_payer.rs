//! Who pays for a scheduled task in a chat, shared by task creation and runs.

use bot_adapters::chat_config::ChatConfigRepository;
use bot_adapters::chat_limits::ChatLimitRepository;
use bot_core::chat_config::ChatConfig;
use bot_core::credit_units::{CreditUnits, display_credit_units};
use bot_core::locale::Locale;

use crate::dispatcher::GroupAuthorizer;
use crate::error_text;
use crate::task_tools::task_credit_insufficient;

/// How one member's task is paid for in the chat it runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskPayer {
    /// The group the task runs in, whether or not it pays.
    pub group_chat_id: Option<i64>,
    /// "Tareas con créditos del grupo" is on for that group.
    pub group_pays: bool,
    pub hourly_limit: i64,
    pub group_pays_first: bool,
}

impl TaskPayer {
    /// Private chats and tasks outside a group: personal credits only.
    pub const PERSONAL: Self = Self {
        group_chat_id: None,
        group_pays: false,
        hourly_limit: 0,
        group_pays_first: false,
    };

    /// The group whose credits may pay, if any.
    #[must_use]
    pub const fn paying_chat(&self) -> Option<i64> {
        if self.group_pays {
            self.group_chat_id
        } else {
            None
        }
    }

    /// A message from a member and a task of theirs are paid the same way.
    /// `own_limit` is the member's own hourly limit, set with /limit.
    #[must_use]
    pub fn for_group(group_chat_id: i64, config: &ChatConfig, own_limit: Option<i64>) -> Self {
        Self {
            group_chat_id: Some(group_chat_id),
            group_pays: config.group_pays_tasks,
            hourly_limit: own_limit.unwrap_or(config.creditless_user_hourly_limit),
            group_pays_first: config.group_pays_first,
        }
    }
}

/// Telegram group and supergroup chat IDs are negative.
#[must_use]
pub fn group_chat_id(chat_id: &str) -> Option<i64> {
    chat_id.parse::<i64>().ok().filter(|id| *id < 0)
}

/// Reads the chat settings when a task is created and again when it runs, so
/// switching the setting off stops group payment for existing tasks.
pub trait TaskPayerPolicy: Send {
    fn payer(&mut self, chat_id: &str, user_id: i64) -> Result<TaskPayer, String>;
}

pub struct ChatTaskPayerPolicy {
    config: ChatConfigRepository,
    limits: ChatLimitRepository,
    admins: Option<Box<dyn GroupAuthorizer + Send>>,
}

impl ChatTaskPayerPolicy {
    #[must_use]
    pub fn new(database_url: &str) -> Self {
        Self {
            config: ChatConfigRepository::new(database_url),
            limits: ChatLimitRepository::new(database_url),
            admins: None,
        }
    }

    /// Chat admins keep the group's limit, like their messages do.
    #[must_use]
    pub fn with_admin_check(mut self, admins: Box<dyn GroupAuthorizer + Send>) -> Self {
        self.admins = Some(admins);
        self
    }
}

/// A member's own /limit replaces the group's limit, except for chat admins
/// (and when the admin lookup fails the member limit stays, as for messages).
/// The lookup only happens when the member has an own limit.
fn member_limit(
    chat_id: &str,
    user_id: i64,
    own_limit: Option<i64>,
    admins: Option<&mut dyn GroupAuthorizer>,
) -> Option<i64> {
    let own_limit = own_limit?;
    let is_admin =
        admins.is_some_and(|admins| admins.authorize(chat_id, &user_id.to_string()).is_admin);
    if is_admin { None } else { Some(own_limit) }
}

impl TaskPayerPolicy for ChatTaskPayerPolicy {
    fn payer(&mut self, chat_id: &str, user_id: i64) -> Result<TaskPayer, String> {
        let Some(group) = group_chat_id(chat_id) else {
            return Ok(TaskPayer::PERSONAL);
        };
        let config = self
            .config
            .get(chat_id)
            .map_err(error_text)?
            .unwrap_or_default();
        let own_limit = self
            .limits
            .hourly_limit(group, user_id)
            .map_err(error_text)?;
        let own_limit = member_limit(
            chat_id,
            user_id,
            own_limit,
            self.admins
                .as_mut()
                .map(|admins| &mut **admins as &mut dyn GroupAuthorizer),
        );
        Ok(TaskPayer::for_group(group, &config, own_limit))
    }
}

/// What a member and their group have to pay with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskFunds {
    pub personal: i64,
    pub group: i64,
    /// Messages the group already paid for this member in the last hour.
    pub hourly_used: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskShortfall {
    /// Personal credits only, in a private chat.
    Personal,
    /// In a group where tasks don't use the group's credits.
    GroupOff,
    /// Personal and group credits are both short.
    Both,
    /// The group could pay but the member used their hourly allowance.
    HourlyLimit { limit: i64 },
}

/// Why `required` credit units can't be paid, or `None` when they can.
/// Mirrors how a run pays: the member's credits, or the group's while the
/// member is under their hourly allowance.
#[must_use]
pub fn task_shortfall(
    payer: &TaskPayer,
    funds: &TaskFunds,
    required: i64,
) -> Option<TaskShortfall> {
    if funds.personal >= required {
        return None;
    }
    if payer.group_chat_id.is_none() {
        return Some(TaskShortfall::Personal);
    }
    if !payer.group_pays {
        return Some(TaskShortfall::GroupOff);
    }
    if funds.group < required {
        return Some(TaskShortfall::Both);
    }
    let under_limit = payer.hourly_limit < 0 || funds.hourly_used < payer.hourly_limit;
    (!under_limit).then_some(TaskShortfall::HourlyLimit {
        limit: payer.hourly_limit,
    })
}

#[must_use]
pub fn task_shortfall_message(
    shortfall: TaskShortfall,
    funds: &TaskFunds,
    required: i64,
    locale: Locale,
) -> String {
    let balance = display_credit_units(CreditUnits::new(funds.personal));
    let group = display_credit_units(CreditUnits::new(funds.group));
    let need = display_credit_units(CreditUnits::new(required));
    match (shortfall, locale) {
        (TaskShortfall::Personal, _) => task_credit_insufficient(funds.personal, required, locale),
        (TaskShortfall::GroupOff, Locale::Es) => format!(
            "Tus créditos personales no alcanzan para esta tarea: tenés {balance} y necesitás {need}. En este grupo las tareas no usan los créditos del grupo (hay {group}). Un admin lo puede activar en /config → Tareas con créditos del grupo, o cargá con /topup."
        ),
        (TaskShortfall::GroupOff, Locale::En) => format!(
            "Your personal credits are not enough for this task: you have {balance} and need {need}. In this group, tasks do not use the group credits (there are {group}). An admin can turn that on in /config → Tasks on group credits, or add credits with /topup."
        ),
        (TaskShortfall::Both, Locale::Es) => format!(
            "Ni tus créditos personales ni los del grupo alcanzan para esta tarea: tenés {balance}, el grupo tiene {group} y necesitás {need}. Cargá con /topup."
        ),
        (TaskShortfall::Both, Locale::En) => format!(
            "Neither your personal credits nor the group credits are enough for this task: you have {balance}, the group has {group} and you need {need}. Add credits with /topup."
        ),
        (TaskShortfall::HourlyLimit { limit }, Locale::Es) => format!(
            "Tus créditos personales no alcanzan para esta tarea (tenés {balance}, necesitás {need}) y ya usaste tus {limit} mensajes por hora con créditos del grupo. Cargá con /topup o probá más tarde."
        ),
        (TaskShortfall::HourlyLimit { limit }, Locale::En) => format!(
            "Your personal credits are not enough for this task (you have {balance}, you need {need}) and you already used your {limit} messages per hour on group credits. Add credits with /topup or try later."
        ),
    }
}

#[cfg(test)]
mod tests {
    use bot_core::chat_config::ChatConfig;
    use bot_core::locale::Locale;

    use super::{
        TaskFunds, TaskPayer, TaskShortfall, group_chat_id, task_shortfall, task_shortfall_message,
    };

    fn group_payer(group_pays: bool, hourly_limit: i64) -> TaskPayer {
        TaskPayer {
            group_chat_id: Some(-100),
            group_pays,
            hourly_limit,
            group_pays_first: false,
        }
    }

    fn funds(personal: i64, group: i64, hourly_used: i64) -> TaskFunds {
        TaskFunds {
            personal,
            group,
            hourly_used,
        }
    }

    struct Admins {
        is_admin: bool,
        asked: Vec<(String, String)>,
    }

    impl crate::dispatcher::GroupAuthorizer for Admins {
        fn authorize(
            &mut self,
            chat_id: &str,
            user_id: &str,
        ) -> crate::dispatcher::GroupAuthorizationDecision {
            self.asked.push((chat_id.to_owned(), user_id.to_owned()));
            crate::dispatcher::GroupAuthorizationDecision {
                is_admin: self.is_admin,
                diagnostics: Vec::new(),
            }
        }
    }

    #[test]
    fn admins_keep_the_group_limit_and_members_keep_their_own() {
        let mut admin = Admins {
            is_admin: true,
            asked: Vec::new(),
        };
        assert_eq!(
            super::member_limit("-7", 9, Some(1), Some(&mut admin)),
            None
        );
        assert_eq!(admin.asked, [("-7".to_owned(), "9".to_owned())]);
        let mut member = Admins {
            is_admin: false,
            asked: Vec::new(),
        };
        assert_eq!(
            super::member_limit("-7", 9, Some(1), Some(&mut member)),
            Some(1)
        );
        assert_eq!(super::member_limit("-7", 9, Some(1), None), Some(1));
        // No own limit: nobody is looked up.
        assert_eq!(super::member_limit("-7", 9, None, Some(&mut member)), None);
        assert_eq!(member.asked.len(), 1);
    }

    #[test]
    fn only_negative_chat_ids_are_groups() {
        assert_eq!(group_chat_id("-1001"), Some(-1001));
        assert_eq!(group_chat_id("42"), None);
        assert_eq!(group_chat_id("not-a-chat"), None);
    }

    #[test]
    fn group_payment_follows_the_setting_and_the_members_own_limit() {
        let config = ChatConfig {
            creditless_user_hourly_limit: 4,
            group_pays_first: true,
            ..ChatConfig::default()
        };
        let payer = TaskPayer::for_group(-7, &config, None);
        assert_eq!(payer.paying_chat(), Some(-7));
        assert_eq!(payer.hourly_limit, 4);
        assert!(payer.group_pays_first);
        assert_eq!(TaskPayer::for_group(-7, &config, Some(1)).hourly_limit, 1);
        let off = ChatConfig {
            group_pays_tasks: false,
            ..config
        };
        let payer = TaskPayer::for_group(-7, &off, None);
        assert_eq!(payer.paying_chat(), None);
        assert_eq!(payer.group_chat_id, Some(-7));
        assert_eq!(TaskPayer::PERSONAL.paying_chat(), None);
    }

    #[test]
    fn shortfall_matches_how_a_run_pays() {
        let cases = [
            (TaskPayer::PERSONAL, funds(10, 999, 0), None),
            (
                TaskPayer::PERSONAL,
                funds(9, 999, 0),
                Some(TaskShortfall::Personal),
            ),
            (group_payer(true, 5), funds(10, 0, 5), None),
            (group_payer(true, 5), funds(0, 10, 4), None),
            (group_payer(true, -1), funds(0, 10, 500), None),
            (
                group_payer(true, 5),
                funds(0, 9, 4),
                Some(TaskShortfall::Both),
            ),
            (
                group_payer(true, 5),
                funds(0, 10, 5),
                Some(TaskShortfall::HourlyLimit { limit: 5 }),
            ),
            (
                group_payer(false, 5),
                funds(0, 999, 0),
                Some(TaskShortfall::GroupOff),
            ),
            (group_payer(false, 5), funds(10, 0, 0), None),
        ];
        for (payer, funds, expected) in cases {
            assert_eq!(task_shortfall(&payer, &funds, 10), expected, "{payer:?}");
        }
    }

    #[test]
    fn shortfall_messages_name_whose_credits_are_short() {
        let funds = funds(500, 1_000, 0);
        let expected = [
            (
                TaskShortfall::GroupOff,
                Locale::Es,
                "Tus créditos personales no alcanzan para esta tarea: tenés 5.00 y necesitás 20.00. En este grupo las tareas no usan los créditos del grupo (hay 10.00). Un admin lo puede activar en /config → Tareas con créditos del grupo, o cargá con /topup.",
            ),
            (
                TaskShortfall::GroupOff,
                Locale::En,
                "Your personal credits are not enough for this task: you have 5.00 and need 20.00. In this group, tasks do not use the group credits (there are 10.00). An admin can turn that on in /config → Tasks on group credits, or add credits with /topup.",
            ),
            (
                TaskShortfall::Both,
                Locale::Es,
                "Ni tus créditos personales ni los del grupo alcanzan para esta tarea: tenés 5.00, el grupo tiene 10.00 y necesitás 20.00. Cargá con /topup.",
            ),
            (
                TaskShortfall::Both,
                Locale::En,
                "Neither your personal credits nor the group credits are enough for this task: you have 5.00, the group has 10.00 and you need 20.00. Add credits with /topup.",
            ),
            (
                TaskShortfall::HourlyLimit { limit: 3 },
                Locale::Es,
                "Tus créditos personales no alcanzan para esta tarea (tenés 5.00, necesitás 20.00) y ya usaste tus 3 mensajes por hora con créditos del grupo. Cargá con /topup o probá más tarde.",
            ),
            (
                TaskShortfall::HourlyLimit { limit: 3 },
                Locale::En,
                "Your personal credits are not enough for this task (you have 5.00, you need 20.00) and you already used your 3 messages per hour on group credits. Add credits with /topup or try later.",
            ),
        ];
        for (shortfall, locale, text) in expected {
            assert_eq!(
                task_shortfall_message(shortfall, &funds, 2_000, locale),
                text
            );
        }
        assert_eq!(
            task_shortfall_message(TaskShortfall::Personal, &funds, 2_000, Locale::En),
            crate::task_tools::task_credit_insufficient(500, 2_000, Locale::En)
        );
    }
}
