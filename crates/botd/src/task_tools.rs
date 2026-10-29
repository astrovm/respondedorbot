//! Native task creation, listing, and cancellation AI tools.

use std::collections::BTreeMap;
use std::sync::Arc;

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::openrouter_chat::OpenRouterPricingCache;
use bot_adapters::redis_creditless_cap::{RedisCreditlessCap, creditless_cap_key};
use bot_adapters::redis_task_store::RedisTaskStore;
use bot_adapters::task_record::TaskRecordDocument;
use bot_core::credit_units::{CreditUnits, display_credit_units};
use bot_core::locale::Locale;
use bot_core::scheduled_tasks::{
    ScheduledTask, TaskId, TaskSchedule, initial_next_run, parse_weekday,
};
use bot_core::task_commands::format_task_summary;
use bot_core::task_triggers::{
    IntegerInput, TaskTrigger, TaskTriggerInput, TriggerConfigInput, TriggerError,
    parse_task_trigger,
};
use chrono::{DateTime, SecondsFormat, Utc, Weekday};
use serde_json::Value;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::native_ai::estimate_task_reserve_credit_units;
use crate::task_payer::{
    ChatTaskPayerPolicy, TaskFunds, TaskPayer, TaskPayerPolicy, task_shortfall,
    task_shortfall_message,
};
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest, json_integer};

const TASK_RECORD_TTL_SECONDS: i64 = 86_400 * 3_650;

pub trait TaskToolStore {
    fn list(&mut self, chat_id: &str) -> Result<Vec<ScheduledTask>, String>;
    fn save(&mut self, document: &TaskRecordDocument, ttl_seconds: i64) -> Result<(), String>;
    fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String>;
}

impl TaskToolStore for RedisTaskStore {
    fn list(&mut self, chat_id: &str) -> Result<Vec<ScheduledTask>, String> {
        self.list_chat_tasks(chat_id)
            .map(|documents| {
                documents
                    .into_iter()
                    .map(|document| document.task)
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    fn save(&mut self, document: &TaskRecordDocument, ttl_seconds: i64) -> Result<(), String> {
        self.save_task(document, ttl_seconds)
            .map(|_saved| ())
            .map_err(|error| error.to_string())
    }

    fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String> {
        self.cancel_task(task_id.as_str(), chat_id)
            .map_err(|error| error.to_string())
    }
}

/// How a member's task would be paid in this chat, and what they have to pay
/// with. Scheduled runs read the same settings again when they happen.
pub trait TaskFundsSource {
    fn funds(&mut self, chat_id: &str, user_id: i64) -> Result<(TaskPayer, TaskFunds), String>;
}

pub struct BillingTaskFunds {
    repository: BillingRepository,
    cap: RedisCreditlessCap,
    policy: ChatTaskPayerPolicy,
}

impl BillingTaskFunds {
    #[must_use]
    pub fn new(database_url: &str, cap: RedisCreditlessCap) -> Self {
        Self {
            repository: BillingRepository::new(database_url),
            cap,
            policy: ChatTaskPayerPolicy::new(database_url),
        }
    }
}

impl TaskFundsSource for BillingTaskFunds {
    fn funds(&mut self, chat_id: &str, user_id: i64) -> Result<(TaskPayer, TaskFunds), String> {
        let payer = self.policy.payer(chat_id, user_id)?;
        let personal = self
            .repository
            .get_balance("user", user_id)
            .map_err(|error| error.to_string())?;
        let group = match payer.group_chat_id {
            Some(group) => self
                .repository
                .get_balance("chat", group)
                .map_err(|error| error.to_string())?,
            None => 0,
        };
        let hourly_used = match payer.paying_chat() {
            Some(group) => self
                .cap
                .count(&creditless_cap_key(&group.to_string(), user_id))
                .map_err(|error| error.to_string())?
                .unwrap_or(0),
            None => 0,
        };
        Ok((
            payer,
            TaskFunds {
                personal,
                group,
                hourly_used,
            },
        ))
    }
}

pub trait TaskIdSource {
    fn next_id(&mut self) -> Result<TaskId, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RandomTaskIdSource;

impl TaskIdSource for RandomTaskIdSource {
    fn next_id(&mut self) -> Result<TaskId, String> {
        // Eight hex digits always form a valid ID.
        TaskId::new(format!("{:08x}", rand::random::<u32>()))
            .ok()
            .ok_or(String::from("generated task ID is invalid"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskToolContext {
    pub chat_id: String,
    pub user_name: String,
    pub user_id: Option<i64>,
    pub timezone_offset: i32,
    pub locale: Locale,
}

pub struct TaskSetTool<Store, Balance, Ids, Now> {
    store: Store,
    balance: Balance,
    ids: Ids,
    now: Now,
    context: TaskToolContext,
    openrouter_pricing: Option<Arc<OpenRouterPricingCache>>,
}

impl<Store, Balance, Ids, Now> TaskSetTool<Store, Balance, Ids, Now> {
    #[must_use]
    pub const fn new(
        store: Store,
        balance: Balance,
        ids: Ids,
        now: Now,
        context: TaskToolContext,
    ) -> Self {
        Self {
            store,
            balance,
            ids,
            now,
            context,
            openrouter_pricing: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.openrouter_pricing = Some(pricing);
        self
    }
}

impl<Store, Balance, Ids, Now> ExternalToolExecutor for TaskSetTool<Store, Balance, Ids, Now>
where
    Store: TaskToolStore,
    Balance: TaskFundsSource,
    Ids: TaskIdSource,
    Now: FnMut() -> i64,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::TaskSet {
            text,
            delay_seconds,
            interval_seconds,
            trigger_config,
        } = request
        else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.context.locale,
                "task_set",
            ));
        };
        if self.context.chat_id.is_empty() {
            return ToolExecutionResult::output(no_chat(self.context.locale));
        }
        let trigger = match parse_task_trigger(TaskTriggerInput {
            delay_seconds: optional_integer(delay_seconds),
            interval_seconds: optional_integer(interval_seconds),
            config: trigger_config_input(trigger_config.as_ref()),
        }) {
            Ok(trigger) => trigger,
            Err(error) => {
                return ToolExecutionResult::output(trigger_error(error, self.context.locale));
            }
        };
        let Some(user_id) = self.context.user_id else {
            return ToolExecutionResult::output(credit_user(self.context.locale));
        };
        let schedule = task_schedule(&trigger);
        // Chat history keeps only reply text, so a later message cannot see
        // which tasks an earlier one created. Check the stored tasks instead.
        let same_text = match self.store.list(&self.context.chat_id) {
            Ok(tasks) => tasks
                .into_iter()
                .filter(|task| same_task_text(&task.text, &text))
                .collect::<Vec<_>>(),
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    list_error(self.context.locale),
                    vec![format!("task duplicate check failed: {error}")],
                );
            }
        };
        if let Some(existing) = same_text.iter().find(|task| {
            task.schedule != TaskSchedule::Once && same_schedule(&task.schedule, &schedule)
        }) {
            return ToolExecutionResult::output(task_exists(existing, self.context.locale));
        }
        let locale_code = locale_code(self.context.locale);
        let required = match estimate_task_reserve_credit_units(
            &text,
            locale_code,
            self.openrouter_pricing.as_deref(),
        ) {
            Ok(required) => required.max(1),
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    task_cost_error(self.context.locale),
                    vec![format!("task reserve estimate failed: {error}")],
                );
            }
        };
        let (payer, funds) = match self.balance.funds(&self.context.chat_id, user_id) {
            Ok(funds) => funds,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    task_credit_check(self.context.locale),
                    vec![format!("task credit check failed: {error}")],
                );
            }
        };
        if let Some(shortfall) = task_shortfall(&payer, &funds, required) {
            return ToolExecutionResult::output(task_shortfall_message(
                shortfall,
                &funds,
                required,
                self.context.locale,
            ));
        }

        let now = (self.now)();
        let delay = match trigger {
            TaskTrigger::Delay { seconds } => Some(seconds),
            _ => None,
        };
        let next_run_at =
            match initial_next_run(&schedule, self.context.timezone_offset, now, delay) {
                Ok(next_run_at) => next_run_at,
                Err(error) => {
                    return ToolExecutionResult::with_diagnostics(
                        create_error(self.context.locale),
                        vec![format!("task next run calculation failed: {error}")],
                    );
                }
            };
        let task_id = match self.ids.next_id() {
            Ok(task_id) => task_id,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    create_error(self.context.locale),
                    vec![format!("task ID generation failed: {error}")],
                );
            }
        };
        let run_date = matches!(schedule, TaskSchedule::Once)
            .then(|| timestamp_text(next_run_at))
            .transpose();
        let run_date = match run_date {
            Ok(value) => value,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    create_error(self.context.locale),
                    vec![error],
                );
            }
        };
        // The model needs the ID to cancel or change the task later.
        let id_line = format!("\nID: {}", task_id.as_str());
        let document = TaskRecordDocument {
            task: ScheduledTask {
                id: task_id,
                chat_id: self.context.chat_id.clone(),
                text: text.clone(),
                user_name: self.context.user_name.clone(),
                user_id: Some(user_id),
                schedule,
                timezone_offset: self.context.timezone_offset,
                locale: locale_code.to_owned(),
                schedule_anchor_at: Some(now),
                next_run_at: Some(next_run_at),
                last_execution_id: None,
            },
            run_date,
            extra: BTreeMap::new(),
        };
        if let Err(error) = self.store.save(&document, TASK_RECORD_TTL_SECONDS) {
            return ToolExecutionResult::with_diagnostics(
                create_error(self.context.locale),
                vec![format!("task persistence failed: {error}")],
            );
        }
        let mut result = ToolExecutionResult::confirmed_output(created(
            &describe_trigger(&trigger, self.context.locale),
            &text,
            self.context.locale,
        ));
        result.output.push_str(&id_line);
        if !same_text.is_empty() {
            result
                .output
                .push_str(&same_text_note(&same_text, self.context.locale));
        }
        result
    }
}

fn same_task_text(left: &str, right: &str) -> bool {
    left.split_whitespace()
        .map(str::to_lowercase)
        .eq(right.split_whitespace().map(str::to_lowercase))
}

fn same_schedule(left: &TaskSchedule, right: &TaskSchedule) -> bool {
    match (left, right) {
        (
            TaskSchedule::Cron {
                hour: left_hour,
                minute: left_minute,
                weekdays: left_weekdays,
                day: left_day,
            },
            TaskSchedule::Cron {
                hour: right_hour,
                minute: right_minute,
                weekdays: right_weekdays,
                day: right_day,
            },
        ) => {
            let sorted = |weekdays: &[Weekday]| {
                let mut days = weekdays
                    .iter()
                    .map(Weekday::num_days_from_monday)
                    .collect::<Vec<_>>();
                days.sort_unstable();
                days.dedup();
                days
            };
            (left_hour, left_minute, left_day) == (right_hour, right_minute, right_day)
                && sorted(left_weekdays) == sorted(right_weekdays)
        }
        _ => match (interval_length(left), interval_length(right)) {
            (Some(left), Some(right)) => left == right,
            _ => left == right,
        },
    }
}

/// Day and second intervals of the same length repeat the same way.
fn interval_length(schedule: &TaskSchedule) -> Option<i64> {
    match schedule {
        TaskSchedule::IntervalSeconds { seconds } => Some(*seconds),
        TaskSchedule::IntervalDays { days } => days.checked_mul(86_400),
        TaskSchedule::Once | TaskSchedule::Cron { .. } => None,
    }
}

pub struct TaskListTool<Store> {
    store: Store,
    chat_id: String,
    locale: Locale,
}

impl<Store> TaskListTool<Store> {
    #[must_use]
    pub fn new(store: Store, chat_id: &str, locale: Locale) -> Self {
        Self {
            store,
            chat_id: chat_id.to_owned(),
            locale,
        }
    }
}

impl<Store: TaskToolStore> ExternalToolExecutor for TaskListTool<Store> {
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        if request != ExternalToolRequest::TaskList {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "task_list",
            ));
        }
        if self.chat_id.is_empty() {
            return ToolExecutionResult::output(no_chat(self.locale));
        }
        match self.store.list(&self.chat_id) {
            Ok(tasks) if tasks.is_empty() => ToolExecutionResult::output(no_tasks(self.locale)),
            Ok(tasks) => ToolExecutionResult::output(
                tasks
                    .iter()
                    .map(|task| format_task_summary(task, self.locale))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Err(error) => ToolExecutionResult::with_diagnostics(
                list_error(self.locale),
                vec![format!("task list failed: {error}")],
            ),
        }
    }
}

pub struct TaskCancelTool<Store> {
    store: Store,
    chat_id: String,
    locale: Locale,
}

impl<Store> TaskCancelTool<Store> {
    #[must_use]
    pub fn new(store: Store, chat_id: &str, locale: Locale) -> Self {
        Self {
            store,
            chat_id: chat_id.to_owned(),
            locale,
        }
    }
}

impl<Store: TaskToolStore> ExternalToolExecutor for TaskCancelTool<Store> {
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        let ExternalToolRequest::TaskCancel { task_id } = request else {
            return ToolExecutionResult::output(tool_output::incompatible(
                self.locale,
                "task_cancel",
            ));
        };
        if self.chat_id.is_empty() {
            return ToolExecutionResult::output(no_chat(self.locale));
        }
        let task_id = match TaskId::new(task_id) {
            Ok(task_id) => task_id,
            Err(_) => return ToolExecutionResult::output(task_not_found(self.locale)),
        };
        let tasks = match self.store.list(&self.chat_id) {
            Ok(tasks) => tasks,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    list_error(self.locale),
                    vec![format!("task cancel list failed: {error}")],
                );
            }
        };
        if !tasks.iter().any(|task| task.id == task_id) {
            return ToolExecutionResult::output(task_not_found(self.locale));
        }
        match self.store.cancel(&task_id, &self.chat_id) {
            // Confirmed, so the user still learns it is gone if the model
            // fails to answer afterwards.
            Ok(true) => ToolExecutionResult::confirmed_output(task_canceled(&task_id, self.locale)),
            Ok(false) => ToolExecutionResult::output(task_not_found(self.locale)),
            Err(error) => ToolExecutionResult::with_diagnostics(
                bot_core::menu_ui::localized(
                    self.locale,
                    "No pude cancelar la tarea. Probá de nuevo",
                    "I could not cancel the task. Try again",
                ),
                vec![format!("task cancellation failed: {error}")],
            ),
        }
    }
}

fn optional_integer(value: Option<i64>) -> IntegerInput {
    value.map_or(IntegerInput::Missing, IntegerInput::Value)
}

fn value_integer(value: Option<&Value>) -> IntegerInput {
    match value {
        None | Some(Value::Null) => IntegerInput::Missing,
        Some(value) => json_integer(value).map_or(IntegerInput::Invalid, IntegerInput::Value),
    }
}

const INTERVAL_FIELDS: [&str; 5] = ["type", "days", "hours", "minutes", "seconds"];
const CRON_FIELDS: [&str; 5] = ["type", "hour", "minute", "day_of_week", "day"];

fn trigger_config_input(config: Option<&Value>) -> TriggerConfigInput {
    let Some(config) = config.and_then(Value::as_object) else {
        return TriggerConfigInput::Missing;
    };
    let interval = match config.get("type").and_then(Value::as_str) {
        Some("interval") => true,
        Some("cron") => false,
        _ => return TriggerConfigInput::Unsupported,
    };
    let fields = if interval {
        &INTERVAL_FIELDS
    } else {
        &CRON_FIELDS
    };
    // Some models send every schema field and leave the unused ones null.
    // A real unused value, such as `weekdays` instead of `day_of_week`,
    // would otherwise be dropped and change the schedule without a word.
    if let Some(name) = config
        .iter()
        .find(|(name, value)| !value.is_null() && !fields.contains(&name.as_str()))
        .map(|(name, _)| name.clone())
    {
        return TriggerConfigInput::UnknownField { name };
    }
    if interval {
        return TriggerConfigInput::Interval {
            days: value_integer(config.get("days")),
            hours: value_integer(config.get("hours")),
            minutes: value_integer(config.get("minutes")),
            seconds: value_integer(config.get("seconds")),
        };
    }
    TriggerConfigInput::Cron {
        hour: value_integer(config.get("hour")),
        minute: value_integer(config.get("minute")),
        weekdays: config
            .get("day_of_week")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        day: value_integer(config.get("day")),
    }
}

fn task_schedule(trigger: &TaskTrigger) -> TaskSchedule {
    match trigger {
        TaskTrigger::Delay { .. } => TaskSchedule::Once,
        TaskTrigger::IntervalSeconds { seconds } => {
            TaskSchedule::IntervalSeconds { seconds: *seconds }
        }
        TaskTrigger::IntervalDays { days } => TaskSchedule::IntervalDays { days: *days },
        TaskTrigger::Cron {
            hour,
            minute,
            weekdays,
            day,
        } => TaskSchedule::Cron {
            hour: u32::try_from(*hour).unwrap_or_default(),
            minute: u32::try_from(*minute).unwrap_or_default(),
            weekdays: weekdays
                .iter()
                .filter_map(|weekday| parse_weekday(weekday).ok())
                .collect(),
            day: day.and_then(|value| u32::try_from(value).ok()),
        },
    }
}

fn timestamp_text(timestamp: i64) -> Result<String, String> {
    DateTime::<Utc>::from_timestamp(timestamp, 0)
        .map(|value| value.to_rfc3339_opts(SecondsFormat::Secs, true))
        .ok_or_else(|| "task timestamp is outside the supported range".to_owned())
}

fn locale_code(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "es",
        Locale::En => "en",
    }
}

fn trigger_error(error: TriggerError, locale: Locale) -> String {
    match (error, locale) {
        (TriggerError::Required, Locale::Es) => "necesito usar algún parámetro de tiempo: delay_seconds (una vez), interval_seconds (repetir cada tantos segundos) o trigger_config (cron a una hora fija).".to_owned(),
        (TriggerError::Required, Locale::En) => "provide delay_seconds (once), interval_seconds (repeat every few seconds), or trigger_config (cron at a clock time)".to_owned(),
        (TriggerError::UnsupportedType, Locale::Es) => "trigger_config.type tiene que ser 'cron'. Para repetir cada tantos minutos, horas o días usá interval_seconds (3 h = 10800)".to_owned(),
        (TriggerError::UnsupportedType, Locale::En) => "trigger_config.type must be 'cron'. To repeat every few minutes, hours, or days, use interval_seconds (3 h = 10800)".to_owned(),
        (TriggerError::UnknownField { name }, Locale::Es) => format!("trigger_config no usa '{name}'. Para cron los campos son hour, minute, day_of_week y day. Para repetir cada tantos minutos, horas o días usá interval_seconds (3 h = 10800)"),
        (TriggerError::UnknownField { name }, Locale::En) => format!("trigger_config does not use '{name}'. Cron fields are hour, minute, day_of_week, and day. To repeat every few minutes, hours, or days, use interval_seconds (3 h = 10800)"),
        (TriggerError::DelayPositive, Locale::Es) => "delay_seconds debe ser un entero positivo".to_owned(),
        (TriggerError::DelayPositive, Locale::En) => "delay_seconds must be a positive integer".to_owned(),
        (TriggerError::DelayMaximum, Locale::Es) => "el máximo es 10 años".to_owned(),
        (TriggerError::DelayMaximum, Locale::En) => "the maximum delay is 10 years".to_owned(),
        (TriggerError::IntervalMinimum, Locale::Es) => "el intervalo mínimo es 300 segundos (5 min)".to_owned(),
        (TriggerError::IntervalMinimum, Locale::En) => "the minimum interval is 300 seconds (5 minutes)".to_owned(),
        (TriggerError::IntervalMaximum, Locale::Es) => "el intervalo máximo es 90 días (7776000 segundos)".to_owned(),
        (TriggerError::IntervalMaximum, Locale::En) => "the maximum interval is 90 days (7776000 seconds)".to_owned(),
        (TriggerError::IntervalWholeDays, Locale::Es) => "los intervalos de más de 7 días tienen que ser días enteros (múltiplos de 86400 segundos)".to_owned(),
        (TriggerError::IntervalWholeDays, Locale::En) => "intervals over 7 days must be whole days (multiples of 86400 seconds)".to_owned(),
        (TriggerError::IntervalUnitsRequired, Locale::Es) => "para repetir cada tantos minutos, horas o días usá interval_seconds (3 h = 10800)".to_owned(),
        (TriggerError::IntervalUnitsRequired, Locale::En) => "to repeat every few minutes, hours, or days, use interval_seconds (3 h = 10800)".to_owned(),
        (TriggerError::IntervalUnitInvalid, Locale::Es) => "days, hours, minutes y seconds tienen que ser enteros de 0 para arriba. Mejor usá interval_seconds (3 h = 10800)".to_owned(),
        (TriggerError::IntervalUnitInvalid, Locale::En) => "days, hours, minutes, and seconds must be whole numbers from 0 up. Use interval_seconds (3 h = 10800) instead".to_owned(),
        (TriggerError::HourRequired, Locale::Es) => "hour es requerido para trigger cron".to_owned(),
        (TriggerError::HourRequired, Locale::En) => "hour is required for a cron trigger".to_owned(),
        (TriggerError::HourRange, Locale::Es) => "hour debe ser 0-23".to_owned(),
        (TriggerError::HourRange, Locale::En) => "hour must be between 0 and 23".to_owned(),
        (TriggerError::MinuteRequired, Locale::Es) => "minute es requerido para trigger cron".to_owned(),
        (TriggerError::MinuteRequired, Locale::En) => "minute is required for a cron trigger".to_owned(),
        (TriggerError::MinuteRange, Locale::Es) => "minute debe ser 0-59".to_owned(),
        (TriggerError::MinuteRange, Locale::En) => "minute must be between 0 and 59".to_owned(),
        (TriggerError::Weekday { value }, Locale::Es) => format!("day_of_week inválido: {value}"),
        (TriggerError::Weekday { value }, Locale::En) => format!("invalid day_of_week: {value}"),
        (TriggerError::WeekdayEmpty, Locale::Es) => "day_of_week inválido".to_owned(),
        (TriggerError::WeekdayEmpty, Locale::En) => "invalid day_of_week".to_owned(),
        (TriggerError::DayRange, Locale::Es) => "day debe ser 1-31".to_owned(),
        (TriggerError::DayRange, Locale::En) => "day must be between 1 and 31".to_owned(),
    }
}

fn describe_trigger(trigger: &TaskTrigger, locale: Locale) -> String {
    match trigger {
        TaskTrigger::Delay { seconds } => interval_description(*seconds, true, locale),
        TaskTrigger::IntervalSeconds { seconds } => interval_description(*seconds, false, locale),
        TaskTrigger::IntervalDays { days } => match locale {
            Locale::Es => format!("cada {days} días"),
            Locale::En => format!("every {days} days"),
        },
        TaskTrigger::Cron {
            hour,
            minute,
            weekdays,
            day,
        } => {
            let time = format!("{hour:02}:{minute:02}");
            if !weekdays.is_empty() {
                let weekdays = weekdays
                    .iter()
                    .map(|day| localized_weekday(day, locale))
                    .collect::<Vec<_>>()
                    .join(", ");
                match locale {
                    Locale::Es => format!("los {weekdays} a las {time}"),
                    Locale::En => format!("on {weekdays} at {time}"),
                }
            } else if let Some(day) = day {
                match locale {
                    Locale::Es => format!("el día {day} de cada mes a las {time}"),
                    Locale::En => format!("on day {day} of every month at {time}"),
                }
            } else {
                match locale {
                    Locale::Es => format!("todos los días a las {time}"),
                    Locale::En => format!("every day at {time}"),
                }
            }
        }
    }
}

fn interval_description(seconds: i64, delayed: bool, locale: Locale) -> String {
    let (value, es_unit, en_unit) = if seconds >= 86_400 {
        let value = seconds / 86_400;
        (
            value,
            if value == 1 { "día" } else { "días" },
            if value == 1 { "day" } else { "days" },
        )
    } else if seconds >= 3_600 {
        let value = seconds / 3_600;
        (
            value,
            if value == 1 { "hora" } else { "horas" },
            if value == 1 { "hour" } else { "hours" },
        )
    } else {
        let value = seconds / 60;
        (
            value,
            if value == 1 { "minuto" } else { "minutos" },
            if value == 1 { "minute" } else { "minutes" },
        )
    };
    match (delayed, locale) {
        (true, Locale::Es) => format!("en {value} {es_unit}"),
        (true, Locale::En) => format!("in {value} {en_unit}"),
        (false, Locale::Es) if value == 1 => format!("cada {es_unit}"),
        (false, Locale::En) if value == 1 => format!("every {en_unit}"),
        (false, Locale::Es) => format!("cada {value} {es_unit}"),
        (false, Locale::En) => format!("every {value} {en_unit}"),
    }
}

fn localized_weekday(day: &str, locale: Locale) -> &str {
    if locale == Locale::En {
        return day;
    }
    match day {
        "mon" => "lun",
        "tue" => "mar",
        "wed" => "mié",
        "thu" => "jue",
        "fri" => "vie",
        "sat" => "sáb",
        "sun" => "dom",
        other => other,
    }
}

fn no_chat(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not identify the chat"
    } else {
        "no sé en qué chat estoy"
    }
}
fn no_tasks(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "there are no tasks"
    } else {
        "no hay tareas"
    }
}
fn credit_user(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not identify your user to charge for the task"
    } else {
        "No pude identificar tu usuario para cobrar la tarea"
    }
}
pub fn task_credit_check(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not check your personal credits. Try again"
    } else {
        "No pude verificar tus créditos personales. Probá de nuevo"
    }
}
pub fn task_cost_error(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not calculate the task cost. Try again"
    } else {
        "No pude calcular el costo de la tarea. Probá de nuevo"
    }
}
fn create_error(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not create the task"
    } else {
        "no se pudo crear la tarea"
    }
}
fn task_not_found(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "That task does not exist in this chat"
    } else {
        "Esa tarea no existe en este chat"
    }
}

pub fn task_credit_insufficient(balance: i64, required: i64, locale: Locale) -> String {
    let balance = display_credit_units(CreditUnits::new(balance));
    let required = display_credit_units(CreditUnits::new(required));
    match locale {
        Locale::Es => format!(
            "No te alcanzan los créditos personales para esa tarea: tenés {balance} y necesitás {required}\nCargá con /topup antes de crearla"
        ),
        Locale::En => format!(
            "Not enough personal credits for this task: you have {balance} and need {required}\nUse /topup before creating it"
        ),
    }
}

fn created(schedule: &str, text: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("Tarea programada: {schedule}\n{text}"),
        Locale::En => format!("Task scheduled: {schedule}\n{text}"),
    }
}

fn list_error(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not load tasks. Try again"
    } else {
        "No pude cargar las tareas. Probá de nuevo"
    }
}

fn task_exists(task: &ScheduledTask, locale: Locale) -> String {
    let summary = format_task_summary(task, locale);
    match locale {
        Locale::Es => format!(
            "Esa tarea ya existe, no creé otra:\n{summary}\nPara cambiarla, cancelala con task_cancel y creá la nueva"
        ),
        Locale::En => format!(
            "That task already exists, so I did not create another:\n{summary}\nTo change it, cancel it with task_cancel and create the new one"
        ),
    }
}

fn same_text_note(tasks: &[ScheduledTask], locale: Locale) -> String {
    let summaries = tasks
        .iter()
        .map(|task| format_task_summary(task, locale))
        .collect::<Vec<_>>()
        .join("\n");
    match locale {
        Locale::Es => format!(
            "\nEste chat ya tenía otra tarea con el mismo texto:\n{summaries}\nSi la nueva la reemplaza, cancelá la vieja con task_cancel"
        ),
        Locale::En => format!(
            "\nThis chat already had a task with the same text:\n{summaries}\nIf the new one replaces it, cancel the old one with task_cancel"
        ),
    }
}

fn task_canceled(task_id: &TaskId, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("Tarea {} cancelada", task_id.as_str()),
        Locale::En => format!("Task {} canceled", task_id.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use bot_adapters::redis_connection::RedisEndpoint;
    use serde_json::json;

    use super::*;

    #[test]
    fn redis_task_tool_store_round_trips_through_the_public_port() -> TestResult {
        std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or(Ok(()), round_trip_redis_task_store)
    }

    fn round_trip_redis_task_store(port: u16) -> TestResult {
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or(String::from("127.0.0.1")),
            port,
            // Empty passwords are ignored by the Redis client.
            password: std::env::var("TEST_REDIS_PASSWORD").ok(),
        };
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let task_id = TaskId::new(format!("synthetic_{nonce}"))?;
        let chat_id = format!("synthetic-chat-{nonce}");
        let document = TaskRecordDocument {
            task: ScheduledTask {
                id: task_id.clone(),
                chat_id: chat_id.clone(),
                text: "synthetic scheduled task".to_owned(),
                user_name: "synthetic-user".to_owned(),
                user_id: Some(42),
                schedule: TaskSchedule::Once,
                timezone_offset: 0,
                locale: "en".to_owned(),
                schedule_anchor_at: Some(1_700_000_000),
                next_run_at: Some(1_700_000_000),
                last_execution_id: None,
            },
            run_date: None,
            extra: BTreeMap::new(),
        };
        let mut store = RedisTaskStore::new(&endpoint)?;
        TaskToolStore::save(&mut store, &document, 600)?;
        assert_eq!(TaskToolStore::list(&mut store, &chat_id)?, [document.task]);
        assert!(TaskToolStore::cancel(&mut store, &task_id, &chat_id)?);
        assert!(TaskToolStore::list(&mut store, &chat_id)?.is_empty());
        assert!(RandomTaskIdSource.next_id().is_ok());
        Ok(())
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn redis_store_failures_are_reported_as_text() -> TestResult {
        let mut store = RedisTaskStore::new(&RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        })?;
        let task_id = TaskId::new("synthetic")?;
        let document = TaskRecordDocument {
            task: ScheduledTask {
                id: task_id.clone(),
                chat_id: "-100".to_owned(),
                text: "synthetic".to_owned(),
                user_name: "synthetic".to_owned(),
                user_id: Some(7),
                schedule: TaskSchedule::Once,
                timezone_offset: 0,
                locale: "en".to_owned(),
                schedule_anchor_at: None,
                next_run_at: None,
                last_execution_id: None,
            },
            run_date: None,
            extra: BTreeMap::new(),
        };
        let list = TaskToolStore::list(&mut store, "-100");
        let save = TaskToolStore::save(&mut store, &document, 60);
        let cancel = TaskToolStore::cancel(&mut store, &task_id, "-100");
        assert!(matches!(&list, Err(error) if !error.is_empty()), "{list:?}");
        assert!(matches!(&save, Err(error) if !error.is_empty()), "{save:?}");
        assert!(
            matches!(&cancel, Err(error) if !error.is_empty()),
            "{cancel:?}"
        );
        // Random IDs are eight lowercase hex digits.
        let generated = RandomTaskIdSource.next_id()?;
        assert_eq!(generated.as_str().len(), 8);
        assert!(
            generated
                .as_str()
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
        Ok(())
    }

    struct StoreState {
        tasks: Vec<ScheduledTask>,
        saved: Vec<(TaskRecordDocument, i64)>,
        canceled: Vec<(String, String)>,
        list_error: Option<String>,
        save_error: Option<String>,
        cancel_result: Result<bool, String>,
    }

    impl Default for StoreState {
        fn default() -> Self {
            Self {
                tasks: Vec::new(),
                saved: Vec::new(),
                canceled: Vec::new(),
                list_error: None,
                save_error: None,
                cancel_result: Ok(false),
            }
        }
    }

    #[derive(Clone)]
    struct Store(Rc<RefCell<StoreState>>);

    impl TaskToolStore for Store {
        fn list(&mut self, _chat_id: &str) -> Result<Vec<ScheduledTask>, String> {
            let state = self.0.borrow();
            if let Some(error) = &state.list_error {
                return Err(error.clone());
            }
            Ok(state.tasks.clone())
        }

        fn save(&mut self, document: &TaskRecordDocument, ttl_seconds: i64) -> Result<(), String> {
            let mut state = self.0.borrow_mut();
            if let Some(error) = &state.save_error {
                return Err(error.clone());
            }
            state.saved.push((document.clone(), ttl_seconds));
            Ok(())
        }

        fn cancel(&mut self, task_id: &TaskId, chat_id: &str) -> Result<bool, String> {
            let mut state = self.0.borrow_mut();
            state
                .canceled
                .push((task_id.as_str().to_owned(), chat_id.to_owned()));
            state.cancel_result.clone()
        }
    }

    struct Balance {
        result: Result<i64, String>,
        calls: Rc<RefCell<Vec<i64>>>,
        payer: TaskPayer,
        group: i64,
        hourly_used: i64,
    }

    impl Balance {
        fn personal(result: Result<i64, String>) -> Self {
            Self {
                result,
                calls: Rc::new(RefCell::new(Vec::new())),
                payer: TaskPayer::PERSONAL,
                group: 0,
                hourly_used: 0,
            }
        }
    }

    impl TaskFundsSource for Balance {
        fn funds(
            &mut self,
            _chat_id: &str,
            user_id: i64,
        ) -> Result<(TaskPayer, TaskFunds), String> {
            self.calls.borrow_mut().push(user_id);
            let personal = self.result.clone()?;
            Ok((
                self.payer,
                TaskFunds {
                    personal,
                    group: self.group,
                    hourly_used: self.hourly_used,
                },
            ))
        }
    }

    /// Hands out scripted IDs; `Err` entries simulate generator failures.
    struct Ids(Vec<Result<&'static str, &'static str>>);

    impl TaskIdSource for Ids {
        fn next_id(&mut self) -> Result<TaskId, String> {
            let id = self.0.remove(0).map_err(String::from)?;
            TaskId::new(id)
                .ok()
                .ok_or(String::from("invalid synthetic ID"))
        }
    }

    /// Every test shares this clock type, so the tool has one instantiation.
    type Clock = fn() -> i64;

    fn fixed_clock() -> i64 {
        1_700_000_000
    }

    fn context(locale: Locale) -> TaskToolContext {
        TaskToolContext {
            chat_id: "-100".to_owned(),
            user_name: "Synthetic User".to_owned(),
            user_id: Some(7),
            timezone_offset: -3,
            locale,
        }
    }

    fn set_tool(
        state: Rc<RefCell<StoreState>>,
        balance: Result<i64, String>,
        locale: Locale,
    ) -> TaskSetTool<Store, Balance, Ids, Clock> {
        TaskSetTool::new(
            Store(state),
            Balance::personal(balance),
            Ids(vec![Ok("abc12345")]),
            fixed_clock,
            context(locale),
        )
    }

    fn set_request(
        delay_seconds: Option<i64>,
        interval_seconds: Option<i64>,
        trigger_config: Option<Value>,
    ) -> ExternalToolRequest {
        ExternalToolRequest::TaskSet {
            text: "check the synthetic result".to_owned(),
            delay_seconds,
            interval_seconds,
            trigger_config,
        }
    }

    #[test]
    fn creates_a_canonical_one_shot_after_the_credit_precondition() {
        let state = Rc::new(RefCell::new(StoreState {
            cancel_result: Ok(true),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(Some(3_600), None, None), "call");
        assert_eq!(
            result.output,
            "Task scheduled: in 1 hour\ncheck the synthetic result\nID: abc12345"
        );
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("Task scheduled: in 1 hour\ncheck the synthetic result")
        );
        let state = state.borrow();
        assert_eq!(state.saved.len(), 1);
        let (document, ttl) = &state.saved[0];
        assert_eq!(*ttl, TASK_RECORD_TTL_SECONDS);
        assert_eq!(document.task.id.as_str(), "abc12345");
        assert_eq!(document.task.chat_id, "-100");
        assert_eq!(document.task.user_id, Some(7));
        assert_eq!(document.task.schedule, TaskSchedule::Once);
        assert_eq!(document.task.schedule_anchor_at, Some(1_700_000_000));
        assert_eq!(document.task.next_run_at, Some(1_700_003_600));
        assert_eq!(document.run_date.as_deref(), Some("2023-11-14T23:13:20Z"));
    }

    #[test]
    fn creates_normalized_recurring_and_cron_records_with_localized_descriptions() {
        let state = Rc::new(RefCell::new(StoreState {
            cancel_result: Ok(true),
            ..StoreState::default()
        }));
        let mut interval = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        assert_eq!(
            interval
                .execute(set_request(None, Some(86_400), None), "call")
                .output,
            "Tarea programada: cada día\ncheck the synthetic result\nID: abc12345"
        );
        assert_eq!(
            state.borrow().saved[0].0.task.schedule,
            TaskSchedule::IntervalSeconds { seconds: 86_400 }
        );
        assert!(state.borrow().saved[0].0.run_date.is_none());

        let mut cron = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        assert_eq!(
            cron.execute(
                set_request(
                    None,
                    None,
                    Some(json!({"type":"cron", "hour":9, "minute":5, "day_of_week":"lun,wed"})),
                ),
                "call"
            )
            .output,
            "Tarea programada: los lun, mié a las 09:05\ncheck the synthetic result\nID: abc12345"
        );
        assert!(matches!(
            state.borrow().saved[1].0.task.schedule,
            TaskSchedule::Cron {
                hour: 9,
                minute: 5,
                ..
            }
        ));
    }

    #[test]
    fn trigger_user_balance_and_persistence_failures_are_safe_and_diagnostic() {
        let state = Rc::new(RefCell::new(StoreState {
            cancel_result: Ok(true),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        assert_eq!(
            tool.execute(set_request(None, Some(299), None), "call")
                .output,
            "the minimum interval is 300 seconds (5 minutes)"
        );
        assert!(state.borrow().saved.is_empty());

        let mut tool = set_tool(Rc::clone(&state), Ok(0), Locale::En);
        assert!(
            tool.execute(set_request(Some(60), None, None), "call")
                .output
                .contains("Not enough personal credits")
        );
        assert!(state.borrow().saved.is_empty());

        state.borrow_mut().save_error = Some("synthetic Redis failure".to_owned());
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        let result = tool.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, "no se pudo crear la tarea");
        assert!(result.diagnostics[0].contains("synthetic Redis failure"));

        let mut missing_user = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        missing_user.context.user_id = None;
        assert_eq!(
            missing_user
                .execute(set_request(Some(60), None, None), "call")
                .output,
            "I could not identify your user to charge for the task"
        );
    }

    #[test]
    fn interval_configs_in_hours_create_second_intervals_instead_of_failing() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        let result = tool.execute(
            set_request(None, None, Some(json!({"type":"interval", "hours":3}))),
            "call",
        );
        assert_eq!(
            result.output,
            "Tarea programada: cada 3 horas\ncheck the synthetic result\nID: abc12345"
        );
        assert_eq!(
            state.borrow().saved[0].0.task.schedule,
            TaskSchedule::IntervalSeconds { seconds: 10_800 }
        );
        assert_eq!(
            state.borrow().saved[0].0.task.next_run_at,
            Some(1_700_010_800)
        );
    }

    #[test]
    fn long_second_intervals_are_stored_as_whole_days() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(None, Some(86_400 * 30), None), "call");
        assert_eq!(
            result.output,
            "Task scheduled: every 30 days\ncheck the synthetic result\nID: abc12345"
        );
        assert_eq!(
            state.borrow().saved[0].0.task.schedule,
            TaskSchedule::IntervalDays { days: 30 }
        );
    }

    #[test]
    fn wrong_trigger_shapes_point_the_model_at_interval_seconds() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        for (config, needle) in [
            (json!({"type":"interval"}), "interval_seconds (3 h = 10800)"),
            (
                json!({"type":"interval", "days":"x"}),
                "interval_seconds (3 h = 10800)",
            ),
            (
                json!({"type":"every", "hours":3}),
                "interval_seconds (3 h = 10800)",
            ),
            (
                json!({"type":"cron", "hour":9, "minute":0, "weekdays":"mon"}),
                "does not use 'weekdays'",
            ),
        ] {
            let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
            let output = tool
                .execute(set_request(None, None, Some(config)), "call")
                .output;
            assert!(output.contains(needle), "{output}");
        }
        assert!(state.borrow().saved.is_empty());
    }

    fn stored_task(id: &str, text: &str, schedule: TaskSchedule) -> Option<ScheduledTask> {
        let mut task = task(id, "-100")?;
        text.clone_into(&mut task.text);
        task.schedule = schedule;
        Some(task)
    }

    #[test]
    fn repeating_the_same_task_returns_the_existing_one_instead_of_a_copy() {
        let daily = TaskSchedule::IntervalDays { days: 1 };
        let state = Rc::new(RefCell::new(StoreState {
            tasks: stored_task("old12345", "Check the  synthetic RESULT", daily)
                .into_iter()
                .collect(),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(None, Some(86_400), None), "call");
        assert!(
            result
                .output
                .starts_with("That task already exists, so I did not create another:\n[old12345]"),
            "{}",
            result.output
        );
        assert!(result.output.contains("task_cancel"));
        assert!(result.failure_fallback.is_none());
        assert!(state.borrow().saved.is_empty());
    }

    #[test]
    fn same_cron_with_reordered_weekdays_is_a_duplicate() {
        let cron = TaskSchedule::Cron {
            hour: 9,
            minute: 5,
            weekdays: vec![Weekday::Wed, Weekday::Mon],
            day: None,
        };
        let state = Rc::new(RefCell::new(StoreState {
            tasks: stored_task("old12345", "check the synthetic result", cron)
                .into_iter()
                .collect(),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        let output = tool
            .execute(
                set_request(
                    None,
                    None,
                    Some(json!({"type":"cron", "hour":9, "minute":5, "day_of_week":"lun,mie"})),
                ),
                "call",
            )
            .output;
        assert!(output.starts_with("Esa tarea ya existe"), "{output}");
        assert!(state.borrow().saved.is_empty());
    }

    #[test]
    fn same_text_on_a_new_schedule_is_created_with_a_note_for_the_model() {
        let state = Rc::new(RefCell::new(StoreState {
            tasks: [
                stored_task(
                    "old12345",
                    "check the synthetic result",
                    TaskSchedule::IntervalDays { days: 1 },
                ),
                stored_task("once1234", "check the synthetic result", TaskSchedule::Once),
                stored_task(
                    "other123",
                    "something else",
                    TaskSchedule::IntervalSeconds { seconds: 10_800 },
                ),
            ]
            .into_iter()
            .flatten()
            .collect(),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(None, Some(10_800), None), "call");
        assert!(result.output.starts_with(
            "Task scheduled: every 3 hours\ncheck the synthetic result\nID: abc12345\nThis chat already had a task with the same text:\n[old12345]"
        ));
        assert!(result.output.contains("[once1234]"));
        assert!(!result.output.contains("[other123]"));
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("Task scheduled: every 3 hours\ncheck the synthetic result")
        );
        assert_eq!(state.borrow().saved.len(), 1);

        // One-shot tasks with the same text are never treated as copies.
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        let output = tool
            .execute(set_request(Some(60), None, None), "call")
            .output;
        assert!(output.contains("Este chat ya tenía otra tarea"), "{output}");
        assert_eq!(state.borrow().saved.len(), 2);
    }

    #[test]
    fn duplicate_check_failures_stop_before_creating_a_task() {
        let state = Rc::new(RefCell::new(StoreState {
            list_error: Some("synthetic list failure".to_owned()),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, "I could not load tasks. Try again");
        assert!(result.diagnostics[0].contains("synthetic list failure"));
        assert!(state.borrow().saved.is_empty());
    }

    #[test]
    fn schedules_compare_by_how_they_repeat() {
        let cron = |weekdays: Vec<Weekday>, day| TaskSchedule::Cron {
            hour: 9,
            minute: 0,
            weekdays,
            day,
        };
        assert!(same_schedule(
            &TaskSchedule::IntervalDays { days: 1 },
            &TaskSchedule::IntervalSeconds { seconds: 86_400 }
        ));
        assert!(!same_schedule(
            &TaskSchedule::IntervalDays { days: 1 },
            &TaskSchedule::IntervalSeconds { seconds: 10_800 }
        ));
        assert!(!same_schedule(
            &TaskSchedule::IntervalSeconds { seconds: 86_400 },
            &cron(Vec::new(), None)
        ));
        assert!(same_schedule(&TaskSchedule::Once, &TaskSchedule::Once));
        assert!(same_schedule(
            &cron(vec![Weekday::Fri, Weekday::Mon, Weekday::Mon], None),
            &cron(vec![Weekday::Mon, Weekday::Fri], None)
        ));
        assert!(!same_schedule(
            &cron(vec![Weekday::Mon], None),
            &cron(vec![Weekday::Tue], None)
        ));
        assert!(!same_schedule(
            &cron(Vec::new(), Some(1)),
            &cron(Vec::new(), Some(2))
        ));
        assert!(same_task_text(" Check  the result ", "check the RESULT"));
        assert!(!same_task_text("check the result", "check the results"));
    }

    fn task(id: &str, chat_id: &str) -> Option<ScheduledTask> {
        Some(ScheduledTask {
            id: TaskId::new(id).ok()?,
            chat_id: chat_id.to_owned(),
            text: "synthetic task".to_owned(),
            user_name: "Owner".to_owned(),
            user_id: Some(7),
            schedule: TaskSchedule::Once,
            timezone_offset: -3,
            locale: "en".to_owned(),
            schedule_anchor_at: Some(1_700_000_000),
            next_run_at: Some(1_700_003_600),
            last_execution_id: None,
        })
    }

    #[test]
    fn lists_and_cancels_only_tasks_from_the_current_chat() {
        let state = Rc::new(RefCell::new(StoreState {
            tasks: task("abc12345", "-100").into_iter().collect(),
            cancel_result: Ok(true),
            ..StoreState::default()
        }));
        let mut list = TaskListTool::new(Store(Rc::clone(&state)), "-100", Locale::En);
        let output = list.execute(ExternalToolRequest::TaskList, "call").output;
        assert!(output.starts_with("[abc12345] synthetic task (Owner) - "));

        let mut cancel = TaskCancelTool::new(Store(Rc::clone(&state)), "-100", Locale::En);
        let canceled = cancel.execute(
            ExternalToolRequest::TaskCancel {
                task_id: "abc12345".to_owned(),
            },
            "call",
        );
        assert_eq!(canceled.output, "Task abc12345 canceled");
        assert_eq!(
            canceled.failure_fallback.as_deref(),
            Some("Task abc12345 canceled")
        );
        assert_eq!(
            state.borrow().canceled,
            [("abc12345".to_owned(), "-100".to_owned())]
        );

        let mut cancel = TaskCancelTool::new(Store(Rc::clone(&state)), "-100", Locale::Es);
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "different".to_owned(),
                    },
                    "call"
                )
                .output,
            "Esa tarea no existe en este chat"
        );
        assert_eq!(state.borrow().canceled.len(), 1);
    }

    #[test]
    fn list_and_cancel_store_failures_keep_user_output_stable_and_add_diagnostics() {
        let state = Rc::new(RefCell::new(StoreState {
            list_error: Some("synthetic list failure".to_owned()),
            cancel_result: Ok(false),
            ..StoreState::default()
        }));
        let mut list = TaskListTool::new(Store(Rc::clone(&state)), "-100", Locale::Es);
        let result = list.execute(ExternalToolRequest::TaskList, "call");
        assert_eq!(result.output, "No pude cargar las tareas. Probá de nuevo");
        assert!(result.diagnostics[0].contains("synthetic list failure"));

        let mut cancel = TaskCancelTool::new(Store(state), "-100", Locale::En);
        let result = cancel.execute(
            ExternalToolRequest::TaskCancel {
                task_id: "abc12345".to_owned(),
            },
            "call",
        );
        assert_eq!(result.output, "I could not load tasks. Try again");
        assert!(result.diagnostics[0].contains("synthetic list failure"));
    }

    #[test]
    fn rejects_incompatible_requests_and_missing_chat_context() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut set = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        assert!(
            set.execute(ExternalToolRequest::TaskList, "call")
                .output
                .contains("task_set")
        );
        set.context.chat_id.clear();
        assert_eq!(
            set.execute(set_request(Some(60), None, None), "call")
                .output,
            "I could not identify the chat"
        );

        let mut list = TaskListTool::new(Store(Rc::clone(&state)), "", Locale::Es);
        assert!(
            list.execute(
                ExternalToolRequest::TaskCancel {
                    task_id: "synthetic".to_owned(),
                },
                "call"
            )
            .output
            .contains("task_list")
        );
        assert_eq!(
            list.execute(ExternalToolRequest::TaskList, "call").output,
            "no sé en qué chat estoy"
        );

        let mut cancel = TaskCancelTool::new(Store(state), "", Locale::En);
        assert!(
            cancel
                .execute(ExternalToolRequest::TaskList, "call")
                .output
                .contains("task_cancel")
        );
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "synthetic".to_owned(),
                    },
                    "call"
                )
                .output,
            "I could not identify the chat"
        );
    }

    #[test]
    fn reports_balance_id_timestamp_and_cancel_failures() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut balance_failure = set_tool(
            Rc::clone(&state),
            Err("synthetic balance failure".to_owned()),
            Locale::En,
        );
        let result = balance_failure.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, task_credit_check(Locale::En));
        assert!(result.diagnostics[0].contains("synthetic balance failure"));

        let mut id_failure = TaskSetTool::new(
            Store(Rc::clone(&state)),
            Balance::personal(Ok(i64::MAX)),
            Ids(vec![Err("synthetic ID failure")]),
            fixed_clock as Clock,
            context(Locale::Es),
        );
        let result = id_failure.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, create_error(Locale::Es));
        assert!(result.diagnostics[0].contains("synthetic ID failure"));

        let mut timestamp_failure = TaskSetTool::new(
            Store(Rc::clone(&state)),
            Balance::personal(Ok(i64::MAX)),
            Ids(vec![Ok("synthetic")]),
            (|| i64::MAX - 60) as Clock,
            context(Locale::En),
        );
        let result = timestamp_failure.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, create_error(Locale::En));
        assert!(result.diagnostics[0].contains("timestamp"));

        state.borrow_mut().tasks = task("synthetic", "-100").into_iter().collect();
        state.borrow_mut().cancel_result = Ok(false);
        let mut cancel = TaskCancelTool::new(Store(Rc::clone(&state)), "-100", Locale::En);
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "bad id!".to_owned(),
                    },
                    "call"
                )
                .output,
            task_not_found(Locale::En)
        );
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "synthetic".to_owned(),
                    },
                    "call"
                )
                .output,
            task_not_found(Locale::En)
        );
        state.borrow_mut().cancel_result = Err("synthetic cancel failure".to_owned());
        let result = cancel.execute(
            ExternalToolRequest::TaskCancel {
                task_id: "synthetic".to_owned(),
            },
            "call",
        );
        assert_eq!(result.output, "I could not cancel the task. Try again");
        assert!(result.diagnostics[0].contains("synthetic cancel failure"));
    }

    #[test]
    fn trigger_validation_and_descriptions_cover_every_supported_shape() {
        let validation_errors = [
            TriggerError::Required,
            TriggerError::UnsupportedType,
            TriggerError::DelayPositive,
            TriggerError::DelayMaximum,
            TriggerError::UnknownField {
                name: "synthetic".to_owned(),
            },
            TriggerError::IntervalMinimum,
            TriggerError::IntervalMaximum,
            TriggerError::IntervalWholeDays,
            TriggerError::IntervalUnitsRequired,
            TriggerError::IntervalUnitInvalid,
            TriggerError::HourRequired,
            TriggerError::HourRange,
            TriggerError::MinuteRequired,
            TriggerError::MinuteRange,
            TriggerError::Weekday {
                value: "synthetic".to_owned(),
            },
            TriggerError::WeekdayEmpty,
            TriggerError::DayRange,
        ];
        for error in validation_errors {
            assert!(!trigger_error(error.clone(), Locale::Es).is_empty());
            assert!(!trigger_error(error, Locale::En).is_empty());
        }

        assert_eq!(
            trigger_config_input(Some(
                &json!({"type":"interval", "days":2, "hours":"3", "minutes":4.0})
            )),
            TriggerConfigInput::Interval {
                days: IntegerInput::Value(2),
                hours: IntegerInput::Value(3),
                minutes: IntegerInput::Value(4),
                seconds: IntegerInput::Missing,
            }
        );
        assert_eq!(
            trigger_config_input(Some(&json!({"type":"interval", "hours":2.5}))),
            TriggerConfigInput::Interval {
                days: IntegerInput::Missing,
                hours: IntegerInput::Invalid,
                minutes: IntegerInput::Missing,
                seconds: IntegerInput::Missing,
            }
        );
        // Null fields from models that fill in the whole schema are ignored.
        assert_eq!(
            trigger_config_input(Some(
                &json!({"type":"cron", "hour":9, "minute":0, "days":null})
            )),
            TriggerConfigInput::Cron {
                hour: IntegerInput::Value(9),
                minute: IntegerInput::Value(0),
                weekdays: None,
                day: IntegerInput::Missing,
            }
        );
        for (config, name) in [
            (
                json!({"type":"cron", "hour":9, "minute":0, "weekdays":"mon"}),
                "weekdays",
            ),
            (json!({"type":"interval", "hour":3}), "hour"),
            (json!({"type":"interval", "every":"3h"}), "every"),
        ] {
            assert_eq!(
                trigger_config_input(Some(&config)),
                TriggerConfigInput::UnknownField {
                    name: name.to_owned()
                }
            );
        }
        assert_eq!(
            trigger_config_input(Some(&json!({"type":"synthetic"}))),
            TriggerConfigInput::Unsupported
        );
        assert_eq!(
            trigger_config_input(Some(&json!({"type":"cron", "hour":null, "minute":"bad"}))),
            TriggerConfigInput::Cron {
                hour: IntegerInput::Missing,
                minute: IntegerInput::Invalid,
                weekdays: None,
                day: IntegerInput::Missing,
            }
        );

        let triggers = [
            TaskTrigger::IntervalDays { days: 2 },
            TaskTrigger::Cron {
                hour: 9,
                minute: 5,
                weekdays: Vec::new(),
                day: Some(3),
            },
            TaskTrigger::Cron {
                hour: 9,
                minute: 5,
                weekdays: Vec::new(),
                day: None,
            },
        ];
        for trigger in &triggers {
            let _schedule = task_schedule(trigger);
            assert!(!describe_trigger(trigger, Locale::Es).is_empty());
            assert!(!describe_trigger(trigger, Locale::En).is_empty());
        }
        for seconds in [60, 120, 3_600, 7_200, 86_400, 172_800] {
            assert!(!interval_description(seconds, true, Locale::Es).is_empty());
            assert!(!interval_description(seconds, true, Locale::En).is_empty());
            assert!(!interval_description(seconds, false, Locale::Es).is_empty());
            assert!(!interval_description(seconds, false, Locale::En).is_empty());
        }
        for day in ["mon", "tue", "wed", "thu", "fri", "sat", "sun", "synthetic"] {
            assert!(!localized_weekday(day, Locale::Es).is_empty());
            assert_eq!(localized_weekday(day, Locale::En), day);
        }
    }

    #[test]
    fn pricing_estimate_failures_stop_before_the_credit_check_in_both_locales() -> TestResult {
        let pricing = Arc::new(OpenRouterPricingCache::new("synthetic-key", "not-a-url")?);
        for (locale, expected) in [
            (Locale::En, "I could not calculate the task cost. Try again"),
            (
                Locale::Es,
                "No pude calcular el costo de la tarea. Probá de nuevo",
            ),
        ] {
            let state = Rc::new(RefCell::new(StoreState::default()));
            let calls = Rc::new(RefCell::new(Vec::new()));
            let mut tool = TaskSetTool::new(
                Store(Rc::clone(&state)),
                Balance {
                    calls: Rc::clone(&calls),
                    ..Balance::personal(Ok(i64::MAX))
                },
                Ids(Vec::new()),
                fixed_clock as Clock,
                context(locale),
            )
            .with_openrouter_pricing(Arc::clone(&pricing));
            let result = tool.execute(set_request(Some(60), None, None), "call");
            assert_eq!(result.output, expected);
            assert_eq!(task_cost_error(locale), expected);
            assert_eq!(result.diagnostics.len(), 1);
            assert!(
                result.diagnostics[0].starts_with("task reserve estimate failed: "),
                "{:?}",
                result.diagnostics
            );
            assert!(calls.borrow().is_empty());
            assert!(state.borrow().saved.is_empty());
        }
        Ok(())
    }

    #[test]
    fn next_run_overflow_is_reported_without_consuming_an_id_or_persisting() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut tool = TaskSetTool::new(
            Store(Rc::clone(&state)),
            Balance::personal(Ok(i64::MAX)),
            Ids(vec![Err("the ID source must not be consulted")]),
            (|| i64::MAX) as Clock,
            context(Locale::En),
        );
        let result = tool.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, "I could not create the task");
        assert_eq!(result.diagnostics.len(), 1);
        assert!(result.diagnostics[0].starts_with("task next run calculation failed: "));
        assert_eq!(tool.ids.0.len(), 1);
        assert!(state.borrow().saved.is_empty());
    }

    #[test]
    fn spanish_user_and_credit_rejections_are_localized() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut missing_user = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::Es);
        missing_user.context.user_id = None;
        assert_eq!(
            missing_user
                .execute(set_request(Some(60), None, None), "call")
                .output,
            "No pude identificar tu usuario para cobrar la tarea"
        );

        let mut poor = set_tool(Rc::clone(&state), Ok(0), Locale::Es);
        let output = poor
            .execute(set_request(Some(60), None, None), "call")
            .output;
        assert!(
            output.starts_with("No te alcanzan los créditos personales para esa tarea: tenés 0"),
            "{output}"
        );
        assert!(output.ends_with("\nCargá con /topup antes de crearla"));
        assert_eq!(
            task_credit_insufficient(0, 1_000_000, Locale::En),
            format!(
                "Not enough personal credits for this task: you have {} and need {}\nUse /topup before creating it",
                display_credit_units(CreditUnits::new(0)),
                display_credit_units(CreditUnits::new(1_000_000)),
            )
        );
        assert!(state.borrow().saved.is_empty());
    }

    #[test]
    fn english_weekday_cron_description_keeps_canonical_day_names() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut cron = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        assert_eq!(
            cron.execute(
                set_request(
                    None,
                    None,
                    Some(json!({"type":"cron", "hour":7, "minute":30, "day_of_week":"mon,fri"})),
                ),
                "call"
            )
            .output,
            "Task scheduled: on mon, fri at 07:30\ncheck the synthetic result\nID: abc12345"
        );
        assert_eq!(state.borrow().saved.len(), 1);
    }

    #[test]
    fn empty_task_lists_and_spanish_cancellations_are_localized() {
        let state = Rc::new(RefCell::new(StoreState::default()));
        let mut english = TaskListTool::new(Store(Rc::clone(&state)), "-100", Locale::En);
        assert_eq!(
            english
                .execute(ExternalToolRequest::TaskList, "call")
                .output,
            "there are no tasks"
        );
        let mut spanish = TaskListTool::new(Store(Rc::clone(&state)), "-100", Locale::Es);
        assert_eq!(
            spanish
                .execute(ExternalToolRequest::TaskList, "call")
                .output,
            "no hay tareas"
        );

        {
            let mut state = state.borrow_mut();
            state.tasks = task("abc12345", "-100").into_iter().collect();
            state.cancel_result = Ok(true);
        }
        let mut cancel = TaskCancelTool::new(Store(Rc::clone(&state)), "-100", Locale::Es);
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "abc12345".to_owned(),
                    },
                    "call"
                )
                .output,
            "Tarea abc12345 cancelada"
        );
        assert_eq!(
            state.borrow().canceled,
            [("abc12345".to_owned(), "-100".to_owned())]
        );
    }

    #[test]
    fn billing_funds_source_surfaces_repository_failures_as_text() {
        let endpoint = crate::test_env::redis_endpoint().unwrap_or(RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        });
        let cap = RedisCreditlessCap::new(&endpoint);
        assert!(cap.is_ok());
        let Ok(cap) = cap else { return };
        let mut funds = BillingTaskFunds::new(
            "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic?sslmode=disable",
            cap,
        );
        let error = funds.funds("-100", 7);
        assert!(
            matches!(&error, Err(message) if !message.is_empty()),
            "{error:?}"
        );
        let error = funds.funds("7", 7);
        assert!(
            matches!(&error, Err(message) if !message.is_empty()),
            "{error:?}"
        );
    }
}
