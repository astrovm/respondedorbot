//! Native task creation, listing, and cancellation AI tools.

use std::collections::BTreeMap;

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::redis_task_store::RedisTaskStore;
use bot_adapters::task_record::TaskRecordDocument;
use bot_core::credit_units::{CreditUnits, format_credit_units};
use bot_core::locale::Locale;
use bot_core::scheduled_tasks::{
    ScheduledTask, TaskId, TaskSchedule, initial_next_run, parse_weekday,
};
use bot_core::task_commands::format_task_summary;
use bot_core::task_triggers::{
    IntegerInput, TaskTrigger, TaskTriggerInput, TriggerConfigInput, TriggerError,
    parse_task_trigger,
};
use chrono::{DateTime, SecondsFormat, Utc};
use serde_json::Value;

use crate::chat_tool_loop::ToolExecutionResult;
use crate::native_ai::estimate_task_reserve_credit_units;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

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

pub trait PersonalBalanceSource {
    fn balance(&mut self, user_id: i64) -> Result<i64, String>;
}

impl PersonalBalanceSource for BillingRepository {
    fn balance(&mut self, user_id: i64) -> Result<i64, String> {
        self.get_balance("user", user_id)
            .map_err(|error| error.to_string())
    }
}

pub trait TaskIdSource {
    fn next_id(&mut self) -> Result<TaskId, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RandomTaskIdSource;

impl TaskIdSource for RandomTaskIdSource {
    fn next_id(&mut self) -> Result<TaskId, String> {
        TaskId::new(format!("{:08x}", rand::random::<u32>())).map_err(|error| error.to_string())
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
        }
    }
}

impl<Store, Balance, Ids, Now> ExternalToolExecutor for TaskSetTool<Store, Balance, Ids, Now>
where
    Store: TaskToolStore,
    Balance: PersonalBalanceSource,
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
        let locale_code = locale_code(self.context.locale);
        let required = match estimate_task_reserve_credit_units(&text, locale_code) {
            Ok(required) => required.max(1),
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    task_cost_error(self.context.locale),
                    vec![format!("task reserve estimate failed: {error}")],
                );
            }
        };
        let balance = match self.balance.balance(user_id) {
            Ok(balance) => balance,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    task_credit_check(self.context.locale),
                    vec![format!("task credit check failed: {error}")],
                );
            }
        };
        if balance < required {
            return ToolExecutionResult::output(task_credit_insufficient(
                balance,
                required,
                self.context.locale,
            ));
        }

        let now = (self.now)();
        let schedule = task_schedule(&trigger);
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
        let legacy_run_date = matches!(schedule, TaskSchedule::Once)
            .then(|| timestamp_text(next_run_at))
            .transpose();
        let legacy_run_date = match legacy_run_date {
            Ok(value) => value,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    create_error(self.context.locale),
                    vec![error],
                );
            }
        };
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
            legacy_run_date,
            extra: BTreeMap::new(),
        };
        if let Err(error) = self.store.save(&document, TASK_RECORD_TTL_SECONDS) {
            return ToolExecutionResult::with_diagnostics(
                create_error(self.context.locale),
                vec![format!("task persistence failed: {error}")],
            );
        }
        ToolExecutionResult::output(created(
            &describe_trigger(&trigger, self.context.locale),
            &text,
            self.context.locale,
        ))
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
                no_tasks(self.locale),
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
                    task_not_found(self.locale),
                    vec![format!("task cancel list failed: {error}")],
                );
            }
        };
        if !tasks.iter().any(|task| task.id == task_id) {
            return ToolExecutionResult::output(task_not_found(self.locale));
        }
        match self.store.cancel(&task_id, &self.chat_id) {
            Ok(true) => ToolExecutionResult::output(task_canceled(&task_id, self.locale)),
            Ok(false) => ToolExecutionResult::output(task_not_found(self.locale)),
            Err(error) => ToolExecutionResult::with_diagnostics(
                task_not_found(self.locale),
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
        Some(value) => value
            .as_i64()
            .map_or(IntegerInput::Invalid, IntegerInput::Value),
    }
}

fn trigger_config_input(config: Option<&Value>) -> TriggerConfigInput {
    let Some(config) = config.and_then(Value::as_object) else {
        return TriggerConfigInput::Missing;
    };
    match config.get("type").and_then(Value::as_str) {
        Some("interval") => TriggerConfigInput::IntervalDays {
            days: value_integer(config.get("days")),
        },
        Some("cron") => TriggerConfigInput::Cron {
            hour: value_integer(config.get("hour")),
            minute: value_integer(config.get("minute")),
            weekdays: config
                .get("day_of_week")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            day: value_integer(config.get("day")),
        },
        _ => TriggerConfigInput::Unsupported,
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
        (TriggerError::Required, Locale::Es) => "necesito usar algun parametro de tiempo: delay_seconds (una vez), interval_seconds (repetir), o trigger_config.".to_owned(),
        (TriggerError::Required, Locale::En) => "provide delay_seconds, interval_seconds, or trigger_config".to_owned(),
        (TriggerError::UnsupportedType, Locale::Es) => "trigger_config.type debe ser 'interval' o 'cron'".to_owned(),
        (TriggerError::UnsupportedType, Locale::En) => "trigger_config.type must be 'interval' or 'cron'".to_owned(),
        (TriggerError::DelayPositive, Locale::Es) => "delay_seconds debe ser un entero positivo".to_owned(),
        (TriggerError::DelayPositive, Locale::En) => "delay_seconds must be a positive integer".to_owned(),
        (TriggerError::DelayMaximum, Locale::Es) => "el maximo es 10 años".to_owned(),
        (TriggerError::DelayMaximum, Locale::En) => "the maximum delay is 10 years".to_owned(),
        (TriggerError::IntervalMinimum, Locale::Es) => "el intervalo minimo es 300 segundos (5 min)".to_owned(),
        (TriggerError::IntervalMinimum, Locale::En) => "the minimum interval is 300 seconds (5 minutes)".to_owned(),
        (TriggerError::IntervalMaximum, Locale::Es) => "el intervalo maximo es 7 dias".to_owned(),
        (TriggerError::IntervalMaximum, Locale::En) => "the maximum interval is 7 days".to_owned(),
        (TriggerError::DaysRequired, Locale::Es) => "days es requerido para trigger interval".to_owned(),
        (TriggerError::DaysRequired, Locale::En) => "days is required for an interval trigger".to_owned(),
        (TriggerError::DaysPositive, Locale::Es) => "days debe ser un entero positivo".to_owned(),
        (TriggerError::DaysPositive, Locale::En) => "days must be a positive integer".to_owned(),
        (TriggerError::DaysMaximum, Locale::Es) => "el maximo son 90 dias".to_owned(),
        (TriggerError::DaysMaximum, Locale::En) => "the maximum interval is 90 days".to_owned(),
        (TriggerError::HourRequired, Locale::Es) => "hour es requerido para trigger cron".to_owned(),
        (TriggerError::HourRequired, Locale::En) => "hour is required for a cron trigger".to_owned(),
        (TriggerError::HourRange, Locale::Es) => "hour debe ser 0-23".to_owned(),
        (TriggerError::HourRange, Locale::En) => "hour must be between 0 and 23".to_owned(),
        (TriggerError::MinuteRequired, Locale::Es) => "minute es requerido para trigger cron".to_owned(),
        (TriggerError::MinuteRequired, Locale::En) => "minute is required for a cron trigger".to_owned(),
        (TriggerError::MinuteRange, Locale::Es) => "minute debe ser 0-59".to_owned(),
        (TriggerError::MinuteRange, Locale::En) => "minute must be between 0 and 59".to_owned(),
        (TriggerError::Weekday { value }, Locale::Es) => format!("day_of_week invalido: {value}"),
        (TriggerError::Weekday { value }, Locale::En) => format!("invalid day_of_week: {value}"),
        (TriggerError::WeekdayEmpty, Locale::Es) => "day_of_week invalido".to_owned(),
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
            Locale::Es => format!("cada {days} dias"),
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
                    Locale::Es => format!("el dia {day} de cada mes a las {time}"),
                    Locale::En => format!("on day {day} of every month at {time}"),
                }
            } else {
                match locale {
                    Locale::Es => format!("todos los dias a las {time}"),
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
            if value == 1 { "dia" } else { "dias" },
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
        "wed" => "mie",
        "thu" => "jue",
        "fri" => "vie",
        "sat" => "sab",
        "sun" => "dom",
        other => other,
    }
}

fn no_chat(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not identify the chat"
    } else {
        "no se en que chat estoy"
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
        "no pude identificar tu usuario para cobrar la tarea"
    }
}
pub fn task_credit_check(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not check your personal credits, try again"
    } else {
        "no pude verificar tus créditos personales, probá de nuevo"
    }
}
pub fn task_cost_error(locale: Locale) -> &'static str {
    if locale == Locale::En {
        "I could not calculate the task cost, try again"
    } else {
        "no se pudo calcular el costo de la tarea, probá de nuevo"
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
        "that task does not exist in this chat"
    } else {
        "esa tarea no existe en este chat"
    }
}

pub fn task_credit_insufficient(balance: i64, required: i64, locale: Locale) -> String {
    let balance = format_credit_units(CreditUnits::new(balance));
    let required = format_credit_units(CreditUnits::new(required));
    match locale {
        Locale::Es => format!(
            "no tenés créditos personales suficientes para ejecutar esa tarea\n- tenés: {balance}\n- necesitás: {required}\ncargá con /topup antes de crearla"
        ),
        Locale::En => format!(
            "you do not have enough personal credits to run this task\n- available: {balance}\n- required: {required}\nuse /topup before creating it"
        ),
    }
}

fn created(schedule: &str, text: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("listo, tarea programada {schedule}: {text}"),
        Locale::En => format!("task scheduled {schedule}: {text}"),
    }
}

fn task_canceled(task_id: &TaskId, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("tarea {} cancelada", task_id.as_str()),
        Locale::En => format!("task {} canceled", task_id.as_str()),
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
    fn redis_task_tool_store_round_trips_through_the_public_port() -> Result<(), String> {
        let Some(port) = std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
        else {
            return Ok(());
        };
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let task_id =
            TaskId::new(format!("synthetic_{nonce}")).map_err(|error| error.to_string())?;
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
            legacy_run_date: None,
            extra: BTreeMap::new(),
        };
        let mut store = RedisTaskStore::new(&endpoint).map_err(|error| error.to_string())?;
        TaskToolStore::save(&mut store, &document, 600)?;
        assert_eq!(TaskToolStore::list(&mut store, &chat_id)?, [document.task]);
        assert!(TaskToolStore::cancel(&mut store, &task_id, &chat_id)?);
        assert!(TaskToolStore::list(&mut store, &chat_id)?.is_empty());
        assert!(RandomTaskIdSource.next_id().is_ok());
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
    }

    impl PersonalBalanceSource for Balance {
        fn balance(&mut self, user_id: i64) -> Result<i64, String> {
            self.calls.borrow_mut().push(user_id);
            self.result.clone()
        }
    }

    struct Ids(Vec<Result<TaskId, String>>);

    impl TaskIdSource for Ids {
        fn next_id(&mut self) -> Result<TaskId, String> {
            self.0.remove(0)
        }
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
    ) -> TaskSetTool<Store, Balance, Ids, impl FnMut() -> i64> {
        TaskSetTool::new(
            Store(state),
            Balance {
                result: balance,
                calls: Rc::new(RefCell::new(Vec::new())),
            },
            Ids(vec![
                TaskId::new("abc12345").map_err(|error| error.to_string()),
            ]),
            || 1_700_000_000,
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
    fn creates_a_rollback_compatible_one_shot_after_the_credit_precondition() {
        let state = Rc::new(RefCell::new(StoreState {
            cancel_result: Ok(true),
            ..StoreState::default()
        }));
        let mut tool = set_tool(Rc::clone(&state), Ok(i64::MAX), Locale::En);
        let result = tool.execute(set_request(Some(3_600), None, None), "call");
        assert_eq!(
            result.output,
            "task scheduled in 1 hour: check the synthetic result"
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
        assert_eq!(
            document.legacy_run_date.as_deref(),
            Some("2023-11-14T23:13:20Z")
        );
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
            "listo, tarea programada cada 1 dia: check the synthetic result"
        );
        assert_eq!(
            state.borrow().saved[0].0.task.schedule,
            TaskSchedule::IntervalSeconds { seconds: 86_400 }
        );
        assert!(state.borrow().saved[0].0.legacy_run_date.is_none());

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
            "listo, tarea programada los lun, mie a las 09:05: check the synthetic result"
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
                .contains("not have enough personal credits")
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
        assert_eq!(
            cancel
                .execute(
                    ExternalToolRequest::TaskCancel {
                        task_id: "abc12345".to_owned(),
                    },
                    "call"
                )
                .output,
            "task abc12345 canceled"
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
            "esa tarea no existe en este chat"
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
        assert_eq!(result.output, "no hay tareas");
        assert!(result.diagnostics[0].contains("synthetic list failure"));

        let mut cancel = TaskCancelTool::new(Store(state), "-100", Locale::En);
        let result = cancel.execute(
            ExternalToolRequest::TaskCancel {
                task_id: "abc12345".to_owned(),
            },
            "call",
        );
        assert_eq!(result.output, "that task does not exist in this chat");
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
            "no se en que chat estoy"
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
            Balance {
                result: Ok(i64::MAX),
                calls: Rc::new(RefCell::new(Vec::new())),
            },
            Ids(vec![Err("synthetic ID failure".to_owned())]),
            || 1_700_000_000,
            context(Locale::Es),
        );
        let result = id_failure.execute(set_request(Some(60), None, None), "call");
        assert_eq!(result.output, create_error(Locale::Es));
        assert!(result.diagnostics[0].contains("synthetic ID failure"));

        let mut timestamp_failure = TaskSetTool::new(
            Store(Rc::clone(&state)),
            Balance {
                result: Ok(i64::MAX),
                calls: Rc::new(RefCell::new(Vec::new())),
            },
            Ids(vec![
                TaskId::new("synthetic").map_err(|error| error.to_string()),
            ]),
            || i64::MAX - 60,
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
        assert_eq!(result.output, task_not_found(Locale::En));
        assert!(result.diagnostics[0].contains("synthetic cancel failure"));
    }

    #[test]
    fn trigger_validation_and_descriptions_cover_every_supported_shape() {
        let validation_errors = [
            TriggerError::Required,
            TriggerError::UnsupportedType,
            TriggerError::DelayPositive,
            TriggerError::DelayMaximum,
            TriggerError::IntervalMinimum,
            TriggerError::IntervalMaximum,
            TriggerError::DaysRequired,
            TriggerError::DaysPositive,
            TriggerError::DaysMaximum,
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
            trigger_config_input(Some(&json!({"type":"interval", "days":2}))),
            TriggerConfigInput::IntervalDays {
                days: IntegerInput::Value(2)
            }
        );
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
}
