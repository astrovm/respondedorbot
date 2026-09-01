//! Native scheduled-task AI execution with stable billing identities.

use bot_core::ai_response_cleanup::{
    clean_duplicate_response, remove_gordo_prefix, strip_markdown_formatting,
};
use bot_core::scheduled_tasks::ScheduledTask;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::scheduler::{ScheduledTaskExecutor, TaskExecutionDisposition};

const MAX_FALLBACK_RETRIES: usize = 1;
const MAX_EMPTY_RETRIES: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct TaskProviderReply {
    pub text: String,
    pub fallback: bool,
    pub billing_segments: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskProviderFailure<Error> {
    pub source: Error,
    pub billing_segments: Vec<Value>,
}

impl<Error> TaskProviderFailure<Error> {
    #[must_use]
    pub const fn new(source: Error, billing_segments: Vec<Value>) -> Self {
        Self {
            source,
            billing_segments,
        }
    }
}

impl<Error: std::fmt::Display> std::fmt::Display for TaskProviderFailure<Error> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(formatter)
    }
}

pub trait TaskAiProvider {
    type Error: std::fmt::Display;

    fn complete(
        &mut self,
        messages: &[TaskPromptMessage],
        task: &ScheduledTask,
        execution_id: &str,
    ) -> Result<TaskProviderReply, TaskProviderFailure<Self::Error>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPromptMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskReserveOutcome {
    Authorized,
    /// This execution already reached a durable billing settlement. The
    /// scheduler may safely mark it complete without repeating provider I/O
    /// or Telegram delivery after a process crash.
    AlreadySettled,
    Denied {
        message: String,
    },
}

pub trait TaskBilling {
    type Error: std::fmt::Display;

    fn reserve(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        prompt: &[TaskPromptMessage],
    ) -> Result<TaskReserveOutcome, Self::Error>;

    fn settle(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        segments: &[Value],
        reason: &'static str,
    ) -> Result<(), Self::Error>;

    fn refund(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        reason: &'static str,
    ) -> Result<(), Self::Error>;
}

pub trait TaskMessenger {
    type Error: std::fmt::Display;

    fn send(&mut self, chat_id: &str, text: &str) -> Result<(), Self::Error>;
}

const MAX_DELIVERY_ATTEMPTS: u8 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskExecutionState {
    response: Option<String>,
    billing_segments: Vec<Value>,
    kind: TaskExecutionKind,
    billing_finalized: bool,
    delivered: bool,
    #[serde(default)]
    delivery_attempts: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TaskExecutionKind {
    Success,
    Fallback,
    ProviderError,
    Empty,
}

pub trait TaskExecutionJournal {
    type Error: std::fmt::Display;

    fn load(&mut self, execution_id: &str) -> Result<Option<TaskExecutionState>, Self::Error>;
    fn save(&mut self, execution_id: &str, state: &TaskExecutionState) -> Result<(), Self::Error>;
}

pub trait TaskDiagnostics {
    fn record(&mut self, task_id: &str, stage: &'static str, message: &str);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StderrTaskDiagnostics;

impl TaskDiagnostics for StderrTaskDiagnostics {
    fn record(&mut self, task_id: &str, stage: &'static str, message: &str) {
        eprintln!("scheduled task {task_id} {stage}: {message}");
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NativeTaskExecutorError {
    #[error("scheduled-task billing failed during {stage}: {message}")]
    Billing {
        stage: &'static str,
        message: String,
    },
    #[error("scheduled-task delivery failed: {message}")]
    Delivery { message: String },
    #[error("scheduled-task journal failed during {stage}: {message}")]
    Journal {
        stage: &'static str,
        message: String,
    },
}

pub struct NativeTaskExecutor<Provider, Billing, Messenger, Journal, Diagnostics> {
    provider: Provider,
    billing: Billing,
    messenger: Messenger,
    journal: Journal,
    diagnostics: Diagnostics,
}

impl<Provider, Billing, Messenger, Journal, Diagnostics>
    NativeTaskExecutor<Provider, Billing, Messenger, Journal, Diagnostics>
where
    Provider: TaskAiProvider,
    Billing: TaskBilling,
    Messenger: TaskMessenger,
    Journal: TaskExecutionJournal,
    Diagnostics: TaskDiagnostics,
{
    #[must_use]
    pub const fn new(
        provider: Provider,
        billing: Billing,
        messenger: Messenger,
        journal: Journal,
        diagnostics: Diagnostics,
    ) -> Self {
        Self {
            provider,
            billing,
            messenger,
            journal,
            diagnostics,
        }
    }

    fn billing_error(
        stage: &'static str,
        error: impl std::fmt::Display,
    ) -> NativeTaskExecutorError {
        NativeTaskExecutorError::Billing {
            stage,
            message: error.to_string(),
        }
    }

    fn settle_or_refund(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        segments: &[Value],
        settlement_reason: &'static str,
        refund_reason: &'static str,
    ) -> Result<(), NativeTaskExecutorError> {
        if segments.is_empty() {
            self.billing
                .refund(task, execution_id, refund_reason)
                .map_err(|error| Self::billing_error("refund", error))
        } else {
            self.billing
                .settle(task, execution_id, segments, settlement_reason)
                .map_err(|error| Self::billing_error("settle", error))
        }
    }

    fn send_nonfatal(&mut self, task: &ScheduledTask, text: &str) {
        if let Err(error) = self.messenger.send(&task.chat_id, text) {
            self.diagnostics
                .record(task.id.as_str(), "delivery", &error.to_string());
        }
    }

    fn journal_error(
        stage: &'static str,
        error: impl std::fmt::Display,
    ) -> NativeTaskExecutorError {
        NativeTaskExecutorError::Journal {
            stage,
            message: error.to_string(),
        }
    }

    fn finish_saved_result(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
        mut state: TaskExecutionState,
    ) -> Result<TaskExecutionDisposition, NativeTaskExecutorError> {
        if !state.billing_finalized {
            match state.kind {
                TaskExecutionKind::Success => self
                    .billing
                    .settle(task, execution_id, &state.billing_segments, "task_success")
                    .map_err(|error| Self::billing_error("settle", error))?,
                TaskExecutionKind::Fallback => self.settle_or_refund(
                    task,
                    execution_id,
                    &state.billing_segments,
                    "task_fallback_provider_usage",
                    "task_fallback",
                )?,
                TaskExecutionKind::ProviderError => self.settle_or_refund(
                    task,
                    execution_id,
                    &state.billing_segments,
                    "task_error_provider_usage",
                    "task_error",
                )?,
                TaskExecutionKind::Empty => self.settle_or_refund(
                    task,
                    execution_id,
                    &state.billing_segments,
                    "task_empty_provider_usage",
                    "task_empty",
                )?,
            }
            state.billing_finalized = true;
            self.journal
                .save(execution_id, &state)
                .map_err(|error| Self::journal_error("save_billing", error))?;
        }
        if !state.delivered
            && let Some(response) = state.response.as_deref()
        {
            if state.delivery_attempts >= MAX_DELIVERY_ATTEMPTS {
                self.diagnostics.record(
                    task.id.as_str(),
                    "delivery_abandoned",
                    "scheduled-task result reached its delivery retry limit",
                );
                return Ok(TaskExecutionDisposition::Complete);
            }
            if let Err(error) = self
                .messenger
                .send(&task.chat_id, &task_result(task, response))
            {
                let message = error.to_string();
                state.delivery_attempts = state.delivery_attempts.saturating_add(1);
                self.journal
                    .save(execution_id, &state)
                    .map_err(|error| Self::journal_error("save_delivery_failure", error))?;
                let exhausted = state.delivery_attempts >= MAX_DELIVERY_ATTEMPTS;
                self.diagnostics.record(
                    task.id.as_str(),
                    if exhausted {
                        "delivery_abandoned"
                    } else {
                        "delivery"
                    },
                    &message,
                );
                if exhausted {
                    return Ok(TaskExecutionDisposition::Complete);
                }
                return Err(NativeTaskExecutorError::Delivery { message });
            }
            state.delivered = true;
            self.journal
                .save(execution_id, &state)
                .map_err(|error| Self::journal_error("save_delivery", error))?;
        }
        Ok(TaskExecutionDisposition::Complete)
    }

    fn execute_task(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
    ) -> Result<TaskExecutionDisposition, NativeTaskExecutorError> {
        if task.chat_id.is_empty() || task.text.is_empty() || task.user_name.is_empty() {
            self.diagnostics.record(
                task.id.as_str(),
                "validation",
                "chat, text, and user name are required",
            );
            return Ok(TaskExecutionDisposition::Retry);
        }
        if let Some(state) = self
            .journal
            .load(execution_id)
            .map_err(|error| Self::journal_error("load", error))?
        {
            return self.finish_saved_result(task, execution_id, state);
        }
        if task.user_id.is_none() {
            self.send_nonfatal(
                task,
                &execute_failed(task, credit_user_message(task.locale.as_str())),
            );
            return Ok(TaskExecutionDisposition::Complete);
        }

        let mut messages = build_task_messages(&task.text, &task.locale);
        match self
            .billing
            .reserve(task, execution_id, &messages)
            .map_err(|error| Self::billing_error("reserve", error))?
        {
            TaskReserveOutcome::Authorized => {}
            TaskReserveOutcome::AlreadySettled => {
                return Ok(TaskExecutionDisposition::Complete);
            }
            TaskReserveOutcome::Denied { message } => {
                self.send_nonfatal(task, &execute_failed(task, &message));
                return Ok(TaskExecutionDisposition::Complete);
            }
        }

        let mut fallback_retries = 0_usize;
        let mut empty_retries = 0_usize;
        let mut segments = Vec::new();
        loop {
            let reply = match self.provider.complete(&messages, task, execution_id) {
                Ok(reply) => reply,
                Err(error) => {
                    let message = error.to_string();
                    segments.extend(error.billing_segments);
                    self.diagnostics
                        .record(task.id.as_str(), "provider", &message);
                    let state = TaskExecutionState {
                        response: None,
                        billing_segments: segments,
                        kind: TaskExecutionKind::ProviderError,
                        billing_finalized: false,
                        delivered: true,
                        delivery_attempts: 0,
                    };
                    self.journal
                        .save(execution_id, &state)
                        .map_err(|error| Self::journal_error("save_provider_failure", error))?;
                    return self.finish_saved_result(task, execution_id, state);
                }
            };
            segments.extend(reply.billing_segments);
            if reply.text.trim().is_empty() {
                if empty_retries < MAX_EMPTY_RETRIES {
                    empty_retries += 1;
                    messages.push(TaskPromptMessage {
                        role: "system",
                        content: force_response(&task.locale).to_owned(),
                    });
                    continue;
                }
                let state = TaskExecutionState {
                    response: None,
                    billing_segments: segments,
                    kind: TaskExecutionKind::Empty,
                    billing_finalized: false,
                    delivered: true,
                    delivery_attempts: 0,
                };
                self.journal
                    .save(execution_id, &state)
                    .map_err(|error| Self::journal_error("save_empty_result", error))?;
                return self.finish_saved_result(task, execution_id, state);
            }
            if reply.fallback && fallback_retries < MAX_FALLBACK_RETRIES {
                fallback_retries += 1;
                messages.push(TaskPromptMessage {
                    role: "system",
                    content: force_nonfallback(&task.locale).to_owned(),
                });
                continue;
            }

            let state = TaskExecutionState {
                response: Some(clean_task_response(&reply.text)),
                billing_segments: segments,
                kind: if reply.fallback {
                    TaskExecutionKind::Fallback
                } else {
                    TaskExecutionKind::Success
                },
                billing_finalized: false,
                delivered: false,
                delivery_attempts: 0,
            };
            self.journal
                .save(execution_id, &state)
                .map_err(|error| Self::journal_error("save_provider_result", error))?;
            return self.finish_saved_result(task, execution_id, state);
        }
    }

    #[cfg(test)]
    fn parts(&self) -> (&Provider, &Billing, &Messenger, &Diagnostics) {
        (
            &self.provider,
            &self.billing,
            &self.messenger,
            &self.diagnostics,
        )
    }
}

impl<Provider, Billing, Messenger, Journal, Diagnostics> ScheduledTaskExecutor
    for NativeTaskExecutor<Provider, Billing, Messenger, Journal, Diagnostics>
where
    Provider: TaskAiProvider,
    Billing: TaskBilling,
    Messenger: TaskMessenger,
    Journal: TaskExecutionJournal,
    Diagnostics: TaskDiagnostics,
{
    type Error = NativeTaskExecutorError;

    fn execute(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
    ) -> Result<TaskExecutionDisposition, Self::Error> {
        self.execute_task(task, execution_id)
    }
}

#[must_use]
pub fn build_task_messages(text: &str, locale: &str) -> Vec<TaskPromptMessage> {
    vec![TaskPromptMessage {
        role: "user",
        content: format!("{text}\n\n{}", format_prompt(locale)),
    }]
}

fn format_prompt(locale: &str) -> &'static str {
    if locale == "en" {
        "INSTRUCTIONS:\n- keep the bot persona\n- use casual English\n- respond without markdown or emojis\n- use a numbered list when the user asks for a list or there are several topics\n- leave a blank line between numbered items\n- give each item a short title and a brief explanation\n- structure the answer so it is easy to read"
    } else {
        "INSTRUCCIONES:\n- mantené el personaje del gordo\n- usá lenguaje coloquial argentino\n- respondé en minúsculas, sin emojis, sin punto final\n- si el usuario pide una lista o hay varios temas, usá lista numerada: 1., 2., 3.\n- dejá una línea en blanco entre cada item numerado\n- cada item debe tener título corto en su propia línea y explicación breve abajo\n- no la pongas toda en una sola frase: estructurala para que sea fácil de leer"
    }
}

fn force_response(locale: &str) -> &'static str {
    if locale == "en" {
        "you must answer the task"
    } else {
        "respondé la tarea, es obligatorio"
    }
}

fn force_nonfallback(locale: &str) -> &'static str {
    if locale == "en" {
        "you must provide a real answer"
    } else {
        "tenés que responder. no hay opcion de no responder"
    }
}

fn credit_user_message(locale: &str) -> &'static str {
    if locale == "en" {
        "I could not identify your user to charge for the task"
    } else {
        "no pude identificar tu usuario para cobrar la tarea"
    }
}

fn execute_failed(task: &ScheduledTask, error: &str) -> String {
    if task.locale == "en" {
        format!(
            "{}, I could not run the task “{}”:\n{error}",
            task.user_name, task.text
        )
    } else {
        format!(
            "{}, no pude ejecutar la tarea «{}»:\n{error}",
            task.user_name, task.text
        )
    }
}

fn task_result(task: &ScheduledTask, response: &str) -> String {
    if task.locale == "en" {
        format!("{}, task “{}”:\n{response}", task.user_name, task.text)
    } else {
        format!("{}, tarea «{}»:\n{response}", task.user_name, task.text)
    }
}

fn clean_task_response(response: &str) -> String {
    let original_has_spacing = response.contains("\n\n");
    let cleaned =
        strip_markdown_formatting(&clean_duplicate_response(&remove_gordo_prefix(response)));
    if !original_has_spacing {
        return cleaned.trim().to_owned();
    }
    let mut lines = Vec::new();
    for line in cleaned.lines().map(str::trim_end) {
        let numbered = line.split_once('.').is_some_and(|(number, rest)| {
            number.trim().parse::<u64>().is_ok() && rest.starts_with(' ')
        });
        if numbered && lines.last().is_some_and(|prior: &String| !prior.is_empty()) {
            lines.push(String::new());
        }
        lines.push(line.to_owned());
    }
    lines.join("\n").trim().to_owned()
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::collections::{HashMap, VecDeque};

    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
    use serde_json::{Value, json};

    use super::{
        NativeTaskExecutor, NativeTaskExecutorError, ScheduledTaskExecutor, TaskAiProvider,
        TaskBilling, TaskDiagnostics, TaskExecutionDisposition, TaskExecutionJournal,
        TaskExecutionKind, TaskExecutionState, TaskMessenger, TaskPromptMessage,
        TaskProviderFailure, TaskProviderReply, TaskReserveOutcome, build_task_messages,
        clean_task_response,
    };

    #[derive(Default)]
    struct Provider {
        replies: VecDeque<Result<TaskProviderReply, TaskProviderFailure<&'static str>>>,
        prompts: Vec<Vec<TaskPromptMessage>>,
        execution_ids: Vec<String>,
    }

    impl TaskAiProvider for Provider {
        type Error = &'static str;

        fn complete(
            &mut self,
            messages: &[TaskPromptMessage],
            _task: &ScheduledTask,
            execution_id: &str,
        ) -> Result<TaskProviderReply, TaskProviderFailure<Self::Error>> {
            self.prompts.push(messages.to_vec());
            self.execution_ids.push(execution_id.to_owned());
            self.replies.pop_front().unwrap_or(Ok(TaskProviderReply {
                text: "synthetic answer".to_owned(),
                fallback: false,
                billing_segments: vec![json!({"kind": "chat"})],
            }))
        }
    }

    #[derive(Default)]
    struct Billing {
        reserve: VecDeque<Result<TaskReserveOutcome, &'static str>>,
        reserve_ids: Vec<String>,
        settlements: Vec<(String, Vec<Value>, &'static str)>,
        refunds: Vec<(String, &'static str)>,
        fail_settlement: bool,
    }

    impl TaskBilling for Billing {
        type Error = &'static str;

        fn reserve(
            &mut self,
            _task: &ScheduledTask,
            execution_id: &str,
            _prompt: &[TaskPromptMessage],
        ) -> Result<TaskReserveOutcome, Self::Error> {
            self.reserve_ids.push(execution_id.to_owned());
            self.reserve
                .pop_front()
                .unwrap_or(Ok(TaskReserveOutcome::Authorized))
        }

        fn settle(
            &mut self,
            _task: &ScheduledTask,
            execution_id: &str,
            segments: &[Value],
            reason: &'static str,
        ) -> Result<(), Self::Error> {
            if self.fail_settlement {
                return Err("settlement failed");
            }
            self.settlements
                .push((execution_id.to_owned(), segments.to_vec(), reason));
            Ok(())
        }

        fn refund(
            &mut self,
            _task: &ScheduledTask,
            execution_id: &str,
            reason: &'static str,
        ) -> Result<(), Self::Error> {
            self.refunds.push((execution_id.to_owned(), reason));
            Ok(())
        }
    }

    #[derive(Default)]
    struct Messenger {
        messages: Vec<(String, String)>,
        fail: bool,
    }

    impl TaskMessenger for Messenger {
        type Error = &'static str;

        fn send(&mut self, chat_id: &str, text: &str) -> Result<(), Self::Error> {
            self.messages.push((chat_id.to_owned(), text.to_owned()));
            if self.fail {
                Err("delivery failed")
            } else {
                Ok(())
            }
        }
    }

    #[derive(Default)]
    struct Journal(HashMap<String, TaskExecutionState>);

    impl TaskExecutionJournal for Journal {
        type Error = &'static str;

        fn load(&mut self, execution_id: &str) -> Result<Option<TaskExecutionState>, Self::Error> {
            Ok(self.0.get(execution_id).cloned())
        }

        fn save(
            &mut self,
            execution_id: &str,
            state: &TaskExecutionState,
        ) -> Result<(), Self::Error> {
            self.0.insert(execution_id.to_owned(), state.clone());
            Ok(())
        }
    }

    #[derive(Default)]
    struct Diagnostics(Vec<(String, &'static str, String)>);

    impl TaskDiagnostics for Diagnostics {
        fn record(&mut self, task_id: &str, stage: &'static str, message: &str) {
            self.0.push((task_id.to_owned(), stage, message.to_owned()));
        }
    }

    fn task(locale: &str) -> ScheduledTask {
        ScheduledTask {
            id: TaskId::new("task123").unwrap_or_else(|error| panic!("task id: {error}")),
            chat_id: "-100123".to_owned(),
            text: "synthetic task".to_owned(),
            user_name: "synthetic-user".to_owned(),
            user_id: Some(42),
            schedule: TaskSchedule::Once,
            timezone_offset: -3,
            locale: locale.to_owned(),
            schedule_anchor_at: Some(1_000),
            next_run_at: Some(1_000),
            last_execution_id: None,
        }
    }

    fn executor(
        provider: Provider,
        billing: Billing,
        messenger: Messenger,
    ) -> NativeTaskExecutor<Provider, Billing, Messenger, Journal, Diagnostics> {
        NativeTaskExecutor::new(
            provider,
            billing,
            messenger,
            Journal::default(),
            Diagnostics::default(),
        )
    }

    #[test]
    fn prompt_and_localized_results_match_the_python_contract() {
        let prompt = build_task_messages("recordame algo", "es");
        assert_eq!(prompt[0].role, "user");
        assert!(
            prompt[0]
                .content
                .starts_with("recordame algo\n\nINSTRUCCIONES:")
        );
        assert!(prompt[0].content.contains("usá lista numerada: 1., 2., 3."));
        let mut executor = executor(
            Provider::default(),
            Billing::default(),
            Messenger::default(),
        );
        assert_eq!(
            executor.execute(&task("es"), "task123:1000"),
            Ok(TaskExecutionDisposition::Complete)
        );
        let (provider, billing, messenger, _) = executor.parts();
        assert_eq!(provider.execution_ids, ["task123:1000"]);
        assert_eq!(billing.reserve_ids, ["task123:1000"]);
        assert_eq!(billing.settlements[0].2, "task_success");
        assert_eq!(
            messenger.messages[0].1,
            "synthetic-user, tarea «synthetic task»:\nsynthetic answer"
        );
    }

    #[test]
    fn denied_reservation_reports_failure_without_provider_io() {
        let billing = Billing {
            reserve: VecDeque::from([Ok(TaskReserveOutcome::Denied {
                message: "saldo insuficiente".to_owned(),
            })]),
            ..Billing::default()
        };
        let mut executor = executor(Provider::default(), billing, Messenger::default());
        assert_eq!(
            executor.execute(&task("es"), "task123:1000"),
            Ok(TaskExecutionDisposition::Complete)
        );
        let (provider, _, messenger, _) = executor.parts();
        assert!(provider.prompts.is_empty());
        assert_eq!(
            messenger.messages[0].1,
            "synthetic-user, no pude ejecutar la tarea «synthetic task»:\nsaldo insuficiente"
        );
    }

    #[test]
    fn settled_execution_does_not_repeat_provider_or_delivery_side_effects() {
        let billing = Billing {
            reserve: VecDeque::from([Ok(TaskReserveOutcome::AlreadySettled)]),
            ..Billing::default()
        };
        let mut executor = executor(Provider::default(), billing, Messenger::default());
        assert_eq!(
            executor.execute(&task("es"), "task123:1000"),
            Ok(TaskExecutionDisposition::Complete)
        );
        let (provider, billing, messenger, _) = executor.parts();
        assert!(provider.prompts.is_empty());
        assert!(billing.settlements.is_empty());
        assert!(billing.refunds.is_empty());
        assert!(messenger.messages.is_empty());
    }

    #[test]
    fn saved_result_retries_delivery_without_repeating_provider_or_billing() {
        let execution_id = "task123:1000";
        let mut journal = Journal::default();
        journal.0.insert(
            execution_id.to_owned(),
            TaskExecutionState {
                response: Some("saved answer".to_owned()),
                billing_segments: vec![json!({"kind": "chat"})],
                kind: TaskExecutionKind::Success,
                billing_finalized: true,
                delivered: false,
                delivery_attempts: 0,
            },
        );
        let mut executor = NativeTaskExecutor::new(
            Provider::default(),
            Billing::default(),
            Messenger::default(),
            journal,
            Diagnostics::default(),
        );

        assert_eq!(
            executor.execute(&task("en"), execution_id),
            Ok(TaskExecutionDisposition::Complete)
        );
        let (provider, billing, messenger, _) = executor.parts();
        assert!(provider.prompts.is_empty());
        assert!(billing.reserve_ids.is_empty());
        assert!(billing.settlements.is_empty());
        assert_eq!(
            messenger.messages[0].1,
            "synthetic-user, task “synthetic task”:\nsaved answer"
        );
    }

    #[test]
    fn delivery_failures_stop_after_the_persisted_retry_budget() {
        let execution_id = "task123:1000";
        let messenger = Messenger {
            fail: true,
            ..Messenger::default()
        };
        let mut executor = executor(Provider::default(), Billing::default(), messenger);

        for _ in 0..2 {
            assert_eq!(
                executor.execute(&task("es"), execution_id),
                Err(NativeTaskExecutorError::Delivery {
                    message: "delivery failed".to_owned(),
                })
            );
        }
        assert_eq!(
            executor.execute(&task("es"), execution_id),
            Ok(TaskExecutionDisposition::Complete)
        );
        // A crash after persisting the final failed attempt must not send again.
        assert_eq!(
            executor.execute(&task("es"), execution_id),
            Ok(TaskExecutionDisposition::Complete)
        );

        let (provider, billing, messenger, diagnostics) = executor.parts();
        assert_eq!(provider.execution_ids, [execution_id]);
        assert_eq!(billing.reserve_ids, [execution_id]);
        assert_eq!(billing.settlements.len(), 1);
        assert_eq!(messenger.messages.len(), 3);
        assert_eq!(
            diagnostics
                .0
                .iter()
                .filter(|(_, stage, _)| *stage == "delivery_abandoned")
                .count(),
            2
        );
    }

    #[test]
    fn retries_empty_and_fallback_replies_with_separate_budgets() {
        let provider = Provider {
            replies: VecDeque::from([
                Ok(TaskProviderReply {
                    text: String::new(),
                    fallback: false,
                    billing_segments: vec![json!({"attempt": 1})],
                }),
                Ok(TaskProviderReply {
                    text: "fallback".to_owned(),
                    fallback: true,
                    billing_segments: vec![json!({"attempt": 2})],
                }),
                Ok(TaskProviderReply {
                    text: "paid answer".to_owned(),
                    fallback: false,
                    billing_segments: vec![json!({"attempt": 3})],
                }),
            ]),
            ..Provider::default()
        };
        let mut executor = executor(provider, Billing::default(), Messenger::default());
        assert!(executor.execute(&task("en"), "task123:1000").is_ok());
        let (provider, billing, messenger, _) = executor.parts();
        assert_eq!(provider.prompts.len(), 3);
        assert_eq!(provider.prompts[1][1].content, "you must answer the task");
        assert_eq!(
            provider.prompts[2][2].content,
            "you must provide a real answer"
        );
        assert_eq!(billing.settlements[0].1.len(), 3);
        assert_eq!(
            messenger.messages[0].1,
            "synthetic-user, task “synthetic task”:\npaid answer"
        );
    }

    #[test]
    fn terminal_fallback_settles_usage_or_refunds_when_no_usage_exists() {
        for segments in [vec![json!({"cost": 1})], Vec::new()] {
            let provider = Provider {
                replies: VecDeque::from([
                    Ok(TaskProviderReply {
                        text: "fallback one".to_owned(),
                        fallback: true,
                        billing_segments: segments.clone(),
                    }),
                    Ok(TaskProviderReply {
                        text: "fallback two".to_owned(),
                        fallback: true,
                        billing_segments: segments.clone(),
                    }),
                ]),
                ..Provider::default()
            };
            let mut executor = executor(provider, Billing::default(), Messenger::default());
            assert!(executor.execute(&task("es"), "task123:1000").is_ok());
            let (_, billing, _, _) = executor.parts();
            if segments.is_empty() {
                assert_eq!(
                    billing.refunds,
                    [("task123:1000".to_owned(), "task_fallback")]
                );
            } else {
                assert_eq!(billing.settlements[0].2, "task_fallback_provider_usage");
                assert_eq!(billing.settlements[0].1.len(), 2);
            }
        }
    }

    #[test]
    fn provider_and_empty_failures_finalize_billing_without_scheduler_retry() {
        let provider = Provider {
            replies: VecDeque::from([Err(TaskProviderFailure::new(
                "provider unavailable",
                vec![json!({"attempt": "partial"})],
            ))]),
            ..Provider::default()
        };
        let mut failed = executor(provider, Billing::default(), Messenger::default());
        assert_eq!(
            failed.execute(&task("es"), "task123:1000"),
            Ok(TaskExecutionDisposition::Complete)
        );
        let (_, billing, _, diagnostics) = failed.parts();
        assert!(billing.refunds.is_empty());
        assert_eq!(billing.settlements[0].2, "task_error_provider_usage");
        assert_eq!(billing.settlements[0].1, [json!({"attempt": "partial"})]);
        assert_eq!(diagnostics.0[0].1, "provider");

        let provider = Provider {
            replies: VecDeque::from([Err(TaskProviderFailure::new(
                "provider unavailable",
                Vec::new(),
            ))]),
            ..Provider::default()
        };
        let mut unused = executor(provider, Billing::default(), Messenger::default());
        assert!(unused.execute(&task("es"), "task123:1000").is_ok());
        assert_eq!(unused.parts().1.refunds[0].1, "task_error");

        let provider = Provider {
            replies: VecDeque::from([
                Ok(TaskProviderReply {
                    text: String::new(),
                    fallback: false,
                    billing_segments: Vec::new(),
                }),
                Ok(TaskProviderReply {
                    text: "   ".to_owned(),
                    fallback: false,
                    billing_segments: Vec::new(),
                }),
            ]),
            ..Provider::default()
        };
        let mut empty = executor(provider, Billing::default(), Messenger::default());
        assert!(empty.execute(&task("es"), "task123:1000").is_ok());
        assert_eq!(empty.parts().1.refunds[0].1, "task_empty");
    }

    #[test]
    fn validation_user_identity_delivery_and_billing_failures_are_explicit() {
        let mut invalid = task("es");
        invalid.user_name.clear();
        let mut validation_executor = executor(
            Provider::default(),
            Billing::default(),
            Messenger::default(),
        );
        assert_eq!(
            validation_executor.execute(&invalid, "task123:1000"),
            Ok(TaskExecutionDisposition::Retry)
        );

        let mut unidentified = task("en");
        unidentified.user_id = None;
        assert_eq!(
            validation_executor.execute(&unidentified, "task123:1000"),
            Ok(TaskExecutionDisposition::Complete)
        );
        assert!(
            validation_executor.parts().2.messages[0]
                .1
                .contains("I could not identify your user")
        );

        let messenger = Messenger {
            fail: true,
            ..Messenger::default()
        };
        let mut delivery = executor(Provider::default(), Billing::default(), messenger);
        assert_eq!(
            delivery.execute(&task("es"), "task123:1000"),
            Err(NativeTaskExecutorError::Delivery {
                message: "delivery failed".to_owned()
            })
        );
        assert_eq!(delivery.parts().3.0[0].1, "delivery");

        let billing = Billing {
            fail_settlement: true,
            ..Billing::default()
        };
        let mut settlement = executor(Provider::default(), billing, Messenger::default());
        assert!(settlement.execute(&task("es"), "task123:1000").is_err());
    }

    #[test]
    fn cleanup_matches_markdown_and_numbered_spacing_contract() {
        assert_eq!(
            clean_task_response("Gordo: **hola**\n## titulo"),
            "hola\ntitulo"
        );
        assert_eq!(
            clean_task_response("1. noticia\ndetalle\n\n2. noticia\ndetalle"),
            "1. noticia\ndetalle\n\n2. noticia\ndetalle"
        );
    }
}
