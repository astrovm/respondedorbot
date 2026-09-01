//! Composition for native scheduled-task verification and execution.

use std::convert::Infallible;

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::openrouter_chat::ReqwestOpenRouterTransport;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_task_store::RedisTaskStore;
use bot_adapters::telegram_http::ReqwestTelegramTransport;
use bot_core::scheduled_tasks::ScheduledTask;
use thiserror::Error;

use crate::composition::{TelegramActionSink, TelegramDeliveryCoordinator};
use crate::native_ai::{
    ActionTaskMessenger, OpenRouterTaskProvider, PRIMARY_CHAT_MODEL, PostgresTaskBilling,
};
use crate::scheduler::{
    ScheduledTaskExecutor, SchedulerMode, SchedulerSettings, SchedulerStep,
    TaskExecutionDisposition, TaskScheduler,
};
use crate::task_executor::{NativeTaskExecutor, StderrTaskDiagnostics};

#[derive(Debug, Default, Clone, Copy)]
pub struct VerificationExecutor;

impl ScheduledTaskExecutor for VerificationExecutor {
    type Error = Infallible;

    fn execute(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
    ) -> Result<TaskExecutionDisposition, Self::Error> {
        Ok(TaskExecutionDisposition::Retry)
    }
}

pub type TaskVerifier = TaskScheduler<RedisTaskStore, VerificationExecutor>;

type ConcreteExecutor = NativeTaskExecutor<
    OpenRouterTaskProvider<ReqwestOpenRouterTransport>,
    PostgresTaskBilling<BillingRepository>,
    ActionTaskMessenger<TelegramActionSink<ReqwestTelegramTransport>>,
    StderrTaskDiagnostics,
>;

pub type ConcreteTaskScheduler = TaskScheduler<RedisTaskStore, ConcreteExecutor>;

pub struct TaskServiceOptions<'a> {
    pub redis_endpoint: &'a RedisEndpoint,
    pub database_url: &'a str,
    pub telegram_token: &'a str,
    pub openrouter_api_key: &'a str,
    pub openrouter_base_url: &'a str,
    pub system_prompt: &'a str,
    pub owner_token: &'a str,
    pub mode: SchedulerMode,
    pub telegram_delivery: TelegramDeliveryCoordinator,
}

#[derive(Debug, Error)]
pub enum TaskServiceError {
    #[error("could not construct the scheduled-task Redis store: {0}")]
    Redis(#[from] bot_adapters::redis_task_store::RedisTaskStoreError),
    #[error("could not construct the OpenRouter transport: {0}")]
    OpenRouter(#[from] bot_adapters::openrouter_chat::OpenRouterChatError),
    #[error("could not construct the Telegram transport: {0:?}")]
    Telegram(bot_adapters::telegram_http::TransportFailureKind),
    #[error("could not construct the scheduled-task engine: {0}")]
    Scheduler(#[from] crate::scheduler::SchedulerError),
}

pub fn build_task_verifier(
    endpoint: &RedisEndpoint,
    owner_token: &str,
) -> Result<TaskVerifier, TaskServiceError> {
    Ok(TaskScheduler::new(
        RedisTaskStore::new(endpoint)?,
        VerificationExecutor,
        SchedulerMode::Verify,
        SchedulerSettings::default(),
        owner_token,
    )?)
}

pub fn verify_tasks_once(
    endpoint: &RedisEndpoint,
    owner_token: &str,
    now: i64,
) -> Result<SchedulerStep, TaskServiceError> {
    Ok(build_task_verifier(endpoint, owner_token)?.step(now)?)
}

pub fn build_task_scheduler(
    options: TaskServiceOptions<'_>,
) -> Result<ConcreteTaskScheduler, TaskServiceError> {
    let provider = OpenRouterTaskProvider::new(
        ReqwestOpenRouterTransport::new()?,
        options.openrouter_api_key,
        options.openrouter_base_url,
        PRIMARY_CHAT_MODEL,
        options.system_prompt,
    );
    let billing = PostgresTaskBilling::new(
        BillingRepository::new(options.database_url),
        PRIMARY_CHAT_MODEL,
    );
    let action_transport = ReqwestTelegramTransport::new().map_err(TaskServiceError::Telegram)?;
    let messenger = ActionTaskMessenger::new(
        TelegramActionSink::new(action_transport, options.telegram_token)
            .with_delivery_coordinator(options.telegram_delivery),
    );
    let executor = NativeTaskExecutor::new(provider, billing, messenger, StderrTaskDiagnostics);
    Ok(TaskScheduler::new(
        RedisTaskStore::new(options.redis_endpoint)?,
        executor,
        options.mode,
        SchedulerSettings::default(),
        options.owner_token,
    )?)
}

#[cfg(test)]
mod tests {
    use bot_adapters::redis_connection::RedisEndpoint;

    use super::{TaskServiceOptions, build_task_scheduler};
    use crate::composition::TelegramDeliveryCoordinator;
    use crate::scheduler::SchedulerMode;

    #[test]
    fn authoritative_composition_is_side_effect_free_until_the_scheduler_steps() {
        let result = build_task_scheduler(TaskServiceOptions {
            redis_endpoint: &RedisEndpoint {
                host: "synthetic.invalid".to_owned(),
                port: 6379,
                password: Some("synthetic-password".to_owned()),
            },
            database_url: "postgresql://synthetic.invalid/database",
            telegram_token: "synthetic-token",
            openrouter_api_key: "synthetic-key",
            openrouter_base_url: "https://synthetic.invalid/api/v1",
            system_prompt: "synthetic persona",
            owner_token: "synthetic-owner",
            mode: SchedulerMode::Authoritative,
            telegram_delivery: TelegramDeliveryCoordinator::default(),
        });
        assert!(result.is_ok());
    }
}
