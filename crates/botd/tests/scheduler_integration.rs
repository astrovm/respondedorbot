use std::collections::BTreeMap;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_task_store::RedisTaskStore;
use bot_adapters::task_record::TaskRecordDocument;
use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};
use botd::scheduler::{
    ScheduledTaskExecutor, SchedulerMode, SchedulerSettings, SchedulerStep,
    TaskExecutionDisposition, TaskObservation, TaskScheduler,
};

fn redis_endpoint() -> Option<RedisEndpoint> {
    let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
    Some(RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    })
}

fn document(
    task_id: &str,
    schedule: TaskSchedule,
    next_run_at: i64,
) -> Result<TaskRecordDocument, Box<dyn Error>> {
    Ok(TaskRecordDocument {
        task: ScheduledTask {
            id: TaskId::new(task_id)?,
            chat_id: "-100987654".to_owned(),
            text: "synthetic scheduled task".to_owned(),
            user_name: "synthetic-user".to_owned(),
            user_id: Some(4242),
            schedule,
            timezone_offset: -3,
            locale: "en".to_owned(),
            schedule_anchor_at: Some(next_run_at),
            next_run_at: Some(next_run_at),
            last_execution_id: None,
        },
        legacy_run_date: None,
        extra: BTreeMap::new(),
    })
}

#[derive(Default)]
struct Executor {
    executions: usize,
}

impl ScheduledTaskExecutor for Executor {
    type Error = &'static str;

    fn execute(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
    ) -> Result<TaskExecutionDisposition, Self::Error> {
        self.executions += 1;
        Ok(TaskExecutionDisposition::Complete)
    }
}

#[test]
fn real_redis_scheduler_claims_executes_advances_and_deletes() -> Result<(), Box<dyn Error>> {
    let Some(endpoint) = redis_endpoint() else {
        return Ok(());
    };
    let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let once_id = format!("once_{}_{}", std::process::id(), suffix);
    let repeat_id = format!("repeat_{}_{}", std::process::id(), suffix);
    let now = i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())?;
    let setup = RedisTaskStore::new(&endpoint)?;
    setup.save_task(&document(&once_id, TaskSchedule::Once, now)?, 600)?;
    setup.save_task(
        &document(
            &repeat_id,
            TaskSchedule::IntervalSeconds { seconds: 600 },
            now,
        )?,
        600,
    )?;

    let owner = format!("integration-owner-{}_{}", std::process::id(), suffix);
    let mut scheduler = TaskScheduler::new(
        RedisTaskStore::new(&endpoint)?,
        Executor::default(),
        SchedulerMode::Authoritative,
        SchedulerSettings::default(),
        owner,
    )?;
    let step = scheduler.step(now)?;
    let SchedulerStep::Observed { tasks, failures } = step else {
        return Err("integration scheduler did not acquire ownership".into());
    };
    assert!(failures.is_empty());
    assert_eq!(tasks.len(), 2);
    assert!(
        tasks
            .iter()
            .all(|task| matches!(task.observation, TaskObservation::Executed { .. }))
    );

    let verify = RedisTaskStore::new(&endpoint)?;
    assert!(verify.load_task(&once_id)?.is_none());
    let recurring = verify
        .load_task(&repeat_id)?
        .ok_or("recurring task was not retained")?;
    assert_eq!(recurring.task.next_run_at, now.checked_add(600));
    assert!(recurring.task.last_execution_id.is_some());

    verify.cancel_task(&repeat_id, "-100987654")?;
    assert!(scheduler.shutdown()?);
    Ok(())
}
