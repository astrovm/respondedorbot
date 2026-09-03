//! Single-owner scheduled-task observation, claiming, and recovery.

use std::fmt::Display;

use bot_adapters::redis_task_store::{
    RedisTaskStore, TASK_SCHEDULER_OWNER_KEY, TaskOccurrenceCompletion,
};
use bot_adapters::task_record::{TaskRecordDocument, encode_task_record};
use bot_core::scheduled_tasks::{DueDecision, ScheduledTask, evaluate_due};
use thiserror::Error;

const DEFAULT_BATCH_LIMIT: usize = 100;
const DEFAULT_OWNER_TTL_SECONDS: i64 = 30;
const DEFAULT_CLAIM_TTL_SECONDS: i64 = 1_800;
const DEFAULT_RECORD_TTL_SECONDS: i64 = 86_400 * 3_650;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerMode {
    Verify,
    Authoritative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerSettings {
    pub batch_limit: usize,
    pub owner_ttl_seconds: i64,
    pub claim_ttl_seconds: i64,
    pub record_ttl_seconds: i64,
}

impl Default for SchedulerSettings {
    fn default() -> Self {
        Self {
            batch_limit: DEFAULT_BATCH_LIMIT,
            owner_ttl_seconds: DEFAULT_OWNER_TTL_SECONDS,
            claim_ttl_seconds: DEFAULT_CLAIM_TTL_SECONDS,
            record_ttl_seconds: DEFAULT_RECORD_TTL_SECONDS,
        }
    }
}

impl SchedulerSettings {
    fn validate(self) -> Result<Self, SchedulerError> {
        if self.batch_limit == 0
            || self.owner_ttl_seconds <= 0
            || self.claim_ttl_seconds <= 0
            || self.record_ttl_seconds <= 0
        {
            return Err(SchedulerError::InvalidSettings);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskExecutionDisposition {
    Complete,
    Retry,
}

pub trait ScheduledTaskExecutor {
    type Error: Display;

    fn execute(
        &mut self,
        task: &ScheduledTask,
        execution_id: &str,
    ) -> Result<TaskExecutionDisposition, Self::Error>;
}

pub trait SchedulerStore {
    type Error: Display;

    fn due_task_ids(&mut self, now: i64, limit: usize) -> Result<Vec<String>, Self::Error>;
    fn load_task(&mut self, task_id: &str) -> Result<Option<TaskRecordDocument>, Self::Error>;
    fn remove_due_task_id(&mut self, task_id: &str) -> Result<bool, Self::Error>;
    fn save_task(
        &mut self,
        document: &TaskRecordDocument,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error>;
    fn acquire_owner(&mut self, token: &str, ttl_seconds: i64) -> Result<bool, Self::Error>;
    fn renew_owner(&mut self, token: &str, ttl_seconds: i64) -> Result<bool, Self::Error>;
    fn release_owner(&mut self, token: &str) -> Result<bool, Self::Error>;
    fn claim_occurrence(
        &mut self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error>;
    fn release_occurrence(
        &mut self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
    ) -> Result<bool, Self::Error>;
    fn complete_occurrence(
        &mut self,
        completion: &SchedulerCompletion<'_>,
    ) -> Result<bool, Self::Error>;
}

pub struct SchedulerCompletion<'a> {
    pub task_id: &'a str,
    pub chat_id: &'a str,
    pub execution_id: &'a str,
    pub claim_token: &'a str,
    pub next_document: Option<&'a TaskRecordDocument>,
    pub record_ttl_seconds: i64,
}

impl SchedulerStore for RedisTaskStore {
    type Error = bot_adapters::redis_task_store::RedisTaskStoreError;

    fn due_task_ids(&mut self, now: i64, limit: usize) -> Result<Vec<String>, Self::Error> {
        RedisTaskStore::due_task_ids(self, now as f64, limit)
    }

    fn load_task(&mut self, task_id: &str) -> Result<Option<TaskRecordDocument>, Self::Error> {
        RedisTaskStore::load_task(self, task_id)
    }

    fn remove_due_task_id(&mut self, task_id: &str) -> Result<bool, Self::Error> {
        RedisTaskStore::remove_due_task_id(self, task_id)
    }

    fn save_task(
        &mut self,
        document: &TaskRecordDocument,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error> {
        RedisTaskStore::save_task(self, document, ttl_seconds)
    }

    fn acquire_owner(&mut self, token: &str, ttl_seconds: i64) -> Result<bool, Self::Error> {
        if self.acquire_lease(TASK_SCHEDULER_OWNER_KEY, token, ttl_seconds)? {
            return Ok(true);
        }
        // Several bounded workers in the same process share one owner identity.
        // They may join that lease, while a different deployment token remains
        // excluded.
        self.renew_lease(TASK_SCHEDULER_OWNER_KEY, token, ttl_seconds)
    }

    fn renew_owner(&mut self, token: &str, ttl_seconds: i64) -> Result<bool, Self::Error> {
        self.renew_lease(TASK_SCHEDULER_OWNER_KEY, token, ttl_seconds)
    }

    fn release_owner(&mut self, token: &str) -> Result<bool, Self::Error> {
        self.release_lease(TASK_SCHEDULER_OWNER_KEY, token)
    }

    fn claim_occurrence(
        &mut self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
        ttl_seconds: i64,
    ) -> Result<bool, Self::Error> {
        RedisTaskStore::claim_occurrence(self, task_id, execution_id, claim_token, ttl_seconds)
    }

    fn release_occurrence(
        &mut self,
        task_id: &str,
        execution_id: &str,
        claim_token: &str,
    ) -> Result<bool, Self::Error> {
        RedisTaskStore::release_occurrence(self, task_id, execution_id, claim_token)
    }

    fn complete_occurrence(
        &mut self,
        completion: &SchedulerCompletion<'_>,
    ) -> Result<bool, Self::Error> {
        let next_payload = completion
            .next_document
            .map(encode_task_record)
            .transpose()?;
        let next_run_score = completion
            .next_document
            .and_then(|document| document.task.next_run_at)
            .map_or(0.0, |timestamp| timestamp as f64);
        RedisTaskStore::complete_occurrence(
            self,
            &TaskOccurrenceCompletion {
                task_id: completion.task_id,
                chat_id: completion.chat_id,
                execution_id: completion.execution_id,
                claim_token: completion.claim_token,
                next_payload: next_payload.as_deref(),
                next_run_score,
                ttl_seconds: completion.record_ttl_seconds,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskObservation {
    Wait,
    Execute { execution_id: String },
    Skip { execution_id: String },
    AlreadyCompleted { execution_id: String },
    Missing,
    ClaimedElsewhere { execution_id: String },
    Executed { execution_id: String },
    RetryRequested { execution_id: String },
    Advanced { execution_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedTask {
    pub task_id: String,
    pub observation: TaskObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskFailure {
    pub task_id: String,
    pub stage: &'static str,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerStep {
    NotOwner,
    Observed {
        tasks: Vec<ObservedTask>,
        failures: Vec<TaskFailure>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchedulerError {
    #[error("scheduler settings must be positive")]
    InvalidSettings,
    #[error("scheduler owner token must not be empty")]
    EmptyOwnerToken,
    #[error("scheduler store failed: {0}")]
    Store(String),
}

pub struct TaskScheduler<Store, Executor> {
    store: Store,
    executor: Executor,
    mode: SchedulerMode,
    settings: SchedulerSettings,
    owner_token: String,
    claim_token: String,
    owns_lease: bool,
}

impl<Store, Executor> TaskScheduler<Store, Executor>
where
    Store: SchedulerStore,
    Executor: ScheduledTaskExecutor,
{
    pub fn new(
        store: Store,
        executor: Executor,
        mode: SchedulerMode,
        settings: SchedulerSettings,
        owner_token: impl Into<String>,
    ) -> Result<Self, SchedulerError> {
        let owner_token = owner_token.into();
        if owner_token.is_empty() {
            return Err(SchedulerError::EmptyOwnerToken);
        }
        Ok(Self {
            store,
            executor,
            mode,
            settings: settings.validate()?,
            owner_token,
            claim_token: String::new(),
            owns_lease: false,
        }
        .with_default_claim_token())
    }

    fn with_default_claim_token(mut self) -> Self {
        self.claim_token.clone_from(&self.owner_token);
        self
    }

    #[must_use]
    pub fn with_claim_token(mut self, claim_token: impl Into<String>) -> Self {
        self.claim_token = claim_token.into();
        self
    }

    fn store_error(error: impl Display) -> SchedulerError {
        SchedulerError::Store(error.to_string())
    }

    fn ensure_owner(&mut self) -> Result<bool, SchedulerError> {
        if self.owns_lease
            && self
                .store
                .renew_owner(&self.owner_token, self.settings.owner_ttl_seconds)
                .map_err(Self::store_error)?
        {
            return Ok(true);
        }
        self.owns_lease = self
            .store
            .acquire_owner(&self.owner_token, self.settings.owner_ttl_seconds)
            .map_err(Self::store_error)?;
        Ok(self.owns_lease)
    }

    fn observation_for(document: &TaskRecordDocument, now: i64) -> Result<TaskObservation, String> {
        match evaluate_due(&document.task, now).map_err(|error| error.to_string())? {
            DueDecision::Wait => Ok(TaskObservation::Wait),
            DueDecision::Skip { .. } => {
                let scheduled_for = document
                    .task
                    .next_run_at
                    .ok_or_else(|| "due task has no next-run timestamp".to_owned())?;
                Ok(TaskObservation::Skip {
                    execution_id: format!("{}:{scheduled_for}", document.task.id.as_str()),
                })
            }
            DueDecision::Execute { execution_id, .. } => {
                if document.task.last_execution_id.as_deref() == Some(&execution_id) {
                    Ok(TaskObservation::AlreadyCompleted { execution_id })
                } else {
                    Ok(TaskObservation::Execute { execution_id })
                }
            }
        }
    }

    fn next_document(
        document: &TaskRecordDocument,
        now: i64,
        execution_id: &str,
    ) -> Result<Option<TaskRecordDocument>, String> {
        let decision = evaluate_due(&document.task, now).map_err(|error| error.to_string())?;
        let next_run_at = match decision {
            DueDecision::Wait => return Ok(Some(document.clone())),
            DueDecision::Skip { next_run_at, .. } | DueDecision::Execute { next_run_at, .. } => {
                next_run_at
            }
        };
        let Some(next_run_at) = next_run_at else {
            return Ok(None);
        };
        let mut next = document.clone();
        next.task.next_run_at = Some(next_run_at);
        next.task.last_execution_id = Some(execution_id.to_owned());
        Ok(Some(next))
    }

    pub fn step(&mut self, now: i64) -> Result<SchedulerStep, SchedulerError> {
        if self.mode == SchedulerMode::Authoritative && !self.ensure_owner()? {
            return Ok(SchedulerStep::NotOwner);
        }
        let task_ids = self
            .store
            .due_task_ids(now, self.settings.batch_limit)
            .map_err(Self::store_error)?;
        let mut tasks = Vec::with_capacity(task_ids.len());
        let mut failures = Vec::new();

        for task_id in task_ids {
            let document = match self.store.load_task(&task_id) {
                Ok(Some(document)) => document,
                Ok(None) => {
                    if self.mode == SchedulerMode::Authoritative
                        && let Err(error) = self.store.remove_due_task_id(&task_id)
                    {
                        failures.push(TaskFailure {
                            task_id: task_id.clone(),
                            stage: "remove_stale_index",
                            error: error.to_string(),
                        });
                    }
                    tasks.push(ObservedTask {
                        task_id,
                        observation: TaskObservation::Missing,
                    });
                    continue;
                }
                Err(error) => {
                    failures.push(TaskFailure {
                        task_id,
                        stage: "load",
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            let observation = match Self::observation_for(&document, now) {
                Ok(observation) => observation,
                Err(error) => {
                    failures.push(TaskFailure {
                        task_id,
                        stage: "evaluate",
                        error,
                    });
                    continue;
                }
            };
            if self.mode == SchedulerMode::Verify {
                tasks.push(ObservedTask {
                    task_id,
                    observation,
                });
                continue;
            }
            if observation == TaskObservation::Wait {
                if let Err(error) = self
                    .store
                    .save_task(&document, self.settings.record_ttl_seconds)
                {
                    failures.push(TaskFailure {
                        task_id,
                        stage: "repair_due_index",
                        error: error.to_string(),
                    });
                    continue;
                }
                tasks.push(ObservedTask {
                    task_id,
                    observation,
                });
                continue;
            }

            let execution_id = match &observation {
                TaskObservation::Execute { execution_id }
                | TaskObservation::Skip { execution_id }
                | TaskObservation::AlreadyCompleted { execution_id } => execution_id.clone(),
                TaskObservation::Wait
                | TaskObservation::Missing
                | TaskObservation::ClaimedElsewhere { .. }
                | TaskObservation::Executed { .. }
                | TaskObservation::RetryRequested { .. }
                | TaskObservation::Advanced { .. } => continue,
            };
            let claimed = match self.store.claim_occurrence(
                document.task.id.as_str(),
                &execution_id,
                &self.claim_token,
                self.settings.claim_ttl_seconds,
            ) {
                Ok(claimed) => claimed,
                Err(error) => {
                    failures.push(TaskFailure {
                        task_id,
                        stage: "claim",
                        error: error.to_string(),
                    });
                    continue;
                }
            };
            if !claimed {
                tasks.push(ObservedTask {
                    task_id,
                    observation: TaskObservation::ClaimedElsewhere { execution_id },
                });
                continue;
            }

            if matches!(observation, TaskObservation::Execute { .. }) {
                match self.executor.execute(&document.task, &execution_id) {
                    Ok(TaskExecutionDisposition::Complete) => {}
                    Ok(TaskExecutionDisposition::Retry) => {
                        if let Err(error) = self.store.release_occurrence(
                            document.task.id.as_str(),
                            &execution_id,
                            &self.claim_token,
                        ) {
                            failures.push(TaskFailure {
                                task_id,
                                stage: "release_retry_claim",
                                error: error.to_string(),
                            });
                            continue;
                        }
                        tasks.push(ObservedTask {
                            task_id,
                            observation: TaskObservation::RetryRequested { execution_id },
                        });
                        continue;
                    }
                    Err(error) => {
                        let release_error = self
                            .store
                            .release_occurrence(
                                document.task.id.as_str(),
                                &execution_id,
                                &self.claim_token,
                            )
                            .err()
                            .map(|release| format!("; claim release failed: {release}"))
                            .unwrap_or_default();
                        failures.push(TaskFailure {
                            task_id,
                            stage: "execute",
                            error: format!("{error}{release_error}"),
                        });
                        continue;
                    }
                }
            }

            let next_document = match Self::next_document(&document, now, &execution_id) {
                Ok(next_document) => next_document,
                Err(error) => {
                    failures.push(TaskFailure {
                        task_id,
                        stage: "advance",
                        error,
                    });
                    continue;
                }
            };
            match self.store.complete_occurrence(&SchedulerCompletion {
                task_id: document.task.id.as_str(),
                chat_id: &document.task.chat_id,
                execution_id: &execution_id,
                claim_token: &self.claim_token,
                next_document: next_document.as_ref(),
                record_ttl_seconds: self.settings.record_ttl_seconds,
            }) {
                Ok(true) => tasks.push(ObservedTask {
                    task_id,
                    observation: if matches!(observation, TaskObservation::Execute { .. }) {
                        TaskObservation::Executed { execution_id }
                    } else {
                        TaskObservation::Advanced { execution_id }
                    },
                }),
                Ok(false) => tasks.push(ObservedTask {
                    task_id,
                    observation: TaskObservation::ClaimedElsewhere { execution_id },
                }),
                Err(error) => failures.push(TaskFailure {
                    task_id,
                    stage: "complete",
                    error: error.to_string(),
                }),
            }
        }
        Ok(SchedulerStep::Observed { tasks, failures })
    }

    pub fn shutdown(&mut self) -> Result<bool, SchedulerError> {
        if !self.owns_lease {
            return Ok(false);
        }
        let released = self
            .store
            .release_owner(&self.owner_token)
            .map_err(Self::store_error)?;
        self.owns_lease = false;
        Ok(released)
    }

    #[cfg(test)]
    fn parts(&self) -> (&Store, &Executor) {
        (&self.store, &self.executor)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::collections::{HashMap, VecDeque};

    use bot_adapters::task_record::TaskRecordDocument;
    use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};

    use super::{
        ObservedTask, ScheduledTaskExecutor, SchedulerCompletion, SchedulerError, SchedulerMode,
        SchedulerSettings, SchedulerStep, SchedulerStore, TaskExecutionDisposition,
        TaskObservation, TaskScheduler,
    };

    #[derive(Clone, Debug, PartialEq)]
    struct Completed {
        task_id: String,
        execution_id: String,
        next: Option<TaskRecordDocument>,
    }

    struct Store {
        due: Vec<String>,
        documents: HashMap<String, TaskRecordDocument>,
        owner_available: bool,
        renew_available: bool,
        claim_available: bool,
        complete_available: bool,
        acquired: usize,
        renewed: usize,
        owner_releases: usize,
        occurrence_releases: usize,
        stale_removed: Vec<String>,
        saved: Vec<TaskRecordDocument>,
        completed: Vec<Completed>,
        fail_due: bool,
        fail_load: bool,
        fail_complete: bool,
        fail_owner: bool,
        fail_remove: bool,
        fail_save: bool,
        fail_claim: bool,
        fail_release_occurrence: bool,
        fail_release_owner: bool,
    }

    impl Default for Store {
        fn default() -> Self {
            Self {
                due: Vec::new(),
                documents: HashMap::new(),
                owner_available: true,
                renew_available: true,
                claim_available: true,
                complete_available: true,
                acquired: 0,
                renewed: 0,
                owner_releases: 0,
                occurrence_releases: 0,
                stale_removed: Vec::new(),
                saved: Vec::new(),
                completed: Vec::new(),
                fail_due: false,
                fail_load: false,
                fail_complete: false,
                fail_owner: false,
                fail_remove: false,
                fail_save: false,
                fail_claim: false,
                fail_release_occurrence: false,
                fail_release_owner: false,
            }
        }
    }

    impl SchedulerStore for Store {
        type Error = &'static str;

        fn due_task_ids(&mut self, _now: i64, _limit: usize) -> Result<Vec<String>, Self::Error> {
            if self.fail_due {
                Err("due failed")
            } else {
                Ok(self.due.clone())
            }
        }

        fn load_task(&mut self, task_id: &str) -> Result<Option<TaskRecordDocument>, Self::Error> {
            if self.fail_load {
                Err("load failed")
            } else {
                Ok(self.documents.get(task_id).cloned())
            }
        }

        fn remove_due_task_id(&mut self, task_id: &str) -> Result<bool, Self::Error> {
            if self.fail_remove {
                return Err("remove failed");
            }
            self.stale_removed.push(task_id.to_owned());
            self.due.retain(|candidate| candidate != task_id);
            Ok(true)
        }

        fn save_task(
            &mut self,
            document: &TaskRecordDocument,
            _ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            if self.fail_save {
                return Err("save failed");
            }
            self.saved.push(document.clone());
            Ok(true)
        }

        fn acquire_owner(&mut self, _token: &str, _ttl_seconds: i64) -> Result<bool, Self::Error> {
            if self.fail_owner {
                return Err("owner failed");
            }
            self.acquired += 1;
            Ok(self.owner_available)
        }

        fn renew_owner(&mut self, _token: &str, _ttl_seconds: i64) -> Result<bool, Self::Error> {
            self.renewed += 1;
            Ok(self.renew_available)
        }

        fn release_owner(&mut self, _token: &str) -> Result<bool, Self::Error> {
            if self.fail_release_owner {
                return Err("owner release failed");
            }
            self.owner_releases += 1;
            Ok(true)
        }

        fn claim_occurrence(
            &mut self,
            _task_id: &str,
            _execution_id: &str,
            _claim_token: &str,
            _ttl_seconds: i64,
        ) -> Result<bool, Self::Error> {
            if self.fail_claim {
                return Err("claim failed");
            }
            Ok(self.claim_available)
        }

        fn release_occurrence(
            &mut self,
            _task_id: &str,
            _execution_id: &str,
            _claim_token: &str,
        ) -> Result<bool, Self::Error> {
            if self.fail_release_occurrence {
                return Err("occurrence release failed");
            }
            self.occurrence_releases += 1;
            Ok(true)
        }

        fn complete_occurrence(
            &mut self,
            completion: &SchedulerCompletion<'_>,
        ) -> Result<bool, Self::Error> {
            if self.fail_complete {
                return Err("complete failed");
            }
            self.completed.push(Completed {
                task_id: completion.task_id.to_owned(),
                execution_id: completion.execution_id.to_owned(),
                next: completion.next_document.cloned(),
            });
            Ok(self.complete_available)
        }
    }

    #[derive(Default)]
    struct Executor {
        calls: Vec<(String, String)>,
        outcomes: VecDeque<Result<TaskExecutionDisposition, &'static str>>,
    }

    impl ScheduledTaskExecutor for Executor {
        type Error = &'static str;

        fn execute(
            &mut self,
            task: &ScheduledTask,
            execution_id: &str,
        ) -> Result<TaskExecutionDisposition, Self::Error> {
            self.calls
                .push((task.id.as_str().to_owned(), execution_id.to_owned()));
            self.outcomes
                .pop_front()
                .unwrap_or(Ok(TaskExecutionDisposition::Complete))
        }
    }

    fn document(id: &str, schedule: TaskSchedule, next_run_at: i64) -> TaskRecordDocument {
        TaskRecordDocument {
            task: ScheduledTask {
                id: TaskId::new(id).unwrap_or_else(|error| panic!("synthetic id: {error}")),
                chat_id: "-100123".to_owned(),
                text: "synthetic task".to_owned(),
                user_name: "synthetic-user".to_owned(),
                user_id: Some(42),
                schedule,
                timezone_offset: -3,
                locale: "es".to_owned(),
                schedule_anchor_at: Some(100),
                next_run_at: Some(next_run_at),
                last_execution_id: None,
            },
            run_date: None,
            extra: Default::default(),
        }
    }

    fn scheduler(
        store: Store,
        executor: Executor,
        mode: SchedulerMode,
    ) -> TaskScheduler<Store, Executor> {
        TaskScheduler::new(
            store,
            executor,
            mode,
            SchedulerSettings::default(),
            "owner-1",
        )
        .unwrap_or_else(|error| panic!("synthetic scheduler: {error}"))
    }

    #[test]
    fn validates_settings_and_owner_identity() {
        assert!(matches!(
            TaskScheduler::new(
                Store::default(),
                Executor::default(),
                SchedulerMode::Verify,
                SchedulerSettings {
                    batch_limit: 0,
                    ..SchedulerSettings::default()
                },
                "owner"
            ),
            Err(SchedulerError::InvalidSettings)
        ));
        assert!(matches!(
            TaskScheduler::new(
                Store::default(),
                Executor::default(),
                SchedulerMode::Verify,
                SchedulerSettings::default(),
                ""
            ),
            Err(SchedulerError::EmptyOwnerToken)
        ));
    }

    #[test]
    fn verify_mode_observes_without_ownership_claims_or_execution() {
        let recurring = document(
            "verify1",
            TaskSchedule::IntervalSeconds { seconds: 600 },
            1_000,
        );
        let mut store = Store {
            due: vec!["verify1".to_owned()],
            ..Store::default()
        };
        store.documents.insert("verify1".to_owned(), recurring);
        let mut scheduler = scheduler(store, Executor::default(), SchedulerMode::Verify);
        assert_eq!(
            scheduler.step(1_100),
            Ok(SchedulerStep::Observed {
                tasks: vec![ObservedTask {
                    task_id: "verify1".to_owned(),
                    observation: TaskObservation::Execute {
                        execution_id: "verify1:1000".to_owned(),
                    },
                }],
                failures: Vec::new(),
            })
        );
        let (store, executor) = scheduler.parts();
        assert_eq!(store.acquired, 0);
        assert!(store.completed.is_empty());
        assert!(executor.calls.is_empty());
    }

    #[test]
    fn authoritative_execution_advances_recurring_tasks_atomically() {
        let recurring = document(
            "repeat1",
            TaskSchedule::IntervalSeconds { seconds: 600 },
            1_000,
        );
        let mut store = Store {
            due: vec!["repeat1".to_owned()],
            ..Store::default()
        };
        store.documents.insert("repeat1".to_owned(), recurring);
        let mut scheduler = scheduler(store, Executor::default(), SchedulerMode::Authoritative);
        assert!(matches!(
            scheduler.step(1_100),
            Ok(SchedulerStep::Observed { failures, .. }) if failures.is_empty()
        ));
        let (store, executor) = scheduler.parts();
        assert_eq!(store.acquired, 1);
        assert_eq!(
            executor.calls,
            [("repeat1".to_owned(), "repeat1:1000".to_owned())]
        );
        assert_eq!(store.completed.len(), 1);
        let next = store.completed[0]
            .next
            .as_ref()
            .unwrap_or_else(|| panic!("recurring task must advance"));
        assert_eq!(next.task.next_run_at, Some(1_600));
        assert_eq!(next.task.last_execution_id.as_deref(), Some("repeat1:1000"));
    }

    #[test]
    fn one_shot_and_late_occurrences_are_removed_without_duplicate_execution() {
        let one_shot = document("once1", TaskSchedule::Once, 1_000);
        let late = document(
            "late1",
            TaskSchedule::IntervalSeconds { seconds: 600 },
            1_000,
        );
        let mut store = Store {
            due: vec!["once1".to_owned(), "late1".to_owned()],
            ..Store::default()
        };
        store.documents.insert("once1".to_owned(), one_shot);
        store.documents.insert("late1".to_owned(), late);
        let mut scheduler = scheduler(store, Executor::default(), SchedulerMode::Authoritative);
        let result = scheduler.step(2_000);
        assert!(matches!(
            result,
            Ok(SchedulerStep::Observed { failures, .. }) if failures.is_empty()
        ));
        let (store, executor) = scheduler.parts();
        assert!(executor.calls.is_empty());
        assert_eq!(store.completed.len(), 2);
        assert!(store.completed[0].next.is_none());
        assert_eq!(
            store.completed[1]
                .next
                .as_ref()
                .and_then(|next| next.task.next_run_at),
            Some(2_200)
        );
    }

    #[test]
    fn unavailable_owner_or_occurrence_claim_prevents_side_effects() {
        let record = document("claimed1", TaskSchedule::Once, 1_000);
        let mut owner_store = Store {
            owner_available: false,
            due: vec!["claimed1".to_owned()],
            ..Store::default()
        };
        owner_store
            .documents
            .insert("claimed1".to_owned(), record.clone());
        let mut owner = scheduler(
            owner_store,
            Executor::default(),
            SchedulerMode::Authoritative,
        );
        assert_eq!(owner.step(1_000), Ok(SchedulerStep::NotOwner));

        let mut claim_store = Store {
            claim_available: false,
            due: vec!["claimed1".to_owned()],
            ..Store::default()
        };
        claim_store.documents.insert("claimed1".to_owned(), record);
        let mut claim = scheduler(
            claim_store,
            Executor::default(),
            SchedulerMode::Authoritative,
        );
        assert!(matches!(
            claim.step(1_000),
            Ok(SchedulerStep::Observed { tasks, .. })
                if matches!(tasks[0].observation, TaskObservation::ClaimedElsewhere { .. })
        ));
        assert!(claim.parts().1.calls.is_empty());
    }

    #[test]
    fn retry_and_executor_failure_release_the_occurrence_claim() {
        let retry = document("retry1", TaskSchedule::Once, 1_000);
        let failed = document("failed1", TaskSchedule::Once, 1_000);
        let mut store = Store {
            due: vec!["retry1".to_owned(), "failed1".to_owned()],
            ..Store::default()
        };
        store.documents.insert("retry1".to_owned(), retry);
        store.documents.insert("failed1".to_owned(), failed);
        let executor = Executor {
            outcomes: VecDeque::from([
                Ok(TaskExecutionDisposition::Retry),
                Err("synthetic execution failure"),
            ]),
            ..Executor::default()
        };
        let mut scheduler = scheduler(store, executor, SchedulerMode::Authoritative);
        let result = scheduler.step(1_000);
        assert!(matches!(
            result,
            Ok(SchedulerStep::Observed { tasks, failures })
                if matches!(tasks[0].observation, TaskObservation::RetryRequested { .. })
                    && failures.len() == 1
                    && failures[0].stage == "execute"
        ));
        let (store, _) = scheduler.parts();
        assert_eq!(store.occurrence_releases, 2);
        assert!(store.completed.is_empty());
    }

    #[test]
    fn repairs_missing_and_future_index_entries_and_renews_then_releases_ownership() {
        let future = document(
            "future1",
            TaskSchedule::IntervalSeconds { seconds: 600 },
            2_000,
        );
        let mut store = Store {
            due: vec!["missing1".to_owned(), "future1".to_owned()],
            ..Store::default()
        };
        store.documents.insert("future1".to_owned(), future);
        let mut scheduler = scheduler(store, Executor::default(), SchedulerMode::Authoritative);
        assert!(scheduler.step(1_000).is_ok());
        assert!(scheduler.step(1_000).is_ok());
        assert_eq!(scheduler.shutdown(), Ok(true));
        assert_eq!(scheduler.shutdown(), Ok(false));
        let (store, _) = scheduler.parts();
        assert_eq!(store.stale_removed, ["missing1"]);
        assert_eq!(store.saved.len(), 2);
        assert_eq!(store.acquired, 1);
        assert_eq!(store.renewed, 1);
        assert_eq!(store.owner_releases, 1);
    }

    #[test]
    fn reports_store_and_per_task_failures_without_claiming_success() {
        let mut due_failure = scheduler(
            Store {
                fail_due: true,
                ..Store::default()
            },
            Executor::default(),
            SchedulerMode::Verify,
        );
        assert_eq!(
            due_failure.step(1_000),
            Err(SchedulerError::Store("due failed".to_owned()))
        );

        let mut load_failure = scheduler(
            Store {
                due: vec!["broken1".to_owned()],
                fail_load: true,
                ..Store::default()
            },
            Executor::default(),
            SchedulerMode::Verify,
        );
        assert!(matches!(
            load_failure.step(1_000),
            Ok(SchedulerStep::Observed { failures, .. })
                if failures.len() == 1 && failures[0].stage == "load"
        ));

        let record = document("complete1", TaskSchedule::Once, 1_000);
        let mut store = Store {
            due: vec!["complete1".to_owned()],
            fail_complete: true,
            ..Store::default()
        };
        store.documents.insert("complete1".to_owned(), record);
        let mut completion_failure =
            scheduler(store, Executor::default(), SchedulerMode::Authoritative);
        assert!(matches!(
            completion_failure.step(1_000),
            Ok(SchedulerStep::Observed { failures, .. })
                if failures.len() == 1 && failures[0].stage == "complete"
        ));

        let invalid = document(
            "invalid1",
            TaskSchedule::IntervalSeconds { seconds: 0 },
            1_000,
        );
        let mut invalid_store = Store {
            due: vec!["invalid1".to_owned()],
            ..Store::default()
        };
        invalid_store
            .documents
            .insert("invalid1".to_owned(), invalid);
        let mut invalid_scheduler =
            scheduler(invalid_store, Executor::default(), SchedulerMode::Verify);
        let invalid_result = invalid_scheduler.step(1_000);
        assert!(
            matches!(
                &invalid_result,
                Ok(SchedulerStep::Observed { failures, .. })
                    if failures.len() == 1 && failures[0].stage == "evaluate"
            ),
            "unexpected invalid-task result: {invalid_result:?}"
        );

        let record = document("lost1", TaskSchedule::Once, 1_000);
        let mut unavailable_store = Store {
            due: vec!["lost1".to_owned()],
            complete_available: false,
            ..Store::default()
        };
        unavailable_store
            .documents
            .insert("lost1".to_owned(), record);
        let mut unavailable = scheduler(
            unavailable_store,
            Executor::default(),
            SchedulerMode::Authoritative,
        );
        assert!(matches!(
            unavailable.step(1_000),
            Ok(SchedulerStep::Observed { tasks, failures })
                if failures.is_empty()
                    && matches!(tasks[0].observation, TaskObservation::ClaimedElsewhere { .. })
        ));
    }

    #[test]
    fn reports_each_owner_index_claim_and_release_failure_boundary() {
        let mut owner = scheduler(
            Store {
                fail_owner: true,
                ..Store::default()
            },
            Executor::default(),
            SchedulerMode::Authoritative,
        );
        assert_eq!(
            owner.step(1_000),
            Err(SchedulerError::Store("owner failed".to_owned()))
        );

        let scenarios = [
            ("missing1", "remove_stale_index", "remove"),
            ("future1", "repair_due_index", "save"),
            ("execute1", "claim", "claim"),
        ];
        for (task_id, expected_stage, failure) in scenarios {
            let mut store = Store {
                due: vec![task_id.to_owned()],
                fail_remove: failure == "remove",
                fail_save: failure == "save",
                fail_claim: failure == "claim",
                ..Store::default()
            };
            if failure != "remove" {
                let next_run_at = if failure == "save" { 2_000 } else { 1_000 };
                store.documents.insert(
                    task_id.to_owned(),
                    document(task_id, TaskSchedule::Once, next_run_at),
                );
            }
            let mut scheduler = scheduler(store, Executor::default(), SchedulerMode::Authoritative);
            assert!(matches!(
                scheduler.step(1_000),
                Ok(SchedulerStep::Observed { failures, .. })
                    if failures.len() == 1 && failures[0].stage == expected_stage
            ));
        }

        for outcome in [
            Ok(TaskExecutionDisposition::Retry),
            Err("synthetic execution failure"),
        ] {
            let mut store = Store {
                due: vec!["release1".to_owned()],
                fail_release_occurrence: true,
                ..Store::default()
            };
            store.documents.insert(
                "release1".to_owned(),
                document("release1", TaskSchedule::Once, 1_000),
            );
            let executor = Executor {
                outcomes: VecDeque::from([outcome]),
                ..Executor::default()
            };
            let mut scheduler = scheduler(store, executor, SchedulerMode::Authoritative);
            assert!(matches!(
                scheduler.step(1_000),
                Ok(SchedulerStep::Observed { failures, .. }) if failures.len() == 1
            ));
        }

        let mut release = scheduler(
            Store {
                fail_release_owner: true,
                ..Store::default()
            },
            Executor::default(),
            SchedulerMode::Authoritative,
        );
        assert!(release.step(1_000).is_ok());
        assert_eq!(
            release.shutdown(),
            Err(SchedulerError::Store("owner release failed".to_owned()))
        );
    }
}
