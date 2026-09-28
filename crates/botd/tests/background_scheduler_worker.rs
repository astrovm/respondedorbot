//! Scheduler and reconciler outcomes crossing the background-worker boundary.

use std::collections::VecDeque;

use bot_adapters::billing_read::UnsettledAiOperation;
use bot_adapters::task_record::TaskRecordDocument;
use bot_core::scheduled_tasks::ScheduledTask;
use botd::background::BackgroundWorker;
use botd::reconciliation::{
    ActiveOperationRegistry, AiBillingReconciler, GenerationSource, ReconciliationSettings,
    ReconciliationStore,
};
use botd::scheduler::{
    ScheduledTaskExecutor, SchedulerCompletion, SchedulerMode, SchedulerSettings, SchedulerStore,
    TaskExecutionDisposition, TaskScheduler,
};
use serde_json::{Map, Value, json};

/// Owns the scheduler lease and lists scripted due tasks whose records cannot
/// be loaded; releasing the lease fails.
struct LoadFailingStore {
    due: VecDeque<Vec<String>>,
}

impl SchedulerStore for LoadFailingStore {
    type Error = String;

    fn due_task_ids(&mut self, _now: i64, _limit: usize) -> Result<Vec<String>, String> {
        Ok(self.due.pop_front().unwrap_or_default())
    }

    fn load_task(&mut self, task_id: &str) -> Result<Option<TaskRecordDocument>, String> {
        Err(format!("synthetic load failure for {task_id}"))
    }

    fn remove_due_task_id(&mut self, _task_id: &str) -> Result<bool, String> {
        Err("unexpected remove".to_owned())
    }

    fn save_task(&mut self, _document: &TaskRecordDocument, _ttl: i64) -> Result<bool, String> {
        Err("unexpected save".to_owned())
    }

    fn acquire_owner(&mut self, _token: &str, _ttl: i64) -> Result<bool, String> {
        Ok(true)
    }

    fn renew_owner(&mut self, _token: &str, _ttl: i64) -> Result<bool, String> {
        Ok(true)
    }

    fn release_owner(&mut self, _token: &str) -> Result<bool, String> {
        Err("synthetic release failure".to_owned())
    }

    fn claim_occurrence(
        &mut self,
        _task_id: &str,
        _execution_id: &str,
        _claim_token: &str,
        _ttl: i64,
    ) -> Result<bool, String> {
        Err("unexpected claim".to_owned())
    }

    fn release_occurrence(
        &mut self,
        _task_id: &str,
        _execution_id: &str,
        _claim_token: &str,
    ) -> Result<bool, String> {
        Err("unexpected occurrence release".to_owned())
    }

    fn complete_occurrence(
        &mut self,
        _completion: &SchedulerCompletion<'_>,
    ) -> Result<bool, String> {
        Err("unexpected completion".to_owned())
    }
}

struct UnusedExecutor;

impl ScheduledTaskExecutor for UnusedExecutor {
    type Error = String;

    fn execute(
        &mut self,
        _task: &ScheduledTask,
        _execution_id: &str,
    ) -> Result<TaskExecutionDisposition, String> {
        Err("unexpected execution".to_owned())
    }
}

#[test]
fn scheduler_outcomes_and_lease_release_cross_the_background_boundary() {
    let scheduler = TaskScheduler::new(
        LoadFailingStore {
            due: VecDeque::from([vec!["first".to_owned(), "second".to_owned()]]),
        },
        UnusedExecutor,
        SchedulerMode::Authoritative,
        SchedulerSettings::default(),
        "synthetic-owner",
    );
    assert!(scheduler.is_ok());
    let Ok(mut scheduler) = scheduler else {
        return;
    };
    assert_eq!(
        BackgroundWorker::run_once(&mut scheduler, 1_700_000_000),
        Err(
            "task first failed at load: synthetic load failure for first; \
             task second failed at load: synthetic load failure for second"
                .to_owned()
        )
    );
    // Nothing is due on the next tick.
    assert_eq!(
        BackgroundWorker::run_once(&mut scheduler, 1_700_000_001),
        Ok(())
    );
    assert_eq!(
        BackgroundWorker::shutdown(&mut scheduler),
        Err("scheduler store failed: synthetic release failure".to_owned())
    );
}

/// Lists scripted batches of unsettled operations; settling always fails.
struct UnsettledStore {
    batches: VecDeque<Vec<UnsettledAiOperation>>,
}

impl ReconciliationStore for UnsettledStore {
    type Error = String;

    fn list_unsettled(&mut self, _limit: i64) -> Result<Vec<UnsettledAiOperation>, String> {
        Ok(self.batches.pop_front().unwrap_or_default())
    }

    fn update_provider_segment(
        &mut self,
        _operation_id: &str,
        _segment_id: &str,
        _segment: &Value,
    ) -> Result<bool, String> {
        Err("unexpected segment update".to_owned())
    }

    fn settle_operation(
        &mut self,
        operation: &UnsettledAiOperation,
        _actual_credit_units: i64,
        _metadata: &Map<String, Value>,
    ) -> Result<bool, String> {
        Err(format!(
            "synthetic settle failure for {}",
            operation.operation_id
        ))
    }
}

struct NoGenerations;

impl GenerationSource for NoGenerations {
    type Error = String;

    fn generation(&mut self, _generation_id: &str) -> Result<Option<Map<String, Value>>, String> {
        Ok(None)
    }
}

fn stale_operation(operation_id: &str) -> UnsettledAiOperation {
    UnsettledAiOperation {
        operation_id: operation_id.to_owned(),
        user_id: 7,
        chat_id: None,
        authorized_credit_units: 10,
        source: "user".to_owned(),
        created_at: "2000-01-01T00:00:00Z".to_owned(),
        last_activity_at: "2000-01-01T00:00:00Z".to_owned(),
        reserve_metadata: json!({}),
        segments: Vec::new(),
    }
}

#[test]
fn reconciliation_outcomes_cross_the_background_boundary() {
    let mut reconciler = AiBillingReconciler::new(
        UnsettledStore {
            batches: VecDeque::from([
                Vec::new(),
                vec![stale_operation("ai:1"), stale_operation("ai:2")],
            ]),
        },
        NoGenerations,
        ActiveOperationRegistry::default(),
        ReconciliationSettings::default(),
    );
    assert_eq!(
        BackgroundWorker::run_once(&mut reconciler, 1_700_000_000),
        Ok(())
    );
    assert_eq!(
        BackgroundWorker::run_once(&mut reconciler, 1_700_000_000),
        Err("operation ai:1: synthetic settle failure for ai:1; \
             operation ai:2: synthetic settle failure for ai:2"
            .to_owned())
    );
}
