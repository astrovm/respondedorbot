use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_task_store::{
    RedisTaskStore, TaskOccurrenceCompletion, task_execution_key,
};
use bot_adapters::task_record::TaskRecordDocument;
use bot_core::scheduled_tasks::{ScheduledTask, TaskId, TaskSchedule};

const TASK_TTL_SECONDS: i64 = 600;

fn endpoint() -> Option<RedisEndpoint> {
    let port = std::env::var("TEST_REDIS_PORT").ok()?.parse().ok()?;
    Some(RedisEndpoint {
        host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
        port,
        password: std::env::var("TEST_REDIS_PASSWORD")
            .ok()
            .filter(|value| !value.is_empty()),
    })
}

#[test]
fn canonical_task_claim_advance_and_cancel_are_atomic() -> Result<(), Box<dyn Error>> {
    let Some(endpoint) = endpoint() else {
        return Ok(());
    };
    let store = RedisTaskStore::new(&endpoint)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let task_id = format!("it{nonce}");
    let chat_id = format!("integration-{nonce}");
    let owner_token = format!("owner-{nonce}");
    let owner_key = format!("task:test:scheduler:owner:{nonce}");
    let claim_token = format!("claim-{nonce}");
    let scheduled_for = 2_000_000_000_i64;
    let execution_id = format!("{task_id}:{scheduled_for}");
    let mut document = TaskRecordDocument {
        task: ScheduledTask {
            id: TaskId::new(&task_id)?,
            chat_id: chat_id.clone(),
            text: "synthetic integration reminder".to_owned(),
            user_name: "synthetic-user".to_owned(),
            user_id: Some(42),
            schedule: TaskSchedule::IntervalSeconds { seconds: 600 },
            timezone_offset: -3,
            locale: "es".to_owned(),
            schedule_anchor_at: Some(scheduled_for - 600),
            next_run_at: Some(scheduled_for),
            last_execution_id: None,
        },
        legacy_run_date: None,
        extra: Default::default(),
    };

    assert!(store.acquire_lease(&owner_key, &owner_token, 30)?);
    assert!(!store.acquire_lease(&owner_key, "other-owner", 30)?);
    assert!(store.renew_lease(&owner_key, &owner_token, 30)?);
    assert!(store.save_task(&document, TASK_TTL_SECONDS)?);
    assert!(
        store
            .due_task_ids(scheduled_for as f64, 100)?
            .contains(&task_id)
    );
    assert_eq!(
        store.load_task(&task_id)?.map(|loaded| loaded.task.text),
        Some("synthetic integration reminder".to_owned())
    );
    assert_eq!(store.list_chat_tasks(&chat_id)?.len(), 1);

    assert!(store.claim_occurrence(&task_id, &execution_id, &claim_token, 30,)?);
    assert!(!store.claim_occurrence(&task_id, &execution_id, "other-claim", 30)?);
    let execution_key = task_execution_key(&execution_id);
    assert!(store.setex(&execution_key, TASK_TTL_SECONDS, "{}")?);
    document.task.last_execution_id = Some(execution_id.clone());
    document.task.next_run_at = Some(scheduled_for + 600);
    let next_payload = bot_adapters::task_record::encode_task_record(&document)?;
    assert!(!store.complete_occurrence(&TaskOccurrenceCompletion {
        task_id: &task_id,
        chat_id: &chat_id,
        execution_id: &execution_id,
        claim_token: "wrong-claim",
        next_payload: Some(&next_payload),
        next_run_score: (scheduled_for + 600) as f64,
        ttl_seconds: TASK_TTL_SECONDS,
    })?);
    assert!(store.complete_occurrence(&TaskOccurrenceCompletion {
        task_id: &task_id,
        chat_id: &chat_id,
        execution_id: &execution_id,
        claim_token: &claim_token,
        next_payload: Some(&next_payload),
        next_run_score: (scheduled_for + 600) as f64,
        ttl_seconds: TASK_TTL_SECONDS,
    })?);
    assert_eq!(
        store
            .load_task(&task_id)?
            .and_then(|loaded| loaded.task.last_execution_id),
        Some(execution_id)
    );
    assert!(store.get(&execution_key)?.is_none());

    assert!(store.cancel_task(&task_id, &chat_id)?);
    assert!(store.load_task(&task_id)?.is_none());
    assert!(store.release_lease(&owner_key, &owner_token)?);
    Ok(())
}
