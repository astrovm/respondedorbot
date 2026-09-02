use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::billing_schema::BillingSchemaRepository;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_creditless_cap::{RedisCreditlessCap, creditless_cap_key};
use botd::compaction_scheduler::PayerSource;
use botd::conversation::{
    ConversationBilling, ProviderSegmentRequest, ReserveDenial, ReserveRequest, SettlementRequest,
};
use botd::conversation_adapters::PostgresConversationBilling;
use botd::native_ai::TaskCreditStore;
use botd::reconciliation::ActiveOperationRegistry;
use serde_json::{Map, json};

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

fn reserve_request(
    user_id: i64,
    chat_id: i64,
    operation_id: &str,
    reservation: &str,
    amount: i64,
    limit: i64,
) -> ReserveRequest {
    ReserveRequest {
        user_id,
        chat_id: Some(chat_id),
        operation_id: operation_id.to_owned(),
        reservation_id: format!("{operation_id}:{reservation}"),
        amount,
        creditless_user_hourly_limit: limit,
        metadata: Map::from_iter([
            ("operation_id".to_owned(), json!(operation_id)),
            ("origin_chat_id".to_owned(), json!(chat_id)),
            ("usage_tag".to_owned(), json!(reservation)),
        ]),
    }
}

#[test]
fn postgres_and_redis_enforce_onboarding_replay_cap_and_refund_policy() -> Result<(), Box<dyn Error>>
{
    let Some(database_url) = std::env::var("TEST_DATABASE_URL").ok() else {
        return Ok(());
    };
    let Some(endpoint) = redis_endpoint() else {
        return Ok(());
    };
    BillingSchemaRepository::new(&database_url).ensure_schema()?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let suffix = i64::try_from(nonce % 100_000_000)?;
    let base_operation = format!("integration-ai-1:{nonce}");
    let blocked_operation = format!("integration-ai-2:{nonce}");
    let refund_operation = format!("integration-ai-refund:{nonce}");
    let abort_operation = format!("integration-ai-abort:{nonce}");
    let crash_operation = format!("integration-ai-crash-window:{nonce}");
    let user_id = 8_100_000_000_000_i64 + suffix;
    let chat_id = -8_200_000_000_000_i64 - suffix;
    let repository = BillingRepository::new(&database_url);
    assert_eq!(repository.mint_user_credits(user_id, 1_000, None)?, 1_000);
    assert!(
        repository
            .transfer_user_to_chat(user_id, chat_id, 1_000)?
            .transferred
    );

    let task_user_id = user_id + 500_000_000;
    let task_operation = format!("integration-task:{nonce}");
    assert_eq!(repository.mint_user_credits(task_user_id, 100, None)?, 100);
    let task_metadata = Map::from_iter([
        ("operation_id".to_owned(), json!(task_operation)),
        ("usage_tag".to_owned(), json!("scheduled_task")),
    ]);
    let charged = TaskCreditStore::charge(
        &repository,
        task_user_id,
        20,
        &task_metadata,
        &format!("{task_operation}:reserve"),
        &task_operation,
    )?;
    assert!(charged.ok);
    assert!(TaskCreditStore::record_segment(
        &repository,
        task_user_id,
        &json!({
            "operation_id": task_operation,
            "segment_id": format!("{task_operation}:segment"),
            "segment": {
                "kind": "chat",
                "model": "unknown/synthetic-model",
                "usage": {"cost": "0.000001"},
                "metadata": {"provider": "openrouter"}
            }
        }),
    )?);
    assert_eq!(
        TaskCreditStore::list_segments(&repository, task_user_id, &task_operation)?.len(),
        1
    );
    assert!(TaskCreditStore::settle_once(
        &repository,
        task_user_id,
        &task_operation,
        5,
        &Map::from_iter([("reason".to_owned(), json!("synthetic_task"))]),
    )?);

    let cap_reader = RedisCreditlessCap::new(&endpoint)?;
    let cap_key = creditless_cap_key(&chat_id.to_string(), user_id);
    let active = ActiveOperationRegistry::default();
    let mut billing = PostgresConversationBilling::new(&database_url)
        .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?)
        .with_active_operations(active.clone());
    let base = reserve_request(user_id, chat_id, &base_operation, "base", 400, 1);
    let admitted = billing.reserve(base.clone())?;
    assert!(admitted.authorized);
    assert!(matches!(admitted.user_balance, 0 | 300));
    assert_eq!(admitted.chat_balance, 600);
    assert_eq!(admitted.source, Some(PayerSource::Chat));
    assert_eq!(admitted.denial, None);
    assert!(active.is_active(&base_operation));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));

    billing.record_segment(ProviderSegmentRequest {
        user_id,
        chat_id: Some(chat_id),
        operation_id: base_operation.clone(),
        segment_id: format!("{base_operation}:segment"),
        segment: json!({
            "kind": "chat",
            "model": "unknown/synthetic-model",
            "usage": {"cost": "0.000001"},
            "metadata": {"provider": "openrouter"}
        }),
    })?;
    assert!(matches!(billing.personal_balance(user_id)?, Some(0 | 300)));

    let replay = billing.reserve(base)?;
    assert!(replay.authorized);
    assert_eq!(replay.source, Some(PayerSource::Chat));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));

    // A new runtime replays the durable PostgreSQL reservation through Redis.
    // The operation marker returns the original admission without incrementing.
    let mut replay_billing = PostgresConversationBilling::new(&database_url)
        .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?);
    let replay = replay_billing.reserve(reserve_request(
        user_id,
        chat_id,
        &base_operation,
        "base",
        400,
        1,
    ))?;
    assert!(replay.authorized);
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));
    let extension = reserve_request(
        user_id,
        chat_id,
        &base_operation,
        "context-extension",
        100,
        1,
    );
    assert!(billing.reserve(extension)?.authorized);
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));
    billing.settle(SettlementRequest {
        user_id,
        chat_id: Some(chat_id),
        operation_id: base_operation.clone(),
        actual_credit_units: 500,
        delivered: true,
        reason: "integration_success".to_owned(),
        billing_segments: vec![json!({
            "kind": "chat",
            "model": "unknown/synthetic-model",
            "usage": {"cost": "0.000001"},
            "metadata": {"provider": "openrouter"}
        })],
    })?;
    assert!(!active.is_active(&base_operation));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));

    let blocked = billing.reserve(reserve_request(
        user_id,
        chat_id,
        &blocked_operation,
        "base",
        400,
        1,
    ))?;
    assert_eq!(
        blocked.denial,
        Some(ReserveDenial::CreditlessHourlyCap { limit: 1 })
    );
    assert!(!blocked.authorized);
    assert!(!active.is_active(&blocked_operation));
    assert_eq!(blocked.chat_balance, 500);
    assert_eq!(cap_reader.count(&cap_key)?, Some(2));

    let refund_user_id = user_id + 100_000_000;
    let refund_chat_id = chat_id - 100_000_000;
    assert_eq!(
        repository.mint_user_credits(refund_user_id, 1_000, None)?,
        1_000
    );
    assert!(
        repository
            .transfer_user_to_chat(refund_user_id, refund_chat_id, 1_000)?
            .transferred
    );
    let refund_key = creditless_cap_key(&refund_chat_id.to_string(), refund_user_id);
    assert!(
        billing
            .reserve(reserve_request(
                refund_user_id,
                refund_chat_id,
                &refund_operation,
                "base",
                400,
                1,
            ))?
            .authorized
    );
    assert_eq!(cap_reader.count(&refund_key)?, Some(1));
    billing.settle(SettlementRequest {
        user_id: refund_user_id,
        chat_id: Some(refund_chat_id),
        operation_id: refund_operation.clone(),
        actual_credit_units: 0,
        delivered: false,
        reason: "integration_refund".to_owned(),
        billing_segments: Vec::new(),
    })?;
    assert_eq!(cap_reader.count(&refund_key)?, Some(0));

    billing.release_operation("synthetic-operation-without-reservation");
    assert_eq!(repository.get_balance("chat", refund_chat_id)?, 1_000);

    assert!(
        billing
            .reserve(reserve_request(
                refund_user_id,
                refund_chat_id,
                &abort_operation,
                "base",
                400,
                1,
            ))?
            .authorized
    );
    assert_eq!(cap_reader.count(&refund_key)?, Some(1));
    assert!(active.is_active(&abort_operation));
    billing.abort_operation(&abort_operation)?;
    assert_eq!(cap_reader.count(&refund_key)?, Some(0));
    assert!(!active.is_active(&abort_operation));

    // An aborted operation can reacquire exactly one slot on replay, then
    // refund that new admission without decrementing twice.
    assert!(
        billing
            .reserve(reserve_request(
                refund_user_id,
                refund_chat_id,
                &abort_operation,
                "base",
                400,
                1,
            ))?
            .authorized
    );
    assert_eq!(cap_reader.count(&refund_key)?, Some(1));
    billing.abort_operation(&abort_operation)?;
    assert_eq!(cap_reader.count(&refund_key)?, Some(0));

    // Simulate a process stopping after PostgreSQL commits the reservation but
    // before Redis admits it. The replacement runtime must consume the cap.
    let crash_user_id = refund_user_id + 100_000_000;
    let crash_chat_id = refund_chat_id - 100_000_000;
    assert_eq!(
        repository.mint_user_credits(crash_user_id, 1_000, None)?,
        1_000
    );
    assert!(
        repository
            .transfer_user_to_chat(crash_user_id, crash_chat_id, 1_000)?
            .transferred
    );
    let crash_key = creditless_cap_key(&crash_chat_id.to_string(), crash_user_id);
    let mut database_only = PostgresConversationBilling::new(&database_url);
    assert!(
        database_only
            .reserve(reserve_request(
                crash_user_id,
                crash_chat_id,
                &crash_operation,
                "base",
                400,
                1,
            ))?
            .authorized
    );
    assert_eq!(cap_reader.count(&crash_key)?, None);
    let mut replacement = PostgresConversationBilling::new(&database_url)
        .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?);
    assert!(
        replacement
            .reserve(reserve_request(
                crash_user_id,
                crash_chat_id,
                &crash_operation,
                "base",
                400,
                1,
            ))?
            .authorized
    );
    assert_eq!(cap_reader.count(&crash_key)?, Some(1));
    replacement.abort_operation(&crash_operation)?;
    assert_eq!(cap_reader.count(&crash_key)?, Some(0));
    Ok(())
}
