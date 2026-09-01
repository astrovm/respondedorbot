use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use bot_adapters::billing_read::BillingRepository;
use bot_adapters::billing_schema::BillingSchemaRepository;
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_creditless_cap::{RedisCreditlessCap, creditless_cap_key};
use botd::compaction_scheduler::PayerSource;
use botd::conversation::{ConversationBilling, ReserveDenial, ReserveRequest, SettlementRequest};
use botd::conversation_adapters::PostgresConversationBilling;
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
    let suffix =
        i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() % 100_000_000)?;
    let user_id = 8_100_000_000_000_i64 + suffix;
    let chat_id = -8_200_000_000_000_i64 - suffix;
    let repository = BillingRepository::new(&database_url);
    assert_eq!(repository.mint_user_credits(user_id, 1_000, None)?, 1_000);
    assert!(
        repository
            .transfer_user_to_chat(user_id, chat_id, 1_000)?
            .transferred
    );

    let cap_reader = RedisCreditlessCap::new(&endpoint)?;
    let cap_key = creditless_cap_key(&chat_id.to_string(), user_id);
    let active = ActiveOperationRegistry::default();
    let mut billing = PostgresConversationBilling::new(&database_url)
        .with_creditless_cap(RedisCreditlessCap::new(&endpoint)?)
        .with_active_operations(active.clone());
    let base = reserve_request(user_id, chat_id, "integration-ai-1", "base", 400, 1);
    let admitted = billing.reserve(base.clone())?;
    assert!(admitted.authorized);
    assert!(matches!(admitted.user_balance, 0 | 300));
    assert_eq!(admitted.chat_balance, 600);
    assert_eq!(admitted.source, Some(PayerSource::Chat));
    assert_eq!(admitted.denial, None);
    assert!(active.is_active("integration-ai-1"));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));

    let replay = billing.reserve(base)?;
    assert!(replay.authorized);
    assert_eq!(replay.source, Some(PayerSource::Chat));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));
    let extension = reserve_request(
        user_id,
        chat_id,
        "integration-ai-1",
        "context-extension",
        100,
        1,
    );
    assert!(billing.reserve(extension)?.authorized);
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));
    billing.settle(SettlementRequest {
        user_id,
        chat_id: Some(chat_id),
        operation_id: "integration-ai-1".to_owned(),
        actual_credit_units: 500,
        delivered: true,
        reason: "integration_success".to_owned(),
        billing_segments: Vec::new(),
    })?;
    assert!(!active.is_active("integration-ai-1"));
    assert_eq!(cap_reader.count(&cap_key)?, Some(1));

    let blocked = billing.reserve(reserve_request(
        user_id,
        chat_id,
        "integration-ai-2",
        "base",
        400,
        1,
    ))?;
    assert_eq!(
        blocked.denial,
        Some(ReserveDenial::CreditlessHourlyCap { limit: 1 })
    );
    assert!(!blocked.authorized);
    assert!(!active.is_active("integration-ai-2"));
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
    let refund_operation = "integration-ai-refund";
    assert!(
        billing
            .reserve(reserve_request(
                refund_user_id,
                refund_chat_id,
                refund_operation,
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
        operation_id: refund_operation.to_owned(),
        actual_credit_units: 0,
        delivered: false,
        reason: "integration_refund".to_owned(),
        billing_segments: Vec::new(),
    })?;
    assert_eq!(cap_reader.count(&refund_key)?, Some(0));
    assert_eq!(repository.get_balance("chat", refund_chat_id)?, 1_000);
    Ok(())
}
