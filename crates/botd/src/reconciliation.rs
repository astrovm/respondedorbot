//! Recovery and exact-once settlement for interrupted AI operations.

use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

use bot_adapters::billing_read::{BillingRepository, UnsettledAiOperation};
use bot_adapters::openrouter_generation::{
    GenerationOutcome, GenerationTransport, ReqwestGenerationTransport, fetch_with,
};
use bot_core::ai_pricing::calculate_billing_for_segments;
use bot_core::ai_usage::{
    ProviderUsageStatus, needs_reconciliation, provider_reported_cost_is_positive,
};
use chrono::DateTime;
use serde_json::{Map, Value, json};

const DEFAULT_BATCH_LIMIT: i64 = 500;
const DEFAULT_RETRY_WINDOW_SECONDS: i64 = 3_600;
const DEFAULT_SAFETY_CREDIT_UNITS: i64 = 10;
const DEFAULT_STALE_SECONDS: i64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconciliationSettings {
    pub batch_limit: i64,
    pub retry_window_seconds: i64,
    pub safety_credit_units: i64,
    pub stale_seconds: i64,
}

impl Default for ReconciliationSettings {
    fn default() -> Self {
        Self {
            batch_limit: DEFAULT_BATCH_LIMIT,
            retry_window_seconds: DEFAULT_RETRY_WINDOW_SECONDS,
            safety_credit_units: DEFAULT_SAFETY_CREDIT_UNITS,
            stale_seconds: DEFAULT_STALE_SECONDS,
        }
    }
}

impl ReconciliationSettings {
    #[must_use]
    pub fn normalized(self) -> Self {
        Self {
            batch_limit: self.batch_limit.max(1),
            retry_window_seconds: self.retry_window_seconds.max(60),
            safety_credit_units: self.safety_credit_units.max(0),
            stale_seconds: self.stale_seconds.max(30),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconciliationFailure {
    pub operation_id: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReconciliationReport {
    pub settled: usize,
    pub pending: usize,
    pub unresolved: usize,
    pub failures: Vec<ReconciliationFailure>,
}

pub trait ReconciliationStore {
    type Error: Display;

    fn list_unsettled(&mut self, limit: i64) -> Result<Vec<UnsettledAiOperation>, Self::Error>;

    fn update_provider_segment(
        &mut self,
        operation_id: &str,
        segment_id: &str,
        segment: &Value,
    ) -> Result<bool, Self::Error>;

    fn settle_operation(
        &mut self,
        operation: &UnsettledAiOperation,
        actual_credit_units: i64,
        metadata: &Map<String, Value>,
    ) -> Result<bool, Self::Error>;
}

pub trait GenerationSource {
    type Error: Display;

    fn generation(
        &mut self,
        generation_id: &str,
    ) -> Result<Option<Map<String, Value>>, Self::Error>;
}

#[derive(Debug, Clone, Default)]
pub struct ActiveOperationRegistry {
    counts: Arc<Mutex<HashMap<String, usize>>>,
}

impl ActiveOperationRegistry {
    pub fn mark_active(&self, operation_id: &str) {
        let operation_id = operation_id.trim();
        if operation_id.is_empty() {
            return;
        }
        if let Ok(mut counts) = self.counts.lock() {
            *counts.entry(operation_id.to_owned()).or_default() += 1;
        }
    }

    pub fn mark_inactive(&self, operation_id: &str) {
        let operation_id = operation_id.trim();
        if operation_id.is_empty() {
            return;
        }
        if let Ok(mut counts) = self.counts.lock() {
            match counts.get_mut(operation_id) {
                Some(count) if *count > 1 => *count -= 1,
                Some(_) => {
                    counts.remove(operation_id);
                }
                None => {}
            }
        }
    }

    #[must_use]
    pub fn is_active(&self, operation_id: &str) -> bool {
        self.counts
            .lock()
            .ok()
            .and_then(|counts| counts.get(operation_id.trim()).copied())
            .is_some_and(|count| count > 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationOutcome {
    Settled,
    Pending,
    Unresolved,
}

pub struct AiBillingReconciler<Store, Generations> {
    store: Store,
    generations: Generations,
    active: ActiveOperationRegistry,
    settings: ReconciliationSettings,
}

impl<Store, Generations> AiBillingReconciler<Store, Generations> {
    #[must_use]
    pub fn new(
        store: Store,
        generations: Generations,
        active: ActiveOperationRegistry,
        settings: ReconciliationSettings,
    ) -> Self {
        Self {
            store,
            generations,
            active,
            settings: settings.normalized(),
        }
    }

    #[must_use]
    pub fn active_operations(&self) -> ActiveOperationRegistry {
        self.active.clone()
    }

    #[must_use]
    pub fn into_parts(self) -> (Store, Generations, ActiveOperationRegistry) {
        (self.store, self.generations, self.active)
    }
}

impl<Store, Generations> AiBillingReconciler<Store, Generations>
where
    Store: ReconciliationStore,
    Generations: GenerationSource,
{
    pub fn run_once(&mut self, now_epoch_seconds: i64) -> Result<ReconciliationReport, String> {
        let operations = self
            .store
            .list_unsettled(self.settings.batch_limit)
            .map_err(|error| error.to_string())?;
        let mut report = ReconciliationReport::default();
        for operation in operations {
            match self.reconcile_operation(&operation, now_epoch_seconds) {
                Ok(OperationOutcome::Settled) => report.settled += 1,
                Ok(OperationOutcome::Pending) => report.pending += 1,
                Ok(OperationOutcome::Unresolved) => report.unresolved += 1,
                Err(error) => {
                    report.pending += 1;
                    report.failures.push(ReconciliationFailure {
                        operation_id: operation.operation_id,
                        error,
                    });
                }
            }
        }
        Ok(report)
    }

    fn reconcile_operation(
        &mut self,
        operation: &UnsettledAiOperation,
        now_epoch_seconds: i64,
    ) -> Result<OperationOutcome, String> {
        if operation
            .reserve_metadata
            .get("background")
            .and_then(Value::as_bool)
            == Some(true)
        {
            return Ok(OperationOutcome::Pending);
        }
        if self.active.is_active(&operation.operation_id) {
            return Ok(OperationOutcome::Pending);
        }
        let age_seconds = age_seconds(&operation.last_activity_at, now_epoch_seconds);
        if age_seconds < self.settings.stale_seconds {
            return Ok(OperationOutcome::Pending);
        }

        let mut entries = operation.segments.clone();
        let mut segments = entries
            .iter()
            .map(|entry| entry.get("segment").cloned().unwrap_or_else(|| json!({})))
            .collect::<Vec<_>>();
        for (index, segment) in segments.clone().iter().enumerate() {
            if !segment_needs_reconciliation(segment) {
                continue;
            }
            let Some(generation_id) = segment
                .get("metadata")
                .and_then(|metadata| metadata.get("provider_generation_id"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
            else {
                continue;
            };
            let Some(generation) = self
                .generations
                .generation(generation_id)
                .map_err(|error| error.to_string())?
            else {
                continue;
            };
            let Some(reconciled) = reconciled_segment(segment, &generation) else {
                continue;
            };
            let segment_id = entries[index]
                .get("segment_id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if segment_id.is_empty() {
                continue;
            }
            self.store
                .update_provider_segment(&operation.operation_id, segment_id, &reconciled)
                .map_err(|error| error.to_string())?;
            entries[index]["segment"] = reconciled.clone();
            segments[index] = reconciled;
        }

        let still_pending = segments.iter().any(segment_needs_reconciliation);
        let expired = age_seconds >= self.settings.retry_window_seconds;
        if still_pending && !expired {
            return Ok(OperationOutcome::Pending);
        }

        let breakdown = calculate_billing_for_segments(&Value::Array(segments.clone()))
            .map_err(|error| error.to_string())?;
        let mut actual = breakdown
            .get("charged_credit_units")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let incomplete_pricing = !segments.is_empty()
            && breakdown.get("pricing_complete").and_then(Value::as_bool) != Some(true);
        if still_pending {
            actual = actual.max(
                operation.authorized_credit_units.min(
                    actual
                        .checked_add(self.settings.safety_credit_units)
                        .unwrap_or(i64::MAX),
                ),
            );
        } else if incomplete_pricing {
            actual = actual.max(operation.authorized_credit_units);
        }
        let unresolved = still_pending || incomplete_pricing;
        let reason = settlement_reason(still_pending, incomplete_pricing, !segments.is_empty());
        let mut metadata = operation
            .reserve_metadata
            .as_object()
            .cloned()
            .unwrap_or_default();
        metadata.extend(Map::from_iter([
            ("operation_id".to_owned(), json!(&operation.operation_id)),
            ("reason".to_owned(), json!(reason)),
            ("billing_segments".to_owned(), json!(segments)),
            (
                "pricing_version".to_owned(),
                breakdown
                    .get("pricing_version")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "raw_usd_micros".to_owned(),
                breakdown.get("raw_usd_micros").cloned().unwrap_or(json!(0)),
            ),
            (
                "markup_multiplier".to_owned(),
                breakdown
                    .get("markup_multiplier")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "model_breakdown".to_owned(),
                breakdown
                    .get("model_breakdown")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            ),
            (
                "tool_breakdown".to_owned(),
                breakdown
                    .get("tool_breakdown")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            ),
            (
                "segment_breakdown".to_owned(),
                breakdown
                    .get("segment_breakdown")
                    .cloned()
                    .unwrap_or_else(|| json!([])),
            ),
            (
                "pricing_complete".to_owned(),
                json!(
                    !unresolved
                        && breakdown.get("pricing_complete").and_then(Value::as_bool) == Some(true)
                ),
            ),
            ("reconciliation_unresolved".to_owned(), json!(unresolved)),
        ]));
        self.store
            .settle_operation(operation, actual, &metadata)
            .map_err(|error| error.to_string())?;
        Ok(if unresolved {
            OperationOutcome::Unresolved
        } else {
            OperationOutcome::Settled
        })
    }
}

impl ReconciliationStore for BillingRepository {
    type Error = bot_adapters::billing_read::BillingError;

    fn list_unsettled(&mut self, limit: i64) -> Result<Vec<UnsettledAiOperation>, Self::Error> {
        self.list_unsettled_ai_operations(limit)
    }

    fn update_provider_segment(
        &mut self,
        operation_id: &str,
        segment_id: &str,
        segment: &Value,
    ) -> Result<bool, Self::Error> {
        self.update_ai_provider_usage(operation_id, segment_id, segment)
    }

    fn settle_operation(
        &mut self,
        operation: &UnsettledAiOperation,
        actual_credit_units: i64,
        metadata: &Map<String, Value>,
    ) -> Result<bool, Self::Error> {
        self.settle_ai_operation_once(
            operation.user_id,
            operation.chat_id,
            &operation.operation_id,
            actual_credit_units,
            metadata,
        )
        .map(|result| result.applied)
    }
}

pub struct OpenRouterGenerationSource<Transport> {
    transport: Transport,
    api_key: String,
}

impl<Transport> OpenRouterGenerationSource<Transport> {
    #[must_use]
    pub fn new(transport: Transport, api_key: &str) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
        }
    }
}

impl<Transport: GenerationTransport> GenerationSource for OpenRouterGenerationSource<Transport> {
    type Error = bot_adapters::openrouter_generation::GenerationError;

    fn generation(
        &mut self,
        generation_id: &str,
    ) -> Result<Option<Map<String, Value>>, Self::Error> {
        fetch_with(&self.transport, &self.api_key, generation_id).map(|outcome| match outcome {
            GenerationOutcome::Pending => None,
            GenerationOutcome::Success { generation } => Some(generation),
        })
    }
}

pub type ProductionAiBillingReconciler =
    AiBillingReconciler<BillingRepository, OpenRouterGenerationSource<ReqwestGenerationTransport>>;

pub fn production_reconciler(
    database_url: &str,
    openrouter_api_key: &str,
    active: ActiveOperationRegistry,
    settings: ReconciliationSettings,
) -> Result<ProductionAiBillingReconciler, String> {
    let transport = ReqwestGenerationTransport::new().map_err(|error| error.to_string())?;
    Ok(AiBillingReconciler::new(
        BillingRepository::new(database_url),
        OpenRouterGenerationSource::new(transport, openrouter_api_key),
        active,
        settings,
    ))
}

fn segment_needs_reconciliation(segment: &Value) -> bool {
    needs_reconciliation(ProviderUsageStatus {
        source: segment
            .get("source")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        cost_is_positive: segment
            .get("usage")
            .and_then(Value::as_object)
            .is_some_and(provider_reported_cost_is_positive),
    })
}

fn positive_number(value: &Value) -> bool {
    value
        .as_f64()
        .or_else(|| value.as_str()?.parse::<f64>().ok())
        .is_some_and(|value| value > 0.0)
}

fn reconciled_segment(segment: &Value, generation: &Map<String, Value>) -> Option<Value> {
    let upstream_cost = generation.get("upstream_inference_cost")?;
    if !positive_number(upstream_cost) {
        return None;
    }
    let total_cost = generation
        .get("total_cost")
        .or_else(|| generation.get("cost"))
        .filter(|value| positive_number(value))
        .unwrap_or(upstream_cost);
    let mut reconciled = segment.as_object()?.clone();
    let mut usage = segment
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut cost_details = usage
        .get("cost_details")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    cost_details.insert("upstream_inference_cost".to_owned(), upstream_cost.clone());
    usage.insert("cost".to_owned(), total_cost.clone());
    usage.insert("cost_details".to_owned(), Value::Object(cost_details));
    for (target, generation_key) in [
        ("prompt_tokens", "tokens_prompt"),
        ("completion_tokens", "tokens_completion"),
    ] {
        let value = generation
            .get(generation_key)
            .cloned()
            .or_else(|| usage.get(target).cloned())
            .unwrap_or(json!(0));
        usage.insert(target.to_owned(), value);
    }
    reconciled.insert("usage".to_owned(), Value::Object(usage));
    if let Some(model) = generation.get("model").filter(|value| !value.is_null()) {
        reconciled.insert("model".to_owned(), model.clone());
    }
    let mut metadata = segment
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    metadata.insert("stream_interrupted".to_owned(), json!(false));
    metadata.insert("provider_usage_pending".to_owned(), json!(false));
    metadata.insert("usage_reconciled".to_owned(), json!(true));
    if let Some(provider) = generation
        .get("provider_name")
        .filter(|value| !value.is_null())
    {
        metadata.insert("upstream_provider".to_owned(), provider.clone());
    }
    reconciled.insert("metadata".to_owned(), Value::Object(metadata));
    Some(Value::Object(reconciled))
}

fn settlement_reason(
    still_pending: bool,
    incomplete_pricing: bool,
    has_segments: bool,
) -> &'static str {
    if still_pending {
        "reconciliation_timeout"
    } else if incomplete_pricing {
        "reconciliation_incomplete_pricing"
    } else if has_segments {
        "recovered_provider_usage"
    } else {
        "unused_stale_reservation"
    }
}

fn age_seconds(raw: &str, now_epoch_seconds: i64) -> i64 {
    parse_timestamp(raw)
        .map(|timestamp| now_epoch_seconds.saturating_sub(timestamp).max(0))
        .unwrap_or_default()
}

fn parse_timestamp(raw: &str) -> Option<i64> {
    DateTime::parse_from_rfc3339(raw)
        .or_else(|_| DateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f%#z"))
        .ok()
        .map(|timestamp| timestamp.timestamp())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::convert::Infallible;

    use bot_adapters::billing_read::UnsettledAiOperation;
    use serde_json::{Map, Value, json};

    use super::{
        ActiveOperationRegistry, AiBillingReconciler, GenerationSource, ReconciliationSettings,
        ReconciliationStore, age_seconds, reconciled_segment, segment_needs_reconciliation,
        settlement_reason,
    };

    #[derive(Default)]
    struct Store {
        operations: Vec<UnsettledAiOperation>,
        updates: Vec<(String, String, Value)>,
        settlements: Vec<(String, i64, Map<String, Value>)>,
    }

    impl ReconciliationStore for Store {
        type Error = Infallible;

        fn list_unsettled(
            &mut self,
            _limit: i64,
        ) -> Result<Vec<UnsettledAiOperation>, Self::Error> {
            Ok(self.operations.clone())
        }

        fn update_provider_segment(
            &mut self,
            operation_id: &str,
            segment_id: &str,
            segment: &Value,
        ) -> Result<bool, Self::Error> {
            self.updates.push((
                operation_id.to_owned(),
                segment_id.to_owned(),
                segment.clone(),
            ));
            Ok(true)
        }

        fn settle_operation(
            &mut self,
            operation: &UnsettledAiOperation,
            actual_credit_units: i64,
            metadata: &Map<String, Value>,
        ) -> Result<bool, Self::Error> {
            self.settlements.push((
                operation.operation_id.clone(),
                actual_credit_units,
                metadata.clone(),
            ));
            Ok(true)
        }
    }

    #[derive(Default)]
    struct Generations {
        values: HashMap<String, VecDeque<Option<Map<String, Value>>>>,
    }

    impl GenerationSource for Generations {
        type Error = Infallible;

        fn generation(
            &mut self,
            generation_id: &str,
        ) -> Result<Option<Map<String, Value>>, Self::Error> {
            Ok(self
                .values
                .get_mut(generation_id)
                .and_then(VecDeque::pop_front)
                .flatten())
        }
    }

    fn operation(id: &str, last_activity_at: &str, segment: Option<Value>) -> UnsettledAiOperation {
        UnsettledAiOperation {
            operation_id: id.to_owned(),
            user_id: 42,
            chat_id: Some(-100),
            authorized_credit_units: 100,
            source: "chat".to_owned(),
            created_at: last_activity_at.to_owned(),
            last_activity_at: last_activity_at.to_owned(),
            reserve_metadata: json!({"trace_id": "synthetic"}),
            segments: segment.map_or_else(Vec::new, |segment| {
                vec![json!({"segment_id": "openrouter:generation-1", "segment": segment})]
            }),
        }
    }

    fn pending_segment() -> Value {
        json!({
            "kind": "chat",
            "model": "synthetic/model",
            "source": "openrouter",
            "usage": {"cost": 0},
            "metadata": {
                "provider_generation_id": "generation-1",
                "stream_interrupted": true,
                "provider_usage_pending": true
            }
        })
    }

    #[test]
    fn positive_upstream_cost_details_do_not_need_reconciliation() {
        let mut segment = pending_segment();
        segment["usage"] = json!({
            "cost_details": {"upstream_inference_cost": "0.001"}
        });

        assert!(!segment_needs_reconciliation(&segment));
    }

    #[test]
    fn unpriced_openrouter_media_does_not_require_producer_flags() {
        let segment = json!({
            "kind": "vision",
            "model": "synthetic/model",
            "source": "openrouter",
            "usage": {"prompt_tokens": 10},
            "metadata": {"provider_generation_id": "generation-1"}
        });

        assert!(segment_needs_reconciliation(&segment));
    }

    #[test]
    fn active_and_fresh_operations_remain_pending_without_provider_io() {
        let active = ActiveOperationRegistry::default();
        active.mark_active("active");
        active.mark_active("active");
        active.mark_inactive("active");
        assert!(active.is_active("active"));
        let store = Store {
            operations: vec![
                operation("active", "2026-08-31T00:00:00Z", Some(pending_segment())),
                operation("fresh", "2026-08-31T00:09:50Z", Some(pending_segment())),
            ],
            ..Store::default()
        };
        let mut reconciler = AiBillingReconciler::new(
            store,
            Generations::default(),
            active.clone(),
            ReconciliationSettings::default(),
        );
        let report = reconciler.run_once(1_788_135_000);
        assert!(report.is_ok());
        assert_eq!(report.map(|report| report.pending), Ok(2));
        let (store, _, _) = reconciler.into_parts();
        assert!(store.updates.is_empty());
        assert!(store.settlements.is_empty());
        active.mark_inactive("active");
        assert!(!active.is_active("active"));
    }

    #[test]
    fn background_operations_are_left_for_their_own_worker() {
        let mut background = operation(
            "background",
            "2026-08-31T00:00:00Z",
            Some(pending_segment()),
        );
        background.reserve_metadata["background"] = json!(true);
        let store = Store {
            operations: vec![background],
            ..Store::default()
        };
        let mut reconciler = AiBillingReconciler::new(
            store,
            Generations::default(),
            ActiveOperationRegistry::default(),
            ReconciliationSettings::default(),
        );

        let report = reconciler.run_once(1_788_138_000).unwrap_or_default();

        assert_eq!(report.pending, 1);
        let (store, _, _) = reconciler.into_parts();
        assert!(store.updates.is_empty());
        assert!(store.settlements.is_empty());
    }

    #[test]
    fn finalized_generation_is_persisted_before_exact_priced_settlement() {
        let store = Store {
            operations: vec![operation(
                "recovered",
                "2026-08-31 00:00:00+00",
                Some(pending_segment()),
            )],
            ..Store::default()
        };
        let generations = Generations {
            values: HashMap::from([(
                "generation-1".to_owned(),
                VecDeque::from([Some(
                    json!({
                        "upstream_inference_cost": 0.00005,
                        "total_cost": 0.00006,
                        "tokens_prompt": 10,
                        "tokens_completion": 5,
                        "provider_name": "Synthetic",
                        "model": "synthetic/model"
                    })
                    .as_object()
                    .cloned()
                    .unwrap_or_default(),
                )]),
            )]),
        };
        let mut reconciler = AiBillingReconciler::new(
            store,
            generations,
            ActiveOperationRegistry::default(),
            ReconciliationSettings::default(),
        );
        let report = reconciler.run_once(1_788_134_700);
        assert!(report.is_ok());
        assert_eq!(report.map(|report| report.settled), Ok(1));
        let (store, _, _) = reconciler.into_parts();
        assert_eq!(store.updates.len(), 1);
        assert_eq!(store.settlements.len(), 1);
        assert!(store.settlements[0].1 > 0);
        assert_eq!(
            store.settlements[0].2["reason"],
            json!("recovered_provider_usage")
        );
        assert_eq!(
            store.settlements[0].2["reconciliation_unresolved"],
            json!(false)
        );
    }

    #[test]
    fn pending_usage_waits_then_expires_with_bounded_safety_charge() {
        let store = Store {
            operations: vec![operation(
                "timeout",
                "2026-08-31T00:00:00Z",
                Some(pending_segment()),
            )],
            ..Store::default()
        };
        let mut reconciler = AiBillingReconciler::new(
            store,
            Generations::default(),
            ActiveOperationRegistry::default(),
            ReconciliationSettings {
                retry_window_seconds: 3_600,
                safety_credit_units: 10,
                stale_seconds: 300,
                ..ReconciliationSettings::default()
            },
        );
        let before_expiry = reconciler.run_once(1_788_137_999);
        assert_eq!(before_expiry.map(|report| report.pending), Ok(1));
        let expired = reconciler.run_once(1_788_138_000);
        assert_eq!(expired.map(|report| report.unresolved), Ok(1));
        let (store, _, _) = reconciler.into_parts();
        assert_eq!(store.settlements.len(), 1);
        assert_eq!(store.settlements[0].1, 10);
        assert_eq!(
            store.settlements[0].2["reason"],
            json!("reconciliation_timeout")
        );
    }

    #[test]
    fn stale_empty_hold_refunds_and_invalid_time_is_treated_as_fresh() {
        let store = Store {
            operations: vec![
                operation("empty", "2026-08-31T00:00:00Z", None),
                operation("invalid", "not-a-time", None),
            ],
            ..Store::default()
        };
        let mut reconciler = AiBillingReconciler::new(
            store,
            Generations::default(),
            ActiveOperationRegistry::default(),
            ReconciliationSettings::default(),
        );
        let report = reconciler.run_once(1_788_134_700);
        assert!(report.is_ok());
        let report = report.unwrap_or_default();
        assert_eq!(report.settled, 1);
        assert_eq!(report.pending, 1);
        let (store, _, _) = reconciler.into_parts();
        assert_eq!(store.settlements[0].1, 0);
        assert_eq!(
            store.settlements[0].2["reason"],
            json!("unused_stale_reservation")
        );
    }

    #[test]
    fn helpers_match_python_reconciliation_shapes_and_time_rules() {
        let segment = pending_segment();
        assert!(segment_needs_reconciliation(&segment));
        let generation = json!({
            "upstream_inference_cost": "0.00001",
            "tokens_prompt": 7,
            "tokens_completion": 3,
            "provider_name": "Synthetic"
        })
        .as_object()
        .cloned()
        .unwrap_or_default();
        let reconciled = reconciled_segment(&segment, &generation).unwrap_or(Value::Null);
        assert_eq!(reconciled["usage"]["cost"], json!("0.00001"));
        assert_eq!(reconciled["metadata"]["usage_reconciled"], json!(true));
        assert!(!segment_needs_reconciliation(&reconciled));
        assert_eq!(
            settlement_reason(false, true, true),
            "reconciliation_incomplete_pricing"
        );
        assert_eq!(age_seconds("2026-08-31 00:00:00+00", 1_788_134_400), 0);
        assert_eq!(age_seconds("invalid", 1_788_134_400), 0);
    }
}
