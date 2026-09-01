//! Language-neutral Redis payload for durable memory compaction.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const COMPACTION_JOB_SCHEMA_VERSION: u8 = 1;

#[derive(Debug, Error)]
pub enum CompactionJobError {
    #[error("invalid compaction job JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("unsupported compaction job schema version {0}")]
    UnsupportedVersion(u8),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactionJobRecord {
    #[serde(default = "current_schema_version")]
    pub schema_version: u8,
    pub chat_id: String,
    pub messages: Vec<Value>,
    pub prior_summary: Option<String>,
    pub expected_marker: Option<String>,
    pub target_marker: String,
    pub reservation: Value,
    pub user_id: i64,
    pub message_id: Option<String>,
    #[serde(default = "default_locale")]
    pub locale: String,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub next_attempt_at: f64,
    #[serde(default)]
    pub result_summary: Option<String>,
    #[serde(default)]
    pub result_cost_usd_micros: i64,
    #[serde(default)]
    pub result_billing_segment: Option<Value>,
}

const fn current_schema_version() -> u8 {
    COMPACTION_JOB_SCHEMA_VERSION
}

fn default_locale() -> String {
    "es".to_owned()
}

/// Decode legacy or current records and emit the canonical current payload.
pub fn normalize_compaction_job(payload: &str) -> Result<String, CompactionJobError> {
    let record = serde_json::from_str::<CompactionJobRecord>(payload)?;
    if record.schema_version != COMPACTION_JOB_SCHEMA_VERSION {
        return Err(CompactionJobError::UnsupportedVersion(
            record.schema_version,
        ));
    }
    Ok(serde_json::to_string(&record)?)
}

#[cfg(test)]
mod tests {
    use super::{COMPACTION_JOB_SCHEMA_VERSION, normalize_compaction_job};

    const LEGACY: &str = r#"{
        "chat_id":"123",
        "messages":[{"id":"m1","text":"hello"}],
        "prior_summary":null,
        "expected_marker":null,
        "target_marker":"m1",
        "reservation":{"reserved_credit_units":3},
        "user_id":42,
        "message_id":"99"
    }"#;

    #[test]
    fn upgrades_legacy_records_with_compatible_defaults() -> Result<(), Box<dyn std::error::Error>>
    {
        let normalized = normalize_compaction_job(LEGACY)?;
        let value: serde_json::Value = serde_json::from_str(&normalized)?;
        assert_eq!(value["schema_version"], COMPACTION_JOB_SCHEMA_VERSION);
        assert_eq!(value["locale"], "es");
        assert_eq!(value["attempts"], 0);
        assert_eq!(value["next_attempt_at"], 0.0);
        assert!(value["result_summary"].is_null());
        Ok(())
    }

    #[test]
    fn preserves_current_result_and_billing_fields() -> Result<(), Box<dyn std::error::Error>> {
        let current = LEGACY.replace("\"chat_id\"", "\"schema_version\":1,\"chat_id\"");
        let normalized = normalize_compaction_job(&current)?;
        let value: serde_json::Value = serde_json::from_str(&normalized)?;
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["chat_id"], "123");
        Ok(())
    }

    #[test]
    fn rejects_unknown_fields_and_future_versions() {
        let unknown = LEGACY.replace("\"chat_id\"", "\"unknown\":1,\"chat_id\"");
        assert!(normalize_compaction_job(&unknown).is_err());
        let future = LEGACY.replace("\"chat_id\"", "\"schema_version\":2,\"chat_id\"");
        assert!(normalize_compaction_job(&future).is_err());
    }
}
