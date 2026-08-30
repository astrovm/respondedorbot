//! Native plans and replies for privileged billing commands.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::admin_reports::{CreditLogLimit, parse_creditlog_limit, truncate_report};
use crate::ai_pricing::model_cache_input_rates;
use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, format_credit_units, parse_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::{SendMessage, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrintCreditsPlan {
    NotHandled,
    Reply(TelegramAction),
    Mint { user_id: i64, amount: i64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreditLogEntry {
    pub user_id: Option<i64>,
    pub chat_id: Option<i64>,
    pub metadata: Value,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreditLogPlan {
    NotHandled,
    LegacyRequired,
    Reply(TelegramAction),
    Load { limit: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintCreditsContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: i64,
    pub admin_user_id: Option<i64>,
    pub billing_available: bool,
    pub locale: Locale,
}

fn reply(chat_id: ChatId, message_id: MessageId, text: &str) -> PrintCreditsPlan {
    let mut message = SendMessage::new(chat_id, text);
    message.reply_to_message_id = Some(message_id);
    PrintCreditsPlan::Reply(TelegramAction::SendMessage(message))
}

#[must_use]
pub fn plan_printcredits_command(
    message_text: &str,
    bot_name: &str,
    context: PrintCreditsContext,
) -> PrintCreditsPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/printcredits" {
        return PrintCreditsPlan::NotHandled;
    }
    if context.admin_user_id != Some(context.user_id) {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "este comando es solo para el admin",
                Locale::En => "this command is only for the admin",
            },
        );
    }
    if !context.billing_available {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "los créditos IA no están disponibles ahora",
                Locale::En => "AI credits are not available right now",
            },
        );
    }
    let amount_token = parsed
        .message_text
        .split_once(' ')
        .map_or(parsed.message_text.as_str(), |(token, _)| token)
        .trim();
    let Some(amount) = parse_credit_units(amount_token) else {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "mandalo bien: /printcredits <monto>",
                Locale::En => "usage: /printcredits <amount>",
            },
        );
    };
    if amount.value() <= 0 {
        return reply(
            context.chat_id,
            context.message_id,
            match context.locale {
                Locale::Es => "el monto tiene que ser mayor a 0",
                Locale::En => "the amount must be greater than 0",
            },
        );
    }
    PrintCreditsPlan::Mint {
        user_id: context.user_id,
        amount: amount.value(),
    }
}

#[must_use]
pub fn printcredits_result_reply(amount: i64, balance: i64, locale: Locale) -> String {
    let amount = format_credit_units(CreditUnits::new(amount));
    let balance = format_credit_units(CreditUnits::new(balance));
    match locale {
        Locale::Es => format!("listo, te imprimí {amount} créditos\nte quedaron {balance}"),
        Locale::En => format!("minted {amount} credits\nyour balance is {balance}"),
    }
}

#[must_use]
pub fn plan_creditlog_command(
    message_text: &str,
    bot_name: &str,
    context: PrintCreditsContext,
) -> CreditLogPlan {
    let parsed = parse_command(message_text, bot_name);
    if parsed.command != "/creditlog" {
        return CreditLogPlan::NotHandled;
    }
    let reply = |text: &str| match reply(context.chat_id, context.message_id, text) {
        PrintCreditsPlan::Reply(action) => CreditLogPlan::Reply(action),
        PrintCreditsPlan::NotHandled | PrintCreditsPlan::Mint { .. } => CreditLogPlan::NotHandled,
    };
    if context.admin_user_id != Some(context.user_id) {
        return reply(match context.locale {
            Locale::Es => "este comando es solo para el admin",
            Locale::En => "this command is only for the admin",
        });
    }
    if !context.billing_available {
        return reply(match context.locale {
            Locale::Es => "los créditos IA no están disponibles ahora",
            Locale::En => "AI credits are not available right now",
        });
    }
    match parse_creditlog_limit(&parsed.message_text) {
        CreditLogLimit::Valid(limit) => CreditLogPlan::Load { limit },
        CreditLogLimit::Invalid => reply(match context.locale {
            Locale::Es => "mandalo bien: /creditlog [límite]",
            Locale::En => "usage: /creditlog [limit]",
        }),
        CreditLogLimit::NeedsLegacyParser => CreditLogPlan::LegacyRequired,
    }
}

fn value_i64(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(value)) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| value.as_f64().map(|value| value as i64))
            .unwrap_or(0),
        Some(Value::String(value)) => value.parse().unwrap_or(0),
        Some(Value::Bool(value)) => i64::from(*value),
        Some(Value::Null) | Some(Value::Array(_)) | Some(Value::Object(_)) | None => 0,
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value_i64(Some(&Value::Number(value.clone()))) != 0,
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn python_string(value: Option<&Value>, default: &str) -> String {
    match value {
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(Value::Array(value)) if !value.is_empty() => Value::Array(value.clone()).to_string(),
        Some(Value::Object(value)) if !value.is_empty() => Value::Object(value.clone()).to_string(),
        Some(Value::Null)
        | Some(Value::String(_))
        | Some(Value::Array(_))
        | Some(Value::Object(_))
        | None => default.to_owned(),
    }
}

fn objects(value: Option<&Value>) -> impl Iterator<Item = &Map<String, Value>> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
}

fn with_hidden_count(summary: String, total: usize, visible: usize, locale: Locale) -> String {
    let hidden = total.saturating_sub(visible);
    if hidden == 0 {
        summary
    } else {
        match locale {
            Locale::Es => format!("{summary}, +{hidden} más"),
            Locale::En => format!("{summary}, +{hidden} more"),
        }
    }
}

fn summarize_models(value: Option<&Value>, locale: Locale) -> String {
    let mut totals = BTreeMap::<String, i64>::new();
    for item in objects(value) {
        let name = python_string(item.get("model"), "?");
        *totals.entry(name).or_default() += value_i64(item.get("usd_micros"));
    }
    if totals.is_empty() {
        return match locale {
            Locale::Es => "sin modelos".to_owned(),
            Locale::En => "no models".to_owned(),
        };
    }
    let mut ordered = totals.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let visible = ordered.len().min(5);
    let summary = ordered[..visible]
        .iter()
        .map(|(name, usd)| format!("{name}={usd}"))
        .collect::<Vec<_>>()
        .join(", ");
    with_hidden_count(summary, ordered.len(), visible, locale)
}

fn summarize_tools(value: Option<&Value>, locale: Locale) -> String {
    let mut totals = BTreeMap::<String, (i64, i64)>::new();
    for item in objects(value) {
        let name = python_string(item.get("tool"), "?");
        let current = totals.entry(name).or_default();
        current.0 += value_i64(item.get("usd_micros"));
        current.1 += value_i64(item.get("count"));
    }
    if totals.is_empty() {
        return match locale {
            Locale::Es => "sin tools".to_owned(),
            Locale::En => "no tools".to_owned(),
        };
    }
    let mut ordered = totals.into_iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .1
            .0
            .cmp(&left.1.0)
            .then_with(|| right.1.1.cmp(&left.1.1))
            .then_with(|| left.0.cmp(&right.0))
    });
    let visible = ordered.len().min(5);
    let summary = ordered[..visible]
        .iter()
        .map(|(name, (usd, count))| format!("{name}={usd} ({count}x)"))
        .collect::<Vec<_>>()
        .join(", ");
    with_hidden_count(summary, ordered.len(), visible, locale)
}

fn summarize_segments(value: Option<&Value>, cache_only: bool) -> Option<String> {
    let mut totals = BTreeMap::<String, usize>::new();
    for item in objects(value) {
        let is_cache = item
            .get("source")
            .and_then(Value::as_str)
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("cache"));
        if is_cache != cache_only {
            continue;
        }
        let kind = python_string(item.get("kind"), "unknown");
        *totals.entry(kind).or_default() += 1;
    }
    (!totals.is_empty()).then(|| {
        totals
            .into_iter()
            .map(|(kind, count)| format!("{kind}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    })
}

fn summarize_model_cache(value: Option<&Value>, locale: Locale) -> Option<String> {
    let mut cached_tokens_total = 0_i64;
    let mut savings_total = 0_i128;
    for item in objects(value) {
        let cached_tokens = value_i64(item.get("input_cached_tokens"));
        if cached_tokens <= 0 {
            continue;
        }
        let model = item
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (input_price, cached_price) = model_cache_input_rates(model).unwrap_or((0, 0));
        cached_tokens_total = cached_tokens_total.saturating_add(cached_tokens);
        if input_price > cached_price {
            savings_total +=
                i128::from(cached_tokens) * i128::from(input_price - cached_price) / 1_000_000;
        }
    }
    (cached_tokens_total > 0).then(|| match locale {
        Locale::Es => format!("cacheados={cached_tokens_total} ahorro_cache={savings_total}"),
        Locale::En => format!("cached_tokens={cached_tokens_total} cache_savings={savings_total}"),
    })
}

fn metadata_credit(metadata: &Map<String, Value>, keys: &[&str]) -> i64 {
    keys.iter()
        .find_map(|key| truthy(metadata.get(*key)).then(|| value_i64(metadata.get(*key))))
        .unwrap_or(0)
}

fn optional_id(metadata: &Map<String, Value>, key: &str, fallback: Option<i64>) -> String {
    metadata.get(key).map_or_else(
        || fallback.map_or_else(|| "None".to_owned(), |value| value.to_string()),
        |value| match value {
            Value::Null => "None".to_owned(),
            _ => python_string(Some(value), "None"),
        },
    )
}

fn format_creditlog_entry(entry: &CreditLogEntry, locale: Locale) -> String {
    let empty = Map::new();
    let metadata = entry.metadata.as_object().unwrap_or(&empty);
    let models = metadata.get("model_breakdown");
    let tools = metadata.get("tool_breakdown");
    let segments = metadata.get("billing_segments");
    let created = if entry.created_at.is_empty() {
        match locale {
            Locale::Es => "sin fecha".to_owned(),
            Locale::En => "no date".to_owned(),
        }
    } else {
        entry
            .created_at
            .replace('T', " ")
            .chars()
            .take(19)
            .collect()
    };
    let command = ["command", "usage_tag"]
        .into_iter()
        .find_map(|key| truthy(metadata.get(key)).then(|| python_string(metadata.get(key), "")))
        .unwrap_or_else(|| match locale {
            Locale::Es => "sin comando".to_owned(),
            Locale::En => "no command".to_owned(),
        });
    let status = if truthy(metadata.get("billing_zero_usage_fallback")) {
        "groq_zero_usage"
    } else if truthy(metadata.get("missing_usage_billing")) {
        "missing_usage"
    } else {
        "ok"
    };
    let labels = match locale {
        Locale::Es => (
            "estado",
            "reservado",
            "cobrado",
            "refund",
            "extra",
            "deuda",
            "requests",
            "sin segmentos",
            "modelos",
            "tools",
        ),
        Locale::En => (
            "status",
            "reserved",
            "charged",
            "refund",
            "extra",
            "debt",
            "requests",
            "no segments",
            "models",
            "tools",
        ),
    };
    let credit =
        |keys: &[&str]| format_credit_units(CreditUnits::new(metadata_credit(metadata, keys)));
    let mut lines = vec![
        format!("{created} | cmd={command} | {}={status}", labels.0),
        format!(
            "chat={} user={} {}={} {}={} {}={} {}={} {}={}",
            optional_id(metadata, "chat_id", entry.chat_id),
            optional_id(metadata, "user_id", entry.user_id),
            labels.1,
            credit(&[
                "reserved_credit_units_total",
                "reserved_credit_units",
                "reserved_credits_total",
                "reserved_credits"
            ]),
            labels.2,
            credit(&["settled_credit_units", "settled_credits"]),
            labels.3,
            credit(&["refunded_credit_units", "refunded_credits"]),
            labels.4,
            credit(&["extra_charged_credit_units", "extra_charged_credits"]),
            labels.5,
            credit(&["debt_applied_credit_units", "debt_applied_credits"]),
        ),
        format!("usd_micros={}", value_i64(metadata.get("raw_usd_micros"))),
        format!(
            "{}: {}",
            labels.6,
            summarize_segments(segments, false).unwrap_or_else(|| labels.7.to_owned())
        ),
    ];
    if let Some(cache_hits) = summarize_segments(segments, true) {
        lines.push(format!("cache_hits: {cache_hits}"));
    }
    if let Some(cache) = summarize_model_cache(models, locale) {
        lines.push(cache);
    }
    lines.push(format!(
        "{}: {}",
        labels.8,
        summarize_models(models, locale)
    ));
    lines.push(format!("{}: {}", labels.9, summarize_tools(tools, locale)));
    lines.join("\n")
}

#[must_use]
pub fn render_creditlog(entries: &[CreditLogEntry], locale: Locale) -> String {
    let title = match locale {
        Locale::Es => "últimas liquidaciones IA:",
        Locale::En => "latest AI settlements:",
    };
    let text = std::iter::once(title.to_owned())
        .chain(
            entries
                .iter()
                .map(|entry| format_creditlog_entry(entry, locale)),
        )
        .collect::<Vec<_>>()
        .join("\n\n");
    truncate_report(
        &text,
        3500,
        match locale {
            Locale::Es => "truncado",
            Locale::En => "truncated",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        CreditLogEntry, CreditLogPlan, PrintCreditsContext, PrintCreditsPlan,
        format_creditlog_entry, plan_creditlog_command, plan_printcredits_command,
        printcredits_result_reply, python_string, render_creditlog, summarize_model_cache,
        summarize_models, summarize_segments, summarize_tools, truthy, value_i64,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};
    use serde_json::json;

    fn reply_text(plan: PrintCreditsPlan) -> Option<String> {
        match plan {
            PrintCreditsPlan::Reply(TelegramAction::SendMessage(message)) => Some(message.text),
            PrintCreditsPlan::NotHandled
            | PrintCreditsPlan::Mint { .. }
            | PrintCreditsPlan::Reply(_) => None,
        }
    }

    fn plan(text: &str, admin: Option<i64>, billing: bool, locale: Locale) -> PrintCreditsPlan {
        plan_printcredits_command(
            text,
            "@bot",
            PrintCreditsContext {
                chat_id: ChatId(202),
                message_id: MessageId(12),
                user_id: 99,
                admin_user_id: admin,
                billing_available: billing,
                locale,
            },
        )
    }

    #[test]
    fn preserves_authorization_billing_and_input_guard_order() {
        assert_eq!(
            reply_text(plan("/printcredits bad", None, false, Locale::Es)),
            Some("este comando es solo para el admin".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits bad", Some(99), false, Locale::En)),
            Some("AI credits are not available right now".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits bad", Some(99), true, Locale::Es)),
            Some("mandalo bien: /printcredits <monto>".to_owned())
        );
        assert_eq!(
            reply_text(plan("/printcredits -1", Some(99), true, Locale::En)),
            Some("the amount must be greater than 0".to_owned())
        );
    }

    #[test]
    fn parses_exact_credit_units_and_formats_bilingual_success() {
        assert_eq!(
            plan(
                "/printcredits@bot 100.0 ignored",
                Some(99),
                true,
                Locale::Es
            ),
            PrintCreditsPlan::Mint {
                user_id: 99,
                amount: 10_000,
            }
        );
        assert_eq!(
            printcredits_result_reply(10_000, 12_000, Locale::Es),
            "listo, te imprimí 100.00 créditos\nte quedaron 120.00"
        );
        assert_eq!(
            printcredits_result_reply(10_000, 12_000, Locale::En),
            "minted 100.00 credits\nyour balance is 120.00"
        );
        assert_eq!(
            plan("/other 1", Some(99), true, Locale::Es),
            PrintCreditsPlan::NotHandled
        );
    }

    #[test]
    fn creditlog_plan_preserves_guards_limits_and_legacy_boundaries() {
        let context = |admin_user_id, billing_available, locale| PrintCreditsContext {
            chat_id: ChatId(202),
            message_id: MessageId(12),
            user_id: 99,
            admin_user_id,
            billing_available,
            locale,
        };
        assert!(matches!(
            plan_creditlog_command("/other", "@bot", context(Some(99), true, Locale::Es)),
            CreditLogPlan::NotHandled
        ));
        assert!(matches!(
            plan_creditlog_command("/creditlog", "@bot", context(None, false, Locale::Es)),
            CreditLogPlan::Reply(TelegramAction::SendMessage(message))
                if message.text == "este comando es solo para el admin"
        ));
        assert!(matches!(
            plan_creditlog_command(
                "/creditlog bad",
                "@bot",
                context(Some(99), false, Locale::En)
            ),
            CreditLogPlan::Reply(TelegramAction::SendMessage(message))
                if message.text == "AI credits are not available right now"
        ));
        assert!(matches!(
            plan_creditlog_command(
                "/creditlog bad",
                "@bot",
                context(Some(99), true, Locale::En)
            ),
            CreditLogPlan::Reply(TelegramAction::SendMessage(message))
                if message.text == "usage: /creditlog [limit]"
        ));
        assert_eq!(
            plan_creditlog_command("/creditlog ２", "@bot", context(Some(99), true, Locale::Es)),
            CreditLogPlan::LegacyRequired
        );
        assert_eq!(
            plan_creditlog_command(
                "/creditlog@bot 200",
                "@bot",
                context(Some(99), true, Locale::Es)
            ),
            CreditLogPlan::Load { limit: 25 }
        );
    }

    #[test]
    fn renders_detailed_spanish_creditlog_with_python_parity() {
        let entry = CreditLogEntry {
            user_id: Some(99),
            chat_id: Some(202),
            created_at: "2026-03-11T17:35:10+00:00".to_owned(),
            metadata: json!({
                "command": "/ask",
                "reserved_credit_units_total": 200,
                "settled_credit_units": 100,
                "refunded_credit_units": 100,
                "extra_charged_credit_units": 0,
                "raw_usd_micros": 390,
                "model_breakdown": [
                    {"model":"deepseek/deepseek-v4-flash-0731","usd_micros":325,"input_cached_tokens":800},
                    {"model":"deepseek/deepseek-v4-flash-0731","usd_micros":65,"input_cached_tokens":100}
                ],
                "tool_breakdown": [
                    {"tool":"web_search","usd_micros":8000,"count":2},
                    {"tool":"python","usd_micros":500,"count":1}
                ],
                "billing_segments": [
                    {"kind":"chat"},{"kind":"chat"},{"kind":"chat"},
                    {"kind":"chat","source":" cache "}
                ]
            }),
        };
        assert_eq!(
            render_creditlog(&[entry], Locale::Es),
            "últimas liquidaciones IA:\n\n2026-03-11 17:35:10 | cmd=/ask | estado=ok\nchat=202 user=99 reservado=2.00 cobrado=1.00 refund=1.00 extra=0.00 deuda=0.00\nusd_micros=390\nrequests: chat=3\ncache_hits: chat=1\ncacheados=900 ahorro_cache=20\nmodelos: deepseek/deepseek-v4-flash-0731=390\ntools: web_search=8000 (2x), python=500 (1x)"
        );
    }

    #[test]
    fn renders_english_defaults_status_sorting_hidden_counts_and_truncation() {
        let models = (0..7)
            .map(|index| json!({"model":format!("m{index}"),"usd_micros":index,"input_cached_tokens":1}))
            .collect::<Vec<_>>();
        let tools = (0..7)
            .map(|index| json!({"tool":format!("t{index}"),"usd_micros":index,"count":7-index}))
            .collect::<Vec<_>>();
        let entry = CreditLogEntry {
            user_id: None,
            chat_id: None,
            created_at: String::new(),
            metadata: json!({
                "usage_tag":"legacy",
                "missing_usage_billing":true,
                "reserved_credits":"2",
                "settled_credits":true,
                "chat_id":null,
                "model_breakdown":models,
                "tool_breakdown":tools,
                "billing_segments":[]
            }),
        };
        let text = format_creditlog_entry(&entry, Locale::En);
        assert!(text.starts_with("no date | cmd=legacy | status=missing_usage"));
        assert!(text.contains("chat=None user=None reserved=0.02 charged=0.01"));
        assert!(text.contains("requests: no segments"));
        assert!(text.contains("models: m6=6, m5=5, m4=4, m3=3, m2=2, +2 more"));
        assert!(
            text.contains("tools: t6=6 (1x), t5=5 (2x), t4=4 (3x), t3=3 (4x), t2=2 (5x), +2 more")
        );

        let huge = CreditLogEntry {
            metadata: json!({"command":"x".repeat(4000),"billing_zero_usage_fallback":true}),
            ..entry
        };
        let truncated = render_creditlog(&[huge], Locale::En);
        assert_eq!(truncated.chars().count(), 3500);
        assert!(truncated.ends_with("\n\n[truncated]"));
    }

    #[test]
    fn formatter_helpers_cover_typed_and_malformed_metadata() {
        assert_eq!(value_i64(Some(&json!(2.9))), 2);
        assert_eq!(value_i64(Some(&json!(true))), 1);
        assert_eq!(value_i64(Some(&json!("bad"))), 0);
        assert_eq!(value_i64(Some(&json!([]))), 0);
        assert!(truthy(Some(&json!([1]))));
        assert!(!truthy(Some(&json!({}))));
        assert_eq!(python_string(Some(&json!(false)), "x"), "False");
        assert_eq!(python_string(Some(&json!([1])), "x"), "[1]");
        assert_eq!(python_string(Some(&json!({"a":1})), "x"), "{\"a\":1}");
        assert_eq!(
            summarize_models(Some(&json!("bad")), Locale::Es),
            "sin modelos"
        );
        assert_eq!(summarize_tools(None, Locale::En), "no tools");
        assert_eq!(
            summarize_segments(Some(&json!([null, {"kind":null}])), false),
            Some("unknown=1".to_owned())
        );
        assert_eq!(
            summarize_model_cache(
                Some(&json!([{"model":"unknown","input_cached_tokens":5}])),
                Locale::En
            ),
            Some("cached_tokens=5 cache_savings=0".to_owned())
        );
    }
}
