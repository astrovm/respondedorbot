//! Charge-history command planning, grouping, formatting, and pagination markup.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use serde_json::{Map, Value};

use crate::command_parsing::parse_command;
use crate::credit_units::{CreditUnits, format_credit_units};
use crate::locale::Locale;
use crate::telegram_actions::{InlineKeyboardButton, InlineKeyboardMarkup, TelegramAction};
use crate::telegram_input::{ChatId, MessageId};

const CHARGE_COMMANDS: [&str; 3] = ["/charges", "/history", "/gastos"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChargesCommandContext {
    pub chat_id: ChatId,
    pub message_id: MessageId,
    pub user_id: Option<i64>,
    pub locale: Locale,
    pub timezone_offset_hours: i64,
    pub billing_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChargesCommandPlan {
    NotHandled,
    Reply(TelegramAction),
    Load {
        user_id: i64,
        limit: usize,
        timezone_minutes: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChargeHistoryEntry {
    pub id: i64,
    pub event_type: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChargeHistoryGroup {
    pub cursor_id: i64,
    pub created_at: String,
    pub entries: Vec<ChargeHistoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChargeHistoryPage {
    pub groups: Vec<ChargeHistoryGroup>,
    pub has_newer: bool,
    pub has_older: bool,
    pub newer_cursor: Option<i64>,
    pub older_cursor: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Component {
    label: String,
    units: i64,
    pending: bool,
}

fn reply(context: ChargesCommandContext, text: &str) -> ChargesCommandPlan {
    let mut message = crate::telegram_actions::SendMessage::new(context.chat_id, text);
    message.reply_to_message_id = Some(context.message_id);
    ChargesCommandPlan::Reply(TelegramAction::SendMessage(message))
}

#[must_use]
pub fn plan_charges_command(
    message_text: &str,
    bot_name: &str,
    context: ChargesCommandContext,
) -> ChargesCommandPlan {
    let parsed = parse_command(message_text, bot_name);
    if !CHARGE_COMMANDS.contains(&parsed.command.as_str()) {
        return ChargesCommandPlan::NotHandled;
    }
    if !context.billing_available {
        return reply(
            context,
            match context.locale {
                Locale::Es => "el cobro de ia no está andando, avisale al admin",
                Locale::En => "AI billing is unavailable, please tell the admin",
            },
        );
    }
    let Some(user_id) = context.user_id else {
        return reply(
            context,
            match context.locale {
                Locale::Es => "no te pude leer el usuario para ver tu saldo",
                Locale::En => "I could not identify your user to load the balance",
            },
        );
    };

    let tokens = parsed.message_text.split_whitespace().collect::<Vec<_>>();
    let limit = match tokens.as_slice() {
        [] => 10,
        [token] => {
            let Some(value) = token
                .parse::<BigInt>()
                .ok()
                .filter(|value| value > &0.into())
            else {
                return reply(context, charges_usage(context.locale));
            };
            usize::try_from(value).unwrap_or(20).min(20)
        }
        _ => return reply(context, charges_usage(context.locale)),
    };
    ChargesCommandPlan::Load {
        user_id,
        limit,
        timezone_minutes: context.timezone_offset_hours.saturating_mul(60),
    }
}

const fn charges_usage(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => "mandalo bien: /charges [cantidad]",
        Locale::En => "usage: /charges [count]",
    }
}

fn object(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

fn integer(value: Option<&Value>) -> i64 {
    value
        .and_then(|value| {
            value
                .as_i64()
                .or_else(|| value.as_str().and_then(|raw| raw.parse().ok()))
        })
        .unwrap_or_default()
}

fn text<'a>(metadata: &'a Map<String, Value>, key: &str) -> &'a str {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn metadata_credit(metadata: &Map<String, Value>, keys: &[&str]) -> i64 {
    keys.iter()
        .find_map(|key| metadata.get(*key).map(|value| integer(Some(value))))
        .unwrap_or_default()
}

fn charged_units(metadata: &Map<String, Value>, event_type: &str) -> i64 {
    if metadata.contains_key("charged_credit_units_total") {
        return integer(metadata.get("charged_credit_units_total")).max(0);
    }
    if event_type == "memory_compaction_settlement" {
        return integer(metadata.get("actual_credit_units")).max(0);
    }
    if event_type == "ai_reserve" {
        return metadata_credit(metadata, &["reserved_credit_units", "reserved_credits"]).max(0);
    }
    let reserved = metadata_credit(
        metadata,
        &[
            "reserved_credit_units_total",
            "reserved_credit_units",
            "reserved_credits_total",
            "reserved_credits",
        ],
    );
    if reserved != 0 {
        return (reserved
            - metadata_credit(metadata, &["refunded_credit_units", "refunded_credits"])
            + metadata_credit(
                metadata,
                &["extra_charged_credit_units", "extra_charged_credits"],
            )
            + metadata_credit(
                metadata,
                &["debt_applied_credit_units", "debt_applied_credits"],
            ))
        .max(0);
    }
    metadata_credit(metadata, &["settled_credit_units", "settled_credits"]).max(0)
}

fn label(locale: Locale, kind: &str) -> &'static str {
    match (locale, kind) {
        (Locale::Es, "title") => "Gastos IA",
        (Locale::En, "title") => "AI expenses",
        (Locale::Es, "empty") => "no tenés gastos IA recientes",
        (Locale::En, "empty") => "you have no recent AI expenses",
        (Locale::Es, "previous") => "‹ Anterior",
        (Locale::En, "previous") => "‹ Previous",
        (Locale::Es, "next") => "Siguiente ›",
        (Locale::En, "next") => "Next ›",
        (Locale::Es, "pending") => "pendiente",
        (Locale::En, "pending") => "pending",
        (Locale::Es, "group") => "grupo",
        (Locale::En, "group") => "group",
        (_, "personal") => "personal",
        (Locale::Es, "response") => "respuesta",
        (Locale::En, "response") => "response",
        (_, "audio") => "audio",
        (Locale::Es, "image") => "imagen",
        (Locale::En, "image") => "image",
        (_, "web") => "web",
        (Locale::Es, "tool") => "herramienta",
        (Locale::En, "tool") => "tool",
        (Locale::Es, "memory") => "memoria",
        (Locale::En, "memory") => "memory",
        (Locale::Es, "no_date") => "sin fecha",
        (Locale::En, "no_date") => "no date",
        _ => "",
    }
}

fn raw_components(metadata: &Map<String, Value>, locale: Locale) -> Vec<(String, i64)> {
    let mut totals = Vec::<(String, i64)>::new();
    let mut add = |component_label: String, amount: i64| {
        if let Some((_, total)) = totals
            .iter_mut()
            .find(|(existing, _)| existing == &component_label)
        {
            *total = total.saturating_add(amount.max(0));
        } else {
            totals.push((component_label, amount.max(0)));
        }
    };
    for item in metadata
        .get("model_breakdown")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let kind = text(item, "kind").to_lowercase();
        let component_label = if kind == "transcribe" || integer(item.get("audio_seconds")) > 0 {
            label(locale, "audio")
        } else if kind == "vision" {
            label(locale, "image")
        } else {
            label(locale, "response")
        };
        add(component_label.to_owned(), integer(item.get("usd_micros")));
    }
    for item in metadata
        .get("tool_breakdown")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
    {
        let tool = text(item, "tool").to_lowercase();
        let count = integer(item.get("count")).max(0);
        let component_label = if tool == "web_search" {
            if count > 1 {
                format!("{} ({count}x)", label(locale, "web"))
            } else {
                label(locale, "web").to_owned()
            }
        } else {
            label(locale, "tool").to_owned()
        };
        add(component_label, integer(item.get("usd_micros")));
    }
    totals.retain(|(_, amount)| *amount > 0);
    totals
}

fn allocate_components(total_units: i64, components: Vec<(String, i64)>) -> Vec<(String, i64)> {
    let total_units = total_units.max(0);
    let raw_total = components
        .iter()
        .map(|(_, raw)| i128::from((*raw).max(0)))
        .sum::<i128>();
    if total_units == 0 || raw_total == 0 {
        return Vec::new();
    }
    let mut allocated = components
        .into_iter()
        .enumerate()
        .filter(|(_, (_, raw))| *raw > 0)
        .map(|(index, (component_label, raw))| {
            let numerator = i128::from(total_units) * i128::from(raw);
            (
                component_label,
                numerator / raw_total,
                numerator % raw_total,
                index,
            )
        })
        .collect::<Vec<_>>();
    let used = allocated
        .iter()
        .map(|(_, units, _, _)| *units)
        .sum::<i128>();
    let leftover = usize::try_from(i128::from(total_units) - used).unwrap_or_default();
    let mut order = (0..allocated.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| (std::cmp::Reverse(allocated[*index].2), allocated[*index].3));
    for index in order.into_iter().take(leftover) {
        allocated[index].1 += 1;
    }
    allocated
        .into_iter()
        .filter_map(|(component_label, units, _, _)| {
            i64::try_from(units)
                .ok()
                .map(|units| (component_label, units))
        })
        .collect()
}

fn activity(metadata: &Map<String, Value>, event_type: &str, locale: Locale) -> &'static str {
    let usage_tag = text(metadata, "usage_tag");
    if event_type == "memory_compaction_settlement" || usage_tag.contains("memory_compaction") {
        label(locale, "memory")
    } else if usage_tag.contains("transcribe") || usage_tag.contains("audio") {
        label(locale, "audio")
    } else if usage_tag.contains("image") || usage_tag.contains("vision") {
        label(locale, "image")
    } else {
        label(locale, "response")
    }
}

fn entry_components(entry: &ChargeHistoryEntry, locale: Locale) -> Vec<Component> {
    let metadata = object(&entry.metadata).cloned().unwrap_or_default();
    let units = charged_units(&metadata, &entry.event_type);
    let pending = metadata
        .get("billing_pending")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || entry.event_type == "ai_reserve";
    if entry.event_type == "memory_compaction_settlement"
        || text(&metadata, "usage_tag").contains("memory_compaction")
    {
        return vec![Component {
            label: label(locale, "memory").to_owned(),
            units,
            pending,
        }];
    }
    let allocated = allocate_components(units, raw_components(&metadata, locale));
    if allocated.is_empty() {
        return vec![Component {
            label: activity(&metadata, &entry.event_type, locale).to_owned(),
            units,
            pending,
        }];
    }
    allocated
        .into_iter()
        .filter(|(_, units)| *units > 0)
        .map(|(component_label, units)| Component {
            label: component_label,
            units,
            pending,
        })
        .collect()
}

fn component_rank(component: &Component, locale: Locale) -> (u8, bool) {
    let rank = if component.label == label(locale, "response") {
        0
    } else if component.label == label(locale, "audio") || component.label == label(locale, "image")
    {
        1
    } else if component.label.starts_with(label(locale, "web")) {
        2
    } else if component.label == label(locale, "memory") {
        4
    } else {
        3
    };
    (rank, component.pending)
}

fn payer_suffix(entries: &[ChargeHistoryEntry], locale: Locale) -> String {
    let mut totals = BTreeMap::from([("user", 0_i64), ("chat", 0_i64)]);
    for entry in entries {
        let metadata = object(&entry.metadata).cloned().unwrap_or_default();
        let mut found = false;
        for payer in metadata
            .get("payer_breakdown")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_object)
        {
            found = true;
            let scope = if payer.get("scope").and_then(Value::as_str) == Some("chat") {
                "chat"
            } else {
                "user"
            };
            if let Some(total) = totals.get_mut(scope) {
                *total = total.saturating_add(integer(payer.get("credit_units")).max(0));
            }
        }
        if found {
            continue;
        }
        let scope = ["payer_scope", "source"]
            .into_iter()
            .find_map(|key| metadata.get(key).and_then(Value::as_str))
            .unwrap_or_default();
        if let Some(total) = totals.get_mut(scope) {
            *total = total.saturating_add(charged_units(&metadata, &entry.event_type));
        }
    }
    let user = totals["user"];
    let chat = totals["chat"];
    if chat <= 0 {
        String::new()
    } else if user <= 0 {
        format!(" · {}", label(locale, "group"))
    } else {
        format!(
            " · {} {} · {} {}",
            label(locale, "group"),
            format_credit_units(CreditUnits::new(chat)),
            label(locale, "personal"),
            format_credit_units(CreditUnits::new(user))
        )
    }
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let adjusted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month, day)
}

fn parse_number(raw: &str, start: usize, end: usize) -> Option<i64> {
    raw.get(start..end)?.parse().ok()
}

const fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

const fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    }
}

fn time_label(raw: &str, timezone_minutes: i64, locale: Locale) -> String {
    if raw.len() < 16 {
        return label(locale, "no_date").to_owned();
    }
    let (year, month, day, hour, minute) = (
        parse_number(raw, 0, 4),
        parse_number(raw, 5, 7),
        parse_number(raw, 8, 10),
        parse_number(raw, 11, 13),
        parse_number(raw, 14, 16),
    );
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute)) =
        (year, month, day, hour, minute)
    else {
        return label(locale, "no_date").to_owned();
    };
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
    {
        return label(locale, "no_date").to_owned();
    }
    let source_offset = raw
        .get(16..)
        .and_then(|tail| {
            let position = tail.rfind(['+', '-'])?;
            let sign = if tail.as_bytes().get(position) == Some(&b'-') {
                -1
            } else {
                1
            };
            let offset = &tail[position + 1..];
            let hours = offset.get(0..2)?.parse::<i64>().ok()?;
            let minutes = offset
                .strip_prefix(&offset[..2])
                .and_then(|rest| rest.strip_prefix(':'))
                .and_then(|rest| rest.get(0..2))
                .and_then(|rest| rest.parse::<i64>().ok())
                .unwrap_or_default();
            Some(sign * (hours * 60 + minutes))
        })
        .unwrap_or_default();
    let absolute_minutes = days_from_civil(year, month, day)
        .saturating_mul(1_440)
        .saturating_add(hour * 60 + minute)
        .saturating_sub(source_offset)
        .saturating_add(timezone_minutes);
    let local_days = absolute_minutes.div_euclid(1_440);
    let local_minutes = absolute_minutes.rem_euclid(1_440);
    let (_, local_month, local_day) = civil_from_days(local_days);
    format!(
        "{local_day:02}/{local_month:02} {:02}:{:02}",
        local_minutes / 60,
        local_minutes % 60
    )
}

#[must_use]
pub fn render_charge_history_page(
    page: &ChargeHistoryPage,
    user_id: i64,
    limit: usize,
    timezone_minutes: i64,
    locale: Locale,
) -> (String, Option<InlineKeyboardMarkup>) {
    if page.groups.is_empty() {
        return (label(locale, "empty").to_owned(), None);
    }
    let mut lines = vec![label(locale, "title").to_owned()];
    for group in &page.groups {
        let mut components = group
            .entries
            .iter()
            .flat_map(|entry| entry_components(entry, locale))
            .collect::<Vec<_>>();
        components.sort_by_key(|component| component_rank(component, locale));
        let total = components.iter().fold(0_i64, |total, component| {
            total.saturating_add(component.units)
        });
        let timestamp = time_label(&group.created_at, timezone_minutes, locale);
        let payer = payer_suffix(&group.entries, locale);
        lines.push(String::new());
        if let [component] = components.as_slice() {
            let pending = if component.pending {
                format!(" · {}", label(locale, "pending"))
            } else {
                String::new()
            };
            lines.push(format!(
                "{timestamp} · {} · {} cr{pending}{payer}",
                component.label,
                format_credit_units(CreditUnits::new(component.units))
            ));
            continue;
        }
        lines.push(format!(
            "{timestamp} · {} cr{payer}",
            format_credit_units(CreditUnits::new(total))
        ));
        for component in components {
            let pending = if component.pending {
                format!(" · {}", label(locale, "pending"))
            } else {
                String::new()
            };
            lines.push(format!(
                "  {} {} cr{pending}",
                component.label,
                format_credit_units(CreditUnits::new(component.units))
            ));
        }
    }
    let mut buttons = Vec::new();
    if page.has_newer
        && let Some(cursor) = page.newer_cursor
    {
        buttons.push(InlineKeyboardButton {
            text: label(locale, "previous").to_owned(),
            url: None,
            callback_data: Some(format!(
                "chg:{user_id}:{limit}:n:{cursor}:{timezone_minutes}"
            )),
        });
    }
    if page.has_older
        && let Some(cursor) = page.older_cursor
    {
        buttons.push(InlineKeyboardButton {
            text: label(locale, "next").to_owned(),
            url: None,
            callback_data: Some(format!(
                "chg:{user_id}:{limit}:o:{cursor}:{timezone_minutes}"
            )),
        });
    }
    let keyboard = (!buttons.is_empty()).then_some(InlineKeyboardMarkup {
        inline_keyboard: vec![buttons],
    });
    (lines.join("\n"), keyboard)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ChargeHistoryEntry, ChargeHistoryGroup, ChargeHistoryPage, ChargesCommandContext,
        ChargesCommandPlan, plan_charges_command, render_charge_history_page,
    };
    use crate::locale::Locale;
    use crate::telegram_actions::TelegramAction;
    use crate::telegram_input::{ChatId, MessageId};

    fn context(locale: Locale) -> ChargesCommandContext {
        ChargesCommandContext {
            chat_id: ChatId(101),
            message_id: MessageId(11),
            user_id: Some(55),
            locale,
            timezone_offset_hours: -3,
            billing_available: true,
        }
    }

    #[test]
    fn command_plan_parses_aliases_defaults_clamps_and_usage() {
        assert_eq!(
            plan_charges_command("/charges 2", "", context(Locale::Es)),
            ChargesCommandPlan::Load {
                user_id: 55,
                limit: 2,
                timezone_minutes: -180,
            }
        );
        assert!(matches!(
            plan_charges_command("/history", "", context(Locale::En)),
            ChargesCommandPlan::Load { limit: 10, .. }
        ));
        assert!(matches!(
            plan_charges_command(
                "/gastos 999999999999999999999999999",
                "",
                context(Locale::Es)
            ),
            ChargesCommandPlan::Load { limit: 20, .. }
        ));
        let ChargesCommandPlan::Reply(TelegramAction::SendMessage(message)) =
            plan_charges_command("/charges 1 2", "", context(Locale::En))
        else {
            return;
        };
        assert_eq!(message.text, "usage: /charges [count]");
    }

    #[test]
    fn renders_itemized_history_timezone_payer_and_next_page() {
        let page = ChargeHistoryPage {
            groups: vec![
                ChargeHistoryGroup {
                    cursor_id: 30,
                    created_at: "2026-08-26T17:32:00+00:00".to_owned(),
                    entries: vec![ChargeHistoryEntry {
                        id: 30,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({
                            "charged_credit_units_total": 8,
                            "payer_scope": "user",
                            "model_breakdown": [{"kind":"chat","usd_micros":30}],
                            "tool_breakdown": [{"tool":"web_search","count":1,"usd_micros":50}]
                        }),
                    }],
                },
                ChargeHistoryGroup {
                    cursor_id: 29,
                    created_at: "2026-08-26 16:00:00+00".to_owned(),
                    entries: vec![ChargeHistoryEntry {
                        id: 29,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({
                            "usage_tag":"auto_audio_media",
                            "charged_credit_units_total":7,
                            "payer_scope":"chat"
                        }),
                    }],
                },
            ],
            has_newer: false,
            has_older: true,
            newer_cursor: Some(30),
            older_cursor: Some(29),
        };
        let (text, keyboard) = render_charge_history_page(&page, 55, 2, -180, Locale::Es);
        assert_eq!(
            text,
            "Gastos IA\n\n26/08 14:32 · 0.08 cr\n  respuesta 0.03 cr\n  web 0.05 cr\n\n26/08 13:00 · audio · 0.07 cr · grupo"
        );
        let Some(keyboard) = keyboard else {
            return;
        };
        assert_eq!(keyboard.inline_keyboard[0][0].text, "Siguiente ›");
        assert_eq!(
            keyboard.inline_keyboard[0][0].callback_data.as_deref(),
            Some("chg:55:2:o:29:-180")
        );
    }

    #[test]
    fn renders_compaction_pending_split_payers_and_empty_page() {
        let page = ChargeHistoryPage {
            groups: vec![ChargeHistoryGroup {
                cursor_id: 12,
                created_at: "2026-08-26T17:00:00+00:00".to_owned(),
                entries: vec![
                    ChargeHistoryEntry {
                        id: 12,
                        event_type: "memory_compaction_settlement".to_owned(),
                        metadata: json!({"actual_credit_units":2,"source":"user"}),
                    },
                    ChargeHistoryEntry {
                        id: 11,
                        event_type: "ai_reserve".to_owned(),
                        metadata: json!({
                            "usage_tag":"memory_compaction:1:m2",
                            "charged_credit_units_total":107,
                            "billing_pending":true,
                            "payer_breakdown":[
                                {"scope":"chat","credit_units":100},
                                {"scope":"user","credit_units":7}
                            ]
                        }),
                    },
                    ChargeHistoryEntry {
                        id: 10,
                        event_type: "ai_settlement_result".to_owned(),
                        metadata: json!({
                            "charged_credit_units_total":8,
                            "payer_breakdown":[
                                {"scope":"user","credit_units":3},
                                {"scope":"chat","credit_units":5}
                            ]
                        }),
                    },
                ],
            }],
            has_newer: false,
            has_older: false,
            newer_cursor: Some(12),
            older_cursor: Some(12),
        };
        assert_eq!(
            render_charge_history_page(&page, 55, 10, 0, Locale::Es).0,
            "Gastos IA\n\n26/08 17:00 · 1.17 cr · grupo 1.05 · personal 0.12\n  respuesta 0.08 cr\n  memoria 0.02 cr\n  memoria 1.07 cr · pendiente"
        );
        let empty = ChargeHistoryPage {
            groups: Vec::new(),
            has_newer: false,
            has_older: false,
            newer_cursor: None,
            older_cursor: None,
        };
        assert_eq!(
            render_charge_history_page(&empty, 55, 10, 0, Locale::En),
            ("you have no recent AI expenses".to_owned(), None)
        );
    }
}
