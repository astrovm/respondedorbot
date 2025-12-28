use chrono::{DateTime, FixedOffset, Local};
use rand::prelude::IndexedRandom;
use rand::Rng;
use redis::AsyncCommands;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

use crate::{ai, config, hacker_news, message_utils};

const AGENT_THOUGHTS_KEY: &str = "agent:thoughts";
const MAX_AGENT_THOUGHTS: usize = 10;
const AGENT_THOUGHT_DISPLAY_LIMIT: usize = 5;
const AGENT_THOUGHT_CHAR_LIMIT: usize = 500;
const AGENT_RECENT_THOUGHT_WINDOW: usize = 5;
const AGENT_REQUIRED_SECTIONS: [&str; 2] = ["HALLAZGOS", "PRÓXIMO PASO"];
const AGENT_EMPTY_RESPONSE_FALLBACK: &str =
    "HALLAZGOS: no se me ocurrió nada nuevo, pintó el vacío.\nPRÓXIMO PASO: meter una búsqueda puntual para traer un dato real y salir de la fiaca.";
const AGENT_REPETITION_RETRY_LIMIT: usize = 3;
const AGENT_LOOP_FALLBACK_PREFIX: &str =
    "HALLAZGOS: registré que estaba en un loop repitiendo";
const AGENT_REPETITION_ESCALATION_HINT: &str =
    "No escribas que estás trabado o en un loop. Ejecutá de inmediato una herramienta \
(web_search o fetch_url) con un tema distinto y registrá datos nuevos \
(números, titulares, precios). Si el tema anterior no se mueve, cambiá a otro interés fuerte del gordo.";

const AGENT_KEYWORD_STOPWORDS: &[&str] = &[
    "ante",
    "aqui",
    "aquel",
    "aquella",
    "aquello",
    "asi",
    "busque",
    "cada",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "cuyo",
    "datos",
    "donde",
    "durante",
    "entre",
    "este",
    "esta",
    "estas",
    "esto",
    "estos",
    "gordo",
    "hallazgos",
    "hacer",
    "hice",
    "investigue",
    "investigando",
    "investigar",
    "luego",
    "mientras",
    "mismo",
    "mucha",
    "mucho",
    "nada",
    "para",
    "pendiente",
    "pero",
    "porque",
    "proximo",
    "queda",
    "seguir",
    "sigue",
    "sobre",
    "solo",
    "todas",
    "todos",
    "todavia",
    "tema",
    "teni",
    "tenemos",
    "tener",
    "tenes",
    "tenia",
    "teniendo",
    "tengo",
    "unas",
    "unos",
    "voy",
];

#[derive(Debug, Clone)]
pub struct AgentThought {
    pub text: String,
    pub timestamp: Option<i64>,
}

pub async fn get_agent_thoughts(redis: &redis::Client) -> Vec<AgentThought> {
    load_agent_thoughts(redis, MAX_AGENT_THOUGHTS).await
}

pub async fn get_agent_memory(redis: &redis::Client, limit: usize) -> Option<String> {
    let thoughts = load_agent_thoughts(redis, limit).await;
    build_agent_thoughts_context_message(&thoughts)
}

pub async fn show_agent_thoughts(redis: &redis::Client) -> String {
    let thoughts = get_agent_thoughts(redis).await;
    let visible: Vec<AgentThought> = thoughts
        .into_iter()
        .take(AGENT_THOUGHT_DISPLAY_LIMIT)
        .collect();
    format_agent_thoughts(&visible)
}

pub fn build_agent_thoughts_context_message(thoughts: &[AgentThought]) -> Option<String> {
    let mut lines = Vec::new();
    let tz = FixedOffset::west_opt(3 * 3600).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());

    for thought in thoughts {
        let text = thought.text.trim();
        if text.is_empty() {
            continue;
        }
        if let Some(ts) = thought.timestamp {
            if let Some(dt) = DateTime::from_timestamp(ts, 0) {
                let local = dt.with_timezone(&tz);
                lines.push(format!("- [{}] {}", local.format("%d/%m %H:%M"), text));
            } else {
                lines.push(format!("- {}", text));
            }
        } else {
            lines.push(format!("- {}", text));
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(format!(
            "MEMORIA AUTÓNOMA (más reciente primero):\n{}\nUsá esta memoria cuando charles con humanos o cuando generes nuevos pensamientos autónomos.",
            lines.join("\n")
        ))
    }
}

pub fn format_agent_thoughts(thoughts: &[AgentThought]) -> String {
    if thoughts.is_empty() {
        return "todavía no tengo pensamientos guardados, dejame que labure un toque.".to_string();
    }

    let mut lines = vec!["🧠 Pensamientos recientes del gordo autónomo:".to_string()];
    let tz = FixedOffset::west_opt(3 * 3600).unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());
    let mut index = 1;

    for thought in thoughts {
        let text = thought.text.trim();
        if text.is_empty() {
            continue;
        }

        let formatted = text.replace("\r\n", "\n").replace('\r', "\n");
        let formatted = formatted.replace('\n', "\n   ");
        if let Some(ts) = thought.timestamp {
            if let Some(dt) = DateTime::from_timestamp(ts, 0) {
                let local = dt.with_timezone(&tz);
                lines.push(format!(
                    "{}. [{}] {}",
                    index,
                    local.format("%d/%m %H:%M"),
                    formatted
                ));
                index += 1;
                continue;
            }
        }
        lines.push(format!("{}. {}", index, formatted));
        index += 1;
    }

    if lines.len() == 1 {
        return "todavía no tengo pensamientos guardados, dejame que labure un toque.".to_string();
    }

    lines.join("\n")
}

pub async fn run_agent_cycle(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Result<Value, String> {
    let thoughts = load_agent_thoughts(redis, MAX_AGENT_THOUGHTS).await;
    let recent_thoughts: Vec<AgentThought> = thoughts
        .iter()
        .take(AGENT_RECENT_THOUGHT_WINDOW)
        .cloned()
        .collect();
    let mut recent_entry_texts: Vec<String> = Vec::new();
    for thought in &recent_thoughts {
        let text = thought.text.trim();
        if !text.is_empty() {
            recent_entry_texts.push(text.to_string());
        }
    }

    let last_entry_text = recent_entry_texts.first().cloned();
    let recent_topic_summaries = summarize_recent_agent_topics(&recent_thoughts, 4);

    let hn_items = hacker_news::get_hn_context(http).await;
    let hn_trimmed: Vec<hacker_news::HnItem> = hn_items.into_iter().take(3).collect();
    let hn_info = if hn_trimmed.is_empty() {
        None
    } else {
        Some(hacker_news::format_hn_items(&hn_trimmed))
    };

    let mut prompt = String::from(
        "Estás operando en modo autónomo. Podés investigar, navegar y usar herramientas. \
Registrá en primera persona qué investigaste, qué encontraste y recién después el próximo paso. \
Devolvé la nota en dos secciones en mayúsculas: \"HALLAZGOS:\" con los datos concretos y \"PRÓXIMO PASO:\" con la acción puntual.",
    );

    if let Some(text) = &last_entry_text {
        if !text.is_empty() {
            prompt.push_str("\n\nÚLTIMA MEMORIA GUARDADA:\n");
            prompt.push_str(&truncate_agent_text(text, 220));
            prompt.push_str("\nResolvé ese pendiente ahora mismo y deja asentado el resultado concreto antes de planear otra cosa.");
        }
    }

    if !recent_topic_summaries.is_empty() {
        let topics_lines = recent_topic_summaries
            .iter()
            .map(|item| format!("- {item}"))
            .collect::<Vec<_>>()
            .join("\n");
        prompt.push_str("\nEstos fueron los últimos temas que trabajaste:\n");
        prompt.push_str(&topics_lines);
        prompt.push_str(
            "\nSolo repetí uno si apareció un dato nuevo y específico; si no, cambiá a otro interés del gordo.",
        );
    }

    if let Some(hn_block) = &hn_info {
        prompt.push_str("\n\nHACKER NEWS HOY:\n");
        prompt.push_str(hn_block);
        prompt.push_str("\nSi alguna nota trae datos frescos que sumen, citá la fuente y metela en los hallazgos.");
    }

    prompt.push_str(
        "\nIncluí datos específicos (números, titulares, fuentes) de lo que investigues y evitá repetir entradas previas. \
Si necesitás info fresca, llamá a la herramienta web_search con un query puntual y resumí el hallazgo. \
Si hace falta leer una nota puntual, llamá a fetch_url con la URL (incluí https://) y anotá lo relevante. \
Máximo 500 caracteres, sin saludar a nadie: es un apunte privado.",
    );

    let system_prompt = config::load_bot_config()
        .map(|cfg| cfg.system_prompt)
        .unwrap_or_else(|_| "Sos un asistente útil.".to_string());
    let time_label = Local::now().format("%A %d/%m/%Y").to_string();
    let agent_memory = get_agent_memory(redis, AGENT_RECENT_THOUGHT_WINDOW).await;

    let context = ai::AiContext {
        system_prompt,
        time_label,
        market_info: None,
        weather_info: None,
        hn_info: hn_info.clone(),
        agent_memory,
    };

    let base_messages = vec![ai::ChatMessage {
        role: "user".to_string(),
        content: prompt,
    }];

    let mut cleaned = request_agent_response(
        http,
        redis,
        &context,
        base_messages.clone(),
        "Autonomous agent execution failed",
    )
    .await?;

    if !agent_sections_are_valid(&cleaned) {
        let original_attempt = cleaned.clone();
        let mut corrective_prompt = build_agent_retry_prompt(Some(&original_attempt));
        let missing_sections: Vec<&str> = AGENT_REQUIRED_SECTIONS
            .iter()
            .copied()
            .filter(|header| {
                extract_agent_section_content(&original_attempt, header, &[]).is_none()
            })
            .collect();
        if !missing_sections.is_empty() {
            let section_list = missing_sections.join(", ");
            corrective_prompt.push_str(&format!(
                " La nota anterior no tenía contenido en: {section_list}. Respetá ambas secciones con información concreta.",
            ));
        }

        let retry_messages = build_agent_retry_messages(
            &base_messages,
            &original_attempt,
            &corrective_prompt,
        );
        cleaned = request_agent_response(
            http,
            redis,
            &context,
            retry_messages,
            "Autonomous agent execution failed (structure retry)",
        )
        .await?;

        if !agent_sections_are_valid(&cleaned) {
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
        }
    }

    let mut matching_recent_text =
        find_repetitive_recent_thought(&cleaned, &recent_entry_texts);
    let mut repetition_attempt = 0;
    while matching_recent_text.is_some() && repetition_attempt < AGENT_REPETITION_RETRY_LIMIT {
        let matching_text = matching_recent_text.clone().unwrap_or_default();
        let mut corrective_prompt = build_agent_retry_prompt(Some(&matching_text));
        if repetition_attempt == AGENT_REPETITION_RETRY_LIMIT - 1 {
            corrective_prompt.push(' ');
            corrective_prompt.push_str(AGENT_REPETITION_ESCALATION_HINT);
        }

        let retry_messages =
            build_agent_retry_messages(&base_messages, &cleaned, &corrective_prompt);
        cleaned = request_agent_response(
            http,
            redis,
            &context,
            retry_messages,
            "Autonomous agent execution failed (retry)",
        )
        .await?;

        if !agent_sections_are_valid(&cleaned) {
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
            matching_recent_text = None;
            break;
        }

        repetition_attempt += 1;
        matching_recent_text =
            find_repetitive_recent_thought(&cleaned, &recent_entry_texts);
    }

    if let Some(matching_text) = matching_recent_text.clone() {
        let fallback = message_utils::clean_duplicate_response(
            &build_agent_fallback_entry(&matching_text),
        );
        let fallback_entry = ensure_agent_response_text(&fallback);

        let comparison_texts: Vec<String> = if recent_entry_texts.is_empty() {
            recent_entry_texts.clone()
        } else {
            let mut filtered = Vec::new();
            let mut skipped = false;
            for text in &recent_entry_texts {
                if !skipped && text == &matching_text {
                    skipped = true;
                    continue;
                }
                filtered.push(text.clone());
            }
            filtered
        };

        if is_loop_fallback_text(&fallback_entry)
            && find_repetitive_recent_thought(&fallback_entry, &comparison_texts).is_some()
        {
            cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
        } else {
            cleaned = fallback_entry;
        }
    }

    if !agent_sections_are_valid(&cleaned) {
        cleaned = AGENT_EMPTY_RESPONSE_FALLBACK.to_string();
    }

    if is_empty_agent_thought_text(&cleaned) {
        return Ok(json!({
            "text": ensure_agent_response_text(&cleaned),
            "persisted": false,
        }));
    }

    let entry = save_agent_thought(redis, &cleaned)
        .await
        .ok_or_else(|| "Failed to persist autonomous agent thought".to_string())?;

    let mut result = json!({
        "text": entry.text,
        "persisted": true,
    });

    if let Some(ts) = entry.timestamp {
        if let Some(dt) = DateTime::from_timestamp(ts, 0) {
            let tz = FixedOffset::west_opt(3 * 3600)
                .unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());
            result["timestamp"] = json!(ts);
            result["iso_time"] = json!(dt.with_timezone(&tz).to_rfc3339());
        }
    }

    Ok(result)
}

async fn load_agent_thoughts(redis: &redis::Client, limit: usize) -> Vec<AgentThought> {
    let mut conn = match redis.get_multiplexed_async_connection().await {
        Ok(conn) => conn,
        Err(_) => return Vec::new(),
    };

    let raw: Vec<String> = match conn
        .lrange(AGENT_THOUGHTS_KEY, 0, (limit.saturating_sub(1)) as isize)
        .await
    {
        Ok(items) => items,
        Err(_) => return Vec::new(),
    };

    raw.into_iter().filter_map(parse_thought).collect()
}

async fn save_agent_thought(redis: &redis::Client, text: &str) -> Option<AgentThought> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    let text = truncate_agent_text(trimmed, AGENT_THOUGHT_CHAR_LIMIT);
    let timestamp = chrono::Utc::now().timestamp();
    let payload = json!({
        "text": text,
        "timestamp": timestamp,
    });

    let mut conn = redis.get_multiplexed_async_connection().await.ok()?;
    let _: () = conn.lpush(AGENT_THOUGHTS_KEY, payload.to_string()).await.ok()?;
    let _: () = conn
        .ltrim(AGENT_THOUGHTS_KEY, 0, (MAX_AGENT_THOUGHTS - 1) as isize)
        .await
        .ok()?;

    Some(AgentThought {
        text,
        timestamp: Some(timestamp),
    })
}

fn parse_thought(raw: String) -> Option<AgentThought> {
    let value = serde_json::from_str::<Value>(&raw).ok()?;
    let text = value
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if text.is_empty() {
        return None;
    }
    let timestamp = parse_timestamp(value.get("timestamp"));
    Some(AgentThought { text, timestamp })
}

fn parse_timestamp(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(num)) => num.as_i64(),
        Some(Value::String(s)) => s.parse::<i64>().ok(),
        _ => None,
    }
}

pub fn normalize_agent_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let mut without_accents = String::new();
    for ch in lowered.nfkd() {
        if !is_combining_mark(ch) {
            without_accents.push(ch);
        }
    }

    let mut cleaned = String::new();
    for ch in without_accents.chars() {
        if ch.is_ascii_alphanumeric() {
            cleaned.push(ch);
        } else {
            cleaned.push(' ');
        }
    }

    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn loop_fallback_marker() -> &'static String {
    static MARKER: OnceLock<String> = OnceLock::new();
    MARKER.get_or_init(|| normalize_agent_text(AGENT_LOOP_FALLBACK_PREFIX))
}

pub fn is_loop_fallback_text(text: &str) -> bool {
    let marker = loop_fallback_marker();
    if marker.is_empty() {
        return false;
    }
    let normalized = normalize_agent_text(text);
    if normalized.is_empty() {
        return false;
    }
    normalized.starts_with(marker)
}

fn empty_fallback_marker() -> &'static String {
    static MARKER: OnceLock<String> = OnceLock::new();
    MARKER.get_or_init(|| normalize_agent_text(AGENT_EMPTY_RESPONSE_FALLBACK))
}

pub fn is_empty_agent_thought_text(text: &str) -> bool {
    let sanitized = text.trim();
    if sanitized.is_empty() {
        return true;
    }
    let marker = empty_fallback_marker();
    if marker.is_empty() {
        return false;
    }
    normalize_agent_text(sanitized) == *marker
}

fn keyword_stopwords() -> &'static HashSet<&'static str> {
    static STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOPWORDS.get_or_init(|| AGENT_KEYWORD_STOPWORDS.iter().copied().collect())
}

fn extract_keywords_from_normalized(normalized: &str) -> HashSet<String> {
    let mut keywords = HashSet::new();
    let stopwords = keyword_stopwords();

    for token in normalized.split_whitespace() {
        if token.len() < 4 {
            continue;
        }
        if stopwords.contains(token) {
            continue;
        }
        if token.chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        if !token.chars().any(|ch| ch.is_ascii_alphabetic()) {
            continue;
        }
        keywords.insert(token.to_string());
    }

    keywords
}

pub fn get_agent_text_features(text: &str) -> (String, HashSet<String>) {
    let normalized = normalize_agent_text(text);
    if normalized.is_empty() {
        return (String::new(), HashSet::new());
    }
    let keywords = extract_keywords_from_normalized(&normalized);
    (normalized, keywords)
}

fn agent_keywords_are_repetitive(
    new_keywords: &HashSet<String>,
    previous_keywords: &HashSet<String>,
) -> bool {
    if new_keywords.is_empty() || previous_keywords.is_empty() {
        return false;
    }

    let overlap: HashSet<&String> = new_keywords
        .intersection(previous_keywords)
        .collect();
    if overlap.len() >= 3 {
        return true;
    }

    let min_len = new_keywords.len().min(previous_keywords.len());
    if min_len <= 1 {
        return false;
    }

    if overlap.len() >= 2 && min_len <= 5 {
        return true;
    }

    let overlap_ratio = overlap.len() as f32 / min_len as f32;
    overlap_ratio >= 0.6
}

fn normalized_texts_are_repetitive(normalized_new: &str, normalized_prev: &str) -> bool {
    if normalized_new.is_empty() || normalized_prev.is_empty() {
        return false;
    }
    if normalized_new == normalized_prev {
        return true;
    }

    let similarity = sequence_matcher_ratio(normalized_new, normalized_prev);
    if similarity >= 0.88 {
        return true;
    }

    let new_tokens: HashSet<&str> = normalized_new.split_whitespace().collect();
    let prev_tokens: HashSet<&str> = normalized_prev.split_whitespace().collect();
    if new_tokens.is_empty() || prev_tokens.is_empty() {
        return false;
    }

    let union_len = new_tokens.union(&prev_tokens).count();
    if union_len == 0 {
        return false;
    }
    let overlap = new_tokens.intersection(&prev_tokens).count() as f32 / union_len as f32;
    overlap >= 0.75
}

pub fn is_repetitive_thought(new_text: &str, previous_text: Option<&str>) -> bool {
    let Some(previous_text) = previous_text else {
        return false;
    };
    if new_text.trim().is_empty() || previous_text.trim().is_empty() {
        return false;
    }

    let (normalized_new, new_keywords) = get_agent_text_features(new_text);
    let (normalized_prev, prev_keywords) = get_agent_text_features(previous_text);
    if normalized_new.is_empty() || normalized_prev.is_empty() {
        return false;
    }
    if normalized_texts_are_repetitive(&normalized_new, &normalized_prev) {
        return true;
    }
    agent_keywords_are_repetitive(&new_keywords, &prev_keywords)
}

pub fn find_repetitive_recent_thought(
    new_text: &str,
    previous_texts: &[String],
) -> Option<String> {
    for candidate in previous_texts {
        let sanitized = candidate.trim();
        if sanitized.is_empty() {
            continue;
        }
        if is_repetitive_thought(new_text, Some(sanitized)) {
            return Some(sanitized.to_string());
        }
    }
    None
}

pub fn summarize_recent_agent_topics(thoughts: &[AgentThought], limit: usize) -> Vec<String> {
    let mut summaries: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for thought in thoughts {
        if summaries.len() >= limit {
            break;
        }

        let text = thought.text.trim();
        if text.is_empty() {
            continue;
        }

        let section_content = extract_agent_section_content(
            text,
            "HALLAZGOS",
            &["PRÓXIMO PASO", "PROXIMO PASO"],
        );
        let snippet_source = section_content.unwrap_or_else(|| text.to_string());
        let snippet_source = snippet_source
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if snippet_source.is_empty() {
            continue;
        }

        let mut end_index = snippet_source.len();
        for (idx, ch) in snippet_source.char_indices() {
            if matches!(ch, '.' | '!' | '?') {
                end_index = idx + ch.len_utf8();
                break;
            }
        }
        let summary = snippet_source
            .chars()
            .take(end_index)
            .take(160)
            .collect::<String>();
        let normalized_summary = summary.to_lowercase();
        if seen.contains(&normalized_summary) {
            continue;
        }
        seen.insert(normalized_summary);
        summaries.push(summary);
    }

    summaries
}

fn normalize_header_value(value: &str) -> String {
    normalize_agent_text(value)
}

pub fn extract_agent_section_content(
    text: &str,
    header: &str,
    other_headers: &[&str],
) -> Option<String> {
    let sanitized = text;
    if sanitized.is_empty() {
        return None;
    }

    let mut normalized_headers: Vec<String> = Vec::new();
    for item in std::iter::once(header).chain(other_headers.iter().copied()) {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        let normalized = normalize_header_value(trimmed);
        if !normalized.is_empty() && !normalized_headers.contains(&normalized) {
            normalized_headers.push(normalized);
        }
    }

    let target_norm = normalize_header_value(header);
    if !target_norm.is_empty() && !normalized_headers.contains(&target_norm) {
        normalized_headers.push(target_norm.clone());
    }

    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current_norm: Option<String> = None;

    for raw_line in sanitized.replace("\r\n", "\n").replace('\r', "\n").split('\n') {
        let stripped = raw_line.trim();
        if stripped.is_empty() {
            if let Some(current) = &current_norm {
                sections.entry(current.clone()).or_default().push(String::new());
            }
            continue;
        }

        let mut parts = stripped.splitn(2, ':');
        let _candidate = parts.next().unwrap_or("").trim();
        let remainder = parts.next().unwrap_or("");
        let normalized_line = normalize_header_value(stripped);
        let mut header_norm = None;
        if normalized_line.starts_with("hallazgos") {
            header_norm = Some("hallazgos".to_string());
        } else if normalized_line.starts_with("proximo paso") {
            header_norm = Some("proximo paso".to_string());
        }
        if header_norm.is_none() {
            header_norm = normalized_headers
                .iter()
                .find(|key| normalized_line.starts_with(key.as_str()))
                .cloned();
        }

        if let Some(found_header) = header_norm {
            current_norm = Some(found_header.clone());
            sections.entry(found_header).or_default();
            if !remainder.trim().is_empty() {
                sections
                    .entry(current_norm.clone().unwrap())
                    .or_default()
                    .push(remainder.trim().to_string());
            }
            continue;
        }

        if let Some(current) = &current_norm {
            sections
                .entry(current.clone())
                .or_default()
                .push(stripped.to_string());
        }
    }

    let content_lines = sections.get(&target_norm)?;
    let content = content_lines.join("\n").trim().to_string();
    if content.is_empty() {
        None
    } else {
        Some(content)
    }
}

pub fn agent_sections_are_valid(text: &str) -> bool {
    if text.trim().is_empty() {
        return false;
    }

    for header in AGENT_REQUIRED_SECTIONS {
        let other_headers: Vec<&str> = AGENT_REQUIRED_SECTIONS
            .iter()
            .copied()
            .filter(|h| *h != header)
            .collect();
        if extract_agent_section_content(text, header, &other_headers).is_none() {
            return false;
        }
    }
    true
}

pub fn get_agent_retry_hint_with_rng<R: Rng + ?Sized>(
    previous_text: Option<&str>,
    rng: &mut R,
) -> String {
    let normalized_previous = normalize_agent_text(previous_text.unwrap_or(""));
    let mut tokens = Vec::new();
    for token in normalized_previous.split_whitespace() {
        if token.is_empty() {
            continue;
        }
        if keyword_stopwords().contains(token) {
            continue;
        }
        if token.len() < 3 {
            continue;
        }
        if token.chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        tokens.push(token.to_string());
    }

    let mut counter: HashMap<String, usize> = HashMap::new();
    for token in tokens {
        *counter.entry(token).or_insert(0) += 1;
    }

    let mut sorted: Vec<(String, usize)> = counter.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    let top_keywords: Vec<String> = sorted.into_iter().take(3).map(|(k, _)| k).collect();

    let avoided_fragment = if top_keywords.is_empty() {
        "Arrancá desde cero sin reciclar la última búsqueda. ".to_string()
    } else {
        format!(
            "Marcá como prohibidos estos términos ya gastados: {}. ",
            top_keywords.join(", ")
        )
    };

    let option_count = rng.random_range(3..=4);
    let ordinal_words = ["primera", "segunda", "tercera", "cuarta", "quinta"];
    let ordinal_index = rng.random_range(0..option_count);
    let ordinal_word = ordinal_words
        .get(ordinal_index)
        .unwrap_or(&ordinal_words[0]);

    let brainstorming_templates = [
        "Anotá {n} búsquedas frescas en temas distintos. Ejecutá la {ordinal} con web_search.",
        "Hacé una mini lluvia de ideas con {n} queries nuevos y corré la {ordinal} usando web_search.",
        "Pensá en {n} consultas posibles que sorprendan al gordo y quedate con la {ordinal} para web_search.",
    ];

    let follow_up_templates = [
        "Traé números, fechas y citá la fuente puntual.",
        "Resumí el dato clave con cifras concretas y quién lo publicó.",
        "Documentá resultados verificables (monto, variación, protagonista) y la fuente exacta.",
    ];

    let brainstorming_prompt = brainstorming_templates
        .choose(rng)
        .unwrap_or(&brainstorming_templates[0])
        .replace("{n}", &option_count.to_string())
        .replace("{ordinal}", ordinal_word);
    let follow_up_prompt = follow_up_templates
        .choose(rng)
        .unwrap_or(&follow_up_templates[0]);

    let letter_choices = "abcdefghijklmnñopqrstuvwxyz";
    let letters: Vec<char> = letter_choices.chars().collect();
    let chosen_letter = letters
        .choose(rng)
        .copied()
        .unwrap_or('A')
        .to_uppercase()
        .to_string();
    let alternate_pool: Vec<char> = letters
        .iter()
        .copied()
        .filter(|ch| ch.to_uppercase().to_string() != chosen_letter)
        .collect();
    let fallback_letter = alternate_pool
        .choose(rng)
        .copied()
        .unwrap_or_else(|| chosen_letter.chars().next().unwrap_or('A'))
        .to_uppercase()
        .to_string();

    let numeric_floor = rng.random_range(8..=24);
    let numeric_multiplier = rng.random_range(3..=9);
    let numeric_target = numeric_floor * numeric_multiplier;

    let constraint_templates = [
        "Sumá una restricción creativa: la búsqueda tiene que incluir un protagonista cuya inicial sea \"{letter}\" y una cifra cerca de {value}.",
        "Obligate a que la consulta nombre algo que empiece con \"{letter}\" y mencione un número alrededor de {value}.",
        "Forzá el query a combinar un actor que arranque con \"{letter}\" más un dato numérico aproximado a {value}.",
    ];

    let pivot_templates = [
        "Si web_search no trae novedad, generá otra lluvia de ideas reemplazando las palabras prohibidas por categorías nuevas y probá con inicial \"{fallback}\".",
        "Si la ejecución devuelve humo, descartá la idea y repetí el proceso con términos distintos que comiencen con \"{fallback}\".",
        "Si no aparecen datos frescos, reseteá las keywords vetadas y buscá una consulta distinta arrancando por \"{fallback}\".",
    ];

    let constraint_prompt = constraint_templates
        .choose(rng)
        .unwrap_or(&constraint_templates[0])
        .replace("{letter}", &chosen_letter)
        .replace("{value}", &numeric_target.to_string());
    let pivot_prompt = pivot_templates
        .choose(rng)
        .unwrap_or(&pivot_templates[0])
        .replace("{fallback}", &fallback_letter);

    format!(
        "{avoided_fragment}{brainstorming_prompt} {follow_up_prompt} {constraint_prompt} {pivot_prompt}"
    )
}

pub fn build_agent_retry_prompt(previous_text: Option<&str>) -> String {
    let mut rng = rand::rng();
    build_agent_retry_prompt_with_rng(previous_text, &mut rng)
}

pub fn build_agent_retry_prompt_with_rng<R: Rng + ?Sized>(
    previous_text: Option<&str>,
    rng: &mut R,
) -> String {
    let preview = truncate_agent_text(previous_text.unwrap_or(""), 160);
    let preview_single_line = preview.replace('\n', " ").trim().to_string();
    let mut base_prompt = String::from(
        "La última nota no sirvió: te repetiste igual que la memoria anterior o no respetaste la estructura obligatoria. \
Antes de escribir otra vez, completá el pendiente y contá resultados concretos. \
Si necesitás info fresca, llamá a la herramienta web_search con un query preciso y resumí lo que encontraste. Si ya cerraste ese tema, cambiá a otro interés fuerte del gordo en vez de seguir clavado en lo mismo. ",
    );

    if !preview_single_line.is_empty() {
        base_prompt.push_str(&format!(
            "Esto fue lo último guardado o la nota fallida: \"{preview_single_line}\". "
        ));
    }

    base_prompt.push_str(
        "Escribí ahora una nota distinta con hechos puntuales y cerrala en dos secciones claras: \"HALLAZGOS:\" con los datos específicos que obtuviste y \"PRÓXIMO PASO:\" con la siguiente acción directa.",
    );

    let hint = get_agent_retry_hint_with_rng(previous_text, rng);
    if hint.is_empty() {
        base_prompt
    } else {
        format!("{base_prompt} {hint}")
    }
}

pub fn build_agent_fallback_entry(previous_text: &str) -> String {
    let sanitized_previous = previous_text.trim();
    let normalized_previous = normalize_agent_text(sanitized_previous);
    let fallback_marker = loop_fallback_marker();

    let preview = truncate_agent_text(sanitized_previous, 120);
    let preview_single_line = preview.replace('\n', " ").trim().to_string();

    let mut include_fragment = true;
    if !normalized_previous.is_empty()
        && !fallback_marker.is_empty()
        && normalized_previous.starts_with(fallback_marker)
    {
        include_fragment = false;
    }

    let loop_fragment = if include_fragment && !preview_single_line.is_empty() {
        format!(" \"{preview_single_line}\"")
    } else {
        String::new()
    };

    format!(
        "{AGENT_LOOP_FALLBACK_PREFIX}{loop_fragment} sin generar avances reales.\nPRÓXIMO PASO: hacer una búsqueda web urgente, anotar los datos específicos que salgan y recién después planear el próximo paso."
    )
}

pub fn ensure_agent_response_text(text: &str) -> String {
    let sanitized = text.trim();
    if sanitized.is_empty() {
        AGENT_EMPTY_RESPONSE_FALLBACK.to_string()
    } else {
        sanitized.to_string()
    }
}

pub fn build_agent_retry_messages(
    base_messages: &[ai::ChatMessage],
    assistant_text: &str,
    corrective_prompt: &str,
) -> Vec<ai::ChatMessage> {
    let mut messages = base_messages.to_vec();
    messages.push(ai::ChatMessage {
        role: "assistant".to_string(),
        content: assistant_text.to_string(),
    });
    messages.push(ai::ChatMessage {
        role: "user".to_string(),
        content: corrective_prompt.to_string(),
    });
    messages
}

async fn request_agent_response(
    http: &reqwest::Client,
    redis: &redis::Client,
    context: &ai::AiContext,
    messages: Vec<ai::ChatMessage>,
    error_context: &str,
) -> Result<String, String> {
    let response = ai::ask_ai(http, redis, context, messages)
        .await
        .ok_or_else(|| error_context.to_string())?;
    let sanitized = message_utils::sanitize_tool_artifacts(&response);
    let cleaned = message_utils::clean_duplicate_response(&sanitized);
    Ok(ensure_agent_response_text(cleaned.trim()))
}

fn truncate_agent_text(text: &str, max_length: usize) -> String {
    message_utils::truncate_text(text.trim(), max_length)
}

fn sequence_matcher_ratio(a: &str, b: &str) -> f32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    if a_len == 0 && b_len == 0 {
        return 1.0;
    }
    if a_len == 0 || b_len == 0 {
        return 0.0;
    }

    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    collect_matching_blocks(&a_chars, 0, a_len, &b_chars, 0, b_len, &mut blocks);
    blocks.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let mut merged: Vec<(usize, usize, usize)> = Vec::new();
    for block in blocks {
        if let Some(last) = merged.last_mut() {
            if last.0 + last.2 == block.0 && last.1 + last.2 == block.1 {
                last.2 += block.2;
                continue;
            }
        }
        merged.push(block);
    }

    let match_size: usize = merged.iter().map(|block| block.2).sum();
    (2.0 * match_size as f32) / (a_len + b_len) as f32
}

fn collect_matching_blocks(
    a: &[char],
    alo: usize,
    ahi: usize,
    b: &[char],
    blo: usize,
    bhi: usize,
    blocks: &mut Vec<(usize, usize, usize)>,
) {
    let (i, j, size) = find_longest_match(a, alo, ahi, b, blo, bhi);
    if size == 0 {
        return;
    }

    if alo < i && blo < j {
        collect_matching_blocks(a, alo, i, b, blo, j, blocks);
    }

    blocks.push((i, j, size));

    let i_end = i + size;
    let j_end = j + size;
    if i_end < ahi && j_end < bhi {
        collect_matching_blocks(a, i_end, ahi, b, j_end, bhi, blocks);
    }
}

fn find_longest_match(
    a: &[char],
    alo: usize,
    ahi: usize,
    b: &[char],
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    if alo >= ahi || blo >= bhi {
        return (alo, blo, 0);
    }

    let b_len = bhi - blo;
    let mut best_i = alo;
    let mut best_j = blo;
    let mut best_size = 0;

    let mut prev = vec![0usize; b_len + 1];
    for i in alo..ahi {
        let mut current = vec![0usize; b_len + 1];
        for (offset, j) in (blo..bhi).enumerate() {
            if a[i] == b[j] {
                let size = prev[offset] + 1;
                current[offset + 1] = size;
                if size > best_size {
                    best_size = size;
                    best_i = i + 1 - size;
                    best_j = j + 1 - size;
                }
            }
        }
        prev = current;
    }

    (best_i, best_j, best_size)
}
