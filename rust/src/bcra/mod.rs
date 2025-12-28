use chrono::Datelike;
use calamine::Reader;
use regex::Regex;
use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use crate::http_cache::cached_get_json;
use crate::redis_store::{redis_get_string, redis_setex_string};

const TTL_BCRA: u64 = 300;
const ITCRM_URL: &str = "https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/ITCRMSerie.xlsx";

#[derive(Debug, Clone)]
pub struct BandLimits {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub lower_change_pct: Option<f64>,
    pub upper_change_pct: Option<f64>,
    pub date: Option<String>,
}

pub async fn get_bcra_variables(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<String> {
    let variables = fetch_latest_variables(http, redis).await?;
    let itcrm = get_latest_itcrm_value_and_date(http, redis).await;
    let tcrm = get_tcrm_100(http, redis).await;
    Some(format_bcra_variables(&variables, itcrm, tcrm))
}

pub async fn get_currency_band_limits(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<BandLimits> {
    let variables = bcra_list_variables(http, redis, Some("Principales Variables")).await?;

    let mut lower_id = None;
    let mut upper_id = None;
    for item in variables {
        let desc = item.get("descripcion").and_then(|v| v.as_str()).unwrap_or("");
        let normalized = normalize_text(desc);
        if !normalized.contains("bandas cambiarias") {
            continue;
        }
        if let Some(id) = item.get("idVariable").and_then(|v| v.as_i64()) {
            if normalized.contains("superior") && upper_id.is_none() {
                upper_id = Some(id);
            }
            if normalized.contains("inferior") && lower_id.is_none() {
                lower_id = Some(id);
            }
        }
    }

    let (Some(lower_id), Some(upper_id)) = (lower_id, upper_id) else {
        return None;
    };

    let lower_series = fetch_series(http, redis, lower_id, 200).await?;
    let upper_series = fetch_series(http, redis, upper_id, 200).await?;

    let mut dates: Vec<String> = lower_series
        .keys()
        .filter(|date| upper_series.contains_key(*date))
        .cloned()
        .collect();
    dates.sort();
    if dates.is_empty() {
        return None;
    }

    let current_date = dates.last()?.clone();
    let lower_val = *lower_series.get(&current_date)?;
    let upper_val = *upper_series.get(&current_date)?;

    let prev_date = dates.iter().rev().nth(1).cloned();
    let lower_change_pct = prev_date
        .as_ref()
        .and_then(|d| lower_series.get(d))
        .and_then(|prev| pct_change(lower_val, *prev));
    let upper_change_pct = prev_date
        .as_ref()
        .and_then(|d| upper_series.get(d))
        .and_then(|prev| pct_change(upper_val, *prev));

    Some(BandLimits {
        lower: Some(lower_val),
        upper: Some(upper_val),
        lower_change_pct,
        upper_change_pct,
        date: Some(current_date),
    })
}

async fn bcra_list_variables(
    http: &reqwest::Client,
    redis: &redis::Client,
    category: Option<&str>,
) -> Option<Vec<Value>> {
    let params = if category.is_some() {
        Some(&[("limit", "2000".to_string())][..])
    } else {
        None
    };

    let data = cached_get_json(
        http,
        redis,
        "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias",
        params,
        None,
        TTL_BCRA,
    )
    .await?;

    let results = data.get("results").and_then(|v| v.as_array())?.clone();
    if category.is_none() {
        return Some(results);
    }

    let cat = normalize_text(category.unwrap_or(""));
    let filtered = results
        .into_iter()
        .filter(|item| {
            item.get("categoria")
                .and_then(|v| v.as_str())
                .map(|s| normalize_text(s).contains(&cat))
                .unwrap_or(false)
        })
        .collect();
    Some(filtered)
}

async fn fetch_latest_variables(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<Vec<(String, String, String)>> {
    let variables = bcra_list_variables(http, redis, Some("Principales Variables")).await?;
    let mut out = Vec::new();
    for item in variables {
        let name = item.get("descripcion").and_then(|v| v.as_str()).unwrap_or("");
        let date = item
            .get("ultFechaInformada")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let value = item.get("ultValorInformado");
        if name.is_empty() || date.is_empty() || value.is_none() {
            continue;
        }
        let value_str = match value.unwrap() {
            Value::String(s) => s.clone(),
            Value::Number(num) => num.to_string(),
            _ => continue,
        };
        out.push((name.to_string(), value_str, date.to_string()));
    }
    Some(out)
}

fn format_bcra_variables(
    variables: &[(String, String, String)],
    itcrm: Option<(f64, String)>,
    tcrm: Option<f64>,
) -> String {
    if variables.is_empty() {
        return "No se pudieron obtener las variables del BCRA".to_string();
    }

    let specs: Vec<(Regex, Box<dyn Fn(&str) -> String>)> = vec![
        (
            Regex::new(r"base\s*monetaria").unwrap(),
            Box::new(|v| format!("🏦 Base monetaria: ${} mill. pesos", format_value(v, false))),
        ),
        (
            Regex::new(r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual")
                .unwrap(),
            Box::new(|v| format!("📈 Inflación mensual: {}", format_value(v, true))),
        ),
        (
            Regex::new(r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual")
                .unwrap(),
            Box::new(|v| format!("📊 Inflación interanual: {}", format_value(v, true))),
        ),
        (
            Regex::new(r"tamar").unwrap(),
            Box::new(|v| format!("📈 TAMAR: {}", format_value(v, true))),
        ),
        (
            Regex::new(r"badlar").unwrap(),
            Box::new(|v| format!("📊 BADLAR: {}", format_value(v, true))),
        ),
        (
            Regex::new(r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor").unwrap(),
            Box::new(|v| format!("💵 Dólar minorista: ${v}")),
        ),
        (
            Regex::new(r"tipo.*cambio.*mayorista").unwrap(),
            Box::new(|v| format!("💱 Dólar mayorista: ${v}")),
        ),
        (
            Regex::new(r"unidad.*valor.*adquisitivo|\buva\b").unwrap(),
            Box::new(|v| format!("💰 UVA: ${v}")),
        ),
        (
            Regex::new(r"coeficiente.*estabilizacion.*referencia|\bcer\b").unwrap(),
            Box::new(|v| format!("📊 CER: {v}")),
        ),
        (
            Regex::new(r"reservas.*internacionales").unwrap(),
            Box::new(|v| format!("🏛️ Reservas: USD {} millones", format_value(v, false))),
        ),
    ];

    let mut lines = Vec::new();
    lines.push("📊 Variables principales BCRA".to_string());
    lines.push("".to_string());

    for (pattern, formatter) in specs {
        for (name, value, date) in variables {
            if pattern.is_match(&normalize_text(name)) {
                let mut line = formatter(value);
                if !date.is_empty() {
                    line.push_str(&format!(" ({})", to_ddmmyy(date)));
                }
                lines.push(line);
                break;
            }
        }
    }

    if let Some((value, date)) = itcrm {
        lines.push(format!("📐 TCRM: {:.2} ({})", value, date));
    }
    if let Some(value) = tcrm {
        lines.push(format!("🧮 TCRM 100: {:.2}", value));
    }

    lines.join("\n")
}

async fn fetch_series(
    http: &reqwest::Client,
    redis: &redis::Client,
    var_id: i64,
    limit: usize,
) -> Option<std::collections::HashMap<String, f64>> {
    let params = Some(&[("limit", limit.to_string())][..]);
    let url = format!("https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/{var_id}");
    let data = cached_get_json(http, redis, &url, params, None, TTL_BCRA).await?;
    let results = data.get("results").and_then(|v| v.as_array())?;

    let mut series = std::collections::HashMap::new();
    for entry in results {
        if let Some(detalle) = entry.get("detalle").and_then(|v| v.as_array()) {
            for row in detalle {
                let date = row.get("fecha").and_then(|v| v.as_str()).unwrap_or("");
                let value = row.get("valor").and_then(|v| v.as_f64());
                if !date.is_empty() {
                    if let Some(val) = value {
                        series.insert(date.to_string(), val);
                    }
                }
            }
        }
    }
    Some(series)
}

fn pct_change(current: f64, prev: f64) -> Option<f64> {
    if prev == 0.0 {
        return None;
    }
    Some(((current - prev) / prev) * 100.0)
}

fn format_value(value_str: &str, is_percentage: bool) -> String {
    let mut clean = value_str.replace('.', "").replace(',', ".");
    if is_percentage {
        clean = value_str.replace(',', ".");
    }
    if let Ok(num) = clean.parse::<f64>() {
        if is_percentage {
            return if num >= 10.0 {
                format!("{num:.1}%")
            } else {
                format!("{num:.2}%")
            };
        }
        if num >= 1_000_000.0 {
            return format!("{:.0}", num / 1000.0).replace(',', ".");
        }
        if num >= 1000.0 {
            return format!("{num:.0}").replace(',', ".");
        }
        return format!("{num:.2}").replace('.', ",");
    }
    if is_percentage {
        format!("{}%", value_str)
    } else {
        value_str.to_string()
    }
}

fn normalize_text(value: &str) -> String {
    value
        .nfkd()
        .collect::<String>()
        .chars()
        .filter(|c| c.is_ascii())
        .collect::<String>()
        .to_lowercase()
}

fn to_ddmmyy(date_iso: &str) -> String {
    let parts: Vec<&str> = date_iso.split('-').collect();
    if parts.len() >= 3 {
        let year = &parts[0][2..];
        return format!("{}/{}/{}", parts[2], parts[1], year);
    }
    date_iso.to_string()
}

async fn get_latest_itcrm_value_and_date(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<(f64, String)> {
    if let Ok(Some(raw)) = get_cached_string(redis, "latest_itcrm_details").await {
        if let Ok(value) = serde_json::from_str::<Value>(&raw) {
            if let (Some(val), Some(date)) = (
                value.get("value").and_then(|v| v.as_f64()),
                value.get("date").and_then(|v| v.as_str()),
            ) {
                return Some((val, date.to_string()));
            }
        }
    }

    let bytes = http.get(ITCRM_URL).send().await.ok()?.bytes().await.ok()?;
    let cursor = std::io::Cursor::new(bytes);
    let mut workbook: calamine::Xlsx<_> = calamine::Xlsx::new(cursor).ok()?;
    let sheet_name = workbook.sheet_names().get(0).cloned()?;
    let range = workbook.worksheet_range(&sheet_name).ok()?;

    for row in range.rows().rev() {
        let date_cell = row.get(0);
        let value_cell = row.get(1);
        if let Some(value_cell) = value_cell {
            if let Some(val) = parse_numeric_cell(value_cell) {
                let date_str = date_cell
                    .and_then(|d| parse_date_cell(d))
                    .unwrap_or_else(|| "".to_string());
                let payload = serde_json::json!({"value": val, "date": date_str});
                let _ = cache_string(redis, "latest_itcrm_details", &payload.to_string(), 1800).await;
                return Some((val, date_str));
            }
        }
    }

    None
}

async fn get_tcrm_100(http: &reqwest::Client, redis: &redis::Client) -> Option<f64> {
    if let Ok(Some(raw)) = get_cached_string(redis, "tcrm_100").await {
        if let Ok(value) = serde_json::from_str::<Value>(&raw) {
            if let Some(val) = value.get("data").and_then(|v| v.as_f64()) {
                return Some(val);
            }
        }
    }

    let (itcrm_value, itcrm_date) = get_latest_itcrm_value_and_date(http, redis).await?;
    let date_iso = to_iso_date(&itcrm_date)?;
    let mayorista = get_variable_value_for_date(http, redis, "tipo de cambio mayorista", &date_iso).await?;
    if itcrm_value == 0.0 {
        return None;
    }
    let result = mayorista * 100.0 / itcrm_value;

    let payload = serde_json::json!({"timestamp": chrono::Utc::now().timestamp(), "data": result});
    let _ = cache_string(redis, "tcrm_100", &payload.to_string(), 300).await;

    Some(result)
}

async fn get_variable_value_for_date(
    http: &reqwest::Client,
    redis: &redis::Client,
    desc_substr: &str,
    date_iso: &str,
) -> Option<f64> {
    let vars = bcra_list_variables(http, redis, Some("Principales Variables")).await?;
    let target = normalize_text(desc_substr);
    let mut var_id = None;
    for entry in vars {
        let desc = entry.get("descripcion").and_then(|v| v.as_str()).unwrap_or("");
        if normalize_text(desc).contains(&target) {
            var_id = entry.get("idVariable").and_then(|v| v.as_i64());
            break;
        }
    }
    let var_id = var_id?;
    let params = Some(&[
        ("desde", date_iso.to_string()),
        ("hasta", date_iso.to_string()),
        ("limit", "1".to_string()),
    ][..]);
    let url = format!("https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/{var_id}");
    let data = cached_get_json(http, redis, &url, params, None, TTL_BCRA).await?;
    let results = data.get("results").and_then(|v| v.as_array())?;
    if results.is_empty() {
        return None;
    }
    let detalle = results[0].get("detalle").and_then(|v| v.as_array())?;
    let row = detalle.get(0)?;
    row.get("valor").and_then(|v| v.as_f64())
}

fn parse_numeric_cell(cell: &calamine::Data) -> Option<f64> {
    match cell {
        calamine::Data::Float(f) => Some(*f),
        calamine::Data::Int(i) => Some(*i as f64),
        calamine::Data::String(s) => s.replace('.', "").replace(',', ".").parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_date_cell(cell: &calamine::Data) -> Option<String> {
    match cell {
        calamine::Data::String(s) => Some(normalize_date_str(s)),
        _ => None,
    }
}

fn normalize_date_str(input: &str) -> String {
    let cleaned = input.trim();
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"] {
        if let Ok(dt) = chrono::NaiveDate::parse_from_str(cleaned, fmt) {
            return format!("{}/{}/{}", dt.day(), dt.month(), dt.year() % 100);
        }
    }
    cleaned.to_string()
}

fn to_iso_date(ddmmyy: &str) -> Option<String> {
    let parts: Vec<&str> = ddmmyy.split('/').collect();
    if parts.len() != 3 {
        return None;
    }
    let day = parts[0];
    let month = parts[1];
    let year = parts[2];
    let year_full = if year.len() == 2 {
        format!("20{year}")
    } else {
        year.to_string()
    };
    Some(format!("{}-{}-{}", year_full, month, day))
}

async fn get_cached_string(
    redis: &redis::Client,
    key: &str,
) -> redis::RedisResult<Option<String>> {
    let mut conn = redis.get_multiplexed_async_connection().await?;
    redis_get_string(&mut conn, key).await
}

async fn cache_string(
    redis: &redis::Client,
    key: &str,
    value: &str,
    ttl_seconds: u64,
) -> redis::RedisResult<bool> {
    let mut conn = redis.get_multiplexed_async_connection().await?;
    redis_setex_string(&mut conn, key, ttl_seconds, value).await
}
