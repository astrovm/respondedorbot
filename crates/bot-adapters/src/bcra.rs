//! BCRA, BondTerminal, and ITCRM adapters with legacy-compatible Redis payloads.

use std::{collections::BTreeMap, io::Cursor, thread, time::Duration};

use bot_core::{
    bcra::{BcraBands, BcraSnapshot, BcraVariable, CountryRisk, ItcrmDetails, render_bcra},
    cache_policy::request_cache_history_key,
    locale::Locale,
};
use calamine::{Data, Reader, Xlsx};
use reqwest::blocking::Client;
use serde_json::{Map, Value, json};
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

use crate::dollar::{hour_key, tcrm_history};
use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_json_string, python_request_cache_key,
};

const BCRA_URL: &str = "https://api.bcra.gob.ar/estadisticas/v4.0";
const RISK_URL: &str = "https://bondterminal.com/api/riesgo-pais";
const ITCRM_URL: &str = "https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/ITCRMSerie.xlsx";
const TTL: i64 = 300;
const ITCRM_CACHE_TTL_SECONDS: i64 = 1_800;
/// The whole ITCRM series, for long `/dolar` periods. BCRA adds a day at a
/// time, so a day-old copy is fresh enough; a week-old one still serves.
const ITCRM_FULL_HISTORY_KEY: &str = "itcrm_full_history:v1";
const ITCRM_FULL_HISTORY_FRESH_SECONDS: i64 = 86_400;
const ITCRM_FULL_HISTORY_TTL_SECONDS: i64 = 7 * 86_400;
/// Past days do not change, so a looked-up wholesale rate keeps for a day.
const BASELINE_TTL_SECONDS: i64 = 86_400;
/// BCRA publishes the ITCRM about a month late, and wholesale skips
/// weekends and holidays. Older values stand in up to these gaps.
const ITCRM_MAX_GAP_DAYS: i64 = 45;
const WHOLESALE_LOOKBACK_DAYS: i64 = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BcraRequest {
    Variables,
    Series {
        id: i64,
        limit: i64,
    },
    Value {
        id: i64,
        date: String,
    },
    /// The latest value from `from` to `to`, both ISO dates.
    Latest {
        id: i64,
        from: String,
        to: String,
    },
    CountryRisk,
    Itcrm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: Vec<u8>,
}

pub use crate::http_client::TransportFailureKind;
use crate::http_client::classify_error;

pub trait BcraTransport {
    fn get(&self, request: &BcraRequest) -> Result<HttpResponse, TransportFailureKind>;
    fn before_retry(&self) {}
}

pub struct ReqwestBcraTransport {
    client: Client,
    bcra_url: String,
    risk_url: String,
    itcrm_url: String,
}

impl ReqwestBcraTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        static CLIENT: OnceLock<Client> = OnceLock::new();
        crate::http_client::shared_client(&CLIENT, || {
            Client::builder().timeout(Duration::from_secs(10)).build()
        })
        .map(|client| Self {
            client,
            bcra_url: BCRA_URL.to_owned(),
            risk_url: RISK_URL.to_owned(),
            itcrm_url: ITCRM_URL.to_owned(),
        })
        .map_err(|_| TransportFailureKind::Request)
    }

    #[cfg(test)]
    fn with_urls(
        bcra_url: &str,
        risk_url: &str,
        itcrm_url: &str,
    ) -> Result<Self, TransportFailureKind> {
        Self::new().map(|mut transport| {
            transport.bcra_url = bcra_url.trim_end_matches('/').to_owned();
            transport.risk_url = risk_url.to_owned();
            transport.itcrm_url = itcrm_url.to_owned();
            transport
        })
    }
}

impl BcraTransport for ReqwestBcraTransport {
    fn get(&self, request: &BcraRequest) -> Result<HttpResponse, TransportFailureKind> {
        let request = match request {
            BcraRequest::Variables => self
                .client
                .get(format!("{}/monetarias", self.bcra_url))
                .query(&[("limit", "2000")]),
            BcraRequest::Series { id, limit } => self
                .client
                .get(format!("{}/monetarias/{id}", self.bcra_url))
                .query(&[("limit", limit.to_string())]),
            BcraRequest::Value { id, date } => self
                .client
                .get(format!("{}/monetarias/{id}", self.bcra_url))
                .query(&[
                    ("desde", date.as_str()),
                    ("hasta", date.as_str()),
                    ("limit", "1"),
                ]),
            BcraRequest::Latest { id, from, to } => self
                .client
                .get(format!("{}/monetarias/{id}", self.bcra_url))
                .query(&[
                    ("desde", from.as_str()),
                    ("hasta", to.as_str()),
                    ("limit", "1"),
                ]),
            BcraRequest::CountryRisk => self.client.get(&self.risk_url),
            BcraRequest::Itcrm => self.client.get(&self.itcrm_url),
        };
        let response = request.send().map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .bytes()
            .map(|body| HttpResponse {
                status_code,
                body: body.to_vec(),
            })
            .map_err(classify_error)
    }
    fn before_retry(&self) {
        thread::sleep(Duration::from_millis(500));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BcraLoad {
    pub text: Option<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DollarReferencesLoad {
    pub tcrm_100: Option<f64>,
    pub tcrm_100_previous_day: Option<f64>,
    pub bands: Option<BcraBands>,
    pub itcrm: Option<ItcrmDetails>,
    pub diagnostics: Vec<String>,
}

fn norm(value: &str) -> String {
    value
        .nfkd()
        .filter(char::is_ascii)
        .collect::<String>()
        .to_lowercase()
}

fn floor(value: i64, divisor: i64) -> i64 {
    let (q, r) = (value / divisor, value % divisor);
    if r < 0 { q - 1 } else { q }
}

fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = floor(z, 146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn days(year: i64, month: i64, day: i64) -> i64 {
    let y = year - i64::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + (153 * mp + 2) / 5 + day - 1 - 719_468
}

pub(crate) fn parse_date(value: &str) -> Option<i64> {
    let value = value.trim().split(['T', ' ']).next()?;
    if let Some((y, rest)) = value.split_once('-') {
        let (m, d) = rest.split_once('-')?;
        return Some(days(y.parse().ok()?, m.parse().ok()?, d.parse().ok()?));
    }
    let mut p = value.split('/');
    let (d, m, mut y) = (
        p.next()?.parse().ok()?,
        p.next()?.parse().ok()?,
        p.next()?.parse().ok()?,
    );
    if y < 100 {
        y += 2_000;
    }
    Some(days(y, m, d))
}

pub(crate) fn iso(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{y:04}-{m:02}-{d:02}")
}
fn display(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{d:02}/{m:02}/{:02}", y.rem_euclid(100))
}
pub(crate) fn ba_day(now: i64) -> i64 {
    floor(now.saturating_sub(10_800), 86_400)
}
fn now_iso(now: i64) -> String {
    let (y, m, d) = civil(floor(now, 86_400));
    let s = now.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        s / 3600,
        (s % 3600) / 60,
        s % 60
    )
}

fn number(value: Option<&Value>) -> Option<f64> {
    let value = match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s
            .parse()
            .or_else(|_| s.replace('.', "").replace(',', ".").parse())
            .ok(),
        _ => None,
    }?;
    value.is_finite().then_some(value)
}

fn spanish(value: f64) -> String {
    let rendered = format!("{value:.2}");
    let (whole, decimals) = rendered.split_once('.').unwrap_or((&rendered, "00"));
    let mut grouped = String::new();
    for (index, character) in whole.chars().enumerate() {
        if index > 0 && (whole.len() - index).is_multiple_of(3) {
            grouped.push('.');
        }
        grouped.push(character);
    }
    format!("{grouped},{decimals}")
}

fn arguments(request: &BcraRequest) -> Option<String> {
    match request {
        BcraRequest::Variables => Some(format!(
            "{{\"api_url\": \"{BCRA_URL}/monetarias\", \"headers\": null, \"parameters\": {{\"limit\": \"2000\"}}}}"
        )),
        BcraRequest::Series { id, limit } => Some(format!(
            "{{\"api_url\": \"{BCRA_URL}/monetarias/{id}\", \"headers\": null, \"parameters\": {{\"limit\": {}}}}}",
            python_json_string(&limit.to_string())
        )),
        BcraRequest::Value { id, date } => Some(format!(
            "{{\"api_url\": \"{BCRA_URL}/monetarias/{id}\", \"headers\": null, \"parameters\": {{\"desde\": {}, \"hasta\": {}, \"limit\": \"1\"}}}}",
            python_json_string(date),
            python_json_string(date)
        )),
        BcraRequest::Latest { id, from, to } => Some(format!(
            "{{\"api_url\": \"{BCRA_URL}/monetarias/{id}\", \"headers\": null, \"parameters\": {{\"desde\": {}, \"hasta\": {}, \"limit\": \"1\"}}}}",
            python_json_string(from),
            python_json_string(to)
        )),
        BcraRequest::CountryRisk => Some(format!(
            "{{\"api_url\": \"{RISK_URL}\", \"headers\": null, \"parameters\": null}}"
        )),
        BcraRequest::Itcrm => None,
    }
}

fn request_json<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    request: &BcraRequest,
    now: i64,
    diagnostics: &mut Vec<String>,
) -> Option<Value> {
    request_json_for(transport, cache, request, TTL, now, diagnostics)
}

fn request_json_for<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    request: &BcraRequest,
    ttl_seconds: i64,
    now: i64,
    diagnostics: &mut Vec<String>,
) -> Option<Value> {
    let key = python_request_cache_key(&arguments(request)?);
    let load = load_cached_json(
        cache,
        &key,
        ttl_seconds,
        now,
        &format!("BCRA {request:?}"),
        || {
            transport
                .get(request)
                .map_err(|e| format!("transport {e:?}"))
                .and_then(|r| {
                    String::from_utf8(r.body)
                        .map(|body| JsonHttpResponse {
                            status_code: r.status_code,
                            body,
                        })
                        .map_err(|e| format!("invalid UTF-8: {e}"))
                })
        },
        || transport.before_retry(),
    );
    diagnostics.extend(load.diagnostics);
    load.data
}

fn items(value: &Value) -> Vec<&Map<String, Value>> {
    value
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|v| {
            v.get("categoria")
                .and_then(Value::as_str)
                .is_some_and(|c| norm(c).contains("principales variables"))
        })
        .collect()
}

fn latest(items: &[&Map<String, Value>]) -> Vec<BcraVariable> {
    let mut variables = Vec::<BcraVariable>::new();
    for variable in items.iter().filter_map(|v| {
        let description = v.get("descripcion")?.as_str()?.trim();
        let raw_date = v.get("ultFechaInformada")?.as_str()?.trim();
        let value = number(v.get("ultValorInformado"))?;
        (!description.is_empty() && !raw_date.is_empty()).then(|| BcraVariable {
            description: description.to_owned(),
            value: spanish(value),
            date: parse_date(raw_date).map_or_else(|| raw_date.to_owned(), display),
        })
    }) {
        if let Some(existing) = variables
            .iter_mut()
            .find(|existing| existing.description == variable.description)
        {
            *existing = variable;
        } else {
            variables.push(variable);
        }
    }
    variables
}

fn variables_value(variables: &[BcraVariable]) -> Value {
    Value::Object(Map::from_iter(variables.iter().map(|v| {
        (
            v.description.clone(),
            json!({"value":v.value,"date":v.date}),
        )
    })))
}
fn parse_variables(value: &Value) -> Vec<BcraVariable> {
    value
        .as_object()
        .into_iter()
        .flat_map(|v| v.iter())
        .filter(|(k, _)| !k.starts_with('_'))
        .filter_map(|(k, v)| {
            Some(BcraVariable {
                description: k.clone(),
                value: v.get("value")?.as_str()?.to_owned(),
                date: v
                    .get("date")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            })
        })
        .collect()
}

fn read<C: RequestCache>(cache: &mut C, key: &str, d: &mut Vec<String>) -> Option<Value> {
    match cache.get(key) {
        Ok(Some(raw)) => match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v.get("data").cloned().or(Some(v)),
            Err(e) => {
                d.push(format!("invalid BCRA cache {key}: {e}"));
                None
            }
        },
        Ok(None) => None,
        Err(e) => {
            d.push(format!("could not read BCRA cache {key}: {e}"));
            None
        }
    }
}
fn store<C: RequestCache>(
    cache: &mut C,
    key: &str,
    data: &Value,
    ttl: i64,
    grace: i64,
    now: i64,
    d: &mut Vec<String>,
) {
    let value = json!({"data":data,"fetched_at":now_iso(now)}).to_string();
    for (key, ttl) in [
        (key.to_owned(), ttl),
        (format!("{key}:last_success"), ttl + grace),
    ] {
        if let Err(e) = cache.set(&key, &value, ttl) {
            d.push(format!("could not write BCRA cache {key}: {e}"));
        }
    }
}

fn series(value: &Value) -> BTreeMap<i64, f64> {
    let mut out = BTreeMap::new();
    for row in value
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("detalle").and_then(Value::as_array))
        .flatten()
    {
        if let (Some(day), Some(value)) = (
            row.get("fecha")
                .and_then(Value::as_str)
                .and_then(parse_date),
            number(row.get("valor")),
        ) {
            out.insert(day, value);
        }
    }
    out
}

pub(crate) fn current_band_date(value: &Value, now: i64) -> bool {
    value
        .get("date_iso")
        .and_then(Value::as_str)
        .or_else(|| value.get("date").and_then(Value::as_str))
        .and_then(parse_date)
        .is_some_and(|day| day <= ba_day(now))
}

fn bands<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    vars: &[&Map<String, Value>],
    now: i64,
    d: &mut Vec<String>,
) -> Option<BcraBands> {
    if let Some(v) = read(cache, "bcra_currency_band_limits", d)
        && current_band_date(&v, now)
    {
        return Some(BcraBands {
            lower: number(v.get("lower"))?,
            upper: number(v.get("upper"))?,
            date: v.get("date")?.as_str()?.to_owned(),
            lower_change: number(v.get("lower_change_pct")),
            upper_change: number(v.get("upper_change_pct")),
        });
    }
    let mut ids = (None, None);
    for v in vars {
        let desc = norm(
            v.get("descripcion")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        let id = v.get("idVariable").and_then(Value::as_i64);
        if desc.contains("bandas cambiarias") {
            if desc.contains("inferior") {
                ids.0 = id;
            } else if desc.contains("superior") {
                ids.1 = id;
            }
        }
    }
    let lower = series(&request_json(
        transport,
        cache,
        &BcraRequest::Series {
            id: ids.0?,
            limit: 200,
        },
        now,
        d,
    )?);
    let upper = series(&request_json(
        transport,
        cache,
        &BcraRequest::Series {
            id: ids.1?,
            limit: 200,
        },
        now,
        d,
    )?);
    let common = lower
        .keys()
        .filter(|day| upper.contains_key(day) && **day <= ba_day(now))
        .copied()
        .collect::<Vec<_>>();
    let current = *common.last()?;
    let previous = common.iter().rev().nth(1).copied();
    let (lo, up) = (*lower.get(&current)?, *upper.get(&current)?);
    let change = |map: &BTreeMap<i64, f64>, current: f64| {
        previous
            .and_then(|p| map.get(&p).copied())
            .filter(|v| *v != 0.0)
            .map(|v| (current - v) / v * 100.0)
    };
    let result = BcraBands {
        lower: lo,
        upper: up,
        date: display(current),
        lower_change: change(&lower, lo),
        upper_change: change(&upper, up),
    };
    store(
        cache,
        "bcra_currency_band_limits",
        &json!({"date":result.date,"date_iso":iso(current),"lower":lo,"upper":up,"lower_change_pct":result.lower_change,"upper_change_pct":result.upper_change}),
        TTL,
        3600,
        now,
        d,
    );
    Some(result)
}

/// Load the BCRA values consumed by the dollar command independently.
///
/// Dollar commands use this so currency bands and TCRM do not depend on
/// `/bcra` having populated the shared cache first.
#[must_use]
pub fn load_dollar_references<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
) -> DollarReferencesLoad {
    let mut diagnostics = Vec::new();
    let list = request_json(
        transport,
        cache,
        &BcraRequest::Variables,
        now,
        &mut diagnostics,
    );
    let items = list.as_ref().map(items).unwrap_or_default();
    let variables = latest(&items);
    let bands = bands(transport, cache, &items, now, &mut diagnostics);
    let mut indices = None;
    let itcrm = itcrm(transport, cache, now, &mut indices, &mut diagnostics);
    persist_market(cache, &variables, itcrm.as_ref(), now, &mut diagnostics);
    let tcrm_100 = calculate_tcrm(&variables, itcrm.as_ref());
    let tcrm_100_previous_day = if tcrm_100.is_some()
        && tcrm_history(cache, now.saturating_sub(86_400), &mut diagnostics).is_none()
    {
        tcrm_daily_baseline(transport, cache, &items, indices, now, &mut diagnostics)
    } else {
        None
    };
    DollarReferencesLoad {
        tcrm_100,
        tcrm_100_previous_day,
        bands,
        itcrm,
        diagnostics,
    }
}

fn risk_label(value: &str) -> Option<String> {
    let day = parse_date(value.get(..10)?)?;
    let time = value.get(11..16)?;
    let (h, m) = time.split_once(':')?;
    let minutes = h.parse::<i64>().ok()? * 60 + m.parse::<i64>().ok()? - 180;
    let adjusted = day + floor(minutes, 1440);
    let (_, month, date) = civil(adjusted);
    Some(format!(
        "{date:02}/{month:02} {:02}:{:02}",
        minutes.rem_euclid(1440) / 60,
        minutes.rem_euclid(1440) % 60
    ))
}
fn risk<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
    d: &mut Vec<String>,
) -> Option<CountryRisk> {
    let v = request_json(transport, cache, &BcraRequest::CountryRisk, now, d)?;
    Some(CountryRisk {
        value_bps: number(v.get("weightedSpreadBps"))?,
        delta_one_day: v.get("deltas").and_then(|v| number(v.get("oneDay"))),
        valuation_label: ["valuationDate", "asOf", "lastDataTickIso"]
            .iter()
            .find_map(|k| v.get(*k).and_then(Value::as_str))
            .and_then(risk_label),
    })
}

fn cell_date(value: &Data) -> Option<i64> {
    match value {
        Data::DateTime(v) => {
            let (y, m, d, _, _, _, _) = v.to_ymd_hms_milli();
            Some(days(i64::from(y), i64::from(m), i64::from(d)))
        }
        Data::String(v) | Data::DateTimeIso(v) => parse_date(v),
        _ => None,
    }
}
fn cell_number(value: &Data) -> Option<f64> {
    let value = match value {
        Data::Float(v) => Some(*v),
        Data::Int(v) => Some(*v as f64),
        Data::String(v) => v
            .parse()
            .or_else(|_| v.replace('.', "").replace(',', ".").parse())
            .ok(),
        _ => None,
    }?;
    value.is_finite().then_some(value)
}
fn parse_workbook(bytes: &[u8]) -> Result<BTreeMap<i64, f64>, String> {
    let mut book = Xlsx::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let range = book
        .worksheet_range_at(0)
        .ok_or_else(|| "ITCRM workbook has no sheet".to_owned())?
        .map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for row in range.rows() {
        if let (Some(day), Some(value)) = (
            row.first().and_then(cell_date),
            row.get(1).and_then(cell_number),
        ) {
            out.insert(day, value);
        }
    }
    if out.is_empty() {
        Err("ITCRM workbook has no usable rows".to_owned())
    } else {
        Ok(out)
    }
}
fn itcrm<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
    history: &mut Option<BTreeMap<i64, f64>>,
    d: &mut Vec<String>,
) -> Option<ItcrmDetails> {
    if let Some(v) = read(cache, "latest_itcrm_details", d)
        && let Some(value) = number(v.get("value")).filter(|value| *value > 0.0)
        && let Some(date) = v.get("date").and_then(Value::as_str)
        && parse_date(date).is_some_and(|date| date <= ba_day(now))
    {
        return Some(ItcrmDetails {
            value,
            date: date.to_owned(),
        });
    }
    let values = itcrm_series(transport, cache, now, d)?;
    let result = values
        .range(..=ba_day(now))
        .next_back()
        .map(|(day, value)| ItcrmDetails {
            value: *value,
            date: display(*day),
        });
    *history = Some(values);
    let result = result?;
    if let Err(error) = cache.set(
        "latest_itcrm_details",
        &json!({"value":result.value,"date":result.date}).to_string(),
        ITCRM_CACHE_TTL_SECONDS,
    ) {
        d.push(format!(
            "could not write BCRA cache latest_itcrm_details: {error}"
        ));
    }
    Some(result)
}

fn itcrm_series<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
    d: &mut Vec<String>,
) -> Option<BTreeMap<i64, f64>> {
    if let Some(cached) = read(cache, "itcrm_daily_history", d) {
        let values = series(&cached);
        if !values.is_empty() {
            return Some(values);
        }
    }
    download_itcrm(transport, cache, now, d)
}

/// Downloads the ITCRM workbook and refreshes both caches: the last 32 days
/// for the daily TCRM, and the whole series for long periods.
fn download_itcrm<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
    d: &mut Vec<String>,
) -> Option<BTreeMap<i64, f64>> {
    let response = match transport.get(&BcraRequest::Itcrm) {
        Ok(r) if r.status_code < 400 => r,
        Ok(r) => {
            d.push(format!("ITCRM HTTP {}", r.status_code));
            return None;
        }
        Err(e) => {
            d.push(format!("ITCRM transport {e:?}"));
            return None;
        }
    };
    let values = match parse_workbook(&response.body) {
        Ok(v) => v,
        Err(e) => {
            d.push(e);
            return None;
        }
    };
    let rows = values
        .iter()
        .rev()
        .take(32)
        .map(|(day, value)| json!({"fecha":iso(*day),"valor":value}))
        .collect::<Vec<_>>();
    if let Err(error) = cache.set(
        "itcrm_daily_history",
        &json!({"results":[{"detalle":rows}]}).to_string(),
        ITCRM_CACHE_TTL_SECONDS,
    ) {
        d.push(format!(
            "could not write BCRA cache itcrm_daily_history: {error}"
        ));
    }
    let pairs = values
        .iter()
        .map(|(day, value)| json!([day, value]))
        .collect::<Vec<_>>();
    if let Err(error) = cache.set(
        ITCRM_FULL_HISTORY_KEY,
        &json!({"timestamp": now, "data": pairs}).to_string(),
        ITCRM_FULL_HISTORY_TTL_SECONDS,
    ) {
        d.push(format!(
            "could not write BCRA cache {ITCRM_FULL_HISTORY_KEY}: {error}"
        ));
    }
    Some(values)
}

/// The whole ITCRM series since 1997. A stale copy serves when BCRA fails.
fn full_itcrm_series<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now: i64,
    d: &mut Vec<String>,
) -> Option<BTreeMap<i64, f64>> {
    let cached = cache
        .get(ITCRM_FULL_HISTORY_KEY)
        .map_err(|error| {
            d.push(format!(
                "could not read BCRA cache {ITCRM_FULL_HISTORY_KEY}: {error}"
            ));
        })
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| {
            let timestamp = value.get("timestamp")?.as_i64()?;
            let values = value
                .get("data")?
                .as_array()?
                .iter()
                .filter_map(|pair| {
                    let day = pair.get(0)?.as_i64()?;
                    let value = pair.get(1)?.as_f64()?;
                    (value.is_finite() && value > 0.0).then_some((day, value))
                })
                .collect::<BTreeMap<_, _>>();
            (!values.is_empty()).then_some((timestamp, values))
        });
    if let Some((timestamp, values)) = &cached
        && now.saturating_sub(*timestamp) < ITCRM_FULL_HISTORY_FRESH_SECONDS
    {
        return Some(values.clone());
    }
    download_itcrm(transport, cache, now, d).or_else(|| cached.map(|(_, values)| values))
}

fn wholesale_id(variables: &[&Map<String, Value>]) -> Option<i64> {
    variables
        .iter()
        .find(|variable| {
            variable
                .get("descripcion")
                .and_then(Value::as_str)
                .is_some_and(|description| norm(description).contains("tipo de cambio mayorista"))
        })?
        .get("idVariable")?
        .as_i64()
}

/// Official BCRA values on a past day, for `/dolar` periods longer than the
/// hourly snapshots: the wholesale rate (Com. A 3500, since 2002) and the
/// TCRM 100 built from it and the ITCRM of that day.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DollarBaselines {
    pub wholesale: Option<f64>,
    pub tcrm_100: Option<f64>,
    pub diagnostics: Vec<String>,
}

#[must_use]
pub fn load_dollar_baselines<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    target_unix: i64,
    now: i64,
) -> DollarBaselines {
    let mut d = Vec::new();
    let day = ba_day(target_unix);
    let wholesale = request_json(transport, cache, &BcraRequest::Variables, now, &mut d)
        .as_ref()
        .map(items)
        .and_then(|variables| wholesale_id(&variables))
        .and_then(|id| {
            let request = BcraRequest::Latest {
                id,
                from: iso(day - WHOLESALE_LOOKBACK_DAYS),
                to: iso(day),
            };
            request_json_for(
                transport,
                cache,
                &request,
                BASELINE_TTL_SECONDS,
                now,
                &mut d,
            )
        })
        .and_then(|value| series(&value).into_iter().next_back())
        .map(|(_, value)| value)
        .filter(|value| *value > 0.0);
    let tcrm_100 = wholesale.and_then(|wholesale| {
        let indices = full_itcrm_series(transport, cache, now, &mut d)?;
        let (index_day, index) = indices.range(..=day).next_back()?;
        (day - index_day <= ITCRM_MAX_GAP_DAYS)
            .then(|| tcrm_value(wholesale, *index))
            .flatten()
    });
    DollarBaselines {
        wholesale,
        tcrm_100,
        diagnostics: d,
    }
}

fn tcrm_daily_baseline<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    variables: &[&Map<String, Value>],
    indices: Option<BTreeMap<i64, f64>>,
    now: i64,
    d: &mut Vec<String>,
) -> Option<f64> {
    let id = wholesale_id(variables)?;
    let wholesale = series(&request_json(
        transport,
        cache,
        &BcraRequest::Series { id, limit: 32 },
        now,
        d,
    )?);
    let indices = indices.or_else(|| itcrm_series(transport, cache, now, d))?;
    // Daily observations carry forward across weekends and holidays.
    let day = ba_day(now.saturating_sub(86_400));
    let (_, wholesale) = wholesale.range(..=day).next_back()?;
    let (_, index) = indices.range(..=day).next_back()?;
    tcrm_value(*wholesale, *index)
}

fn tcrm_value(wholesale: f64, index: f64) -> Option<f64> {
    if wholesale <= 0.0 || index <= 0.0 {
        return None;
    }
    let result = wholesale * 100.0 / index;
    (result.is_finite() && result > 0.0).then_some(result)
}

fn calculate_tcrm(variables: &[BcraVariable], itcrm: Option<&ItcrmDetails>) -> Option<f64> {
    let wholesale = variables
        .iter()
        .find(|value| norm(&value.description).contains("tipo de cambio mayorista"))
        .and_then(|value| number(Some(&Value::String(value.value.clone()))))?;
    let index = itcrm?.value;
    tcrm_value(wholesale, index)
}

fn persist_market<C: RequestCache>(
    cache: &mut C,
    variables: &[BcraVariable],
    itcrm: Option<&ItcrmDetails>,
    now: i64,
    d: &mut Vec<String>,
) {
    let wholesale = variables
        .iter()
        .find(|v| norm(&v.description).contains("tipo de cambio mayorista"));
    if let Some(v) = wholesale.and_then(|v| parse_date(&v.date).map(|day| (v, day))) {
        let key = format!("bcra_mayorista:{}", iso(v.1));
        if let Some(value) = number(Some(&Value::String(v.0.value.clone())))
            && let Err(e) = cache.set(
                &key,
                &json!({"value":value,"date":v.0.date}).to_string(),
                86400,
            )
        {
            d.push(format!("could not write {key}: {e}"));
        }
    }
    let Some(tcrm) = calculate_tcrm(variables, itcrm) else {
        return;
    };
    let payload = json!({"timestamp":now,"data":tcrm}).to_string();
    for key in [
        "tcrm_100".to_owned(),
        request_cache_history_key(&hour_key(now), "tcrm_100"),
    ] {
        if let Err(e) = cache.set(&key, &payload, 259200) {
            d.push(format!("could not write {key}: {e}"));
        }
    }
}

#[must_use]
pub fn load_bcra<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    locale: Locale,
    now: i64,
) -> BcraLoad {
    let mut diagnostics = Vec::new();
    let cached = read(cache, "bcra_variables", &mut diagnostics);
    let (variables, list, stale) = if let Some(v) = cached {
        (parse_variables(&v), None, false)
    } else if let Some(v) = request_json(
        transport,
        cache,
        &BcraRequest::Variables,
        now,
        &mut diagnostics,
    ) {
        let vars = items(&v);
        let latest = latest(&vars);
        store(
            cache,
            "bcra_variables",
            &variables_value(&latest),
            TTL,
            21600,
            now,
            &mut diagnostics,
        );
        (latest, Some(v), false)
    } else if let Some(v) = read(cache, "bcra_variables:last_success", &mut diagnostics) {
        (parse_variables(&v), None, true)
    } else {
        (Vec::new(), None, false)
    };
    let list = list.or_else(|| {
        request_json(
            transport,
            cache,
            &BcraRequest::Variables,
            now,
            &mut diagnostics,
        )
    });
    let vars = list.as_ref().map(items).unwrap_or_default();
    let bands = bands(transport, cache, &vars, now, &mut diagnostics);
    let country_risk = risk(transport, cache, now, &mut diagnostics);
    let itcrm = itcrm(transport, cache, now, &mut None, &mut diagnostics);
    persist_market(cache, &variables, itcrm.as_ref(), now, &mut diagnostics);
    let snapshot = BcraSnapshot {
        variables,
        bands,
        itcrm,
        country_risk,
        stale,
    };
    let text =
        (!snapshot.variables.is_empty()).then(|| render_bcra(&snapshot, locale, ba_day(now)));
    BcraLoad { text, diagnostics }
}

#[cfg(test)]
mod tests {
    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    use super::*;
    use base64::Engine as _;
    use std::{
        cell::RefCell,
        collections::{HashMap, VecDeque},
        io::{Read, Write},
        net::TcpListener,
        thread,
    };
    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
        fail_sets: bool,
        fail_gets: bool,
    }
    impl RequestCache for Cache {
        type Error = &'static str;
        fn get(&mut self, k: &str) -> Result<Option<String>, Self::Error> {
            if self.fail_gets {
                Err("synthetic cache read failure")
            } else {
                Ok(self.values.get(k).cloned())
            }
        }
        fn set(&mut self, k: &str, v: &str, _: i64) -> Result<(), Self::Error> {
            if self.fail_sets {
                return Err("synthetic cache write failure");
            }
            self.values.insert(k.to_owned(), v.to_owned());
            Ok(())
        }

        fn take(&mut self, k: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.values.remove(k))
        }

        fn claim(&mut self, k: &str, v: &str, _: i64) -> Result<bool, Self::Error> {
            if self.fail_sets {
                return Err("synthetic cache write failure");
            }
            if self.values.contains_key(k) {
                Ok(false)
            } else {
                self.values.insert(k.to_owned(), v.to_owned());
                Ok(true)
            }
        }
    }

    #[test]
    fn cache_take_removes_values() {
        let mut cache = Cache::default();
        cache.values.insert("k".to_owned(), "v".to_owned());
        assert_eq!(cache.take("k"), Ok(Some("v".to_owned())));
        assert_eq!(cache.take("k"), Ok(None));
        assert_eq!(cache.claim("k", "v", 60), Ok(true));
        assert_eq!(cache.claim("k", "v", 60), Ok(false));
    }
    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
    }
    impl BcraTransport for Transport {
        fn get(&self, _: &BcraRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }
    fn response(v: Value) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: v.to_string().into_bytes(),
        })
    }
    #[test]
    fn long_dollar_baselines_use_the_wholesale_rate_and_itcrm_of_that_day() {
        struct Recorded {
            requests: RefCell<Vec<BcraRequest>>,
            responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        }
        impl BcraTransport for Recorded {
            fn get(&self, request: &BcraRequest) -> Result<HttpResponse, TransportFailureKind> {
                self.requests.borrow_mut().push(request.clone());
                self.responses
                    .borrow_mut()
                    .pop_front()
                    .unwrap_or(Err(TransportFailureKind::Request))
            }
        }
        let variables = json!({"results":[
            {"idVariable":4,"descripcion":"Tipo de cambio minorista (promedio vendedor)","categoria":"Principales Variables"},
            {"idVariable":5,"descripcion":"Tipo de cambio mayorista de referencia","categoria":"Principales Variables"}
        ]});
        // The shape of a real BCRA answer for 2016-11-10 to 2016-11-20.
        let wholesale = json!({"status":200,"results":[{"idVariable":5,"detalle":[
            {"fecha":"2016-11-18","valor":15.4817}
        ]}]});
        let day = |iso: &str| parse_date(iso).unwrap_or_default();
        let itcrm =
            |iso: &str, now: i64| json!({"timestamp": now, "data": [[day(iso), 80.0]]}).to_string();
        // 2016-11-20 12:00 UTC, a Sunday: Friday's rate stands in.
        let target = 1_479_643_200;
        let now = 1_795_132_800;
        let transport = Recorded {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                response(variables.clone()),
                response(wholesale.clone()),
            ])),
        };
        let mut cache = Cache::default();
        cache
            .values
            .insert(ITCRM_FULL_HISTORY_KEY.to_owned(), itcrm("2016-11-01", now));
        let baselines = load_dollar_baselines(&transport, &mut cache, target, now);
        assert_eq!(baselines.wholesale, Some(15.4817));
        assert_eq!(
            baselines.tcrm_100.map(|value| (value * 1e6).round() / 1e6),
            Some(19.352_125)
        );
        assert!(
            baselines.diagnostics.is_empty(),
            "{:?}",
            baselines.diagnostics
        );
        assert_eq!(
            transport.requests.borrow()[1],
            BcraRequest::Latest {
                id: 5,
                from: "2016-11-10".to_owned(),
                to: "2016-11-20".to_owned(),
            }
        );

        // An ITCRM more than 45 days older gives no TCRM, and no wholesale
        // rate (like before 2002) gives neither.
        let transport = Recorded {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                response(variables.clone()),
                response(wholesale),
            ])),
        };
        let mut cache = Cache::default();
        cache
            .values
            .insert(ITCRM_FULL_HISTORY_KEY.to_owned(), itcrm("2016-09-01", now));
        let baselines = load_dollar_baselines(&transport, &mut cache, target, now);
        assert_eq!(
            (baselines.wholesale.is_some(), baselines.tcrm_100),
            (true, None)
        );
        let transport = Recorded {
            requests: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::from([
                response(variables),
                response(json!({"results":[{"idVariable":5,"detalle":[]}]})),
            ])),
        };
        let baselines = load_dollar_baselines(&transport, &mut Cache::default(), 0, now);
        assert_eq!(baselines.wholesale, None);
        assert_eq!(baselines.tcrm_100, None);
        // The ITCRM workbook is not downloaded when there is no rate to use.
        assert_eq!(transport.requests.borrow().len(), 2);
    }

    #[test]
    fn a_stale_full_itcrm_copy_is_refreshed_and_still_serves_when_bcra_fails() {
        let now = 1_795_132_800;
        let stale = json!({"timestamp": now - 2 * 86_400, "data": [[20_000, 90.0]]}).to_string();
        let mut cache = Cache::default();
        cache
            .values
            .insert(ITCRM_FULL_HISTORY_KEY.to_owned(), stale);
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([Err(TransportFailureKind::Timeout)])),
        };
        let mut d = Vec::new();
        let values = full_itcrm_series(&transport, &mut cache, now, &mut d);
        assert_eq!(values, Some(BTreeMap::from([(20_000, 90.0)])));
        assert_eq!(d, ["ITCRM transport Timeout"]);
    }

    #[test]
    fn cache_key_and_time_contract() {
        assert_eq!(
            python_request_cache_key(&arguments(&BcraRequest::Variables).unwrap_or_default()),
            "request_cache:a0cbcd2d36185480185569e392c8fa794529abcc8114f3550bc5dadefe48b4e5"
        );
        assert_eq!(
            risk_label("2025-10-29T15:34:00Z").as_deref(),
            Some("29/10 12:34")
        );
        assert!(parse_workbook(b"invalid").is_err());
    }

    #[test]
    fn tcrm_snapshots_use_the_same_utc_hour_as_the_dollar_reader() {
        let now = 1_725_000_000;
        let mut cache = Cache::default();
        let mut diagnostics = Vec::new();
        persist_market(
            &mut cache,
            &[BcraVariable {
                description: "Tipo de cambio mayorista".to_owned(),
                value: "1100".to_owned(),
                date: "2024-08-30".to_owned(),
            }],
            Some(&ItcrmDetails {
                value: 100.0,
                date: "2024-08-30".to_owned(),
            }),
            now,
            &mut diagnostics,
        );
        assert!(diagnostics.is_empty());
        let key = request_cache_history_key("2024-08-30-06", "tcrm_100");
        assert!(cache.values.contains_key(&key), "{:?}", cache.values.keys());
    }

    #[test]
    fn daily_tcrm_uses_only_observations_available_on_the_target_date() {
        let now = days(2024, 9, 1) * 86_400 + 15 * 3_600;
        let variable = json!({"idVariable":5,"descripcion":"Tipo de cambio mayorista"});
        let variables = variable.as_object().into_iter().collect::<Vec<_>>();
        let request = BcraRequest::Series { id: 5, limit: 32 };
        let key = python_request_cache_key(&arguments(&request).unwrap_or_default());
        for (date, wholesale, index, expected) in [
            ("2024-08-30", 1000.0, 125.0, Some(800.0)),
            ("2024-09-01", 1000.0, 125.0, None),
            ("2024-08-30", 1000.0, 0.0, None),
            ("2024-08-30", -1000.0, 125.0, None),
            ("2024-08-30", f64::MAX, 0.01, None),
        ] {
            let mut cache = Cache::default();
            cache.values.insert(
                key.clone(),
                json!({"timestamp":now,"data":{"results":[{"detalle":[
                    {"fecha":date,"valor":wholesale},
                    {"fecha":"2024-09-02","valor":5000.0}
                ]}]}})
                .to_string(),
            );
            cache.values.insert(
                "itcrm_daily_history".to_owned(),
                json!({"results":[{"detalle":[
                    {"fecha":date,"valor":index},
                    {"fecha":"2024-09-02","valor":250.0}
                ]}]})
                .to_string(),
            );
            let transport = Transport {
                responses: RefCell::new(VecDeque::new()),
            };
            let mut diagnostics = Vec::new();
            assert_eq!(
                tcrm_daily_baseline(
                    &transport,
                    &mut cache,
                    &variables,
                    None,
                    now,
                    &mut diagnostics
                ),
                expected
            );
            assert!(diagnostics.is_empty(), "{diagnostics:?}");
        }
    }

    proptest::proptest! {
        #[test]
        fn daily_tcrm_scales_with_wholesale_and_inversely_with_the_index(
            wholesale in 1.0_f64..100_000.0,
            index in 1.0_f64..1000.0,
            factor in 0.1_f64..10.0,
        ) {
            proptest::prop_assert!(tcrm_value(wholesale, index)
                .zip(tcrm_value(wholesale * factor, index * factor))
                .is_some_and(|(value, scaled)| (value - scaled).abs() <= value * 1e-12));
        }
    }

    const ITCRM_WORKBOOK: &str = "UEsDBBQAAAAIAPuUIl1bma6u5QAAAAsCAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbK2RvVLDMBCEX0WjNhOdk4KCsZ0i0AYKXuCQz7HG+hudEszbIzuBggnQUN1Iu3vfalTvJmfFmRKb4Bu5UZXctfXLeyQWRfHcyCHneA/AeiCHrEIkX5Q+JIe5HNMRIuoRjwTbqroDHXwmn9d53iHb+oF6PNksHqdyfaEksizF/mKcWY3EGK3RmIsOZ999o6yvBFWSi4cHE3lVDBJuEmblZ8A191SenUxH4hlTPqArLpgsvIU0voYwqt+X3GgZ+t5o6oI+uRJRHBNhxwNRdlYtUzk0fvU3fzEzLGPzz0W+9n/2gOW72w9QSwMEFAAAAAgA+5QiXUuDozqWAAAABQEAAAsAAABfcmVscy8ucmVsc43PPQ7CMAwF4KtEPkDdMjCgpl1YuiIuEFL3R23iyAlQbk9GihgY/fz0Wa7bza3qQRJn9hqqooS2qS+0mpSDOM0hqtzwUcOUUjghRjuRM7HgQD5vBhZnUh5lxGDsYkbCQ1keUT4N2Juq6zVI11egrq9A/9g8DLOlM9u7I59+nPhqZNnISEnDtuKTZbkxL0VGAZsadw82b1BLAwQUAAAACAD7lCJdLWvSpKMAAAD4AAAADwAAAHhsL3dvcmtib29rLnhtbI2PORKDMAxFr+LRATCkSMEYUyQNRZpMLuCACB7wMpKzHD8eCH0qLX/0vr5qP24RLyS2wTdQFSW0Wr0DzfcQZpFFzw1MKcVaSu4ndIaLENFnZQzkTMojPSRHQjPwhJjcIg9leZTOWA8boaZ/GGEcbY/n0D8d+rRBCBeT8ms82cig1erAvyq8cdhAdztdLyDWVTfkECCotrmhbqhAaiX3K7kH019QSwMEFAAAAAgA+5QiXW026XSaAAAABgEAABoAAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc43POw7CMAwG4KtEPkDdMjCgpl1YWBEXiFK3qdo8FJvX7YkYEJUYmCz/tj7Lbf/wq7pR5jkGDU1VQ9+1Z1qNlIDdnFiVjcAanEg6ILJ15A1XMVEokzFmb6S0ecJk7GImwl1d7zF/G7A11WnQkE9DA+ryTPSPHcdxtnSM9uopyI8TeI95YUckBTV5ItHwiRjfpamKCti1uPmwewFQSwMEFAAAAAgA+5QiXZR5P5S8AAAAYgEAABgAAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWyFkEEKwjAQRa9SstZMEq2ipBHFG3iCUKMNNklJhurxTUVKBcXdzP88HjNy93Bt0ZuYbPAV4ZSRnZL3EG+pMQaL3PpUkQax2wKkujFOJxo643NzCdFpzGu8Quqi0ecX5FoQjK3AaeuJkq/sqFErGcO9iNmS03oY9pwUWBHrW+vNCWPObVISlWBiNWebOeMSUEkYUqjf1OEXxalYLGflJwJZOprFaBb/zOKbeeB7xUXJ6Dp7+qkCJofC+EH1BFBLAQIUAxQAAAAIAPuUIl1bma6u5QAAAAsCAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQDFAAAAAgA+5QiXUuDozqWAAAABQEAAAsAAAAAAAAAAAAAAIABFgEAAF9yZWxzLy5yZWxzUEsBAhQDFAAAAAgA+5QiXS1r0qSjAAAA+AAAAA8AAAAAAAAAAAAAAIAB1QEAAHhsL3dvcmtib29rLnhtbFBLAQIUAxQAAAAIAPuUIl1tNul0mgAAAAYBAAAaAAAAAAAAAAAAAACAAaUCAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc1BLAQIUAxQAAAAIAPuUIl2UeT+UvAAAAGIBAAAYAAAAAAAAAAAAAACAAXcDAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWxQSwUGAAAAAAUABQBFAQAAaQQAAAAA";

    #[test]
    fn parses_the_official_workbook_shape_and_exposes_latest_itcrm() -> TestResult {
        let bytes = base64::engine::general_purpose::STANDARD.decode(ITCRM_WORKBOOK)?;
        let values = parse_workbook(&bytes)?;
        assert_eq!(values.len(), 2);
        assert_eq!(
            values.last_key_value().map(|(_, value)| *value),
            Some(1250.75)
        );

        let mut latest_itcrm = None;
        for offline in [false, true] {
            let transport = Transport {
                responses: RefCell::new(VecDeque::from([
                    response(json!({"results":[{
                        "categoria":"Principales Variables",
                        "idVariable":5,
                        "descripcion":"Tipo de cambio mayorista",
                        "ultFechaInformada":"2026-09-02",
                        "ultValorInformado":1450.0
                    }]})),
                    Ok(HttpResponse {
                        status_code: 200,
                        body: bytes.clone(),
                    }),
                    response(json!({"results":[]})),
                ])),
            };
            let mut cache = Cache {
                fail_gets: offline,
                fail_sets: offline,
                ..Cache::default()
            };
            cache.values.insert(
                "itcrm_daily_history".to_owned(),
                json!({"results":[]}).to_string(),
            );
            let load = load_dollar_references(&transport, &mut cache, 1_788_321_600);
            assert_eq!(
                load.itcrm.as_ref().map(|details| details.value),
                Some(1250.75)
            );
            assert_eq!(cache.values.contains_key("latest_itcrm_details"), !offline);
            assert_eq!(load.tcrm_100, Some(1450.0 * 100.0 / 1250.75));
            assert_eq!(load.diagnostics.is_empty(), !offline);
            latest_itcrm = load.itcrm;
        }

        let mut failing_cache = Cache {
            fail_sets: true,
            ..Cache::default()
        };
        let mut diagnostics = Vec::new();
        persist_market(
            &mut failing_cache,
            &[BcraVariable {
                description: "Tipo de cambio mayorista".to_owned(),
                value: "1450.0".to_owned(),
                date: "2026-09-02".to_owned(),
            }],
            latest_itcrm.as_ref(),
            1_788_321_600,
            &mut diagnostics,
        );
        assert!(diagnostics.iter().any(|entry| entry.contains("tcrm_100")));
        Ok(())
    }

    #[test]
    fn itcrm_provider_failures_are_diagnostic_and_cached_values_avoid_io() {
        for provider_result in [
            Ok(HttpResponse {
                status_code: 503,
                body: Vec::new(),
            }),
            Err(TransportFailureKind::Connection),
        ] {
            let transport = Transport {
                responses: RefCell::new(VecDeque::from([provider_result])),
            };
            let mut diagnostics = Vec::new();
            assert!(
                itcrm(
                    &transport,
                    &mut Cache::default(),
                    1_788_321_600,
                    &mut None,
                    &mut diagnostics
                )
                .is_none()
            );
            assert_eq!(diagnostics.len(), 1);
        }

        let transport = Transport {
            responses: RefCell::new(VecDeque::new()),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            "latest_itcrm_details".to_owned(),
            json!({"value": 1250.75, "date": "02/09/26"}).to_string(),
        );
        let mut diagnostics = Vec::new();
        assert_eq!(
            itcrm(
                &transport,
                &mut cache,
                1_788_321_600,
                &mut None,
                &mut diagnostics
            ),
            Some(ItcrmDetails {
                value: 1250.75,
                date: "02/09/26".to_owned(),
            })
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn current_itcrm_excludes_future_and_malformed_cached_details() {
        let now = days(2024, 9, 1) * 86_400 + 15 * 3600;
        for latest in [
            None,
            Some(json!({"date":"2024-09-02","value":250.0})),
            Some(json!({"value":125.0})),
        ] {
            let mut cache = Cache::default();
            cache.values.insert(python_request_cache_key(&arguments(&BcraRequest::Variables).unwrap_or_default()),
                json!({"timestamp":now,"data":{"results":[{
                    "categoria":"Principales Variables", "idVariable":5,
                    "descripcion":"Tipo de cambio mayorista", "ultFechaInformada":"2024-08-30", "ultValorInformado":1000.0
                }]}}).to_string());
            if let Some(latest) = latest {
                cache
                    .values
                    .insert("latest_itcrm_details".to_owned(), latest.to_string());
            }
            cache.values.insert(
                "itcrm_daily_history".to_owned(),
                json!({"results":[{"detalle":[
                    {"fecha":"2024-08-30","valor":125.0}, {"fecha":"2024-09-02","valor":250.0}
                ]}]})
                .to_string(),
            );
            cache.values.insert(python_request_cache_key(&arguments(&BcraRequest::Series { id: 5, limit: 32 }).unwrap_or_default()),
                json!({"timestamp":now,"data":{"results":[{"detalle":[{"fecha":"2024-08-30","valor":1000.0}]}]}}).to_string());
            let transport = Transport {
                responses: RefCell::new(VecDeque::new()),
            };
            let load = load_dollar_references(&transport, &mut cache, now);
            assert_eq!(load.tcrm_100, Some(800.0));
            assert_eq!(load.tcrm_100_previous_day, Some(800.0));
            assert!(load.diagnostics.is_empty(), "{:?}", load.diagnostics);
        }
    }

    #[test]
    fn uncached_itcrm_response_supplies_daily_history_without_a_second_fetch() -> TestResult {
        let now = days(2026, 9, 2) * 86_400 + 15 * 3600;
        let bytes = base64::engine::general_purpose::STANDARD.decode(ITCRM_WORKBOOK)?;
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                response(
                    json!({"results":[{"categoria":"Principales Variables","idVariable":5,
                "descripcion":"Tipo de cambio mayorista","ultFechaInformada":"2026-09-02","ultValorInformado":1450.0}]}),
                ),
                Ok(HttpResponse {
                    status_code: 200,
                    body: bytes,
                }),
                response(json!({"results":[{"detalle":[{"fecha":"2026-09-01","valor":1400.0}]}]})),
                Ok(HttpResponse {
                    status_code: 403,
                    body: Vec::new(),
                }),
            ])),
        };
        let mut cache = Cache {
            fail_gets: true,
            fail_sets: true,
            ..Cache::default()
        };
        let load = load_dollar_references(&transport, &mut cache, now);
        assert_eq!(load.tcrm_100, Some(1450.0 * 100.0 / 1250.75));
        assert_eq!(load.tcrm_100_previous_day, Some(1400.0 * 100.0 / 1234.5));
        assert_eq!(transport.responses.borrow().len(), 1);
        assert!(
            !load
                .diagnostics
                .iter()
                .any(|entry| entry.contains("ITCRM HTTP"))
        );
        Ok(())
    }

    #[test]
    fn cache_and_cell_boundaries_preserve_diagnostics_and_numeric_formats() {
        struct FailingCache;
        impl RequestCache for FailingCache {
            type Error = &'static str;

            fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
                Err("synthetic read failure")
            }

            fn set(&mut self, _key: &str, _value: &str, _ttl: i64) -> Result<(), Self::Error> {
                Err("synthetic write failure")
            }

            fn take(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
                Err("synthetic read failure")
            }

            fn claim(&mut self, _key: &str, _value: &str, _ttl: i64) -> Result<bool, Self::Error> {
                Err("synthetic write failure")
            }
        }

        let mut diagnostics = Vec::new();
        let mut cache = FailingCache;
        assert!(read(&mut cache, "synthetic", &mut diagnostics).is_none());
        assert_eq!(
            RequestCache::take(&mut cache, "synthetic"),
            Err("synthetic read failure")
        );
        assert_eq!(
            RequestCache::claim(&mut cache, "synthetic", "1", 60),
            Err("synthetic write failure")
        );
        store(
            &mut cache,
            "synthetic",
            &json!({"value": 1}),
            60,
            60,
            1_700_000_000,
            &mut diagnostics,
        );
        assert_eq!(diagnostics.len(), 3);

        // An unreadable full ITCRM copy is reported, then BCRA is asked.
        let mut unreadable = Cache {
            fail_gets: true,
            ..Cache::default()
        };
        let mut itcrm_diagnostics = Vec::new();
        let offline = Transport {
            responses: RefCell::new(VecDeque::new()),
        };
        assert!(
            full_itcrm_series(
                &offline,
                &mut unreadable,
                1_700_000_000,
                &mut itcrm_diagnostics
            )
            .is_none()
        );
        assert_eq!(
            itcrm_diagnostics.first().map(String::as_str),
            Some("could not read BCRA cache itcrm_full_history:v1: synthetic cache read failure")
        );

        let mut invalid = Cache::default();
        invalid
            .values
            .insert("synthetic".to_owned(), "not json".to_owned());
        assert!(read(&mut invalid, "synthetic", &mut diagnostics).is_none());

        assert_eq!(cell_number(&Data::Float(42.5)), Some(42.5));
        assert_eq!(cell_number(&Data::Int(42)), Some(42.0));
        assert_eq!(
            cell_number(&Data::String("1.234,5".to_owned())),
            Some(1234.5)
        );
        assert_eq!(cell_number(&Data::Empty), None);
        assert!(cell_date(&Data::String("2025-09-19".to_owned())).is_some());
        assert!(cell_date(&Data::DateTimeIso("2025-09-19".to_owned())).is_some());
        assert_eq!(cell_date(&Data::Empty), None);

        assert!(
            arguments(&BcraRequest::Value {
                id: 1,
                date: "2025-09-19".to_owned(),
            })
            .is_some()
        );
        assert!(arguments(&BcraRequest::Itcrm).is_none());
    }
    #[test]
    fn loads_all_json_sources_and_writes_compatible_values() {
        let variables = json!({"results":[{"categoria":"Principales Variables","idVariable":1,"descripcion":"Tipo de cambio mayorista de referencia","ultFechaInformada":"2025-09-19","ultValorInformado":1180.25},{"categoria":"Principales Variables","idVariable":2,"descripcion":"Reservas internacionales","ultFechaInformada":"2025-09-19","ultValorInformado":25000},{"categoria":"Principales Variables","idVariable":7,"descripcion":"Tasa de interés BADLAR de bancos privados","ultFechaInformada":"2025-09-19","ultValorInformado":23.94},{"categoria":"Principales Variables","idVariable":35,"descripcion":"Tasa de interés BADLAR de bancos privados","ultFechaInformada":"2025-09-19","ultValorInformado":26.73},{"categoria":"Principales Variables","idVariable":29,"descripcion":"Mediana de la variación interanual próximos 12 meses del índice de precios al consumidor del relevamiento de expectativas de mercado","ultFechaInformada":"2025-09-19","ultValorInformado":21.8},{"categoria":"Principales Variables","idVariable":1187,"descripcion":"Régimen de bandas cambiarias Límite inferior","ultFechaInformada":"2025-09-19","ultValorInformado":944.32},{"categoria":"Principales Variables","idVariable":1188,"descripcion":"Régimen de bandas cambiarias Límite superior","ultFechaInformada":"2025-09-19","ultValorInformado":1481.7}]});
        let lower = json!({"results":[{"detalle":[{"fecha":"2025-09-18","valor":930},{"fecha":"2025-09-19","valor":944.32}]}]});
        let upper = json!({"results":[{"detalle":[{"fecha":"2025-09-18","valor":1470},{"fecha":"2025-09-19","valor":1481.7}]}]});
        let risk = json!({"weightedSpreadBps":"685.21","deltas":{"oneDay":"-12.3"},"valuationDate":"2025-10-29T15:34:00Z"});
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                response(variables),
                response(lower),
                response(upper),
                response(risk),
                Err(TransportFailureKind::Connection),
            ])),
        };
        let mut cache = Cache::default();
        let load = load_bcra(&transport, &mut cache, Locale::Es, 1_758_297_600);
        let text = load.text.unwrap_or_default();
        for expected in [
            "Dólar mayorista: $1,180.25",
            "Reservas: USD 25,000 millones",
            "Inflación esperada: 21.8%",
            "BADLAR: 26.7%",
            "Bandas cambiarias: piso $944.32 / techo $1,481.7",
            "Riesgo país: 685 bps",
        ] {
            assert!(text.contains(expected), "{text}");
        }
        for key in [
            "bcra_variables",
            "bcra_variables:last_success",
            "bcra_currency_band_limits",
            "bcra_mayorista:2025-09-19",
        ] {
            assert!(cache.values.contains_key(key), "{key}");
        }
    }

    #[test]
    fn rejects_future_or_undated_band_cache_entries() {
        let now = 1_758_297_600;
        for (date, expected) in [
            (Some("20/09/25"), false),
            (Some("19/09/25"), true),
            (Some("18/09/25"), true),
            (Some("invalid"), false),
            (None, false),
        ] {
            let transport = Transport {
                responses: RefCell::new(VecDeque::new()),
            };
            let mut cache = Cache::default();
            cache.values.insert(
                "bcra_currency_band_limits".to_owned(),
                json!({"date":date,"lower":900,"upper":1500}).to_string(),
            );
            assert_eq!(
                bands(&transport, &mut cache, &[], now, &mut Vec::new()).is_some(),
                expected,
                "date={date:?}"
            );
        }
    }

    #[test]
    fn dollar_references_load_without_running_the_full_bcra_command() {
        let variables = json!({"results":[{"categoria":"Principales Variables","idVariable":1187,"descripcion":"Régimen de bandas cambiarias. Límite inferior","ultFechaInformada":"2025-09-19","ultValorInformado":944.32},{"categoria":"Principales Variables","idVariable":1188,"descripcion":"Régimen de bandas cambiarias. Límite superior","ultFechaInformada":"2025-09-19","ultValorInformado":1481.7},{"categoria":"Principales Variables","idVariable":5,"descripcion":"Tipo de cambio mayorista","ultFechaInformada":"2025-09-19","ultValorInformado":1450.0}]});
        let lower = json!({"results":[{"detalle":[{"fecha":"2025-09-18","valor":930},{"fecha":"2025-09-19","valor":944.32}]}]});
        let upper = json!({"results":[{"detalle":[{"fecha":"2025-09-18","valor":1470},{"fecha":"2025-09-19","valor":1481.7}]}]});
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                response(variables),
                response(lower),
                response(upper),
            ])),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            "latest_itcrm_details".to_owned(),
            json!({"value":100.0,"date":"19/09/25"}).to_string(),
        );

        cache.values.insert(
            "bcra_currency_band_limits".to_owned(),
            json!({"date":"20/09/25","date_iso":"2025-09-20","lower":9999,"upper":99999})
                .to_string(),
        );
        let load = load_dollar_references(&transport, &mut cache, 1_758_297_600);

        assert_eq!(load.bands.as_ref().map(|bands| bands.lower), Some(944.32));
        assert_eq!(load.bands.as_ref().map(|bands| bands.upper), Some(1481.7));
        assert_eq!(load.itcrm.as_ref().map(|itcrm| itcrm.value), Some(100.0));
        assert_eq!(load.tcrm_100, Some(1450.0));
        assert!(cache.values.contains_key("bcra_currency_band_limits"));
        assert!(cache.values.contains_key("tcrm_100"));
    }
    #[test]
    fn stale_survives_failures() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from(vec![Err(TransportFailureKind::Timeout); 8])),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            "bcra_variables:last_success".to_owned(),
            r#"{"data":{"Reservas internacionales":{"value":"25.000,00","date":"19/09/25"}}}"#
                .to_owned(),
        );
        let load = load_bcra(&transport, &mut cache, Locale::En, 1_758_297_600);
        let text = load.text.unwrap_or_default();
        assert!(text.contains("Reserves: USD 25,000 million"));
        assert!(text.contains("There is no new BCRA update"));
    }

    #[test]
    fn reqwest_transport_preserves_all_bcra_request_shapes() -> TestResult {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> TestResult {
            for expected in [
                "/bcra/monetarias?limit=2000",
                "/bcra/monetarias/7?limit=20",
                "/bcra/monetarias/8?desde=2026-09-02&hasta=2026-09-02&limit=1",
                "/bcra/monetarias/5?desde=2016-11-10&hasta=2016-11-20&limit=1",
                "/risk",
                "/itcrm",
            ] {
                let (mut stream, _) = listener.accept()?;
                let mut request = [0_u8; 2_048];
                let bytes = stream.read(&mut request).unwrap_or_default();
                assert!(
                    String::from_utf8_lossy(&request[..bytes])
                        .starts_with(&format!("GET {expected} HTTP/1.1"))
                );
                let body = b"synthetic";
                let written = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                written?;
                stream.write_all(body)?;
            }
            Ok(())
        });
        let base = format!("http://{address}");
        let transport = ReqwestBcraTransport::with_urls(
            &format!("{base}/bcra"),
            &format!("{base}/risk"),
            &format!("{base}/itcrm"),
        )
        .ok()
        .ok_or("unexpected error")?;
        for request in [
            BcraRequest::Variables,
            BcraRequest::Series { id: 7, limit: 20 },
            BcraRequest::Value {
                id: 8,
                date: "2026-09-02".to_owned(),
            },
            BcraRequest::Latest {
                id: 5,
                from: "2016-11-10".to_owned(),
                to: "2016-11-20".to_owned(),
            },
            BcraRequest::CountryRisk,
            BcraRequest::Itcrm,
        ] {
            let response = transport.get(&request).ok().ok_or("unexpected error")?;
            assert_eq!(response.status_code, 200);
            assert_eq!(response.body, b"synthetic");
        }
        transport.before_retry();
        assert!(matches!(server.join(), Ok(Ok(()))));
        let unavailable = ReqwestBcraTransport::with_urls(
            "http://127.0.0.1:1/bcra",
            "http://127.0.0.1:1/risk",
            "http://127.0.0.1:1/itcrm",
        )
        .ok()
        .ok_or("unexpected error")?;
        assert!(unavailable.get(&BcraRequest::Variables).is_err());
        Ok(())
    }

    #[test]
    fn numbers_accept_only_finite_numeric_json_and_locale_strings() {
        assert_eq!(number(None), None);
        assert_eq!(number(Some(&json!(true))), None);
        assert_eq!(number(Some(&json!(null))), None);
        assert_eq!(number(Some(&json!("not a number"))), None);
        assert_eq!(number(Some(&json!("1.234,5"))), Some(1234.5));
        assert_eq!(number(Some(&json!("NaN"))), None);
        assert_eq!(number(Some(&json!(12))), Some(12.0));
    }

    fn band_variables() -> Value {
        json!({"results":[
            {"categoria":"Principales Variables","idVariable":1187,"descripcion":"Régimen de bandas cambiarias Límite inferior","ultFechaInformada":"2025-09-19","ultValorInformado":944.32},
            {"categoria":"Principales Variables","idVariable":1188,"descripcion":"Régimen de bandas cambiarias Límite superior","ultFechaInformada":"2025-09-19","ultValorInformado":1481.7}
        ]})
    }

    #[test]
    fn bands_are_missing_when_either_limit_series_cannot_be_fetched() {
        let variables = band_variables();
        let vars = items(&variables);
        let lower = json!({"results":[{"detalle":[{"fecha":"2025-09-19","valor":944.32}]}]});
        for responses in [
            vec![
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Timeout),
            ],
            vec![
                response(lower),
                Err(TransportFailureKind::Connection),
                Err(TransportFailureKind::Connection),
            ],
        ] {
            let transport = Transport {
                responses: RefCell::new(VecDeque::from(responses)),
            };
            let mut cache = Cache::default();
            let mut diagnostics = Vec::new();
            assert_eq!(
                bands(
                    &transport,
                    &mut cache,
                    &vars,
                    1_758_297_600,
                    &mut diagnostics
                ),
                None
            );
            assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
            assert!(transport.responses.borrow().is_empty());
            assert!(!cache.values.contains_key("bcra_currency_band_limits"));
        }
    }

    #[test]
    fn workbook_dates_and_unusable_workbooks_are_explicit() -> TestResult {
        assert_eq!(
            cell_date(&Data::DateTime(calamine::ExcelDateTime::new(
                45_919.0,
                calamine::ExcelDateTimeType::DateTime,
                false,
            ))),
            parse_date("2025-09-19")
        );
        const HEADER_ONLY_WORKBOOK: &str = "UEsDBBQAAAAIAA1nPF1bma6u5QAAAAsCAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbK2RvVLDMBCEX0WjNhOdk4KCsZ0i0AYKXuCQz7HG+hudEszbIzuBggnQUN1Iu3vfalTvJmfFmRKb4Bu5UZXctfXLeyQWRfHcyCHneA/AeiCHrEIkX5Q+JIe5HNMRIuoRjwTbqroDHXwmn9d53iHb+oF6PNksHqdyfaEksizF/mKcWY3EGK3RmIsOZ999o6yvBFWSi4cHE3lVDBJuEmblZ8A191SenUxH4hlTPqArLpgsvIU0voYwqt+X3GgZ+t5o6oI+uRJRHBNhxwNRdlYtUzk0fvU3fzEzLGPzz0W+9n/2gOW72w9QSwMEFAAAAAgADWc8XUuDozqWAAAABQEAAAsAAABfcmVscy8ucmVsc43PPQ7CMAwF4KtEPkDdMjCgpl1YuiIuEFL3R23iyAlQbk9GihgY/fz0Wa7bza3qQRJn9hqqooS2qS+0mpSDOM0hqtzwUcOUUjghRjuRM7HgQD5vBhZnUh5lxGDsYkbCQ1keUT4N2Juq6zVI11egrq9A/9g8DLOlM9u7I59+nPhqZNnISEnDtuKTZbkxL0VGAZsadw82b1BLAwQUAAAACAANZzxdLWvSpKMAAAD4AAAADwAAAHhsL3dvcmtib29rLnhtbI2PORKDMAxFr+LRATCkSMEYUyQNRZpMLuCACB7wMpKzHD8eCH0qLX/0vr5qP24RLyS2wTdQFSW0Wr0DzfcQZpFFzw1MKcVaSu4ndIaLENFnZQzkTMojPSRHQjPwhJjcIg9leZTOWA8boaZ/GGEcbY/n0D8d+rRBCBeT8ms82cig1erAvyq8cdhAdztdLyDWVTfkECCotrmhbqhAaiX3K7kH019QSwMEFAAAAAgADWc8XW026XSaAAAABgEAABoAAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc43POw7CMAwG4KtEPkDdMjCgpl1YWBEXiFK3qdo8FJvX7YkYEJUYmCz/tj7Lbf/wq7pR5jkGDU1VQ9+1Z1qNlIDdnFiVjcAanEg6ILJ15A1XMVEokzFmb6S0ecJk7GImwl1d7zF/G7A11WnQkE9DA+ryTPSPHcdxtnSM9uopyI8TeI95YUckBTV5ItHwiRjfpamKCti1uPmwewFQSwMEFAAAAAgADWc8XZNurY6kAAAA+AAAABgAAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWx1j00KwjAQRq8ScgCndeFC0hR/EFy4US8Q4miCTVKSwXp8p0VEF26Gmffx+BjVPkMnHpiLT7GR9aySrVZDyvfiEElwGksjHVG/BCjWYTBllnqMnFxTDob4zDcofUZzmaTQwbyqFhCMj1KriW0NGa1yGkTmFqZ2XFa1FNRIHzsf8USZuS9akd6hdUYBaQUjAPsW1v+E/XlzPPwKwG08v+rh85d+AVBLAQIUAxQAAAAIAA1nPF1bma6u5QAAAAsCAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQDFAAAAAgADWc8XUuDozqWAAAABQEAAAsAAAAAAAAAAAAAAIABFgEAAF9yZWxzLy5yZWxzUEsBAhQDFAAAAAgADWc8XS1r0qSjAAAA+AAAAA8AAAAAAAAAAAAAAIAB1QEAAHhsL3dvcmtib29rLnhtbFBLAQIUAxQAAAAIAA1nPF1tNul0mgAAAAYBAAAaAAAAAAAAAAAAAACAAaUCAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc1BLAQIUAxQAAAAIAA1nPF2Tbq2OpAAAAPgAAAAYAAAAAAAAAAAAAACAAXcDAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWxQSwUGAAAAAAUABQBFAQAAUQQAAAAA";
        let bytes = base64::engine::general_purpose::STANDARD.decode(HEADER_ONLY_WORKBOOK)?;
        assert_eq!(
            parse_workbook(&bytes),
            Err("ITCRM workbook has no usable rows".to_owned())
        );
        for body in [bytes, b"not a workbook".to_vec()] {
            let transport = Transport {
                responses: RefCell::new(VecDeque::from([Ok(HttpResponse {
                    status_code: 200,
                    body,
                })])),
            };
            let mut cache = Cache::default();
            let mut diagnostics = Vec::new();
            assert_eq!(
                itcrm(
                    &transport,
                    &mut cache,
                    1_788_321_600,
                    &mut None,
                    &mut diagnostics
                ),
                None
            );
            assert_eq!(diagnostics.len(), 1);
            assert!(!cache.values.contains_key("latest_itcrm_details"));
        }
        Ok(())
    }

    #[test]
    fn workbooks_without_a_readable_first_sheet_are_rejected() -> TestResult {
        const SHEETLESS_WORKBOOK: &str = "UEsDBBQAAAAIALlrPF1bma6u5QAAAAsCAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbK2RvVLDMBCEX0WjNhOdk4KCsZ0i0AYKXuCQz7HG+hudEszbIzuBggnQUN1Iu3vfalTvJmfFmRKb4Bu5UZXctfXLeyQWRfHcyCHneA/AeiCHrEIkX5Q+JIe5HNMRIuoRjwTbqroDHXwmn9d53iHb+oF6PNksHqdyfaEksizF/mKcWY3EGK3RmIsOZ999o6yvBFWSi4cHE3lVDBJuEmblZ8A191SenUxH4hlTPqArLpgsvIU0voYwqt+X3GgZ+t5o6oI+uRJRHBNhxwNRdlYtUzk0fvU3fzEzLGPzz0W+9n/2gOW72w9QSwMEFAAAAAgAuWs8XUuDozqWAAAABQEAAAsAAABfcmVscy8ucmVsc43PPQ7CMAwF4KtEPkDdMjCgpl1YuiIuEFL3R23iyAlQbk9GihgY/fz0Wa7bza3qQRJn9hqqooS2qS+0mpSDOM0hqtzwUcOUUjghRjuRM7HgQD5vBhZnUh5lxGDsYkbCQ1keUT4N2Juq6zVI11egrq9A/9g8DLOlM9u7I59+nPhqZNnISEnDtuKTZbkxL0VGAZsadw82b1BLAwQUAAAACAC5azxdHyUyI4EAAADDAAAADwAAAHhsL3dvcmtib29rLnhtbI2OQQ7CIBBFr9JwgA66cNFQuvEiWAchBYbMUPX4Ehv3Ln9e8v4zyzun4YkskcqsTqNWizUv4u1GtA0dFplVaK1OALIGzE5Gqlg68cTZtT75AVIZ3V0CYssJzlpfILtY1GGY+B8HeR9XvNK6ZyztkDAm13qahFhFWfN9ELAGfo32A1BLAwQUAAAACAC5azxdbTbpdJoAAAAGAQAAGgAAAHhsL19yZWxzL3dvcmtib29rLnhtbC5yZWxzjc87DsIwDAbgq0Q+QN0yMKCmXVhYEReIUrep2jwUm9ftiRgQlRiYLP+2Pstt//CrulHmOQYNTVVD37VnWo2UgN2cWJWNwBqcSDogsnXkDVcxUSiTMWZvpLR5wmTsYibCXV3vMX8bsDXVadCQT0MD6vJM9I8dx3G2dIz26inIjxN4j3lhRyQFNXki0fCJGN+lqYoK2LW4+bB7AVBLAQIUAxQAAAAIALlrPF1bma6u5QAAAAsCAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQDFAAAAAgAuWs8XUuDozqWAAAABQEAAAsAAAAAAAAAAAAAAIABFgEAAF9yZWxzLy5yZWxzUEsBAhQDFAAAAAgAuWs8XR8lMiOBAAAAwwAAAA8AAAAAAAAAAAAAAIAB1QEAAHhsL3dvcmtib29rLnhtbFBLAQIUAxQAAAAIALlrPF1tNul0mgAAAAYBAAAaAAAAAAAAAAAAAACAAYMCAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc1BLBQYAAAAABAAEAP8AAABVAwAAAAA=";
        const MALFORMED_SHEET_WORKBOOK: &str = "UEsDBBQAAAAIAMFrPF1bma6u5QAAAAsCAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbK2RvVLDMBCEX0WjNhOdk4KCsZ0i0AYKXuCQz7HG+hudEszbIzuBggnQUN1Iu3vfalTvJmfFmRKb4Bu5UZXctfXLeyQWRfHcyCHneA/AeiCHrEIkX5Q+JIe5HNMRIuoRjwTbqroDHXwmn9d53iHb+oF6PNksHqdyfaEksizF/mKcWY3EGK3RmIsOZ999o6yvBFWSi4cHE3lVDBJuEmblZ8A191SenUxH4hlTPqArLpgsvIU0voYwqt+X3GgZ+t5o6oI+uRJRHBNhxwNRdlYtUzk0fvU3fzEzLGPzz0W+9n/2gOW72w9QSwMEFAAAAAgAwWs8XUuDozqWAAAABQEAAAsAAABfcmVscy8ucmVsc43PPQ7CMAwF4KtEPkDdMjCgpl1YuiIuEFL3R23iyAlQbk9GihgY/fz0Wa7bza3qQRJn9hqqooS2qS+0mpSDOM0hqtzwUcOUUjghRjuRM7HgQD5vBhZnUh5lxGDsYkbCQ1keUT4N2Juq6zVI11egrq9A/9g8DLOlM9u7I59+nPhqZNnISEnDtuKTZbkxL0VGAZsadw82b1BLAwQUAAAACADBazxdLWvSpKMAAAD4AAAADwAAAHhsL3dvcmtib29rLnhtbI2PORKDMAxFr+LRATCkSMEYUyQNRZpMLuCACB7wMpKzHD8eCH0qLX/0vr5qP24RLyS2wTdQFSW0Wr0DzfcQZpFFzw1MKcVaSu4ndIaLENFnZQzkTMojPSRHQjPwhJjcIg9leZTOWA8boaZ/GGEcbY/n0D8d+rRBCBeT8ms82cig1erAvyq8cdhAdztdLyDWVTfkECCotrmhbqhAaiX3K7kH019QSwMEFAAAAAgAwWs8XW026XSaAAAABgEAABoAAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc43POw7CMAwG4KtEPkDdMjCgpl1YWBEXiFK3qdo8FJvX7YkYEJUYmCz/tj7Lbf/wq7pR5jkGDU1VQ9+1Z1qNlIDdnFiVjcAanEg6ILJ15A1XMVEokzFmb6S0ecJk7GImwl1d7zF/G7A11WnQkE9DA+ryTPSPHcdxtnSM9uopyI8TeI95YUckBTV5ItHwiRjfpamKCti1uPmwewFQSwMEFAAAAAgAwWs8XRXT2ueAAAAAqAAAABgAAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWxNjksOgzAMRK8S5QAYuuiiMkGVepGIug0q+ciOAsevYVF1Y43nzcjGaY+racSy5DTaoevt5HDL/JFAVI3SJKMNtZYbgMyBopcuF0pKXpmjr7ryG6Qw+edZiitc+v4K0S/JOjy9h6/eIefNsF5Rdz7E/VDNDQjNISjV+ReH3x/uC1BLAQIUAxQAAAAIAMFrPF1bma6u5QAAAAsCAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQDFAAAAAgAwWs8XUuDozqWAAAABQEAAAsAAAAAAAAAAAAAAIABFgEAAF9yZWxzLy5yZWxzUEsBAhQDFAAAAAgAwWs8XS1r0qSjAAAA+AAAAA8AAAAAAAAAAAAAAIAB1QEAAHhsL3dvcmtib29rLnhtbFBLAQIUAxQAAAAIAMFrPF1tNul0mgAAAAYBAAAaAAAAAAAAAAAAAACAAaUCAAB4bC9fcmVscy93b3JrYm9vay54bWwucmVsc1BLAQIUAxQAAAAIAMFrPF0V09rngAAAAKgAAAAYAAAAAAAAAAAAAACAAXcDAAB4bC93b3Jrc2hlZXRzL3NoZWV0MS54bWxQSwUGAAAAAAUABQBFAQAALQQAAAAA";
        let sheetless = base64::engine::general_purpose::STANDARD.decode(SHEETLESS_WORKBOOK)?;
        assert_eq!(
            parse_workbook(&sheetless),
            Err("ITCRM workbook has no sheet".to_owned())
        );
        let malformed =
            base64::engine::general_purpose::STANDARD.decode(MALFORMED_SHEET_WORKBOOK)?;
        let error = parse_workbook(&malformed).err().unwrap_or_default();
        assert!(!error.is_empty());
        assert_ne!(error, "ITCRM workbook has no usable rows");
        Ok(())
    }

    #[test]
    fn dates_rows_and_band_descriptions_tolerate_partial_provider_data() {
        assert_eq!(parse_date("19/09/2025"), parse_date("19/09/25"));
        assert_eq!(parse_date("2025-09-19"), parse_date("19/09/25"));
        let rows = json!({"results":[{"detalle":[
            {"fecha":"2025-09-18","valor":930},
            {"fecha":"2025-09-19"},
            {"valor":944.32}
        ]}]});
        assert_eq!(series(&rows).into_values().collect::<Vec<_>>(), vec![930.0]);
        let variables = json!({"results":[
            {"categoria":"Principales Variables","idVariable":1,"descripcion":"Reservas","ultFechaInformada":"sometime","ultValorInformado":25000},
            {"categoria":"Principales Variables","idVariable":9,"descripcion":"Régimen de bandas cambiarias (referencia)","ultFechaInformada":"2025-09-19","ultValorInformado":1}
        ]});
        let items = items(&variables);
        assert_eq!(latest(&items)[0].date, "sometime");
        let transport = Transport {
            responses: RefCell::new(VecDeque::new()),
        };
        let mut diagnostics = Vec::new();
        assert_eq!(
            bands(
                &transport,
                &mut Cache::default(),
                &items,
                1_758_297_600,
                &mut diagnostics
            ),
            None
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn provider_bodies_must_be_utf8_and_cache_writes_report_failures() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                Ok(HttpResponse {
                    status_code: 200,
                    body: vec![0xff, 0xfe],
                }),
                Ok(HttpResponse {
                    status_code: 200,
                    body: vec![0xff],
                }),
            ])),
        };
        let mut diagnostics = Vec::new();
        assert_eq!(
            request_json(
                &transport,
                &mut Cache::default(),
                &BcraRequest::CountryRisk,
                1_758_297_600,
                &mut diagnostics,
            ),
            None
        );
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics
                .iter()
                .all(|entry| entry.contains("invalid UTF-8"))
        );

        let mut failing = Cache {
            fail_sets: true,
            ..Cache::default()
        };
        let mut diagnostics = Vec::new();
        store(
            &mut failing,
            "bcra_variables",
            &json!({}),
            60,
            60,
            1_758_297_600,
            &mut diagnostics,
        );
        assert_eq!(
            diagnostics,
            [
                "could not write BCRA cache bcra_variables: synthetic cache write failure",
                "could not write BCRA cache bcra_variables:last_success: synthetic cache write failure",
            ]
        );
    }

    #[test]
    fn bcra_without_fresh_or_stale_variables_renders_nothing() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::new()),
        };
        let load = load_bcra(&transport, &mut Cache::default(), Locale::Es, 1_758_297_600);
        assert_eq!(load.text, None);
        assert!(!load.diagnostics.is_empty());
    }

    #[test]
    fn tcrm_requires_a_positive_itcrm_index() {
        let variables = [BcraVariable {
            description: "Tipo de cambio mayorista".to_owned(),
            value: "1450.0".to_owned(),
            date: "02/09/26".to_owned(),
        }];
        for value in [0.0, -1.0] {
            let itcrm = ItcrmDetails {
                value,
                date: "02/09/26".to_owned(),
            };
            assert_eq!(calculate_tcrm(&variables, Some(&itcrm)), None);
        }
        let itcrm = ItcrmDetails {
            value: 145.0,
            date: "02/09/26".to_owned(),
        };
        assert_eq!(calculate_tcrm(&variables, Some(&itcrm)), Some(1000.0));
        assert_eq!(calculate_tcrm(&variables, None), None);
    }

    #[test]
    fn fresh_cached_variables_render_without_refetching_the_variable_list_first() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::new()),
        };
        let mut cache = Cache::default();
        cache.values.insert(
            "bcra_variables".to_owned(),
            r#"{"data":{"Reservas internacionales":{"value":"25.000,00","date":"19/09/25"},"_meta":{"value":"ignored"}}}"#
                .to_owned(),
        );
        let load = load_bcra(&transport, &mut cache, Locale::En, 1_758_297_600);
        let text = load.text.unwrap_or_default();
        assert!(text.contains("Reserves: USD 25,000 million"), "{text}");
        assert!(!text.contains("There is no new BCRA update"), "{text}");
        assert!(!text.contains("ignored"), "{text}");
    }

    #[test]
    fn cache_claim_propagates_write_failures() {
        let mut cache = Cache {
            fail_sets: true,
            ..Cache::default()
        };
        assert_eq!(
            cache.claim("k", "v", 60),
            Err("synthetic cache write failure")
        );
        assert!(cache.values.is_empty());
    }
}
