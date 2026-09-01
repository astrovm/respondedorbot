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
use unicode_normalization::UnicodeNormalization;

use crate::request_cache::{
    JsonHttpResponse, RequestCache, load_cached_json, python_json_string, python_request_cache_key,
};

const BCRA_URL: &str = "https://api.bcra.gob.ar/estadisticas/v4.0";
const RISK_URL: &str = "https://bondterminal.com/api/riesgo-pais";
const ITCRM_URL: &str = "https://www.bcra.gob.ar/Pdfs/PublicacionesEstadisticas/ITCRMSerie.xlsx";
const TTL: i64 = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BcraRequest {
    Variables,
    Series { id: i64, limit: i64 },
    Value { id: i64, date: String },
    CountryRisk,
    Itcrm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

pub trait BcraTransport {
    fn get(&self, request: &BcraRequest) -> Result<HttpResponse, TransportFailureKind>;
    fn before_retry(&self) {}
}

pub struct ReqwestBcraTransport {
    client: Client,
}

impl ReqwestBcraTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(Duration::from_secs(10))
            .danger_accept_invalid_certs(true)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl BcraTransport for ReqwestBcraTransport {
    fn get(&self, request: &BcraRequest) -> Result<HttpResponse, TransportFailureKind> {
        let request = match request {
            BcraRequest::Variables => self
                .client
                .get(format!("{BCRA_URL}/monetarias"))
                .query(&[("limit", "2000")]),
            BcraRequest::Series { id, limit } => self
                .client
                .get(format!("{BCRA_URL}/monetarias/{id}"))
                .query(&[("limit", limit.to_string())]),
            BcraRequest::Value { id, date } => self
                .client
                .get(format!("{BCRA_URL}/monetarias/{id}"))
                .query(&[
                    ("desde", date.as_str()),
                    ("hasta", date.as_str()),
                    ("limit", "1"),
                ]),
            BcraRequest::CountryRisk => self.client.get(RISK_URL),
            BcraRequest::Itcrm => self.client.get(ITCRM_URL),
        };
        let response = request.send().map_err(classify)?;
        let status_code = response.status().as_u16();
        response
            .bytes()
            .map(|body| HttpResponse {
                status_code,
                body: body.to_vec(),
            })
            .map_err(classify)
    }
    fn before_retry(&self) {
        thread::sleep(Duration::from_millis(500));
    }
}

fn classify(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BcraLoad {
    pub text: Option<String>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DollarReferencesLoad {
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

fn parse_date(value: &str) -> Option<i64> {
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

fn iso(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{y:04}-{m:02}-{d:02}")
}
fn display(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{d:02}/{m:02}/{:02}", y.rem_euclid(100))
}
fn ba_day(now: i64) -> i64 {
    floor(now.saturating_sub(10_800), 86_400)
}
fn hour_key(now: i64) -> String {
    let local = now.saturating_sub(10_800);
    let (y, m, d) = civil(floor(local, 86_400));
    format!(
        "{y:04}-{m:02}-{d:02}-{:02}",
        floor(local, 3_600).rem_euclid(24)
    )
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
    let key = python_request_cache_key(&arguments(request)?);
    let load = load_cached_json(
        cache,
        &key,
        TTL,
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

fn bands<T: BcraTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    vars: &[&Map<String, Value>],
    now: i64,
    d: &mut Vec<String>,
) -> Option<BcraBands> {
    if let Some(v) = read(cache, "bcra_currency_band_limits", d) {
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
    let itcrm = itcrm(transport, cache, &mut diagnostics);
    persist_market(cache, &variables, itcrm.as_ref(), now, &mut diagnostics);
    DollarReferencesLoad {
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
    d: &mut Vec<String>,
) -> Option<ItcrmDetails> {
    if let Some(v) = read(cache, "latest_itcrm_details", d) {
        return Some(ItcrmDetails {
            value: number(v.get("value"))?,
            date: v.get("date")?.as_str()?.to_owned(),
        });
    }
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
    let (day, value) = values.last_key_value()?;
    let result = ItcrmDetails {
        value: *value,
        date: display(*day),
    };
    if let Err(error) = cache.set(
        "latest_itcrm_details",
        &json!({"value":result.value,"date":result.date}).to_string(),
        1_800,
    ) {
        d.push(format!(
            "could not write BCRA cache latest_itcrm_details: {error}"
        ));
    }
    Some(result)
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
    let Some((wholesale, itcrm)) = wholesale
        .and_then(|v| number(Some(&Value::String(v.value.clone()))))
        .zip(itcrm)
        .filter(|(_, i)| i.value != 0.0)
    else {
        return;
    };
    let payload = json!({"timestamp":now,"data":wholesale*100.0/itcrm.value}).to_string();
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
    let itcrm = itcrm(transport, cache, &mut diagnostics);
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
    use super::*;
    use std::{
        cell::RefCell,
        collections::{HashMap, VecDeque},
    };
    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
    }
    impl RequestCache for Cache {
        type Error = &'static str;
        fn get(&mut self, k: &str) -> Result<Option<String>, Self::Error> {
            Ok(self.values.get(k).cloned())
        }
        fn set(&mut self, k: &str, v: &str, _: i64) -> Result<(), Self::Error> {
            self.values.insert(k.to_owned(), v.to_owned());
            Ok(())
        }
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
            "dólar mayorista: $1.180,25",
            "reservas: USD 25.000 millones",
            "inflación esperada: 21.8%",
            "BADLAR: 26.7%",
            "bandas cambiarias: piso $944.32 / techo $1481.7",
            "riesgo país: 685 bps",
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

        let load = load_dollar_references(&transport, &mut cache, 1_758_297_600);

        assert_eq!(load.bands.as_ref().map(|bands| bands.lower), Some(944.32));
        assert_eq!(load.bands.as_ref().map(|bands| bands.upper), Some(1481.7));
        assert_eq!(load.itcrm.as_ref().map(|itcrm| itcrm.value), Some(100.0));
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
        assert!(text.contains("reserves: USD 25.000 million"));
        assert!(text.contains("there is no new BCRA update"));
    }

    #[test]
    fn live_official_sources_parse_when_explicitly_enabled() {
        if std::env::var("BCRA_LIVE_TEST").as_deref() != Ok("1") {
            return;
        }
        let transport = ReqwestBcraTransport::new();
        assert!(transport.is_ok());
        let Ok(transport) = transport else { return };
        let load = load_bcra(&transport, &mut Cache::default(), Locale::Es, 1_788_043_200);
        assert!(load.text.is_some(), "diagnostics={:?}", load.diagnostics);
    }
}
