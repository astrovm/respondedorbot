//! Daily dollar closes from ArgentinaDatos, for `/dolar` periods longer than
//! the bot's own hourly snapshots. Blue, official and wholesale go back to
//! 2011, CCL to 2013, MEP to 2018, card to 2019 and crypto to 2023.
//!
//! ArgentinaDatos has no history for the Bitcoin and USDC rows, and no other
//! source has CriptoYa's numbers, so the bot saves its own daily closes of
//! them from November 2026 on.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::bcra::{ba_day, parse_date};
use crate::dollar::DollarTransport;
use crate::request_cache::RequestCache;

pub(crate) const HISTORY_URL: &str = "https://api.argentinadatos.com/v1/cotizaciones/dolares";
const CACHE_KEY: &str = "dollar_daily_history:v1";
/// ArgentinaDatos adds one close per day, so a few hours old is fresh.
const FRESH_SECONDS: i64 = 6 * 3_600;
/// A stale copy still answers when ArgentinaDatos is down.
const CACHE_TTL_SECONDS: i64 = 30 * 86_400;
/// The refresh job reports a failed download only once the copy is this
/// old. Before that the copy still answers and the next cycle tries again.
const REPORT_AFTER_SECONDS: i64 = 2 * 86_400;
/// Without a copy, one reply claims the download and the others skip the
/// history for this long. A failed download keeps the claim, so an outage
/// does not make each `/dolar` wait for a timeout.
const ATTEMPT_KEY: &str = "dollar_daily_history:attempt";
const ATTEMPT_SECONDS: i64 = 600;
/// A close stands in for the next days (weekends, holidays, a lagging feed),
/// but not for a week, so a series that ended (like `solidario`) or has not
/// started yet gives no baseline.
const MAX_GAP_DAYS: i64 = 7;
/// The bot's own daily closes, in the same shape as the ArgentinaDatos copy.
const OWN_CLOSES_KEY: &str = "dollar_own_daily_closes:v1";
/// Each refresh renews it, so the closes only expire after a long outage.
const OWN_CLOSES_TTL_SECONDS: i64 = 10 * 365 * 86_400;

/// The ArgentinaDatos `casa` for each `/dolar` row it has a history for.
/// The crypto history is the USDT quote, so it only stands for USDT.
#[must_use]
pub fn daily_source(rate_name: &str) -> Option<&'static str> {
    match rate_name {
        "Mayorista" => Some("mayorista"),
        "Oficial" => Some("oficial"),
        "Tarjeta" => Some("tarjeta"),
        "MEP" => Some("bolsa"),
        "CCL" => Some("contadoconliqui"),
        "Blue" => Some("blue"),
        "USDT" => Some("cripto"),
        _ => None,
    }
}

/// The name each `/dolar` row without an ArgentinaDatos history is saved
/// under in the bot's own daily closes.
#[must_use]
pub fn own_source(rate_name: &str) -> Option<&'static str> {
    match rate_name {
        "Bitcoin" => Some("ccb"),
        "USDC" => Some("usdc"),
        _ => None,
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DailyHistory {
    /// Selling price by Buenos Aires day (days since 1970-01-01), per `casa`.
    series: HashMap<String, BTreeMap<i64, f64>>,
}

#[derive(Deserialize)]
struct ProviderRow {
    casa: String,
    venta: Option<f64>,
    fecha: String,
}

impl DailyHistory {
    /// Reads the ArgentinaDatos `/v1/cotizaciones/dolares` array.
    pub fn parse_provider(body: &str) -> Result<Self, String> {
        let rows: Vec<ProviderRow> = serde_json::from_str(body)
            .map_err(|error| format!("invalid ArgentinaDatos dollar history: {error}"))?;
        let mut series: HashMap<String, BTreeMap<i64, f64>> = HashMap::new();
        for row in rows {
            if let (Some(day), Some(price)) = (
                parse_date(&row.fecha),
                row.venta.filter(|price| price.is_finite() && *price > 0.0),
            ) {
                series.entry(row.casa).or_default().insert(day, price);
            }
        }
        if series.is_empty() {
            return Err("ArgentinaDatos dollar history has no rates".to_owned());
        }
        Ok(Self { series })
    }

    /// The close of `source` on `day`, or the last one before it within a week.
    #[must_use]
    pub fn close(&self, source: &str, day: i64) -> Option<f64> {
        let (found, price) = self.series.get(source)?.range(..=day).next_back()?;
        (day - found <= MAX_GAP_DAYS).then_some(*price)
    }

    fn to_cache(&self, now_unix: i64) -> String {
        let data = self
            .series
            .iter()
            .map(|(source, closes)| {
                let pairs = closes
                    .iter()
                    .map(|(day, price)| json!([day, price]))
                    .collect::<Vec<_>>();
                (source.clone(), Value::Array(pairs))
            })
            .collect::<serde_json::Map<_, _>>();
        json!({"timestamp": now_unix, "data": data}).to_string()
    }

    fn from_cache(raw: &str) -> Option<(i64, Self)> {
        let value = serde_json::from_str::<Value>(raw).ok()?;
        let timestamp = value.get("timestamp")?.as_i64()?;
        let mut series = HashMap::new();
        for (source, pairs) in value.get("data")?.as_object()? {
            let closes = pairs
                .as_array()?
                .iter()
                .filter_map(|pair| {
                    let day = pair.get(0)?.as_i64()?;
                    let price = pair.get(1)?.as_f64()?;
                    (price.is_finite() && price > 0.0).then_some((day, price))
                })
                .collect::<BTreeMap<_, _>>();
            series.insert(source.clone(), closes);
        }
        (!series.is_empty()).then_some((timestamp, Self { series }))
    }
}

/// The Buenos Aires day `hours_ago` hours before `now_unix`.
#[must_use]
pub fn baseline_day(now_unix: i64, hours_ago: i64) -> i64 {
    ba_day(now_unix.saturating_sub(hours_ago.saturating_mul(3_600)))
}

/// The cached copy of the daily history and when it was downloaded.
fn cached_history<C: RequestCache>(
    cache: &mut C,
    diagnostics: &mut Vec<String>,
) -> Option<(i64, DailyHistory)> {
    match cache.get(CACHE_KEY) {
        Ok(raw) => raw.as_deref().and_then(DailyHistory::from_cache),
        Err(error) => {
            diagnostics.push(format!(
                "could not read dollar cache key {CACHE_KEY}: {error}"
            ));
            None
        }
    }
}

/// Downloads the history and caches it. A failed cache write is only a
/// diagnostic; a failed download is the error.
fn download_history<T: DollarTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Result<DailyHistory, String> {
    let history = transport
        .get_daily_history()
        .map_err(|error| format!("ArgentinaDatos dollar history transport {error:?}"))
        .and_then(|response| {
            if response.status_code >= 400 {
                Err(format!(
                    "ArgentinaDatos dollar history HTTP {}",
                    response.status_code
                ))
            } else {
                DailyHistory::parse_provider(&response.body)
            }
        })?;
    if let Err(error) = cache.set(CACHE_KEY, &history.to_cache(now_unix), CACHE_TTL_SECONDS) {
        diagnostics.push(format!(
            "could not write dollar cache key {CACHE_KEY}: {error}"
        ));
    }
    Ok(history)
}

/// The daily history for a `/dolar` reply. Any cached copy answers, even a
/// stale one, because the refresh job keeps it fresh and a reply should not
/// wait for a 3 MB download. Without a copy one reply downloads it, and
/// the others skip the history until it lands or a few minutes pass.
pub fn daily_history_for_reply<T: DollarTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<DailyHistory> {
    if let Some((_, history)) = cached_history(cache, diagnostics) {
        return Some(history);
    }
    match cache.claim(ATTEMPT_KEY, "1", ATTEMPT_SECONDS) {
        Ok(true) => {}
        Ok(false) => {
            diagnostics.push(
                "ArgentinaDatos dollar history was requested in the last 10 minutes".to_owned(),
            );
            return None;
        }
        // Without a working cache every reply downloads, as before.
        Err(error) => diagnostics.push(format!(
            "could not claim dollar cache key {ATTEMPT_KEY}: {error}"
        )),
    }
    download_history(transport, cache, now_unix, diagnostics)
        .map_err(|error| diagnostics.push(error))
        .ok()
}

/// The bot's own daily closes. A read failure or a value that does not
/// parse is a diagnostic and an error, so nothing overwrites the closes.
fn read_own_closes<C: RequestCache>(
    cache: &mut C,
    diagnostics: &mut Vec<String>,
) -> Result<Option<DailyHistory>, ()> {
    let raw = cache.get(OWN_CLOSES_KEY).map_err(|error| {
        diagnostics.push(format!(
            "could not read dollar cache key {OWN_CLOSES_KEY}: {error}"
        ));
    })?;
    let Some(raw) = raw else {
        return Ok(None);
    };
    match DailyHistory::from_cache(&raw) {
        Some((_, history)) => Ok(Some(history)),
        None => {
            diagnostics.push(format!("invalid dollar cache key {OWN_CLOSES_KEY}"));
            Err(())
        }
    }
}

/// The bot's own daily closes, for a `/dolar` reply.
pub fn own_closes_for_reply<C: RequestCache>(
    cache: &mut C,
    diagnostics: &mut Vec<String>,
) -> Option<DailyHistory> {
    read_own_closes(cache, diagnostics).ok().flatten()
}

/// Saves `prices` (own source name and price) as the closes of the day
/// they were priced on. The price refresh job calls this on each run, so
/// the last run of a day leaves the day's close. Nothing is written when
/// the saved closes cannot be read or parsed, so they are never wiped.
pub fn record_own_closes<C: RequestCache>(
    cache: &mut C,
    prices: &[(&str, f64)],
    priced_at: i64,
    diagnostics: &mut Vec<String>,
) {
    if prices.is_empty() {
        return;
    }
    let Ok(saved) = read_own_closes(cache, diagnostics) else {
        return;
    };
    let mut history = saved.unwrap_or_default();
    let day = ba_day(priced_at);
    for (source, price) in prices {
        history
            .series
            .entry((*source).to_owned())
            .or_default()
            .insert(day, *price);
    }
    if let Err(error) = cache.set(
        OWN_CLOSES_KEY,
        &history.to_cache(priced_at),
        OWN_CLOSES_TTL_SECONDS,
    ) {
        diagnostics.push(format!(
            "could not write dollar cache key {OWN_CLOSES_KEY}: {error}"
        ));
    }
}

/// Keeps the cached daily history fresh, for the price refresh job. A
/// failed download only becomes a diagnostic (and so an admin report) when
/// there is no copy or the copy is days old; until then it is logged.
pub fn refresh_daily_history<T: DollarTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) {
    let age =
        cached_history(cache, diagnostics).map(|(timestamp, _)| now_unix.saturating_sub(timestamp));
    if age.is_some_and(|age| (0..FRESH_SECONDS).contains(&age)) {
        return;
    }
    if let Err(error) = download_history(transport, cache, now_unix, diagnostics) {
        match age.filter(|age| *age < REPORT_AFTER_SECONDS) {
            Some(age) => eprintln!("{error}; the cached copy is {}h old", age / 3_600),
            None => diagnostics.push(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};

    use super::{
        ATTEMPT_KEY, CACHE_KEY, DailyHistory, OWN_CLOSES_KEY, baseline_day,
        daily_history_for_reply, daily_source, own_closes_for_reply, own_source, record_own_closes,
        refresh_daily_history,
    };
    use crate::bcra::parse_date;
    use crate::dollar::{DollarTransport, HttpResponse, TransportFailureKind};
    use crate::request_cache::RequestCache;

    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
        fail_reads: bool,
        fail_writes: bool,
    }

    impl RequestCache for Cache {
        type Error = String;
        fn get(&mut self, key: &str) -> Result<Option<String>, String> {
            if self.fail_reads {
                return Err("synthetic read failure".to_owned());
            }
            Ok(self.values.get(key).cloned())
        }
        fn set(&mut self, key: &str, value: &str, _: i64) -> Result<(), String> {
            if self.fail_writes {
                return Err("synthetic write failure".to_owned());
            }
            self.values.insert(key.to_owned(), value.to_owned());
            Ok(())
        }
        fn take(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.values.remove(key))
        }
        fn claim(&mut self, key: &str, value: &str, _: i64) -> Result<bool, String> {
            if self.fail_writes {
                return Err("synthetic write failure".to_owned());
            }
            Ok(self
                .values
                .insert(key.to_owned(), value.to_owned())
                .is_none())
        }
    }

    struct History(RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>);

    impl History {
        fn with(responses: Vec<Result<HttpResponse, TransportFailureKind>>) -> Self {
            Self(RefCell::new(responses.into()))
        }
        fn calls_left(&self) -> usize {
            self.0.borrow().len()
        }
    }

    impl DollarTransport for History {
        fn get(&self) -> Result<HttpResponse, TransportFailureKind> {
            Err(TransportFailureKind::Request)
        }
        fn get_daily_history(&self) -> Result<HttpResponse, TransportFailureKind> {
            self.0
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn ok(body: &str) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: body.to_owned(),
        })
    }

    /// The shape of a real ArgentinaDatos response, trimmed.
    const BODY: &str = r#"[
        {"casa":"blue","compra":4,"venta":4,"fecha":"2011-01-03"},
        {"casa":"blue","compra":15.9,"venta":16,"fecha":"2016-11-18"},
        {"casa":"oficial","compra":15.2,"venta":15.6,"fecha":"2016-11-18"},
        {"casa":"blue","compra":1630,"venta":1650,"fecha":"2026-11-20"},
        {"casa":"solidario","compra":null,"venta":null,"fecha":"2026-11-20"},
        {"casa":"cripto","compra":1740,"venta":0,"fecha":"2026-11-20"}
    ]"#;

    fn day(iso: &str) -> i64 {
        parse_date(iso).unwrap_or_default()
    }

    #[test]
    fn test_fakes_take_claim_and_refuse_live_rates() {
        let mut cache = Cache::default();
        assert_eq!(cache.claim("k", "v", 60), Ok(true));
        assert_eq!(cache.claim("k", "v", 60), Ok(false));
        assert_eq!(cache.take("k"), Ok(Some("v".to_owned())));
        assert_eq!(
            History::with(Vec::new()).get(),
            Err(TransportFailureKind::Request)
        );
    }

    #[test]
    fn closes_carry_forward_for_a_week_and_skip_empty_prices() -> Result<(), String> {
        let history = DailyHistory::parse_provider(BODY)?;
        assert_eq!(history.close("blue", day("2011-01-03")), Some(4.0));
        // A Sunday uses Friday's close.
        assert_eq!(history.close("blue", day("2016-11-20")), Some(16.0));
        assert_eq!(history.close("blue", day("2016-11-25")), Some(16.0));
        assert_eq!(history.close("blue", day("2016-11-26")), None);
        assert_eq!(history.close("blue", day("2010-12-31")), None);
        assert_eq!(history.close("oficial", day("2016-11-18")), Some(15.6));
        assert_eq!(history.close("solidario", day("2026-11-20")), None);
        assert_eq!(history.close("cripto", day("2026-11-20")), None);
        assert_eq!(history.close("mayorista", day("2026-11-20")), None);
        assert!(DailyHistory::parse_provider("[]").is_err());
        assert!(DailyHistory::parse_provider("{}").is_err());
        Ok(())
    }

    #[test]
    fn only_rates_with_a_matching_history_get_a_source() {
        assert_eq!(daily_source("MEP"), Some("bolsa"));
        assert_eq!(daily_source("CCL"), Some("contadoconliqui"));
        assert_eq!(daily_source("USDT"), Some("cripto"));
        for name in ["USDC", "Bitcoin", "TCRM 100", "Banda piso"] {
            assert_eq!(daily_source(name), None, "{name}");
        }
        // The bot saves its own closes only for the rows without a source.
        assert_eq!(own_source("Bitcoin"), Some("ccb"));
        assert_eq!(own_source("USDC"), Some("usdc"));
        for name in ["USDT", "Blue", "TCRM 100"] {
            assert_eq!(own_source(name), None, "{name}");
        }
    }

    #[test]
    fn own_closes_keep_each_days_last_price_and_survive_cache_failures() {
        // 2024-08-30 06:40 UTC is 03:40 in Buenos Aires.
        let now = 1_725_000_000;
        let today = day("2024-08-30");
        let mut cache = Cache::default();
        let mut diagnostics = Vec::new();
        assert_eq!(own_closes_for_reply(&mut cache, &mut diagnostics), None);
        record_own_closes(
            &mut cache,
            &[("ccb", 1400.0)],
            now - 86_400,
            &mut diagnostics,
        );
        record_own_closes(
            &mut cache,
            &[("ccb", 1450.0), ("usdc", 1460.0)],
            now,
            &mut diagnostics,
        );
        record_own_closes(
            &mut cache,
            &[("ccb", 1470.0)],
            now + 3_600,
            &mut diagnostics,
        );
        let closes = own_closes_for_reply(&mut cache, &mut diagnostics).unwrap_or_default();
        assert_eq!(closes.close("ccb", today - 1), Some(1400.0));
        assert_eq!(closes.close("ccb", today), Some(1470.0));
        assert_eq!(closes.close("usdc", today), Some(1460.0));
        assert_eq!(closes.close("usdc", today - 1), None);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");

        // No prices touch nothing, and an unreadable copy is not replaced.
        let saved = cache.values.get(OWN_CLOSES_KEY).cloned();
        cache.fail_reads = true;
        record_own_closes(&mut cache, &[], now, &mut diagnostics);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        record_own_closes(&mut cache, &[("ccb", 1.0)], now, &mut diagnostics);
        assert_eq!(cache.values.get(OWN_CLOSES_KEY), saved.as_ref());
        assert_eq!(own_closes_for_reply(&mut cache, &mut diagnostics), None);
        assert_eq!(
            diagnostics,
            vec![
                format!("could not read dollar cache key {OWN_CLOSES_KEY}: synthetic read failure");
                2
            ]
        );
        cache.fail_reads = false;
        // A copy that does not parse is kept as it is for someone to look at.
        let broken = r#"{"timestamp":1,"data":{"ccb":[[19000,1400.0]],"usdc":{}}}"#;
        cache
            .values
            .insert(OWN_CLOSES_KEY.to_owned(), broken.to_owned());
        let mut diagnostics = Vec::new();
        record_own_closes(&mut cache, &[("ccb", 1.0)], now, &mut diagnostics);
        assert_eq!(own_closes_for_reply(&mut cache, &mut diagnostics), None);
        assert_eq!(
            cache.values.get(OWN_CLOSES_KEY).map(String::as_str),
            Some(broken)
        );
        assert_eq!(
            diagnostics,
            vec![format!("invalid dollar cache key {OWN_CLOSES_KEY}"); 2]
        );
        cache.values.remove(OWN_CLOSES_KEY);
        cache.fail_writes = true;
        let mut diagnostics = Vec::new();
        record_own_closes(&mut cache, &[("ccb", 1.0)], now, &mut diagnostics);
        assert_eq!(
            diagnostics,
            [format!(
                "could not write dollar cache key {OWN_CLOSES_KEY}: synthetic write failure"
            )]
        );
    }

    #[test]
    fn baseline_days_use_buenos_aires_time() {
        // 2026-11-20 02:00 UTC is still the 19th in Buenos Aires.
        let now = 1_795_132_800 + 2 * 3_600;
        assert_eq!(baseline_day(now, 0), day("2026-11-19"));
        assert_eq!(baseline_day(now, 24 * 365), day("2025-11-19"));
    }

    fn stored(cache: &Cache) -> Option<(i64, DailyHistory)> {
        cache
            .values
            .get(CACHE_KEY)
            .and_then(|raw| DailyHistory::from_cache(raw))
    }

    #[test]
    fn the_refresh_job_downloads_stale_copies_and_reports_only_old_ones() -> Result<(), String> {
        let now = 1_795_132_800;
        let expected = DailyHistory::parse_provider(BODY)?;
        let transport = History::with(vec![
            ok(BODY),
            Err(TransportFailureKind::Timeout),
            Ok(HttpResponse {
                status_code: 503,
                body: String::new(),
            }),
            ok(BODY),
        ]);
        let mut cache = Cache::default();
        let mut diagnostics = Vec::new();
        refresh_daily_history(&transport, &mut cache, now, &mut diagnostics);
        assert_eq!(stored(&cache), Some((now, expected.clone())));
        // Fresh: no request.
        refresh_daily_history(&transport, &mut cache, now + 3_600, &mut diagnostics);
        assert_eq!(transport.calls_left(), 3);
        // Stale but under two days old: the failure is only logged.
        refresh_daily_history(&transport, &mut cache, now + 7 * 3_600, &mut diagnostics);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        // Two days old: the failure is reported.
        refresh_daily_history(&transport, &mut cache, now + 2 * 86_400, &mut diagnostics);
        assert_eq!(diagnostics, ["ArgentinaDatos dollar history HTTP 503"]);
        assert_eq!(stored(&cache), Some((now, expected.clone())));
        // A later success replaces the copy.
        refresh_daily_history(&transport, &mut cache, now + 3 * 86_400, &mut diagnostics);
        assert_eq!(stored(&cache), Some((now + 3 * 86_400, expected)));
        assert_eq!(diagnostics.len(), 1);

        // Without a copy every failure is reported, and invalid data too.
        let mut empty = Cache::default();
        let mut diagnostics = Vec::new();
        refresh_daily_history(
            &History::with(vec![ok("not json")]),
            &mut empty,
            now,
            &mut diagnostics,
        );
        assert!(diagnostics[0].starts_with("invalid ArgentinaDatos dollar history"));
        assert!(empty.values.is_empty());
        Ok(())
    }

    #[test]
    fn replies_use_any_copy_and_one_reply_downloads_a_missing_one() -> Result<(), String> {
        let now = 1_795_132_800;
        let expected = DailyHistory::parse_provider(BODY)?;
        let mut cache = Cache::default();
        let mut diagnostics = Vec::new();
        // No copy, ArgentinaDatos down: one reply asks, the next skips it.
        let down = History::with(vec![Err(TransportFailureKind::Timeout), ok(BODY)]);
        assert_eq!(
            daily_history_for_reply(&down, &mut cache, now, &mut diagnostics),
            None
        );
        assert_eq!(
            daily_history_for_reply(&down, &mut cache, now + 60, &mut diagnostics),
            None
        );
        assert_eq!(down.calls_left(), 1);
        assert_eq!(
            diagnostics,
            [
                "ArgentinaDatos dollar history transport Timeout",
                "ArgentinaDatos dollar history was requested in the last 10 minutes",
            ]
        );

        // Once the claim expires, a download fills the cache.
        cache.values.remove(ATTEMPT_KEY);
        let mut diagnostics = Vec::new();
        assert_eq!(
            daily_history_for_reply(&down, &mut cache, now, &mut diagnostics),
            Some(expected.clone())
        );
        // A week-old copy still answers without a request.
        let none = History::with(Vec::new());
        assert_eq!(
            daily_history_for_reply(&none, &mut cache, now + 7 * 86_400, &mut diagnostics),
            Some(expected.clone())
        );
        assert!(diagnostics.is_empty(), "{diagnostics:?}");

        // A broken cache still downloads, and each failure is reported.
        let mut broken = Cache {
            fail_reads: true,
            fail_writes: true,
            ..Cache::default()
        };
        let fetched = daily_history_for_reply(
            &History::with(vec![ok(BODY)]),
            &mut broken,
            now,
            &mut diagnostics,
        );
        assert_eq!(fetched, Some(expected));
        assert_eq!(
            diagnostics,
            [
                "could not read dollar cache key dollar_daily_history:v1: synthetic read failure",
                "could not claim dollar cache key dollar_daily_history:attempt: synthetic write failure",
                "could not write dollar cache key dollar_daily_history:v1: synthetic write failure",
            ]
        );
        let mut diagnostics = Vec::new();
        assert_eq!(
            daily_history_for_reply(&none, &mut broken, now, &mut diagnostics),
            None
        );
        assert_eq!(
            diagnostics[2],
            "ArgentinaDatos dollar history transport Request"
        );
        Ok(())
    }
}
