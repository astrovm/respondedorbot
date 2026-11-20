//! Daily dollar closes from ArgentinaDatos, for `/dolar` periods longer than
//! the bot's own hourly snapshots. Blue, official and wholesale go back to
//! 2011, CCL to 2013, MEP to 2018, card to 2019 and crypto to 2023.

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
/// A close stands in for the next days (weekends, holidays, a lagging feed),
/// but not for a week, so a series that ended (like `solidario`) or has not
/// started yet gives no baseline.
const MAX_GAP_DAYS: i64 = 7;

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

/// The daily history from the cache, refreshed from ArgentinaDatos once it
/// is a few hours old. A failed refresh falls back to the stale copy.
pub fn load_daily_history<T: DollarTransport, C: RequestCache>(
    transport: &T,
    cache: &mut C,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<DailyHistory> {
    let cached = match cache.get(CACHE_KEY) {
        Ok(raw) => raw.as_deref().and_then(DailyHistory::from_cache),
        Err(error) => {
            diagnostics.push(format!(
                "could not read dollar cache key {CACHE_KEY}: {error}"
            ));
            None
        }
    };
    if let Some((timestamp, history)) = &cached
        && (0..FRESH_SECONDS).contains(&now_unix.saturating_sub(*timestamp))
    {
        return Some(history.clone());
    }
    let fetched = transport
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
        });
    match fetched {
        Ok(history) => {
            if let Err(error) = cache.set(CACHE_KEY, &history.to_cache(now_unix), CACHE_TTL_SECONDS)
            {
                diagnostics.push(format!(
                    "could not write dollar cache key {CACHE_KEY}: {error}"
                ));
            }
            Some(history)
        }
        Err(error) => {
            diagnostics.push(error);
            cached.map(|(_, history)| history)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};

    use super::{CACHE_KEY, DailyHistory, baseline_day, daily_source, load_daily_history};
    use crate::bcra::parse_date;
    use crate::dollar::{DollarTransport, HttpResponse, TransportFailureKind};
    use crate::request_cache::RequestCache;

    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
        fail_reads: bool,
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
            self.values.insert(key.to_owned(), value.to_owned());
            Ok(())
        }
        fn take(&mut self, key: &str) -> Result<Option<String>, String> {
            Ok(self.values.remove(key))
        }
        fn claim(&mut self, key: &str, value: &str, _: i64) -> Result<bool, String> {
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
    }

    #[test]
    fn baseline_days_use_buenos_aires_time() {
        // 2026-11-20 02:00 UTC is still the 19th in Buenos Aires.
        let now = 1_795_132_800 + 2 * 3_600;
        assert_eq!(baseline_day(now, 0), day("2026-11-19"));
        assert_eq!(baseline_day(now, 24 * 365), day("2025-11-19"));
    }

    #[test]
    fn the_cache_is_reused_while_fresh_and_a_stale_copy_survives_failures() {
        let now = 1_795_132_800;
        let transport = History::with(vec![
            ok(BODY),
            Err(TransportFailureKind::Timeout),
            Ok(HttpResponse {
                status_code: 503,
                body: String::new(),
            }),
        ]);
        let mut cache = Cache::default();
        let mut diagnostics = Vec::new();
        let first = load_daily_history(&transport, &mut cache, now, &mut diagnostics);
        assert!(first.is_some());
        assert!(cache.values.contains_key(CACHE_KEY));
        // Fresh: no request, same data from the compact cache copy.
        let cached = load_daily_history(&transport, &mut cache, now + 60, &mut diagnostics);
        assert_eq!(cached, first);
        assert_eq!(transport.calls_left(), 2);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");

        let later = now + 7 * 3_600;
        let stale = load_daily_history(&transport, &mut cache, later, &mut diagnostics);
        assert_eq!(stale, first);
        assert_eq!(
            diagnostics,
            ["ArgentinaDatos dollar history transport Timeout"]
        );
        let stale = load_daily_history(&transport, &mut cache, later, &mut diagnostics);
        assert_eq!(stale, first);
        assert_eq!(diagnostics[1], "ArgentinaDatos dollar history HTTP 503");

        let mut broken = Cache {
            fail_reads: true,
            ..Cache::default()
        };
        let mut diagnostics = Vec::new();
        let fetched = load_daily_history(
            &History::with(vec![ok(BODY)]),
            &mut broken,
            now,
            &mut diagnostics,
        );
        assert_eq!(fetched, first);
        assert_eq!(
            diagnostics,
            ["could not read dollar cache key dollar_daily_history:v1: synthetic read failure"]
        );
        let mut empty = Cache::default();
        let mut diagnostics = Vec::new();
        let invalid = load_daily_history(
            &History::with(vec![ok("not json")]),
            &mut empty,
            now,
            &mut diagnostics,
        );
        assert_eq!(invalid, None);
        assert!(diagnostics[0].starts_with("invalid ArgentinaDatos dollar history"));
    }
}
