//! Typed Open-Meteo adapter with Python-compatible Redis request caching.

use std::thread;
use std::time::Duration;

use bot_core::cache_policy::{request_cache_key, request_cache_ttl};
use bot_core::weather::{
    DEFAULT_WEATHER_LOCATION, WeatherObservation, search_key, select_forecast_hour,
    select_location_candidate,
};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::redis_json_cache::{RedisJsonCache, RedisJsonCacheError};

const GEOCODING_URL: &str = "https://geocoding-api.open-meteo.com/v1/search";
const FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const CACHE_TTL_SECONDS: i64 = 1_800;

#[derive(Debug, Clone, PartialEq)]
pub enum WeatherRequest {
    Geocode { name: String },
    Forecast { latitude: f64, longitude: f64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status_code: u16,
    pub body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportFailureKind {
    Timeout,
    Connection,
    Request,
}

pub trait WeatherTransport {
    fn get(&self, request: &WeatherRequest) -> Result<HttpResponse, TransportFailureKind>;

    fn before_retry(&self) {}
}

pub trait WeatherCache {
    type Error: std::fmt::Display;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error>;

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error>;
}

impl WeatherCache for RedisJsonCache {
    type Error = RedisJsonCacheError;

    fn get(&mut self, key: &str) -> Result<Option<String>, Self::Error> {
        RedisJsonCache::get(self, key)
    }

    fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
        RedisJsonCache::set(self, key, value, Some(ttl_seconds)).map(|_stored| ())
    }
}

pub struct ReqwestWeatherTransport {
    client: Client,
}

impl ReqwestWeatherTransport {
    pub fn new() -> Result<Self, TransportFailureKind> {
        Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map(|client| Self { client })
            .map_err(|_| TransportFailureKind::Request)
    }
}

impl WeatherTransport for ReqwestWeatherTransport {
    fn get(&self, request: &WeatherRequest) -> Result<HttpResponse, TransportFailureKind> {
        let builder = match request {
            WeatherRequest::Geocode { name } => self.client.get(GEOCODING_URL).query(&[
                ("name", name.as_str()),
                ("count", "10"),
                ("language", "es"),
                ("format", "json"),
            ]),
            WeatherRequest::Forecast {
                latitude,
                longitude,
            } => self.client.get(FORECAST_URL).query(&[
                ("latitude", latitude.to_string()),
                ("longitude", longitude.to_string()),
                ("current", "weather_code".to_owned()),
                (
                    "hourly",
                    "apparent_temperature,precipitation_probability,weather_code,cloud_cover,visibility"
                        .to_owned(),
                ),
                ("timezone", "auto".to_owned()),
                ("forecast_days", "2".to_owned()),
            ]),
        };
        let response = builder.send().map_err(classify_error)?;
        let status_code = response.status().as_u16();
        response
            .text()
            .map(|body| HttpResponse { status_code, body })
            .map_err(classify_error)
    }

    fn before_retry(&self) {
        thread::sleep(Duration::from_millis(500));
    }
}

fn classify_error(error: reqwest::Error) -> TransportFailureKind {
    if error.is_timeout() {
        TransportFailureKind::Timeout
    } else if error.is_connect() {
        TransportFailureKind::Connection
    } else {
        TransportFailureKind::Request
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeatherLoad {
    pub observation: Option<WeatherObservation>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CachedResponse {
    timestamp: i64,
    data: Value,
}

fn python_string(value: &str) -> String {
    let mut encoded = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\u{08}' => encoded.push_str("\\b"),
            '\u{0c}' => encoded.push_str("\\f"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if character <= '\u{1f}' => {
                encoded.push_str(&format!("\\u{:04x}", u32::from(character)));
            }
            character if character.is_ascii() => encoded.push(character),
            character => {
                let codepoint = u32::from(character);
                if codepoint <= 0xffff {
                    encoded.push_str(&format!("\\u{codepoint:04x}"));
                } else {
                    let adjusted = codepoint - 0x1_0000;
                    let high = 0xd800 + (adjusted >> 10);
                    let low = 0xdc00 + (adjusted & 0x3ff);
                    encoded.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    encoded.push('"');
    encoded
}

fn python_cache_arguments(request: &WeatherRequest) -> String {
    match request {
        WeatherRequest::Geocode { name } => format!(
            "{{\"api_url\": \"{GEOCODING_URL}\", \"headers\": null, \"parameters\": {{\"count\": 10, \"format\": \"json\", \"language\": \"es\", \"name\": {}}}}}",
            python_string(name)
        ),
        WeatherRequest::Forecast {
            latitude,
            longitude,
        } => format!(
            "{{\"api_url\": \"{FORECAST_URL}\", \"headers\": null, \"parameters\": {{\"current\": \"weather_code\", \"forecast_days\": 2, \"hourly\": \"apparent_temperature,precipitation_probability,weather_code,cloud_cover,visibility\", \"latitude\": {latitude}, \"longitude\": {longitude}, \"timezone\": \"auto\"}}}}"
        ),
    }
}

fn compatible_cache_key(request: &WeatherRequest) -> String {
    let hash = Sha256::digest(python_cache_arguments(request).as_bytes());
    let encoded = hash
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    request_cache_key(&encoded)
}

fn cached_request<T: WeatherTransport, C: WeatherCache>(
    transport: &T,
    cache: &mut C,
    request: &WeatherRequest,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<Value> {
    let key = compatible_cache_key(request);
    let cached = match cache.get(&key) {
        Ok(Some(raw)) => match serde_json::from_str::<CachedResponse>(&raw) {
            Ok(cached) => Some(cached),
            Err(error) => {
                diagnostics.push(format!("invalid weather cache key {key}: {error}"));
                return None;
            }
        },
        Ok(None) => None,
        Err(error) => {
            diagnostics.push(format!("could not read weather cache key {key}: {error}"));
            return None;
        }
    };
    if let Some(cached) = &cached
        && now_unix.saturating_sub(cached.timestamp) <= CACHE_TTL_SECONDS
    {
        return Some(cached.data.clone());
    }

    for attempt in 0..2 {
        let fetched = match transport.get(request) {
            Ok(response) if response.status_code < 400 => {
                serde_json::from_str::<Value>(&response.body).map_err(|_| "invalid JSON")
            }
            Ok(response) => Err(if response.status_code >= 500 {
                "server HTTP error"
            } else {
                "HTTP error"
            }),
            Err(TransportFailureKind::Timeout) => Err("timeout"),
            Err(TransportFailureKind::Connection) => Err("connection error"),
            Err(TransportFailureKind::Request) => Err("request error"),
        };
        if let Ok(data) = fetched {
            let value = json!({"timestamp": now_unix, "data": data});
            let encoded = match serde_json::to_string(&value) {
                Ok(encoded) => encoded,
                Err(error) => {
                    diagnostics.push(format!("could not encode weather cache value: {error}"));
                    return cached.map(|cached| cached.data);
                }
            };
            match cache.set(&key, &encoded, request_cache_ttl(CACHE_TTL_SECONDS)) {
                Ok(()) => return value.get("data").cloned(),
                Err(error) => diagnostics.push(format!(
                    "could not write weather cache key {key} on attempt {}: {error}",
                    attempt + 1
                )),
            }
        } else if let Err(error) = fetched {
            diagnostics.push(format!(
                "weather request {request:?} attempt {} failed: {error}",
                attempt + 1
            ));
        }
        if attempt == 0 {
            transport.before_retry();
        }
    }
    cached.map(|cached| cached.data)
}

#[derive(Debug, Clone)]
struct Location {
    name: String,
    admin1: String,
    country: String,
    country_code: String,
    latitude: f64,
    longitude: f64,
}

fn falsey_string(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn locations(data: &Value) -> Vec<Location> {
    data.get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let item = item.as_object()?;
            Some(Location {
                name: falsey_string(item.get("name")),
                admin1: falsey_string(item.get("admin1")),
                country: falsey_string(item.get("country")),
                country_code: falsey_string(item.get("country_code")),
                latitude: item.get("latitude")?.as_f64()?,
                longitude: item.get("longitude")?.as_f64()?,
            })
        })
        .collect()
}

fn search_locations<T: WeatherTransport, C: WeatherCache>(
    transport: &T,
    cache: &mut C,
    name: &str,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Vec<Location> {
    cached_request(
        transport,
        cache,
        &WeatherRequest::Geocode {
            name: name.to_owned(),
        },
        now_unix,
        diagnostics,
    )
    .map_or_else(Vec::new, |data| locations(&data))
}

fn resolve_location<T: WeatherTransport, C: WeatherCache>(
    transport: &T,
    cache: &mut C,
    requested: &str,
    now_unix: i64,
    diagnostics: &mut Vec<String>,
) -> Option<Location> {
    let query = requested.trim();
    let query = if query.is_empty() {
        DEFAULT_WEATHER_LOCATION
    } else {
        query
    };
    if matches!(query.to_lowercase().as_str(), "buenos aires" | "caba") {
        return Some(Location {
            name: DEFAULT_WEATHER_LOCATION.to_owned(),
            admin1: String::new(),
            country: "Argentina".to_owned(),
            country_code: String::new(),
            latitude: -34.6037,
            longitude: -58.3816,
        });
    }

    let parts = query
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let search_name = if parts.len() > 1 { parts[0] } else { query };
    let mut qualifiers = if parts.len() > 1 {
        parts[1..].to_vec()
    } else {
        Vec::new()
    };
    let mut results = search_locations(transport, cache, search_name, now_unix, diagnostics);
    if results.is_empty() && qualifiers.is_empty() && query.contains(' ') {
        let words = query.split_whitespace().collect::<Vec<_>>();
        if let Some((qualifier, name)) = words.split_last() {
            let shorter_name = name.join(" ");
            results = search_locations(transport, cache, &shorter_name, now_unix, diagnostics);
            qualifiers.push(qualifier);
        }
    }
    if results.is_empty() {
        return None;
    }
    let wanted = qualifiers.into_iter().map(search_key).collect::<Vec<_>>();
    let candidate_keys = results
        .iter()
        .map(|result| {
            search_key(&format!(
                "{} {} {}",
                result.admin1, result.country, result.country_code
            ))
        })
        .collect::<Vec<_>>();
    select_location_candidate(&wanted, &candidate_keys)
        .and_then(|index| results.into_iter().nth(index))
}

fn location_label(location: &Location) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in [&location.name, &location.admin1, &location.country] {
        if !part.is_empty() && !parts.contains(&part.as_str()) {
            parts.push(part.as_str());
        }
    }
    parts.join(", ")
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

fn buenos_aires_hour(now_unix: i64) -> String {
    let local_seconds = now_unix.saturating_sub(3 * 3_600);
    let days = local_seconds.div_euclid(86_400);
    let seconds = local_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{:02}", seconds / 3_600)
}

fn number_text(value: &Value) -> Option<String> {
    match value {
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn hour_key(value: &str) -> Option<String> {
    value.get(..13).map(str::to_owned)
}

fn observation(data: &Value, location: &Location, now_unix: i64) -> Option<WeatherObservation> {
    let hourly = data.get("hourly")?;
    let times = hourly.get("time")?.as_array()?;
    let hours = times
        .iter()
        .map(Value::as_str)
        .map(|time| time.and_then(hour_key))
        .collect::<Option<Vec<_>>>()?;
    let provider_hour = data
        .get("current")
        .and_then(|current| current.get("time"))
        .and_then(Value::as_str)
        .and_then(|time| time.get(..13));
    let index = select_forecast_hour(&hours, provider_hour, &buenos_aires_hour(now_unix))?;
    let at = |key: &str| hourly.get(key)?.as_array()?.get(index);
    let visibility_meters = at("visibility")?.as_f64()?;
    if !visibility_meters.is_finite() {
        return None;
    }
    Some(WeatherObservation {
        location: location_label(location),
        apparent_temperature: number_text(at("apparent_temperature")?)?,
        precipitation_probability: number_text(at("precipitation_probability")?)?,
        weather_code: at("weather_code")?.as_i64()?,
        cloud_cover: number_text(at("cloud_cover")?)?,
        visibility_meters,
    })
}

#[must_use]
pub fn load_weather<T: WeatherTransport, C: WeatherCache>(
    transport: &T,
    cache: &mut C,
    requested: &str,
    now_unix: i64,
) -> WeatherLoad {
    let mut diagnostics = Vec::new();
    let Some(location) = resolve_location(transport, cache, requested, now_unix, &mut diagnostics)
    else {
        return WeatherLoad {
            observation: None,
            diagnostics,
        };
    };
    let data = cached_request(
        transport,
        cache,
        &WeatherRequest::Forecast {
            latitude: location.latitude,
            longitude: location.longitude,
        },
        now_unix,
        &mut diagnostics,
    );
    let observation = data.and_then(|data| observation(&data, &location, now_unix));
    if observation.is_none() {
        diagnostics.push(format!(
            "weather response had no current row for {requested}"
        ));
    }
    WeatherLoad {
        observation,
        diagnostics,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::convert::Infallible;

    use super::{
        HttpResponse, TransportFailureKind, WeatherCache, WeatherRequest, WeatherTransport,
        compatible_cache_key, load_weather, python_cache_arguments,
    };

    struct Transport {
        responses: RefCell<VecDeque<Result<HttpResponse, TransportFailureKind>>>,
        requests: RefCell<Vec<WeatherRequest>>,
    }

    impl WeatherTransport for Transport {
        fn get(&self, request: &WeatherRequest) -> Result<HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            self.responses
                .borrow_mut()
                .pop_front()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    #[derive(Default)]
    struct Cache {
        gets: VecDeque<Result<Option<String>, Infallible>>,
        sets: Vec<(String, String, i64)>,
    }

    impl WeatherCache for Cache {
        type Error = Infallible;

        fn get(&mut self, _key: &str) -> Result<Option<String>, Self::Error> {
            self.gets.pop_front().unwrap_or(Ok(None))
        }

        fn set(&mut self, key: &str, value: &str, ttl_seconds: i64) -> Result<(), Self::Error> {
            self.sets
                .push((key.to_owned(), value.to_owned(), ttl_seconds));
            Ok(())
        }
    }

    fn response(body: &str) -> Result<HttpResponse, TransportFailureKind> {
        Ok(HttpResponse {
            status_code: 200,
            body: body.to_owned(),
        })
    }

    fn forecast() -> &'static str {
        r#"{"current":{"time":"2026-01-02T10:00"},"hourly":{"time":["2026-01-02T10:00"],"apparent_temperature":[19.5],"precipitation_probability":[20],"weather_code":[1],"cloud_cover":[30],"visibility":[15000]}}"#
    }

    #[test]
    fn cache_arguments_match_python_sorted_json_and_stable_keys() {
        let request = WeatherRequest::Geocode {
            name: "Córdoba".to_owned(),
        };
        assert_eq!(
            python_cache_arguments(&request),
            r#"{"api_url": "https://geocoding-api.open-meteo.com/v1/search", "headers": null, "parameters": {"count": 10, "format": "json", "language": "es", "name": "C\u00f3rdoba"}}"#
        );
        assert_eq!(
            compatible_cache_key(&request),
            "request_cache:8cc40912aa2d6e1b8f8911a575b6869652043c00193a71cf924af0880627d7c0"
        );
        assert_eq!(
            compatible_cache_key(&WeatherRequest::Forecast {
                latitude: -34.6037,
                longitude: -58.3816,
            }),
            "request_cache:0857f9d173445969e435442d8f114d15e528962b5ca6006051145d368e872007"
        );
    }

    #[test]
    fn default_location_skips_geocoding_and_reads_current_forecast() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([response(forecast())])),
            requests: RefCell::default(),
        };
        let mut cache = Cache::default();
        let load = load_weather(&transport, &mut cache, "", 1_767_345_000);
        let observation = load.observation.as_ref();
        assert_eq!(
            observation.map(|value| value.location.as_str()),
            Some("Buenos Aires, Argentina")
        );
        assert_eq!(
            observation.map(|value| value.apparent_temperature.as_str()),
            Some("19.5")
        );
        assert_eq!(transport.requests.borrow().len(), 1);
        assert_eq!(cache.sets[0].2, 1_800);
    }

    #[test]
    fn resolves_qualified_location_and_preserves_provider_order() {
        let geocoding = r#"{"results":[{"name":"Example City","country":"Otherland","latitude":1.0,"longitude":2.0},{"name":"Example City","admin1":"North","country":"Exampleland","country_code":"EX","latitude":3.0,"longitude":4.0}]}"#;
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([response(geocoding), response(forecast())])),
            requests: RefCell::default(),
        };
        let mut cache = Cache::default();
        let load = load_weather(
            &transport,
            &mut cache,
            "Example City, Exampleland",
            1_767_345_000,
        );
        assert_eq!(
            load.observation.map(|value| value.location),
            Some("Example City, North, Exampleland".to_owned())
        );
        assert!(matches!(
            transport.requests.borrow().get(1),
            Some(WeatherRequest::Forecast {
                latitude: 3.0,
                longitude: 4.0
            })
        ));
    }

    #[test]
    fn fresh_cache_is_authoritative_and_stale_cache_survives_retry_failures() {
        let fresh = format!(r#"{{"timestamp":100,"data":{}}}"#, forecast());
        let stale = format!(r#"{{"timestamp":1,"data":{}}}"#, forecast());
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                Err(TransportFailureKind::Timeout),
                Err(TransportFailureKind::Connection),
            ])),
            requests: RefCell::default(),
        };
        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some(fresh))]),
            ..Cache::default()
        };
        assert!(
            load_weather(&transport, &mut cache, "CABA", 100)
                .observation
                .is_some()
        );
        assert!(transport.requests.borrow().is_empty());

        let mut cache = Cache {
            gets: VecDeque::from([Ok(Some(stale))]),
            ..Cache::default()
        };
        let load = load_weather(&transport, &mut cache, "CABA", 10_000);
        assert!(load.observation.is_some());
        assert_eq!(transport.requests.borrow().len(), 2);
        assert_eq!(load.diagnostics.len(), 2);
    }

    #[test]
    fn missing_locations_and_malformed_forecasts_are_safe() {
        let transport = Transport {
            responses: RefCell::new(VecDeque::from([
                response(r#"{"results":[]}"#),
                response(r#"{"results":[]}"#),
            ])),
            requests: RefCell::default(),
        };
        let mut cache = Cache::default();
        assert!(
            load_weather(&transport, &mut cache, "Missing Place", 100)
                .observation
                .is_none()
        );

        let transport = Transport {
            responses: RefCell::new(VecDeque::from([response(
                r#"{"hourly":{"time":["ééééééé"],"apparent_temperature":[1],"precipitation_probability":[2],"weather_code":[3],"cloud_cover":[4],"visibility":[5]}}"#,
            )])),
            requests: RefCell::default(),
        };
        assert!(
            load_weather(&transport, &mut Cache::default(), "CABA", 100)
                .observation
                .is_none()
        );
    }
}
