//! Weather command planning, rendering, and deterministic provider selection.

use unicode_normalization::UnicodeNormalization;

use crate::locale::Locale;

pub const DEFAULT_WEATHER_LOCATION: &str = "Buenos Aires";

#[derive(Debug, Clone, PartialEq)]
pub struct WeatherObservation {
    pub location: String,
    pub apparent_temperature: String,
    pub precipitation_probability: String,
    pub weather_code: i64,
    pub cloud_cover: String,
    pub visibility_meters: f64,
}

#[must_use]
pub fn classify_weather_command(command: &str) -> bool {
    matches!(command, "/clima" | "/weather")
}

#[must_use]
pub fn requested_location(message_text: &str) -> &str {
    let location = message_text.trim();
    if location.is_empty() {
        DEFAULT_WEATHER_LOCATION
    } else {
        location
    }
}

#[must_use]
pub fn search_key(value: &str) -> String {
    value
        .to_lowercase()
        .nfkd()
        .filter(|character| !unicode_normalization::char::is_combining_mark(*character))
        .collect()
}

#[must_use]
pub fn weather_description(code: i64, locale: Locale) -> &'static str {
    match (code, locale) {
        (0, Locale::Es) => "despejado",
        (0, Locale::En) => "clear",
        (1, Locale::Es) => "mayormente despejado",
        (1, Locale::En) => "mostly clear",
        (2, Locale::Es) => "parcialmente nublado",
        (2, Locale::En) => "partly cloudy",
        (3, Locale::Es) => "nublado",
        (3, Locale::En) => "cloudy",
        (45, Locale::Es) => "neblina",
        (45, Locale::En) => "foggy",
        (48, Locale::Es) => "niebla",
        (48, Locale::En) => "freezing fog",
        (51, Locale::Es) => "llovizna leve",
        (51, Locale::En) => "light drizzle",
        (53, Locale::Es) => "llovizna moderada",
        (53, Locale::En) => "moderate drizzle",
        (55, Locale::Es) => "llovizna intensa",
        (55, Locale::En) => "heavy drizzle",
        (56, Locale::Es) => "llovizna helada leve",
        (56, Locale::En) => "light freezing drizzle",
        (57, Locale::Es) => "llovizna helada intensa",
        (57, Locale::En) => "heavy freezing drizzle",
        (61, Locale::Es) => "lluvia leve",
        (61, Locale::En) => "light rain",
        (63, Locale::Es) => "lluvia moderada",
        (63, Locale::En) => "moderate rain",
        (65, Locale::Es) => "lluvia intensa",
        (65, Locale::En) => "heavy rain",
        (66, Locale::Es) => "lluvia helada leve",
        (66, Locale::En) => "light freezing rain",
        (67, Locale::Es) => "lluvia helada intensa",
        (67, Locale::En) => "heavy freezing rain",
        (71, Locale::Es) => "nevada leve",
        (71, Locale::En) => "light snow",
        (73, Locale::Es) => "nevada moderada",
        (73, Locale::En) => "moderate snow",
        (75, Locale::Es) => "nevada intensa",
        (75, Locale::En) => "heavy snow",
        (77, Locale::Es) => "granizo",
        (77, Locale::En) => "snow grains",
        (80, Locale::Es) => "lluvia leve intermitente",
        (80, Locale::En) => "light rain showers",
        (81, Locale::Es) => "lluvia moderada intermitente",
        (81, Locale::En) => "moderate rain showers",
        (82, Locale::Es) => "lluvia fuerte intermitente",
        (82, Locale::En) => "heavy rain showers",
        (85, Locale::Es) => "nevada leve intermitente",
        (85, Locale::En) => "light snow showers",
        (86, Locale::Es) => "nevada intensa intermitente",
        (86, Locale::En) => "heavy snow showers",
        (95, Locale::Es) => "tormenta",
        (95, Locale::En) => "thunderstorm",
        (96, Locale::Es) => "tormenta con granizo leve",
        (96, Locale::En) => "thunderstorm with light hail",
        (99, Locale::Es) => "tormenta con granizo intenso",
        (99, Locale::En) => "thunderstorm with heavy hail",
        (_, Locale::Es) => "clima raro",
        (_, Locale::En) => "unusual weather",
    }
}

#[must_use]
pub fn render_weather(observation: &WeatherObservation, locale: Locale) -> String {
    let visibility = format!("{:.1}km", observation.visibility_meters / 1_000.0);
    let description = weather_description(observation.weather_code, locale);
    match locale {
        Locale::Es => format!(
            "- Lugar: {}\n- Temperatura aparente: {}°C\n- Probabilidad de lluvia: {}%\n- Estado: {}\n- Nubosidad: {}%\n- Visibilidad: {}",
            observation.location,
            observation.apparent_temperature,
            observation.precipitation_probability,
            description,
            observation.cloud_cover,
            visibility
        ),
        Locale::En => format!(
            "- Location: {}\n- Feels like: {}°C\n- Chance of rain: {}%\n- Condition: {}\n- Cloud cover: {}%\n- Visibility: {}",
            observation.location,
            observation.apparent_temperature,
            observation.precipitation_probability,
            description,
            observation.cloud_cover,
            visibility
        ),
    }
}

#[must_use]
pub fn weather_load_error(location: &str, locale: Locale) -> String {
    match locale {
        Locale::Es => format!("no se pudo obtener el clima de {location}"),
        Locale::En => format!("I could not load the weather for {location}"),
    }
}

/// Choose the best geocoding candidate from adapter-normalized search keys.
///
/// Each candidate key contains its administrative region, country, and country
/// code. Qualifier keys come from the requested location. The first candidate
/// wins when no qualifier matches or when several candidates have the same
/// score, matching the existing provider-order behavior.
#[must_use]
pub fn select_location_candidate(
    qualifier_keys: &[String],
    candidate_keys: &[String],
) -> Option<usize> {
    if candidate_keys.is_empty() {
        return None;
    }
    if qualifier_keys.is_empty() {
        return Some(0);
    }

    let mut best_index = 0;
    let mut best_score = 0;
    for (index, candidate) in candidate_keys.iter().enumerate() {
        let score = qualifier_keys
            .iter()
            .filter(|qualifier| candidate.contains(qualifier.as_str()))
            .count();
        if score > best_score {
            best_index = index;
            best_score = score;
        }
    }
    Some(if best_score == 0 { 0 } else { best_index })
}

/// Select the hourly forecast row matching the provider clock or local clock.
///
/// The adapter validates and normalizes each ISO timestamp to an hourly key.
/// Provider time is checked before local time for each row, preserving provider
/// order when either clock identifies an earlier row.
#[must_use]
pub fn select_forecast_hour(
    forecast_hours: &[String],
    provider_hour: Option<&str>,
    local_hour: &str,
) -> Option<usize> {
    forecast_hours
        .iter()
        .position(|hour| provider_hour.is_some_and(|current| hour == current) || hour == local_hour)
}

#[cfg(test)]
mod tests {
    use super::{
        WeatherObservation, classify_weather_command, render_weather, requested_location,
        search_key, select_forecast_hour, select_location_candidate, weather_description,
        weather_load_error,
    };
    use crate::locale::Locale;

    #[test]
    fn location_selection_defaults_to_the_first_provider_result() {
        assert_eq!(
            select_location_candidate(&[], &["first".to_owned(), "second".to_owned()]),
            Some(0)
        );
        assert_eq!(
            select_location_candidate(
                &["missing".to_owned()],
                &["first".to_owned(), "second".to_owned()]
            ),
            Some(0)
        );
        assert_eq!(select_location_candidate(&["any".to_owned()], &[]), None);
    }

    #[test]
    fn location_selection_uses_all_matching_qualifiers_and_keeps_ties_stable() {
        let qualifiers = vec!["exampleland".to_owned(), "north".to_owned()];
        let candidates = vec![
            "north otherland no".to_owned(),
            "south exampleland ex".to_owned(),
            "north exampleland ex".to_owned(),
            "north exampleland duplicate".to_owned(),
        ];
        assert_eq!(select_location_candidate(&qualifiers, &candidates), Some(2));
    }

    #[test]
    fn forecast_selection_prefers_the_first_matching_row() {
        let hours = vec![
            "2026-01-02T09".to_owned(),
            "2026-01-02T10".to_owned(),
            "2026-01-02T10".to_owned(),
        ];
        assert_eq!(
            select_forecast_hour(&hours, Some("2026-01-02T10"), "2026-01-02T09"),
            Some(0)
        );
        assert_eq!(
            select_forecast_hour(&hours, Some("2026-01-02T11"), "2026-01-02T10"),
            Some(1)
        );
    }

    #[test]
    fn forecast_selection_reports_missing_rows() {
        assert_eq!(
            select_forecast_hour(&["2026-01-02T09".to_owned()], None, "2026-01-02T10"),
            None
        );
    }

    #[test]
    fn command_defaults_and_search_normalization_match_python() {
        assert!(classify_weather_command("/clima"));
        assert!(classify_weather_command("/weather"));
        assert!(!classify_weather_command("/weathering"));
        assert_eq!(requested_location("  "), "Buenos Aires");
        assert_eq!(requested_location("  Córdoba, AR  "), "Córdoba, AR");
        assert_eq!(search_key("CÓRDOBA, Ñ"), "cordoba, n");
    }

    #[test]
    fn renders_localized_weather_and_unknown_codes() {
        let observation = WeatherObservation {
            location: "Example City, Exampleland".to_owned(),
            apparent_temperature: "19.5".to_owned(),
            precipitation_probability: "20".to_owned(),
            weather_code: 1,
            cloud_cover: "30".to_owned(),
            visibility_meters: 15_000.0,
        };
        assert_eq!(
            render_weather(&observation, Locale::Es),
            "- Lugar: Example City, Exampleland\n- Temperatura aparente: 19.5°C\n- Probabilidad de lluvia: 20%\n- Estado: mayormente despejado\n- Nubosidad: 30%\n- Visibilidad: 15.0km"
        );
        assert!(render_weather(&observation, Locale::En).contains("Condition: mostly clear"));
        assert_eq!(weather_description(999, Locale::Es), "clima raro");
        assert_eq!(weather_description(999, Locale::En), "unusual weather");
        assert_eq!(
            weather_load_error("Rosario", Locale::En),
            "I could not load the weather for Rosario"
        );
        assert_eq!(
            weather_load_error("Rosario", Locale::Es),
            "no se pudo obtener el clima de Rosario"
        );
    }

    #[test]
    fn every_open_meteo_description_is_localized() {
        for code in [
            0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82,
            85, 86, 95, 96, 99,
        ] {
            assert_ne!(weather_description(code, Locale::Es), "clima raro");
            assert_ne!(weather_description(code, Locale::En), "unusual weather");
        }
    }
}
