//! Locale selection shared by native command and callback flows.

use chrono::{Datelike, NaiveDate, Weekday};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locale {
    Es,
    En,
}

impl Locale {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Es => "es",
            Self::En => "en",
        }
    }
}

#[must_use]
pub fn format_date(date: NaiveDate, locale: Locale) -> String {
    format!(
        "{} {}",
        weekday_name(date.weekday(), locale),
        date.format("%d/%m/%Y")
    )
}

const fn weekday_name(weekday: Weekday, locale: Locale) -> &'static str {
    match (locale, weekday) {
        (Locale::Es, Weekday::Mon) => "lunes",
        (Locale::Es, Weekday::Tue) => "martes",
        (Locale::Es, Weekday::Wed) => "miércoles",
        (Locale::Es, Weekday::Thu) => "jueves",
        (Locale::Es, Weekday::Fri) => "viernes",
        (Locale::Es, Weekday::Sat) => "sábado",
        (Locale::Es, Weekday::Sun) => "domingo",
        (Locale::En, Weekday::Mon) => "Monday",
        (Locale::En, Weekday::Tue) => "Tuesday",
        (Locale::En, Weekday::Wed) => "Wednesday",
        (Locale::En, Weekday::Thu) => "Thursday",
        (Locale::En, Weekday::Fri) => "Friday",
        (Locale::En, Weekday::Sat) => "Saturday",
        (Locale::En, Weekday::Sun) => "Sunday",
    }
}

#[must_use]
pub fn normalize_locale(value: &str, default: Locale) -> Locale {
    let normalized = value.trim().to_lowercase().replace('_', "-");
    if normalized == "en" || normalized.starts_with("en-") {
        Locale::En
    } else if normalized == "es" || normalized.starts_with("es-") {
        Locale::Es
    } else {
        default
    }
}

#[must_use]
pub fn resolve_locale(
    configured: Option<&str>,
    telegram_language_code: Option<&str>,
    chat_type: &str,
) -> Locale {
    match configured.unwrap_or("auto").trim().to_lowercase().as_str() {
        "en" => Locale::En,
        "es" => Locale::Es,
        _ if chat_type == "private" => {
            normalize_locale(telegram_language_code.unwrap_or_default(), Locale::Es)
        }
        _ => Locale::Es,
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::{Locale, format_date, normalize_locale, resolve_locale};

    #[test]
    fn normalizes_language_and_region_codes() {
        assert_eq!(normalize_locale("en-US", Locale::Es), Locale::En);
        assert_eq!(normalize_locale(" ES_ar ", Locale::En), Locale::Es);
        assert_eq!(normalize_locale("pt-BR", Locale::En), Locale::En);
        assert_eq!(Locale::Es.code(), "es");
        assert_eq!(Locale::En.code(), "en");
    }

    #[test]
    fn configured_locale_wins_and_auto_uses_private_telegram_language() {
        assert_eq!(
            resolve_locale(Some("auto"), Some("en-US"), "private"),
            Locale::En
        );
        assert_eq!(
            resolve_locale(Some("auto"), Some("en"), "group"),
            Locale::Es
        );
        assert_eq!(
            resolve_locale(Some("es"), Some("en"), "private"),
            Locale::Es
        );
        assert_eq!(resolve_locale(Some("en"), Some("es"), "group"), Locale::En);
    }

    #[test]
    fn formats_weekdays_in_the_selected_locale() {
        let Some(date) = NaiveDate::from_ymd_opt(2026, 9, 1) else {
            return;
        };
        assert_eq!(format_date(date, Locale::Es), "martes 01/09/2026");
        assert_eq!(format_date(date, Locale::En), "Tuesday 01/09/2026");
    }
}
