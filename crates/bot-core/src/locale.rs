//! Locale selection shared by native command and callback flows.

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
    use super::{Locale, normalize_locale, resolve_locale};

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
}
