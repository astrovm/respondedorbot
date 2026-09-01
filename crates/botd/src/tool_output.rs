//! Localized public output shared by native AI tools.

use bot_core::locale::Locale;

#[must_use]
pub fn incompatible(locale: Locale, tool: &str) -> String {
    match locale {
        Locale::Es => format!("la herramienta '{tool}' recibió una solicitud incompatible"),
        Locale::En => format!("tool '{tool}' received an incompatible request"),
    }
}

#[must_use]
pub fn failed(locale: Locale, tool: &str) -> String {
    match locale {
        Locale::Es => format!("falló la herramienta '{tool}'"),
        Locale::En => format!("tool '{tool}' failed"),
    }
}

#[must_use]
pub fn unavailable(locale: Locale, tool: &str) -> String {
    match locale {
        Locale::Es => format!("la herramienta '{tool}' no está disponible"),
        Locale::En => format!("tool '{tool}' is unavailable"),
    }
}

#[must_use]
pub fn unknown(locale: Locale, tool: &str) -> String {
    match locale {
        Locale::Es => format!("herramienta desconocida: {tool}"),
        Locale::En => format!("unknown tool: {tool}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_shared_message_in_both_locales() {
        assert_eq!(
            incompatible(Locale::Es, "weather"),
            "la herramienta 'weather' recibió una solicitud incompatible"
        );
        assert_eq!(
            incompatible(Locale::En, "weather"),
            "tool 'weather' received an incompatible request"
        );
        assert_eq!(
            failed(Locale::Es, "weather"),
            "falló la herramienta 'weather'"
        );
        assert_eq!(failed(Locale::En, "weather"), "tool 'weather' failed");
        assert_eq!(
            unavailable(Locale::Es, "weather"),
            "la herramienta 'weather' no está disponible"
        );
        assert_eq!(
            unavailable(Locale::En, "weather"),
            "tool 'weather' is unavailable"
        );
        assert_eq!(
            unknown(Locale::Es, "weather"),
            "herramienta desconocida: weather"
        );
        assert_eq!(unknown(Locale::En, "weather"), "unknown tool: weather");
    }
}
