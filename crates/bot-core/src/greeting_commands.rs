//! Command classification and localized fallbacks for Giphy greetings.

use crate::locale::Locale;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreetingCategory {
    Morning,
    Night,
}

impl GreetingCategory {
    pub const SEARCH_TERMS: usize = 4;

    #[must_use]
    pub const fn cache_name(self) -> &'static str {
        match self {
            Self::Morning => "gm",
            Self::Night => "gn",
        }
    }

    #[must_use]
    pub const fn search_terms(self) -> [&'static str; Self::SEARCH_TERMS] {
        match self {
            Self::Morning => [
                "good morning",
                "buenos dias",
                "morning coffee",
                "rise and shine",
            ],
            Self::Night => ["good night", "buenas noches", "sweet dreams", "go to sleep"],
        }
    }
}

#[must_use]
pub fn classify_greeting_command(command: &str) -> Option<GreetingCategory> {
    match command {
        "/gm" => Some(GreetingCategory::Morning),
        "/gn" => Some(GreetingCategory::Night),
        _ => None,
    }
}

#[must_use]
pub fn greeting_fallback(category: GreetingCategory, locale: Locale) -> &'static str {
    match (category, locale) {
        (GreetingCategory::Morning, Locale::Es) => "buen día boludo",
        (GreetingCategory::Morning, Locale::En) => "good morning",
        (GreetingCategory::Night, Locale::Es) => "buenas noches boludo",
        (GreetingCategory::Night, Locale::En) => "good night",
    }
}

#[cfg(test)]
mod tests {
    use super::{GreetingCategory, classify_greeting_command, greeting_fallback};
    use crate::locale::Locale;

    #[test]
    fn classifies_both_commands_and_preserves_terms_and_fallbacks() {
        assert_eq!(
            classify_greeting_command("/gm"),
            Some(GreetingCategory::Morning)
        );
        assert_eq!(
            classify_greeting_command("/gn"),
            Some(GreetingCategory::Night)
        );
        assert_eq!(classify_greeting_command("/other"), None);
        assert_eq!(
            GreetingCategory::Morning.search_terms(),
            [
                "good morning",
                "buenos dias",
                "morning coffee",
                "rise and shine"
            ]
        );
        assert_eq!(GreetingCategory::Night.cache_name(), "gn");
        assert_eq!(
            greeting_fallback(GreetingCategory::Morning, Locale::Es),
            "buen día boludo"
        );
        assert_eq!(
            greeting_fallback(GreetingCategory::Night, Locale::En),
            "good night"
        );
    }
}
