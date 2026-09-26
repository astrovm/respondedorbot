//! Process-wide cache for regular expressions built from static patterns.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use regex::Regex;

static CACHE: LazyLock<Mutex<HashMap<&'static str, Option<Regex>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Return the compiled pattern, compiling it only the first time it is seen.
/// Cloning a `Regex` shares its compiled program, so repeat calls are cheap.
#[must_use]
pub fn cached_regex(pattern: &'static str) -> Option<Regex> {
    let Ok(mut cache) = CACHE.lock() else {
        return Regex::new(pattern).ok();
    };
    cache
        .entry(pattern)
        .or_insert_with(|| Regex::new(pattern).ok())
        .clone()
}

#[cfg(test)]
mod tests {
    use super::cached_regex;

    #[test]
    fn compiles_once_and_reuses_patterns() {
        let first = cached_regex(r"^a\d+$");
        let second = cached_regex(r"^a\d+$");
        assert!(first.as_ref().is_some_and(|regex| regex.is_match("a12")));
        assert_eq!(
            first.map(|regex| regex.as_str().to_owned()),
            second.map(|regex| regex.as_str().to_owned())
        );
    }

    #[test]
    fn remembers_invalid_patterns_as_missing() {
        assert!(cached_regex("(").is_none());
        assert!(cached_regex("(").is_none());
    }
}
