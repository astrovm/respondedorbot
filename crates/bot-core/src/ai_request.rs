//! Pure normalization applied to stored assistant text before an AI request.

#[must_use]
pub fn sanitize_assistant_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|character| !(0x1F000..=0x1FFFF).contains(&u32::from(*character)))
        .collect::<String>()
        .trim_end_matches('.')
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::sanitize_assistant_text;
    use proptest::prelude::*;

    #[test]
    fn lowercases_removes_supplementary_symbols_and_trailing_dots() {
        assert_eq!(sanitize_assistant_text("HOLA😀..."), "hola");
        assert_eq!(sanitize_assistant_text("İSTANBUL."), "i\u{307}stanbul");
        assert_eq!(sanitize_assistant_text("Keep.   "), "keep.   ");
        assert_eq!(sanitize_assistant_text("🀀CARD"), "card");
    }

    proptest! {
        #[test]
        fn sanitization_is_idempotent_and_excludes_the_legacy_range(value in ".{0,256}") {
            let once = sanitize_assistant_text(&value);
            prop_assert_eq!(sanitize_assistant_text(&once), once.clone());
            let excludes_legacy_range = once.chars().all(|character| {
                !(0x1F000..=0x1FFFF).contains(&u32::from(character))
            });
            prop_assert!(excludes_legacy_range);
        }
    }
}
