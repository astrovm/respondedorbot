//! Final normalization for text converted into a Telegram command.

use std::sync::OnceLock;

use unicode_normalization::UnicodeNormalization;

use crate::locale::Locale;

fn emoji_name(name: &str, value: &str, locale: Locale) -> String {
    let localized = match (locale, value) {
        (Locale::Es, "😄") => "cara sonriendo con ojos sonrientes",
        (Locale::Es, "💥") => "colisión",
        _ => name,
    };
    localized
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('_')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn demojize(input: &str, locale: Locale) -> String {
    let mut result = String::with_capacity(input.len());
    let mut start = 0;
    while start < input.len() {
        let mut matched = None;
        for (end, _) in input[start..].char_indices().skip(1) {
            let end = start + end;
            if let Some(emoji) = emojis::get(&input[start..end]) {
                matched = Some((end, emoji));
            }
        }
        if let Some(emoji) = emojis::get(&input[start..]) {
            matched = Some((input.len(), emoji));
        }
        if let Some((end, emoji)) = matched {
            result.push('_');
            result.push_str(&emoji_name(emoji.name(), emoji.as_str(), locale));
            result.push('_');
            start = end;
        } else {
            let character = input[start..].chars().next().unwrap_or_default();
            result.push(character);
            start += character.len_utf8();
        }
    }
    result
}

fn contains_japanese(input: &str) -> bool {
    input.chars().any(|character| {
        matches!(
            character as u32,
            0x3040..=0x30ff
                | 0x31f0..=0x31ff
                | 0xff65..=0xff9f
                | 0x3400..=0x4dbf
                | 0x4e00..=0x9fff
                | 0xf900..=0xfaff
                | 0x20000..=0x3134f
        )
    })
}

fn romanize_japanese(input: &str) -> String {
    let normalized = input.nfkc().collect::<String>();
    static ROMANIZER: OnceLock<ib_romaji::HepburnRomanizer> = OnceLock::new();
    let romanizer = ROMANIZER.get_or_init(ib_romaji::HepburnRomanizer::default);
    let mut result = String::with_capacity(normalized.len());
    let mut start = 0;
    while start < normalized.len() {
        let remainder = &normalized[start..];
        if let Some((length, romaji)) = romanizer.romanize_vec(remainder).first().copied() {
            result.push_str(romaji);
            start += length;
        } else {
            let character = remainder.chars().next().unwrap_or_default();
            result.push(character);
            start += character.len_utf8();
        }
    }
    result
}

/// Expand emoji names and romanize Japanese text before command normalization.
#[must_use]
pub fn preprocess_command_text(input: &str, locale: Locale) -> String {
    let demojized = demojize(input, locale);
    if contains_japanese(&demojized) {
        romanize_japanese(&demojized)
    } else {
        demojized
    }
}

fn is_word_character(character: char) -> bool {
    character == '_' || character.is_alphanumeric()
}

fn replace_enye(input: &str) -> String {
    let characters: Vec<_> = input.chars().collect();
    let mut result = String::with_capacity(input.len());
    for (index, character) in characters.iter().copied().enumerate() {
        if character != 'Ñ' {
            result.push(character);
            continue;
        }
        let has_word_before = index
            .checked_sub(1)
            .and_then(|previous| characters.get(previous))
            .is_some_and(|value| is_word_character(*value));
        let has_word_after = characters
            .get(index + 1)
            .is_some_and(|value| is_word_character(*value));
        result.push_str(if has_word_before || has_word_after {
            "NI"
        } else {
            "ENIE"
        });
    }
    result
}

fn ascii_without_diacritics(input: &str) -> String {
    input.nfd().filter(char::is_ascii).collect()
}

fn collapse_ascii_whitespace(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut in_whitespace = false;
    for character in input.chars() {
        if character.is_ascii_whitespace() {
            if !in_whitespace {
                result.push(' ');
            }
            in_whitespace = true;
        } else {
            result.push(character);
            in_whitespace = false;
        }
    }
    result
}

fn translate_punctuation(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut remaining = input;
    while let Some(character) = remaining.chars().next() {
        if remaining.starts_with("...") {
            result.push_str("_PUNTOSSUSPENSIVOS_");
            remaining = &remaining[3..];
            continue;
        }
        match character {
            ' ' | '\n' => result.push('_'),
            '?' => result.push_str("_SIGNODEPREGUNTA_"),
            '!' => result.push_str("_SIGNODEEXCLAMACION_"),
            '.' => result.push_str("_PUNTO_"),
            _ => result.push(character),
        }
        remaining = &remaining[character.len_utf8()..];
    }
    result
}

fn clean_command(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut last_was_underscore = false;
    for character in input.chars() {
        if character == '_' {
            if !last_was_underscore {
                result.push(character);
            }
            last_was_underscore = true;
        } else if character.is_ascii_alphanumeric() {
            result.push(character);
            last_was_underscore = false;
        }
    }
    result.trim_matches('_').to_owned()
}

/// Normalize adapter-preprocessed text into a slash command.
#[must_use]
pub fn normalize_command_text(input: &str) -> Option<String> {
    let uppercase = input.to_uppercase();
    let replaced_enye = replace_enye(&uppercase);
    let ascii = ascii_without_diacritics(&replaced_enye);
    let single_spaced = collapse_ascii_whitespace(&ascii);
    let translated = translate_punctuation(&single_spaced);
    let cleaned = clean_command(&translated);
    (!cleaned.is_empty()).then(|| format!("/{cleaned}"))
}

#[cfg(test)]
mod tests {
    use super::{normalize_command_text, preprocess_command_text};
    use crate::locale::Locale;

    #[test]
    fn matches_normalization_contract() -> Result<(), serde_json::Error> {
        #[derive(serde::Deserialize)]
        struct Contract {
            normalization: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            input: String,
            expected: Option<String>,
        }
        let contract: Contract = serde_json::from_str(include_str!(
            "../../../contracts/command_normalization.json"
        ))?;
        for case in contract.normalization {
            assert_eq!(normalize_command_text(&case.input), case.expected);
        }
        Ok(())
    }

    #[test]
    fn preserves_non_overlapping_ellipsis_and_underscore_order() {
        assert_eq!(
            normalize_command_text("a....__b"),
            Some("/A_PUNTOSSUSPENSIVOS_PUNTO_B".to_owned())
        );
    }

    #[test]
    fn preprocesses_localized_emoji_and_japanese_without_python() {
        assert_eq!(
            preprocess_command_text("😄hello 😄 world", Locale::Es),
            "_cara_sonriendo_con_ojos_sonrientes_hello _cara_sonriendo_con_ojos_sonrientes_ world"
        );
        assert_eq!(
            preprocess_command_text("もうすぐです", Locale::Es),
            "mousugudesu"
        );
        assert_eq!(preprocess_command_text("ｶﾀｶﾅ", Locale::Es), "katakana");
    }
}
