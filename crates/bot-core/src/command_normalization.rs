//! Final normalization for text converted into a Telegram command.

use std::{collections::HashMap, sync::OnceLock};

use unicode_normalization::UnicodeNormalization;

use crate::locale::Locale;

fn spanish_emoji_names() -> &'static HashMap<String, String> {
    static NAMES: OnceLock<HashMap<String, String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        serde_json::from_str(include_str!("../data/emoji_es.json")).unwrap_or_default()
    })
}

fn emoji_name(name: &str, value: &str, locale: Locale) -> String {
    let localized = match locale {
        Locale::Es => spanish_emoji_names()
            .get(value)
            .map_or("emoji", |localized| localized.trim_matches(':')),
        Locale::En => name,
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

fn transliterate_remaining_scripts(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for character in input.chars() {
        let latin_with_diacritic = character
            .to_string()
            .nfd()
            .any(|part| part.is_ascii_alphanumeric());
        if character.is_ascii() || matches!(character, 'ñ' | 'Ñ') || latin_with_diacritic {
            result.push(character);
        } else if let Some(transliterated) = deunicode::deunicode_char(character) {
            result.push_str(transliterated);
        }
    }
    result
}

/// Expand emoji names and romanize Unicode text before command normalization.
#[must_use]
pub fn preprocess_command_text(input: &str, locale: Locale) -> String {
    let demojized = demojize(input, locale);
    let japanese_romanized = if contains_japanese(&demojized) {
        romanize_japanese(&demojized)
    } else {
        demojized
    };
    transliterate_remaining_scripts(&japanese_romanized)
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
    use super::{normalize_command_text, preprocess_command_text, spanish_emoji_names};
    use crate::locale::Locale;

    #[test]
    fn normalizes_supported_command_text() {
        let cases = [
            ("h3llo W0RLD", Some("/H3LLO_W0RLD")),
            (
                "hello! world? or... mmm ...bye.",
                Some(
                    "/HELLO_SIGNODEEXCLAMACION_WORLD_SIGNODEPREGUNTA_OR_PUNTOSSUSPENSIVOS_MMM_PUNTOSSUSPENSIVOS_BYE_PUNTO",
                ),
            ),
            ("  hello   world ", Some("/HELLO_WORLD")),
            (
                "_cara_sonriendo_con_ojos_sonrientes_hello _cara_sonriendo_con_ojos_sonrientes_ world",
                Some(
                    "/CARA_SONRIENDO_CON_OJOS_SONRIENTES_HELLO_CARA_SONRIENDO_CON_OJOS_SONRIENTES_WORLD",
                ),
            ),
            ("hola ñandú ñ", Some("/HOLA_NIANDU_ENIE")),
            ("hola\nlinea\n", Some("/HOLA_LINEA")),
            ("mousugudesu", Some("/MOUSUGUDESU")),
            ("katakana", Some("/KATAKANA")),
            ("💥", None),
            ("", None),
        ];

        for (input, expected) in cases {
            assert_eq!(normalize_command_text(input).as_deref(), expected);
        }
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
            preprocess_command_text("prueba ❤️", Locale::Es),
            "prueba _corazón_rojo_"
        );
        assert_eq!(
            normalize_command_text(&preprocess_command_text("prueba ❤️", Locale::Es)),
            Some("/PRUEBA_CORAZON_ROJO".to_owned())
        );
        assert_eq!(
            preprocess_command_text("もうすぐです", Locale::Es),
            "mousugudesu"
        );
        assert_eq!(preprocess_command_text("ｶﾀｶﾅ", Locale::Es), "katakana");
        assert_eq!(
            normalize_command_text(&preprocess_command_text("テスト", Locale::Es)),
            Some("/TESUTO".to_owned())
        );
    }

    #[test]
    fn transliterates_other_writing_systems_into_commands() {
        let cases = [
            ("Привет мир", "/PRIVET_MIR"),
            ("Καλημέρα κόσμε", "/KALEMERA_KOSME"),
            ("안녕하세요", "/ANNYEONGHASEYO"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_command_text(&preprocess_command_text(input, Locale::Es)).as_deref(),
                Some(expected)
            );
        }

        assert_eq!(
            normalize_command_text(&preprocess_command_text("ñandú", Locale::Es)),
            Some("/NIANDU".to_owned())
        );
    }

    #[test]
    fn spanish_emoji_names_cover_unicode_seventeen_without_english_fallbacks() {
        let cases = [
            ("❤️", "_corazón_rojo_"),
            ("🇦🇷", "_bandera_argentina_"),
            (
                "👩🏽‍💻",
                "_profesional_de_la_tecnología_mujer_tono_de_piel_medio_",
            ),
            ("🪉", "_arpa_"),
            ("🫆", "_huella_dactilar_"),
        ];
        for (emoji, expected) in cases {
            assert_eq!(preprocess_command_text(emoji, Locale::Es), expected);
        }

        let missing = emojis::iter()
            .filter(|emoji| !spanish_emoji_names().contains_key(emoji.as_str()))
            .map(emojis::Emoji::as_str)
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "Spanish names are missing for recognized emoji: {missing:?}"
        );
        assert!(spanish_emoji_names().len() >= 5_000);
        let unsupported_non_components = spanish_emoji_names()
            .keys()
            .filter(|emoji| emojis::get(emoji).is_none())
            .filter(|emoji| {
                !matches!(
                    emoji.as_str(),
                    "🏻" | "🏼" | "🏽" | "🏾" | "🏿" | "🦰" | "🦱" | "🦳" | "🦲"
                )
            })
            .collect::<Vec<_>>();
        assert!(
            unsupported_non_components.is_empty(),
            "Spanish names contain unsupported emoji: {unsupported_non_components:?}"
        );
    }
}
