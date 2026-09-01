//! Arbitrary-precision base conversion for the `/convertbase` command.

use num_bigint::{BigInt, BigUint};
use unicode_normalization::UnicodeNormalization;

/// A localized validation outcome or successful conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BaseConversion {
    Success {
        number: String,
        source: u32,
        result: String,
        target: u32,
    },
    Usage,
    AlphanumericRequired,
    SourceRange {
        input: String,
    },
    TargetRange {
        input: String,
    },
    NumbersRequired,
}

/// Numeric text that cannot be represented by the native parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedNumericInput;

/// Parse and execute the base-conversion command.
pub fn convert_base(input: &str) -> Result<BaseConversion, UnsupportedNumericInput> {
    let parts: Vec<_> = input.split(',').collect();
    if parts.len() != 3 {
        return Ok(BaseConversion::Usage);
    }
    let number = parts[0].trim();
    let source_input = parts[1].trim();
    let target_input = parts[2].trim();
    let normalized_number = number.nfkc().collect::<String>();
    let normalized_source = source_input.nfkc().collect::<String>();
    let normalized_target = target_input.nfkc().collect::<String>();
    if !normalized_number.is_ascii()
        || !normalized_source.is_ascii()
        || !normalized_target.is_ascii()
    {
        return Err(UnsupportedNumericInput);
    }

    let Some(source_integer) = BigInt::parse_bytes(normalized_source.as_bytes(), 10) else {
        return Ok(BaseConversion::NumbersRequired);
    };
    let Some(target_integer) = BigInt::parse_bytes(normalized_target.as_bytes(), 10) else {
        return Ok(BaseConversion::NumbersRequired);
    };
    if !normalized_number
        .chars()
        .all(|character| character.is_ascii_alphanumeric())
    {
        return Ok(BaseConversion::AlphanumericRequired);
    }

    let minimum = BigInt::from(2_u8);
    let maximum = BigInt::from(36_u8);
    if source_integer < minimum || source_integer > maximum {
        return Ok(BaseConversion::SourceRange {
            input: source_input.to_owned(),
        });
    }
    if target_integer < minimum || target_integer > maximum {
        return Ok(BaseConversion::TargetRange {
            input: target_input.to_owned(),
        });
    }
    let Some(source) = source_integer.to_u32_digits().1.first().copied() else {
        return Ok(BaseConversion::NumbersRequired);
    };
    let Some(target) = target_integer.to_u32_digits().1.first().copied() else {
        return Ok(BaseConversion::NumbersRequired);
    };

    let mut value = BigUint::from(0_u8);
    for character in normalized_number.chars() {
        let Some(digit) = character.to_digit(36) else {
            return Ok(BaseConversion::AlphanumericRequired);
        };
        value *= source;
        value += digit;
    }

    let mut digits = Vec::new();
    while value != BigUint::from(0_u8) {
        let remainder = (&value % target)
            .to_u32_digits()
            .first()
            .copied()
            .unwrap_or(0);
        let Some(character) = char::from_digit(remainder, 36) else {
            return Ok(BaseConversion::NumbersRequired);
        };
        digits.push(character.to_ascii_uppercase());
        value /= target;
    }
    digits.reverse();

    Ok(BaseConversion::Success {
        number: number.to_owned(),
        source,
        result: digits.into_iter().collect(),
        target,
    })
}

#[cfg(test)]
mod tests {
    use super::{BaseConversion, convert_base};

    #[test]
    fn converts_without_machine_integer_limits() {
        assert_eq!(
            convert_base("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF, 16, 2"),
            Ok(BaseConversion::Success {
                number: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF".to_owned(),
                source: 16,
                result: "1".repeat(128),
                target: 2,
            })
        );
    }

    #[test]
    fn preserves_zero_as_an_empty_legacy_result() {
        assert_eq!(
            convert_base("0,2,10"),
            Ok(BaseConversion::Success {
                number: "0".to_owned(),
                source: 2,
                result: String::new(),
                target: 10,
            })
        );
    }

    #[test]
    fn preserves_legacy_digit_outside_source_base_behavior() {
        assert_eq!(
            convert_base("2,2,10"),
            Ok(BaseConversion::Success {
                number: "2".to_owned(),
                source: 2,
                result: "2".to_owned(),
                target: 10,
            })
        );
    }

    #[test]
    fn validates_structure_characters_and_ranges_in_order() {
        assert_eq!(convert_base("101,2"), Ok(BaseConversion::Usage));
        assert_eq!(
            convert_base("101,base,10"),
            Ok(BaseConversion::NumbersRequired)
        );
        assert_eq!(
            convert_base("10!,2,10"),
            Ok(BaseConversion::AlphanumericRequired)
        );
        assert_eq!(
            convert_base("101,999999999999999999999999999,10"),
            Ok(BaseConversion::SourceRange {
                input: "999999999999999999999999999".to_owned()
            })
        );
        assert_eq!(
            convert_base("101,2,-3"),
            Ok(BaseConversion::TargetRange {
                input: "-3".to_owned()
            })
        );
    }

    #[test]
    fn accepts_compatibility_decimal_digits_without_changing_display_text() {
        assert_eq!(
            convert_base("１２,10,16"),
            Ok(BaseConversion::Success {
                number: "１２".to_owned(),
                source: 10,
                result: "C".to_owned(),
                target: 16,
            })
        );
    }
}
