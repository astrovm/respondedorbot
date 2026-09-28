//! Arbitrary-precision base conversion for the `/convertbase` command.

use num_bigint::{BigInt, BigUint};
use unicode_normalization::UnicodeNormalization;

const DIGITS: &[u8; 36] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

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
    // `to_digit(36)` accepts exactly the ASCII alphanumerics.
    let Some(number_digits) = normalized_number
        .chars()
        .map(|character| character.to_digit(36))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(BaseConversion::AlphanumericRequired);
    };

    let base = |value: &BigInt| {
        u32::try_from(value)
            .ok()
            .filter(|base| (2..=36).contains(base))
    };
    let Some(source) = base(&source_integer) else {
        return Ok(BaseConversion::SourceRange {
            input: source_input.to_owned(),
        });
    };
    let Some(target) = base(&target_integer) else {
        return Ok(BaseConversion::TargetRange {
            input: target_input.to_owned(),
        });
    };

    let mut value = BigUint::from(0_u8);
    for digit in number_digits {
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
        // The remainder is below `target`, which is at most 36.
        digits.push(char::from(DIGITS[remainder as usize]));
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

    #[test]
    fn rejects_non_ascii_bases_and_non_numeric_targets() {
        assert_eq!(
            convert_base("10,２é,10"),
            Err(super::UnsupportedNumericInput)
        );
        assert_eq!(convert_base("10,2,x"), Ok(BaseConversion::NumbersRequired));
        assert_eq!(
            convert_base("zz,36,36"),
            Ok(BaseConversion::Success {
                number: "zz".to_owned(),
                source: 36,
                result: "ZZ".to_owned(),
                target: 36,
            })
        );
    }
}
