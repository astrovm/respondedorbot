//! Fixed-precision credit amounts.

use thiserror::Error;

/// Number of stored units in one displayed credit.
pub const CREDIT_SCALE: i64 = 100;

/// Scale used by legacy tenth-credit records.
pub const LEGACY_CREDIT_SCALE: i64 = 10;

/// A credit amount stored in hundredths without floating-point arithmetic.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CreditUnits(i64);

impl CreditUnits {
    /// Construct an amount from stored hundredth-credit units.
    #[must_use]
    pub const fn new(units: i64) -> Self {
        Self(units)
    }

    /// Return the stored hundredth-credit units.
    #[must_use]
    pub const fn value(self) -> i64 {
        self.0
    }
}

/// A fixed-precision credit conversion failure.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum CreditUnitError {
    /// The requested value cannot be represented by the storage type.
    #[error("credit amount is outside the supported range")]
    Overflow,
    /// The source scale cannot be converted exactly to hundredths.
    #[error("unsupported credit scale")]
    UnsupportedScale,
}

/// Convert whole credits into internal hundredth-credit units.
pub fn whole_credits_to_units(credits: i64) -> Result<CreditUnits, CreditUnitError> {
    credits
        .checked_mul(CREDIT_SCALE)
        .map(CreditUnits::new)
        .ok_or(CreditUnitError::Overflow)
}

/// Convert stored units from a known scale into hundredth-credit units.
pub fn rescale_credit_units(
    units: i64,
    source_scale: Option<i64>,
) -> Result<CreditUnits, CreditUnitError> {
    let source_scale = source_scale
        .filter(|scale| *scale != 0)
        .unwrap_or(LEGACY_CREDIT_SCALE);
    if source_scale <= 0 || CREDIT_SCALE % source_scale != 0 {
        return Err(CreditUnitError::UnsupportedScale);
    }

    units
        .checked_mul(CREDIT_SCALE / source_scale)
        .map(CreditUnits::new)
        .ok_or(CreditUnitError::Overflow)
}

/// Parse a finite decimal credit amount that resolves exactly to hundredths.
///
/// The accepted grammar includes an optional sign, decimal point, digit
/// separators, and base-10 exponent. Values with fractions smaller than one
/// stored unit return `None`.
#[must_use]
pub fn parse_credit_units(value: &str) -> Option<CreditUnits> {
    let normalized: String = value
        .trim()
        .chars()
        .filter(|character| *character != '_')
        .collect();
    if normalized.is_empty() {
        return None;
    }

    let (negative, unsigned) = match normalized.as_bytes().first() {
        Some(b'-') => (true, &normalized[1..]),
        Some(b'+') => (false, &normalized[1..]),
        _ => (false, normalized.as_str()),
    };
    if unsigned.is_empty() {
        return None;
    }

    let mut exponent_parts = unsigned.split(['e', 'E']);
    let mantissa = exponent_parts.next()?;
    let exponent = match exponent_parts.next() {
        Some(raw) if !raw.is_empty() => raw.parse::<i32>().ok()?,
        Some(_) => return None,
        None => 0,
    };
    if exponent_parts.next().is_some() {
        return None;
    }

    let mut decimal_parts = mantissa.split('.');
    let whole = decimal_parts.next()?;
    let fractional = decimal_parts.next().unwrap_or("");
    if decimal_parts.next().is_some() || (whole.is_empty() && fractional.is_empty()) {
        return None;
    }
    if !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fractional.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }

    let digits = format!("{whole}{fractional}");
    let coefficient = digits.parse::<i128>().ok()?;
    if coefficient == 0 {
        return Some(CreditUnits::new(0));
    }

    let fractional_digits = i32::try_from(fractional.len()).ok()?;
    let scale_power = exponent.checked_add(2)?.checked_sub(fractional_digits)?;
    let scaled = if scale_power >= 0 {
        let power = u32::try_from(scale_power).ok()?;
        coefficient.checked_mul(10_i128.checked_pow(power)?)?
    } else {
        let power = scale_power
            .checked_abs()
            .and_then(|value| u32::try_from(value).ok())?;
        let divisor = 10_i128.checked_pow(power)?;
        if coefficient % divisor != 0 {
            return None;
        }
        coefficient / divisor
    };

    let signed = if negative {
        scaled.checked_neg()?
    } else {
        scaled
    };
    i64::try_from(signed).ok().map(CreditUnits::new)
}

/// Render hundredth-credit units with exactly two decimal places.
#[must_use]
pub fn format_credit_units(units: CreditUnits) -> String {
    let sign = if units.value() < 0 { "-" } else { "" };
    let absolute = units.value().unsigned_abs();
    let whole = absolute / CREDIT_SCALE.unsigned_abs();
    let decimal = absolute % CREDIT_SCALE.unsigned_abs();
    format!("{sign}{whole}.{decimal:02}")
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde::Deserialize;

    use super::{
        CreditUnitError, CreditUnits, format_credit_units, parse_credit_units,
        rescale_credit_units, whole_credits_to_units,
    };

    #[derive(Debug, Deserialize)]
    struct Contract {
        parse: Vec<ParseCase>,
        format: Vec<FormatCase>,
        rescale: Vec<RescaleCase>,
        whole: Vec<WholeCase>,
    }

    #[derive(Debug, Deserialize)]
    struct ParseCase {
        input: String,
        expected: Option<i64>,
    }

    #[derive(Debug, Deserialize)]
    struct FormatCase {
        units: i64,
        expected: String,
    }

    #[derive(Debug, Deserialize)]
    struct RescaleCase {
        units: i64,
        source_scale: Option<i64>,
        expected: Option<i64>,
        error: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    struct WholeCase {
        credits: i64,
        expected: i64,
    }

    fn contract() -> Result<Contract, serde_json::Error> {
        let raw = include_str!("../../../contracts/credit_units.json");
        serde_json::from_str(raw)
    }

    #[test]
    fn matches_parse_contract() -> Result<(), serde_json::Error> {
        for case in contract()?.parse {
            assert_eq!(
                parse_credit_units(&case.input).map(CreditUnits::value),
                case.expected,
                "input={:?}",
                case.input
            );
        }
        Ok(())
    }

    #[test]
    fn matches_format_contract() -> Result<(), serde_json::Error> {
        for case in contract()?.format {
            assert_eq!(
                format_credit_units(CreditUnits::new(case.units)),
                case.expected,
                "units={}",
                case.units
            );
        }
        Ok(())
    }

    #[test]
    fn matches_rescale_contract() -> Result<(), serde_json::Error> {
        for case in contract()?.rescale {
            let result = rescale_credit_units(case.units, case.source_scale);
            assert_ne!(
                case.expected.is_some(),
                case.error.is_some(),
                "rescale contract case must define exactly one result"
            );
            if let Some(expected) = case.expected {
                assert_eq!(result.map(CreditUnits::value), Ok(expected));
            } else {
                assert_eq!(
                    result.map_err(|error| error.to_string()),
                    Err(case.error.unwrap_or_default())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn matches_whole_credit_contract() -> Result<(), serde_json::Error> {
        for case in contract()?.whole {
            assert_eq!(
                whole_credits_to_units(case.credits).map(CreditUnits::value),
                Ok(case.expected)
            );
        }
        Ok(())
    }

    #[test]
    fn reports_overflow_without_panicking() {
        assert_eq!(
            whole_credits_to_units(i64::MAX),
            Err(CreditUnitError::Overflow)
        );
        assert_eq!(
            rescale_credit_units(i64::MAX, Some(10)),
            Err(CreditUnitError::Overflow)
        );
    }

    #[test]
    fn formats_the_minimum_storage_value() {
        assert_eq!(
            format_credit_units(CreditUnits::new(i64::MIN)),
            "-92233720368547758.08"
        );
    }

    proptest! {
        #[test]
        fn formatted_storage_values_parse_exactly(units in any::<i64>()) {
            let amount = CreditUnits::new(units);
            prop_assert_eq!(parse_credit_units(&format_credit_units(amount)), Some(amount));
        }

        #[test]
        fn current_scale_rescaling_is_an_identity(units in any::<i64>()) {
            prop_assert_eq!(
                rescale_credit_units(units, Some(100)),
                Ok(CreditUnits::new(units))
            );
        }
    }
}
