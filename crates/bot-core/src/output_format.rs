//! Shared plain-text formatting for market quotes.
#[must_use]
pub fn price(value: f64) -> String {
    if !value.is_finite() {
        return "N/A".to_owned();
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    if value.abs() < 1e-12 {
        return format!("{value:.4e}");
    }
    let decimals = if value.abs() >= 1.0 {
        4
    } else {
        ((-value.abs().log10()).ceil() as usize + 3).min(16)
    };
    format!("{value:.decimals$}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

#[must_use]
pub fn change(value: Option<f64>) -> String {
    value.filter(|v| v.is_finite()).map_or_else(
        || "N/A".to_owned(),
        |value| {
            let value = if value.abs() < 0.005 { 0.0 } else { value };
            let value = format!("{value:+.2}");
            format!("{}%", value.trim_end_matches('0').trim_end_matches('.'))
        },
    )
}

#[must_use]
pub fn quote(
    symbol: &str,
    amount: f64,
    currency: &str,
    variation: Option<f64>,
    period: &str,
) -> String {
    format!(
        "{symbol}: {} {currency} ({} {})",
        price(amount),
        change(variation),
        if period.trim().is_empty() {
            "24h"
        } else {
            period
        }
    )
}

/// Regroup a plain decimal such as `-24402.5` for the reader: Argentine
/// `-24.402,5` in Spanish, `-24,402.5` in English. Anything that is not a
/// plain decimal is returned unchanged.
#[must_use]
pub fn localized_number(text: &str, locale: crate::locale::Locale) -> String {
    let (sign, unsigned) = match text.chars().next() {
        Some(sign @ ('+' | '-')) => (sign.to_string(), &text[1..]),
        _ => (String::new(), text),
    };
    let (integer, fraction) = unsigned
        .split_once('.')
        .map_or((unsigned, None), |(integer, fraction)| {
            (integer, Some(fraction))
        });
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|fraction| {
            fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return text.to_owned();
    }
    let (group, decimal) = match locale {
        crate::locale::Locale::Es => ('.', ','),
        crate::locale::Locale::En => (',', '.'),
    };
    let mut grouped = String::with_capacity(text.len() + integer.len() / 3);
    for (index, digit) in integer.chars().enumerate() {
        if index > 0 && (integer.len() - index) % 3 == 0 {
            grouped.push(group);
        }
        grouped.push(digit);
    }
    match fraction {
        Some(fraction) => format!("{sign}{grouped}{decimal}{fraction}"),
        None => format!("{sign}{grouped}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn localized_numbers_use_argentine_and_english_grouping() {
        use crate::locale::Locale;
        assert_eq!(localized_number("24402.5", Locale::Es), "24.402,5");
        assert_eq!(localized_number("-1234567", Locale::Es), "-1.234.567");
        assert_eq!(localized_number("+62.68", Locale::Es), "+62,68");
        assert_eq!(localized_number("24402.5", Locale::En), "24,402.5");
        assert_eq!(localized_number("150", Locale::En), "150");
        for raw in ["N/A", "nan", "", "1.", ".5", "1e-20", "+"] {
            assert_eq!(localized_number(raw, Locale::Es), raw);
        }
    }
    #[test]
    fn quotes_preserve_small_prices_units_and_requested_periods() {
        assert_eq!(
            quote("RKH.L", 68.3, "GBp", Some(-12.44), "1m"),
            "RKH.L: 68.3 GBp (-12.44% 1m)"
        );
        assert_eq!(
            quote("TINY", 0.00000525, "USD", None, "7d"),
            "TINY: 0.00000525 USD (N/A 7d)"
        );
        assert_eq!(price(1e-20), "1.0000e-20");
        assert_eq!(price(f64::NAN), "N/A");
        assert_eq!(change(Some(f64::INFINITY)), "N/A");
        assert_eq!(change(Some(-0.001)), "+0%");
        assert_eq!(price(0.0), "0");
    }
}
