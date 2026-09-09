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

#[cfg(test)]
mod tests {
    use super::*;
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
