//! Exact legacy formatting for the `/satoshi` market command.

/// Invalid price input that the command reports through its localized error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidBitcoinPrice;

fn group_integer_digits(value: &str) -> String {
    let (sign, digits) = value
        .strip_prefix('-')
        .map_or(("", value), |digits| ("-", digits));
    let first_group = match digits.len() % 3 {
        0 => 3,
        remainder => remainder,
    };
    let mut result = String::with_capacity(value.len() + (digits.len() / 3));
    result.push_str(sign);
    result.push_str(&digits[..first_group]);
    for chunk in digits.as_bytes()[first_group..].chunks(3) {
        result.push(',');
        result.extend(chunk.iter().map(|byte| char::from(*byte)));
    }
    result
}

/// Format two adapter-provided Bitcoin prices with the legacy output contract.
pub fn format_satoshi_quote(price_usd: f64, price_ars: f64) -> Result<String, InvalidBitcoinPrice> {
    if price_usd == 0.0 || price_ars == 0.0 || price_usd.is_nan() || price_ars.is_nan() {
        return Err(InvalidBitcoinPrice);
    }
    let usd_sats = (100_000_000.0 / price_usd).trunc();
    if !usd_sats.is_finite() {
        return Err(InvalidBitcoinPrice);
    }
    let usd_sats = if usd_sats == 0.0 {
        "0".to_owned()
    } else {
        group_integer_digits(&format!("{usd_sats:.0}"))
    };
    Ok(format!(
        "1 satoshi = ${:.8} USD\n1 satoshi = ${:.4} ARS\n\n$1 USD = {} sats\n$1 ARS = {:.3} sats",
        price_usd / 100_000_000.0,
        price_ars / 100_000_000.0,
        usd_sats,
        100_000_000.0 / price_ars,
    ))
}

#[cfg(test)]
mod tests {
    use super::{InvalidBitcoinPrice, format_satoshi_quote};

    #[test]
    fn matches_positive_and_negative_legacy_formatting() {
        assert_eq!(
            format_satoshi_quote(50_000.0, 10_000_000.0),
            Ok("1 satoshi = $0.00050000 USD\n1 satoshi = $0.1000 ARS\n\n$1 USD = 2,000 sats\n$1 ARS = 10.000 sats".to_owned())
        );
        assert_eq!(
            format_satoshi_quote(-40_000.0, -20_000_000.0),
            Ok("1 satoshi = $-0.00040000 USD\n1 satoshi = $-0.2000 ARS\n\n$1 USD = -2,500 sats\n$1 ARS = -5.000 sats".to_owned())
        );
        assert_eq!(
            format_satoshi_quote(-1.0e30, 1.0),
            Ok("1 satoshi = $-10000000000000000000000.00000000 USD\n1 satoshi = $0.0000 ARS\n\n$1 USD = 0 sats\n$1 ARS = 100000000.000 sats".to_owned())
        );
    }

    #[test]
    fn rejects_values_that_python_cannot_convert_or_divide() {
        assert_eq!(format_satoshi_quote(0.0, 1.0), Err(InvalidBitcoinPrice));
        assert_eq!(
            format_satoshi_quote(f64::NAN, 1.0),
            Err(InvalidBitcoinPrice)
        );
        assert_eq!(
            format_satoshi_quote(f64::MIN_POSITIVE, 1.0),
            Err(InvalidBitcoinPrice)
        );
    }
}
