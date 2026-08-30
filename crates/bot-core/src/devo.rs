//! Input parsing and quote calculations for the `/devo` command.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DevoInput {
    Valid { fee: f64, purchase: f64 },
    Usage,
    InputError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedUnicodeInput;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevoQuotes {
    pub official: f64,
    pub card: f64,
    pub usdt_ask: f64,
    pub usdt_bid: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DevoResult {
    pub profit: String,
    pub fee: String,
    pub official: String,
    pub usdt: String,
    pub card: String,
    pub purchase: Option<DevoPurchase>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DevoPurchase {
    pub usd: String,
    pub ars: String,
    pub usdt: String,
    pub profit_ars: String,
    pub profit_usdt: String,
    pub total_ars: String,
    pub total_usdt: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidDevoQuotes;

fn parse_python_float(input: &str) -> Option<f64> {
    let input = input.trim();
    match input.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => input.parse().ok(),
    }
}

pub fn parse_devo_input(input: &str) -> Result<DevoInput, UnsupportedUnicodeInput> {
    if !input.is_ascii() || input.contains('_') {
        return Err(UnsupportedUnicodeInput);
    }
    let (fee, purchase) = if input.contains(',') {
        let compact = input.replace(' ', "");
        let values: Vec<_> = compact.split(',').collect();
        let Some(fee) = values.first().and_then(|value| parse_python_float(value)) else {
            return Ok(DevoInput::Usage);
        };
        let Some(purchase) = values.get(1).and_then(|value| parse_python_float(value)) else {
            return Ok(DevoInput::Usage);
        };
        (fee / 100.0, purchase)
    } else {
        let Some(fee) = parse_python_float(input) else {
            return Ok(DevoInput::Usage);
        };
        (fee / 100.0, 0.0)
    };
    if fee.is_nan() || fee > 1.0 || purchase.is_nan() || purchase < 0.0 {
        return Ok(DevoInput::InputError);
    }
    Ok(DevoInput::Valid { fee, purchase })
}

fn format_number(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    let value = format!("{value:.2}");
    value.trim_end_matches('0').trim_end_matches('.').to_owned()
}

pub fn calculate_devo(
    fee: f64,
    purchase: f64,
    quotes: DevoQuotes,
) -> Result<DevoResult, InvalidDevoQuotes> {
    let usdt = (quotes.usdt_ask + quotes.usdt_bid) / 2.0;
    if quotes.card == 0.0 || (purchase > 0.0 && usdt == 0.0) {
        return Err(InvalidDevoQuotes);
    }
    let profit = -(fee * usdt + quotes.official - usdt) / quotes.card;
    let purchase_projection = (purchase > 0.0).then(|| {
        let purchase_ars = purchase * quotes.card;
        let purchase_usdt = purchase_ars / usdt;
        let profit_ars = purchase_ars * profit;
        let profit_usdt = profit_ars / usdt;
        DevoPurchase {
            usd: format_number(purchase),
            ars: format_number(purchase_ars),
            usdt: format_number(purchase_usdt),
            profit_ars: format_number(profit_ars),
            profit_usdt: format_number(profit_usdt),
            total_ars: format_number(purchase_ars + profit_ars),
            total_usdt: format_number(purchase_usdt + profit_usdt),
        }
    });
    Ok(DevoResult {
        profit: format_number(profit * 100.0),
        fee: format_number(fee * 100.0),
        official: format_number(quotes.official),
        usdt: format_number(usdt),
        card: format_number(quotes.card),
        purchase: purchase_projection,
    })
}

#[cfg(test)]
mod tests {
    use super::{DevoInput, DevoQuotes, calculate_devo, parse_devo_input};

    #[test]
    fn preserves_legacy_input_priority_and_validation() {
        assert_eq!(
            parse_devo_input("0.5, 100, ignored"),
            Ok(DevoInput::Valid {
                fee: 0.005,
                purchase: 100.0
            })
        );
        assert_eq!(parse_devo_input("invalid"), Ok(DevoInput::Usage));
        assert_eq!(parse_devo_input("0.5,"), Ok(DevoInput::Usage));
        assert_eq!(parse_devo_input("nan"), Ok(DevoInput::InputError));
        assert_eq!(parse_devo_input("101"), Ok(DevoInput::InputError));
        assert_eq!(parse_devo_input("-1,-2"), Ok(DevoInput::InputError));
        assert!(parse_devo_input("０.５").is_err());
        assert!(parse_devo_input("1_0").is_err());
    }

    #[test]
    fn calculates_summary_and_purchase_projection() {
        let quotes = DevoQuotes {
            official: 100.0,
            card: 150.0,
            usdt_ask: 200.0,
            usdt_bid: 190.0,
        };
        let summary = calculate_devo(0.005, 0.0, quotes);
        assert_eq!(summary.map(|value| value.profit), Ok("62.68".to_owned()));

        let purchase = calculate_devo(0.005, 100.0, quotes);
        assert_eq!(
            purchase.map(|result| result.purchase.map(|value| (
                value.usd,
                value.ars,
                value.profit_ars,
                value.total_ars
            ))),
            Ok(Some((
                "100".to_owned(),
                "15000".to_owned(),
                "9402.5".to_owned(),
                "24402.5".to_owned()
            )))
        );
        assert!(
            calculate_devo(
                0.005,
                1.0,
                DevoQuotes {
                    card: 0.0,
                    ..quotes
                }
            )
            .is_err()
        );
    }
}
