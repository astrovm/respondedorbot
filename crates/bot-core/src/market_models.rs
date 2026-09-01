//! Deterministic Bitcoin reference-price models used by market commands.

/// A supported reference-price model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarketModel {
    PowerLaw,
    Rainbow,
}

/// Whether the current price is above or below the model value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Valuation {
    Expensive,
    Cheap,
}

/// Calculated model value and absolute percentage difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MarketModelResult {
    pub model_value: f64,
    pub percentage: f64,
    pub valuation: Valuation,
}

/// Invalid elapsed-day input for a logarithmic or fractional-power model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidElapsedDays;

/// Evaluate one Bitcoin model against the supplied market price.
pub fn evaluate_market_model(
    model: MarketModel,
    elapsed_days: i64,
    market_price: f64,
) -> Result<MarketModelResult, InvalidElapsedDays> {
    if elapsed_days <= 0 {
        return Err(InvalidElapsedDays);
    }
    let days = elapsed_days as f64;
    let model_value = match model {
        MarketModel::PowerLaw => 1.0117e-17 * days.powf(5.82),
        MarketModel::Rainbow => {
            10_f64.powf(2.661_671_550_059_61 * days.ln() - 17.918_376_188_986_4)
        }
    };
    let signed_percentage = ((market_price - model_value) / model_value) * 100.0;
    Ok(MarketModelResult {
        model_value,
        percentage: signed_percentage.abs(),
        valuation: if signed_percentage > 0.0 {
            Valuation::Expensive
        } else {
            Valuation::Cheap
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{MarketModel, Valuation, evaluate_market_model};

    #[test]
    fn matches_fixed_power_law_and_rainbow_examples() {
        let power_law = evaluate_market_model(MarketModel::PowerLaw, 5_475, 50_000.0);
        let rainbow = evaluate_market_model(MarketModel::Rainbow, 5_470, 50_000.0);
        let power_law = power_law.map(|result| {
            (
                format!("{:.2}", result.model_value),
                format!("{:.2}", result.percentage),
                result.valuation,
            )
        });
        let rainbow = rainbow.map(|result| {
            (
                format!("{:.2}", result.model_value),
                format!("{:.2}", result.percentage),
                result.valuation,
            )
        });
        assert_eq!(
            power_law,
            Ok(("57869.18".to_owned(), "13.60".to_owned(), Valuation::Cheap))
        );
        assert_eq!(
            rainbow,
            Ok(("97886.11".to_owned(), "48.92".to_owned(), Valuation::Cheap))
        );
    }

    #[test]
    fn classifies_above_equal_and_below_values() {
        let reference = evaluate_market_model(MarketModel::PowerLaw, 5_475, 0.0)
            .map(|result| result.model_value);
        let Ok(reference) = reference else {
            return;
        };
        assert_eq!(
            evaluate_market_model(MarketModel::PowerLaw, 5_475, reference * 2.0)
                .map(|result| result.valuation),
            Ok(Valuation::Expensive)
        );
        assert_eq!(
            evaluate_market_model(MarketModel::PowerLaw, 5_475, reference)
                .map(|result| result.valuation),
            Ok(Valuation::Cheap)
        );
        assert_eq!(
            evaluate_market_model(MarketModel::PowerLaw, 0, reference),
            Err(super::InvalidElapsedDays)
        );
    }
}
