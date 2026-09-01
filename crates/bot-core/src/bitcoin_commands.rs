//! Native Bitcoin quote and reference-model command rendering.

use crate::locale::Locale;
use crate::market_models::{MarketModel, Valuation, evaluate_market_model};
use crate::satoshi::format_satoshi_quote;

const DAY_SECONDS: i64 = 86_400;
const POWER_LAW_EPOCH_SECONDS: i64 = 1_231_027_200;
const RAINBOW_EPOCH_SECONDS: i64 = 1_231_459_200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitcoinCommand {
    Satoshi,
    PowerLaw,
    Rainbow,
}

#[must_use]
pub fn classify_bitcoin_command(command: &str) -> Option<BitcoinCommand> {
    match command {
        "/satoshi" | "/sat" | "/sats" => Some(BitcoinCommand::Satoshi),
        "/powerlaw" => Some(BitcoinCommand::PowerLaw),
        "/rainbow" => Some(BitcoinCommand::Rainbow),
        _ => None,
    }
}

#[must_use]
pub fn bitcoin_price_error(command: BitcoinCommand, currency: &str, locale: Locale) -> String {
    match (command, currency, locale) {
        (BitcoinCommand::Satoshi, "USD", Locale::Es) => {
            "no pude traer el precio de BTC en USD".to_owned()
        }
        (BitcoinCommand::Satoshi, "ARS", Locale::Es) => {
            "no pude traer el precio de BTC en ARS".to_owned()
        }
        (BitcoinCommand::Satoshi, "USD", Locale::En) => {
            "I could not load the BTC price in USD".to_owned()
        }
        (BitcoinCommand::Satoshi, "ARS", Locale::En) => {
            "I could not load the BTC price in ARS".to_owned()
        }
        (BitcoinCommand::PowerLaw, _, Locale::Es) => {
            "no pude traer el precio de BTC para calcular power law".to_owned()
        }
        (BitcoinCommand::PowerLaw, _, Locale::En) => {
            "I could not load the BTC price for power law".to_owned()
        }
        (BitcoinCommand::Rainbow, _, Locale::Es) => {
            "no pude traer el precio de BTC para calcular rainbow".to_owned()
        }
        (BitcoinCommand::Rainbow, _, Locale::En) => {
            "I could not load the BTC price for rainbow chart".to_owned()
        }
        (BitcoinCommand::Satoshi, _, Locale::Es) => {
            "no pude conseguir el precio de BTC boludo".to_owned()
        }
        (BitcoinCommand::Satoshi, _, Locale::En) => "I could not load the BTC price".to_owned(),
    }
}

#[must_use]
pub fn render_satoshi(price_usd: f64, price_ars: f64, locale: Locale) -> String {
    format_satoshi_quote(price_usd, price_ars)
        .unwrap_or_else(|_error| bitcoin_price_error(BitcoinCommand::Satoshi, "invalid", locale))
}

#[must_use]
pub fn render_market_model(
    command: BitcoinCommand,
    unix_timestamp: i64,
    market_price: f64,
    locale: Locale,
) -> String {
    let (model, epoch) = match command {
        BitcoinCommand::PowerLaw => (MarketModel::PowerLaw, POWER_LAW_EPOCH_SECONDS),
        BitcoinCommand::Rainbow => (MarketModel::Rainbow, RAINBOW_EPOCH_SECONDS),
        BitcoinCommand::Satoshi => {
            return bitcoin_price_error(command, "invalid", locale);
        }
    };
    let elapsed_days = unix_timestamp.saturating_sub(epoch).div_euclid(DAY_SECONDS);
    let Ok(result) = evaluate_market_model(model, elapsed_days, market_price) else {
        return bitcoin_price_error(command, "USD", locale);
    };
    let value = format!("{:.2}", result.model_value);
    let percentage = format!("{:.2}", result.percentage);
    let valuation = match (result.valuation, locale) {
        (Valuation::Expensive, Locale::Es) => format!("{percentage}% caro boludo"),
        (Valuation::Cheap, Locale::Es) => format!("{percentage}% regalado gordo"),
        (Valuation::Expensive, Locale::En) => format!("{percentage}% expensive"),
        (Valuation::Cheap, Locale::En) => format!("{percentage}% undervalued"),
    };
    match (command, locale) {
        (BitcoinCommand::PowerLaw, Locale::Es) => {
            format!("segun power law btc deberia estar en {value} usd ({valuation})")
        }
        (BitcoinCommand::PowerLaw, Locale::En) => {
            format!("power law estimates BTC at {value} USD ({valuation})")
        }
        (BitcoinCommand::Rainbow, Locale::Es) => {
            format!("segun rainbow chart btc deberia estar en {value} usd ({valuation})")
        }
        (BitcoinCommand::Rainbow, Locale::En) => {
            format!("rainbow chart estimates BTC at {value} USD ({valuation})")
        }
        (BitcoinCommand::Satoshi, _) => bitcoin_price_error(command, "invalid", locale),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BitcoinCommand, bitcoin_price_error, classify_bitcoin_command, render_market_model,
        render_satoshi,
    };
    use crate::locale::Locale;

    #[test]
    fn classifies_all_aliases_and_localizes_price_failures() {
        for alias in ["/satoshi", "/sat", "/sats"] {
            assert_eq!(
                classify_bitcoin_command(alias),
                Some(BitcoinCommand::Satoshi)
            );
        }
        assert_eq!(
            classify_bitcoin_command("/powerlaw"),
            Some(BitcoinCommand::PowerLaw)
        );
        assert_eq!(
            classify_bitcoin_command("/rainbow"),
            Some(BitcoinCommand::Rainbow)
        );
        assert_eq!(classify_bitcoin_command("/other"), None);
        assert_eq!(
            bitcoin_price_error(BitcoinCommand::Satoshi, "ARS", Locale::En),
            "I could not load the BTC price in ARS"
        );
        assert_eq!(
            bitcoin_price_error(BitcoinCommand::Rainbow, "USD", Locale::Es),
            "no pude traer el precio de BTC para calcular rainbow"
        );
    }

    #[test]
    fn renders_satoshi_and_both_models_with_exact_legacy_formatting() {
        assert_eq!(
            render_satoshi(50_000.0, 10_000_000.0, Locale::Es),
            "1 satoshi = $0.00050000 USD\n1 satoshi = $0.1000 ARS\n\n$1 USD = 2,000 sats\n$1 ARS = 10.000 sats"
        );
        let power_timestamp = 1_231_027_200 + 5_475 * 86_400;
        assert_eq!(
            render_market_model(
                BitcoinCommand::PowerLaw,
                power_timestamp,
                50_000.0,
                Locale::En
            ),
            "power law estimates BTC at 57869.18 USD (13.60% undervalued)"
        );
        let rainbow_timestamp = 1_231_459_200 + 5_470 * 86_400;
        assert_eq!(
            render_market_model(
                BitcoinCommand::Rainbow,
                rainbow_timestamp,
                50_000.0,
                Locale::Es
            ),
            "segun rainbow chart btc deberia estar en 97886.11 usd (48.92% regalado gordo)"
        );
    }

    #[test]
    fn invalid_prices_and_dates_use_existing_safe_errors() {
        assert_eq!(
            render_satoshi(0.0, 1.0, Locale::En),
            "I could not load the BTC price"
        );
        assert_eq!(
            render_market_model(BitcoinCommand::PowerLaw, 0, 1.0, Locale::Es),
            "no pude traer el precio de BTC para calcular power law"
        );
        assert_eq!(
            render_market_model(BitcoinCommand::Satoshi, 0, 1.0, Locale::Es),
            "no pude conseguir el precio de BTC boludo"
        );
    }
}
