//! Argentine dollar command planning and deterministic formatting.

use crate::locale::Locale;

pub const DOLLAR_TIMEFRAMES: [(&str, i64); 5] =
    [("1h", 1), ("6h", 6), ("12h", 12), ("24h", 24), ("48h", 48)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DollarCommandPlan {
    Load { hours_ago: i64 },
    InvalidTimeframe,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DollarRate {
    pub name: &'static str,
    pub price: f64,
    pub change: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurrencyBands {
    pub lower: f64,
    pub upper: f64,
    pub lower_change: Option<f64>,
    pub upper_change: Option<f64>,
}

#[must_use]
pub fn classify_dollar_command(command: &str) -> bool {
    matches!(command, "/dolar" | "/dollar" | "/usd")
}

#[must_use]
pub fn plan_dollar_command(message_text: &str) -> DollarCommandPlan {
    let token = message_text
        .split_whitespace()
        .next_back()
        .unwrap_or_default()
        .to_lowercase();
    if let Some((_, hours)) = DOLLAR_TIMEFRAMES
        .iter()
        .find(|(timeframe, _)| *timeframe == token)
    {
        return DollarCommandPlan::Load { hours_ago: *hours };
    }
    let is_timeframe = token.len() >= 2
        && matches!(token.as_bytes().last(), Some(b'h' | b'd'))
        && token[..token.len() - 1]
            .bytes()
            .all(|character| character.is_ascii_digit());
    if is_timeframe {
        DollarCommandPlan::InvalidTimeframe
    } else {
        DollarCommandPlan::Load { hours_ago: 24 }
    }
}

#[must_use]
pub fn invalid_timeframe_message(message_text: &str, locale: Locale) -> String {
    let token = message_text
        .split_whitespace()
        .next_back()
        .unwrap_or_default()
        .to_lowercase();
    let valid = DOLLAR_TIMEFRAMES
        .iter()
        .map(|(timeframe, _)| *timeframe)
        .collect::<Vec<_>>()
        .join(", ");
    match locale {
        Locale::Es => format!("timeframe '{token}' no soportado, uso: {valid}"),
        Locale::En => format!("unsupported timeframe '{token}', use: {valid}"),
    }
}

fn trimmed(value: f64, decimals: usize) -> String {
    let formatted = format!("{value:.decimals$}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn signed(value: f64) -> String {
    let prefix = if value >= 0.0 { "+" } else { "" };
    format!("{prefix}{}", trimmed(value, 2))
}

#[must_use]
pub fn render_dollar_rates(
    rates: &[DollarRate],
    bands: Option<&CurrencyBands>,
    hours_ago: i64,
    locale: Locale,
) -> Option<String> {
    if rates.is_empty() {
        return None;
    }
    let mut rates = rates.to_vec();
    if let Some(bands) = bands {
        rates.extend([
            DollarRate {
                name: "Banda piso",
                price: bands.lower,
                change: (hours_ago == 24).then_some(bands.lower_change).flatten(),
            },
            DollarRate {
                name: "Banda techo",
                price: bands.upper,
                change: (hours_ago == 24).then_some(bands.upper_change).flatten(),
            },
        ]);
    }
    rates.sort_by(|left, right| left.price.total_cmp(&right.price));
    let no_history = rates.iter().all(|rate| rate.change.is_none());
    let mut lines = rates
        .iter()
        .map(|rate| {
            let mut line = format!("{}: {}", rate.name, trimmed(rate.price, 2));
            if let Some(change) = rate.change {
                line.push_str(&format!(" ({}% {hours_ago}hs)", signed(change)));
            }
            line
        })
        .collect::<Vec<_>>();
    if hours_ago != 24 && no_history {
        lines.push(String::new());
        lines.push(match locale {
            Locale::Es => format!("(sin datos historicos para {hours_ago}hs todavia)"),
            Locale::En => format!("(no historical data for {hours_ago}h yet)"),
        });
    }
    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{
        CurrencyBands, DOLLAR_TIMEFRAMES, DollarCommandPlan, DollarRate, classify_dollar_command,
        invalid_timeframe_message, plan_dollar_command, render_dollar_rates,
    };
    use crate::locale::Locale;

    #[test]
    fn recognizes_aliases_and_timeframe_contract() {
        for command in ["/dolar", "/dollar", "/usd"] {
            assert!(classify_dollar_command(command));
        }
        assert!(!classify_dollar_command("/usdt"));
        assert_eq!(
            plan_dollar_command("something 6H"),
            DollarCommandPlan::Load { hours_ago: 6 }
        );
        assert_eq!(
            plan_dollar_command(""),
            DollarCommandPlan::Load { hours_ago: 24 }
        );
        assert_eq!(
            plan_dollar_command("7d"),
            DollarCommandPlan::InvalidTimeframe
        );
        assert_eq!(DOLLAR_TIMEFRAMES.len(), 5);
        assert_eq!(
            invalid_timeframe_message("7D", Locale::En),
            "unsupported timeframe '7d', use: 1h, 6h, 12h, 24h, 48h"
        );
    }

    #[test]
    fn renders_sorted_rates_changes_and_bands_like_python() {
        let rates = [
            DollarRate {
                name: "Oficial",
                price: 1420.0,
                change: Some(2.0),
            },
            DollarRate {
                name: "Mayorista",
                price: 1400.0,
                change: Some(7.692_307_692_3),
            },
            DollarRate {
                name: "TCRM 100",
                price: 1410.0,
                change: Some(-0.5),
            },
        ];
        let bands = CurrencyBands {
            lower: 950.12,
            upper: 1460.34,
            lower_change: Some(0.25),
            upper_change: Some(-0.1),
        };
        assert_eq!(
            render_dollar_rates(&rates, Some(&bands), 24, Locale::Es).as_deref(),
            Some(concat!(
                "Banda piso: 950.12 (+0.25% 24hs)\n",
                "Mayorista: 1400 (+7.69% 24hs)\n",
                "TCRM 100: 1410 (-0.5% 24hs)\n",
                "Oficial: 1420 (+2% 24hs)\n",
                "Banda techo: 1460.34 (-0.1% 24hs)"
            ))
        );
    }

    #[test]
    fn non_daily_bands_omit_daily_changes_and_report_missing_history() {
        let rates = [DollarRate {
            name: "Oficial",
            price: 1000.0,
            change: None,
        }];
        let bands = CurrencyBands {
            lower: 900.0,
            upper: 1100.0,
            lower_change: Some(1.0),
            upper_change: Some(2.0),
        };
        assert_eq!(
            render_dollar_rates(&rates, Some(&bands), 6, Locale::En).as_deref(),
            Some(concat!(
                "Banda piso: 900\n",
                "Oficial: 1000\n",
                "Banda techo: 1100\n\n",
                "(no historical data for 6h yet)"
            ))
        );
        assert_eq!(render_dollar_rates(&[], None, 24, Locale::Es), None);
    }
}
