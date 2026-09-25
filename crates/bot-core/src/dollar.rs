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
        Locale::Es => format!("No conozco el período '{token}'. Usá uno de estos: {valid}"),
        Locale::En => format!("Unknown period '{token}'. Use one of: {valid}"),
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
            let name = match (locale, rate.name) {
                (Locale::En, "Oficial") => "Official",
                (Locale::En, "Mayorista") => "Wholesale",
                (Locale::En, "Tarjeta") => "Card",
                (Locale::En, "Banda piso") => "Lower band",
                (Locale::En, "Banda techo") => "Upper band",
                _ => rate.name,
            };
            let number = |value: &str| crate::output_format::localized_number(value, locale);
            let change = rate.change.map_or_else(
                || crate::menu_ui::localized(locale, "sin datos", "no data").to_owned(),
                |change| {
                    let change_text = number(&signed(change));
                    if change_text == "+0" {
                        "= 0%".to_owned()
                    } else if change > 0.0 {
                        format!("▲ {change_text}%")
                    } else {
                        format!("▼ {change_text}%")
                    }
                },
            );
            format!("{name}: ${} ({change})", number(&trimmed(rate.price, 2)))
        })
        .collect::<Vec<_>>();
    lines.insert(0, String::new());
    lines.insert(
        0,
        match locale {
            Locale::Es => format!("💵 Dólar en pesos · variación {hours_ago}h"),
            Locale::En => format!("💵 Dollar in pesos · {hours_ago}h change"),
        },
    );
    if hours_ago != 24 && no_history {
        lines.push(String::new());
        lines.push(match locale {
            Locale::Es => format!("⚠️ Todavía no tengo historial de {hours_ago}h. Probá más tarde"),
            Locale::En => format!("⚠️ No {hours_ago}h history yet. Try again later"),
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
            "Unknown period '7d'. Use one of: 1h, 6h, 12h, 24h, 48h"
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
            DollarRate {
                name: "Blue",
                price: 1415.0,
                change: Some(0.001),
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
            Some(
                "💵 Dólar en pesos · variación 24h\n\nBanda piso: $950,12 (▲ +0,25%)\nMayorista: $1.400 (▲ +7,69%)\nTCRM 100: $1.410 (▼ -0,5%)\nBlue: $1.415 (= 0%)\nOficial: $1.420 (▲ +2%)\nBanda techo: $1.460,34 (▼ -0,1%)"
            )
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
            Some(
                "💵 Dollar in pesos · 6h change\n\nLower band: $900 (no data)\nOfficial: $1,000 (no data)\nUpper band: $1,100 (no data)\n\n⚠️ No 6h history yet. Try again later"
            )
        );
        assert_eq!(render_dollar_rates(&[], None, 24, Locale::Es), None);
    }
}
