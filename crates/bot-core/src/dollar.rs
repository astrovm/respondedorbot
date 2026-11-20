//! Argentine dollar command planning and deterministic formatting.

use crate::locale::Locale;

/// Periods the help text suggests. Any `N` plus `h`, `d`, `w`, `m`/`mo` or
/// `y` up to 100 years also works: the hourly snapshots cover the last days
/// and the daily history covers years back.
pub const DOLLAR_TIMEFRAMES: [&str; 8] = ["1h", "6h", "24h", "48h", "7d", "30d", "1y", "10y"];

/// The last hour the bot's own hourly snapshots still cover. Longer periods
/// compare against daily closes.
pub const HOURLY_HISTORY_HOURS: i64 = 48;

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

/// The BCRA band limits, with their changes over the period shown.
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

fn last_token(message_text: &str) -> String {
    message_text
        .split_whitespace()
        .next_back()
        .unwrap_or_default()
        .to_lowercase()
}

/// A count followed by a unit letter, so `/dolar 0h` or `/dolar 500y` get an
/// error instead of silently showing 24h. Words and bare numbers do not.
fn looks_like_timeframe(token: &str) -> bool {
    let Some(split) = token.find(|character: char| !character.is_ascii_digit()) else {
        return false;
    };
    split > 0 && matches!(&token[split..], "h" | "d" | "w" | "m" | "mo" | "y")
}

#[must_use]
pub fn plan_dollar_command(message_text: &str) -> DollarCommandPlan {
    let token = last_token(message_text);
    if let Some(period) = crate::price_queries::ChartPeriod::parse(&token) {
        return DollarCommandPlan::Load {
            hours_ago: period.seconds / 3_600,
        };
    }
    if looks_like_timeframe(&token) {
        DollarCommandPlan::InvalidTimeframe
    } else {
        DollarCommandPlan::Load { hours_ago: 24 }
    }
}

/// The shortest exact name for a period: `8760` hours is `1y`, `168` is `1w`,
/// `720` is `30d`. Up to 48 hours stays in hours, like the snapshots.
#[must_use]
pub fn period_label(hours: i64) -> String {
    if hours > 0 && hours % (365 * 24) == 0 {
        format!("{}y", hours / (365 * 24))
    } else if hours > 0 && hours % (7 * 24) == 0 {
        format!("{}w", hours / (7 * 24))
    } else if hours > HOURLY_HISTORY_HOURS && hours % 24 == 0 {
        format!("{}d", hours / 24)
    } else {
        format!("{hours}h")
    }
}

#[must_use]
pub fn invalid_timeframe_message(message_text: &str, locale: Locale) -> String {
    let token = last_token(message_text);
    let valid = DOLLAR_TIMEFRAMES.join(", ");
    match locale {
        Locale::Es => {
            format!("No conozco el período '{token}'. Usá {valid} o N[h/d/w/m/y], hasta 100y")
        }
        Locale::En => {
            format!("Unknown period '{token}'. Use {valid} or N[h/d/w/m/y], up to 100y")
        }
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
    let mut rates = rates.to_vec();
    if let Some(bands) = bands {
        rates.extend([
            DollarRate {
                name: "Banda piso",
                price: bands.lower,
                change: bands.lower_change,
            },
            DollarRate {
                name: "Banda techo",
                price: bands.upper,
                change: bands.upper_change,
            },
        ]);
    }
    if rates.is_empty() {
        return None;
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
            let number = |value: &str| crate::output_format::readable_number(value);
            let change = rate
                .change
                .map(|change| {
                    let change_text = number(&signed(change));
                    if change_text == "+0" {
                        " (0%)".to_owned()
                    } else {
                        format!(" ({change_text}%)")
                    }
                })
                .unwrap_or_default();
            let price = number(&trimmed(rate.price, 2));
            match locale {
                Locale::Es => format!("{name}: ${price}{change}"),
                Locale::En => format!("{name}: {price}{change}"),
            }
        })
        .collect::<Vec<_>>();
    let label = period_label(hours_ago);
    lines.insert(0, String::new());
    lines.insert(
        0,
        match locale {
            Locale::Es => format!("Dólar en pesos, variación {label}"),
            Locale::En => format!("Dollar in pesos, {label} change"),
        },
    );
    if hours_ago != 24 && no_history {
        lines.push(String::new());
        lines.push(match (locale, hours_ago > HOURLY_HISTORY_HOURS) {
            (Locale::Es, false) => {
                format!("Todavía no tengo historial de {label}. Probá más tarde")
            }
            (Locale::En, false) => format!("No {label} history yet. Try again later"),
            (Locale::Es, true) => format!("No conseguí cotizaciones de hace {label}"),
            (Locale::En, true) => format!("I could not find rates from {label} ago"),
        });
    }
    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{
        CurrencyBands, DOLLAR_TIMEFRAMES, DollarCommandPlan, DollarRate, classify_dollar_command,
        invalid_timeframe_message, period_label, plan_dollar_command, render_dollar_rates,
    };
    use crate::locale::Locale;

    #[test]
    fn recognizes_aliases_and_timeframe_contract() {
        for command in ["/dolar", "/dollar", "/usd"] {
            assert!(classify_dollar_command(command));
        }
        assert!(!classify_dollar_command("/usdt"));
        for (text, hours_ago) in [
            ("something 6H", 6),
            ("", 24),
            ("/dolar 9h", 9),
            ("/dolar 7d", 168),
            ("/dolar 1w", 168),
            ("/dolar 30d", 720),
            ("/dolar 1m", 720),
            ("/dolar 3mo", 2_160),
            ("/dolar 1y", 8_760),
            ("/dolar 20y", 175_200),
            ("/dolar 100y", 876_000),
        ] {
            assert_eq!(
                plan_dollar_command(text),
                DollarCommandPlan::Load { hours_ago },
                "{text}"
            );
        }
        for text in [
            "/dolar 0h",
            "/dolar 101y",
            "/dolar 0d",
            "/dolar 99999999999999999y",
        ] {
            assert_eq!(
                plan_dollar_command(text),
                DollarCommandPlan::InvalidTimeframe,
                "{text}"
            );
        }
        // Words, bare numbers and unknown units are not timeframes, they fall back to 24h.
        for word in ["hola", "12", "h", "d", "x9d", "1x", "2hs", "y"] {
            assert_eq!(
                plan_dollar_command(&format!("/dolar {word}")),
                DollarCommandPlan::Load { hours_ago: 24 },
                "{word}"
            );
        }
        assert!(DOLLAR_TIMEFRAMES.contains(&"10y"));
        assert_eq!(
            invalid_timeframe_message("0H", Locale::En),
            "Unknown period '0h'. Use 1h, 6h, 24h, 48h, 7d, 30d, 1y, 10y or N[h/d/w/m/y], up to 100y"
        );
    }

    #[test]
    fn period_labels_pick_the_largest_exact_unit() {
        for (hours, label) in [
            (1, "1h"),
            (24, "24h"),
            (48, "48h"),
            (72, "3d"),
            (168, "1w"),
            (336, "2w"),
            (720, "30d"),
            (8_760, "1y"),
            (87_600, "10y"),
            (49, "49h"),
            (0, "0h"),
        ] {
            assert_eq!(period_label(hours), label, "{hours}");
        }
    }

    #[test]
    fn long_periods_without_any_baseline_say_so() {
        let rates = [DollarRate {
            name: "Oficial",
            price: 1000.0,
            change: None,
        }];
        assert_eq!(
            render_dollar_rates(&rates, None, 876_000, Locale::Es).as_deref(),
            Some(
                "Dólar en pesos, variación 100y\n\nOficial: $1,000\n\nNo conseguí cotizaciones de hace 100y"
            )
        );
        assert_eq!(
            render_dollar_rates(&rates, None, 168, Locale::En).as_deref(),
            Some(
                "Dollar in pesos, 1w change\n\nOfficial: 1,000\n\nI could not find rates from 1w ago"
            )
        );
        // 48 hours is still the hourly snapshots' range.
        assert_eq!(
            render_dollar_rates(&rates, None, 48, Locale::En).as_deref(),
            Some(
                "Dollar in pesos, 48h change\n\nOfficial: 1,000\n\nNo 48h history yet. Try again later"
            )
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
                "Dólar en pesos, variación 24h\n\nBanda piso: $950.12 (+0.25%)\nMayorista: $1,400 (+7.69%)\nTCRM 100: $1,410 (-0.5%)\nBlue: $1,415 (0%)\nOficial: $1,420 (+2%)\nBanda techo: $1,460.34 (-0.1%)"
            )
        );
    }

    #[test]
    fn bands_show_the_period_change_they_carry_and_missing_history_is_reported() {
        let rates = [DollarRate {
            name: "Oficial",
            price: 1000.0,
            change: None,
        }];
        let mut bands = CurrencyBands {
            lower: 900.0,
            upper: 1100.0,
            lower_change: None,
            upper_change: None,
        };
        assert_eq!(
            render_dollar_rates(&rates, Some(&bands), 6, Locale::En).as_deref(),
            Some(
                "Dollar in pesos, 6h change\n\nLower band: 900\nOfficial: 1,000\nUpper band: 1,100\n\nNo 6h history yet. Try again later"
            )
        );
        bands.lower_change = Some(-4.5);
        bands.upper_change = Some(1.25);
        assert_eq!(
            render_dollar_rates(&rates, Some(&bands), 720, Locale::Es).as_deref(),
            Some(
                "Dólar en pesos, variación 30d\n\nBanda piso: $900 (-4.5%)\nOficial: $1,000\nBanda techo: $1,100 (+1.25%)"
            )
        );
        assert_eq!(render_dollar_rates(&[], None, 24, Locale::Es), None);
    }

    #[test]
    fn spanish_guards_and_rates_without_bands() {
        assert_eq!(
            invalid_timeframe_message("/dolar 0d", Locale::Es),
            "No conozco el período '0d'. Usá 1h, 6h, 24h, 48h, 7d, 30d, 1y, 10y o N[h/d/w/m/y], hasta 100y"
        );
        let rates = [DollarRate {
            name: "Oficial",
            price: 1000.0,
            change: None,
        }];
        assert_eq!(
            render_dollar_rates(&rates, None, 12, Locale::Es).as_deref(),
            Some(
                "Dólar en pesos, variación 12h\n\nOficial: $1,000\n\nTodavía no tengo historial de 12h. Probá más tarde"
            )
        );
    }

    #[test]
    fn missing_changes_show_only_the_price_in_both_languages() {
        let rates = [DollarRate {
            name: "TCRM 100",
            price: 1814.14,
            change: None,
        }];
        for (locale, expected) in [
            (
                Locale::Es,
                "Dólar en pesos, variación 24h\n\nTCRM 100: $1,814.14",
            ),
            (
                Locale::En,
                "Dollar in pesos, 24h change\n\nTCRM 100: 1,814.14",
            ),
        ] {
            assert_eq!(
                render_dollar_rates(&rates, None, 24, locale).as_deref(),
                Some(expected)
            );
        }
    }

    #[test]
    fn short_timeframes_with_history_skip_the_missing_history_note() {
        let rates = [DollarRate {
            name: "Oficial",
            price: 1000.0,
            change: Some(1.5),
        }];
        assert_eq!(
            render_dollar_rates(&rates, None, 6, Locale::En).as_deref(),
            Some("Dollar in pesos, 6h change\n\nOfficial: 1,000 (+1.5%)")
        );
    }
}
