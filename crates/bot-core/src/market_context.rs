//! Typed market snapshots and compact AI-context formatting.

/// One cryptocurrency quote included in prompt context.
#[derive(Clone, Debug, PartialEq)]
pub struct CryptoQuote {
    pub symbol: String,
    pub price: f64,
    pub change_24h: Option<f64>,
    pub dominance: Option<f64>,
}

/// One Argentine dollar quote included in prompt context.
#[derive(Clone, Debug, PartialEq)]
pub struct DollarQuote {
    pub label: String,
    pub price: f64,
    pub bid: Option<f64>,
}

/// Provider-independent market data used by the AI prompt.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarketSnapshot {
    pub crypto: Vec<CryptoQuote>,
    pub dollars: Vec<DollarQuote>,
}

/// Format a normalized snapshot for the existing AI market-context block.
#[must_use]
pub fn format_market_context(snapshot: &MarketSnapshot) -> String {
    let mut lines = Vec::new();
    if !snapshot.crypto.is_empty() {
        lines.push("PRECIOS DE CRIPTOS:".to_owned());
        lines.extend(snapshot.crypto.iter().map(format_crypto));
    }
    if !snapshot.dollars.is_empty() {
        lines.push("DOLARES:".to_owned());
        lines.extend(snapshot.dollars.iter().map(format_dollar));
    }
    lines.join("\n")
}

fn format_crypto(quote: &CryptoQuote) -> String {
    let mut line = format!("- {}: {} usd", quote.symbol, format_number(quote.price, 2));
    if let Some(change) = quote.change_24h {
        line.push_str(&format!(" ({} 24h)", format_signed(change, 2)));
    }
    if let Some(dominance) = quote.dominance {
        line.push_str(&format!(", dom {}%", format_number(dominance, 1)));
    }
    line
}

fn format_dollar(quote: &DollarQuote) -> String {
    let mut line = format!("- {}: {}", quote.label, format_number(quote.price, 2));
    if let Some(bid) = quote.bid {
        line.push_str(&format!(" (bid {})", format_number(bid, 2)));
    }
    line
}

fn format_number(value: f64, decimals: usize) -> String {
    trim_fraction(format!("{value:.decimals$}"))
}

fn format_signed(value: f64, decimals: usize) -> String {
    trim_fraction(format!("{value:+.decimals$}"))
}

fn trim_fraction(value: String) -> String {
    value.trim_end_matches('0').trim_end_matches('.').to_owned()
}

#[cfg(test)]
mod tests {
    use super::{CryptoQuote, DollarQuote, MarketSnapshot, format_market_context};

    #[test]
    fn formats_all_optional_market_fields() {
        let snapshot = MarketSnapshot {
            crypto: vec![CryptoQuote {
                symbol: "BTC".to_owned(),
                price: 50_000.0,
                change_24h: Some(2.5),
                dominance: Some(52.25),
            }],
            dollars: vec![DollarQuote {
                label: "blue".to_owned(),
                price: 1_200.0,
                bid: Some(1_180.0),
            }],
        };

        assert_eq!(
            format_market_context(&snapshot),
            "PRECIOS DE CRIPTOS:\n- BTC: 50000 usd (+2.5 24h), dom 52.2%\nDOLARES:\n- blue: 1200 (bid 1180)"
        );
    }

    #[test]
    fn omits_empty_sections_and_optional_values() {
        assert!(format_market_context(&MarketSnapshot::default()).is_empty());
        assert_eq!(
            format_market_context(&MarketSnapshot {
                crypto: Vec::new(),
                dollars: vec![DollarQuote {
                    label: "oficial".to_owned(),
                    price: 1_000.5,
                    bid: None,
                }],
            }),
            "DOLARES:\n- oficial: 1000.5"
        );
    }

    #[test]
    fn keeps_explicit_signs_and_negative_zero_behavior() {
        let snapshot = MarketSnapshot {
            crypto: vec![
                CryptoQuote {
                    symbol: "UP".to_owned(),
                    price: 1.005,
                    change_24h: Some(0.0),
                    dominance: None,
                },
                CryptoQuote {
                    symbol: "DOWN".to_owned(),
                    price: -0.0,
                    change_24h: Some(-0.0),
                    dominance: None,
                },
            ],
            dollars: Vec::new(),
        };
        let formatted = format_market_context(&snapshot);
        assert!(formatted.contains("(+0 24h)"));
        assert!(formatted.contains("- DOWN: -0 usd (-0 24h)"));
    }
}
