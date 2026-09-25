//! Arbitrage route selection and calculations for the `/rulo` command.

use crate::locale::Locale;

#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeQuote {
    pub exchange: String,
    pub price: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuloInput {
    pub official: Option<f64>,
    pub mep: Option<f64>,
    pub blue: Option<f64>,
    pub usd_to_usdt: Vec<ExchangeQuote>,
    pub usdt_to_ars: Vec<ExchangeQuote>,
    pub usd_amount: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuloEvaluation {
    OfficialError,
    Routes(RuloPlan),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuloPlan {
    pub official: String,
    pub base_usd: String,
    pub base_ars: String,
    pub routes: Vec<RuloRoute>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuloRoute {
    pub label: &'static str,
    pub sell_price: String,
    pub difference: String,
    pub percentage: String,
    pub details: Vec<RuloDetail>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuloDetail {
    Steps(String),
    Result(String),
    Profit(String),
}

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
        result.push('.');
        result.extend(chunk.iter().map(|byte| char::from(*byte)));
    }
    result
}

fn format_local_currency(value: f64, decimals: usize) -> String {
    let formatted = format!("{value:.decimals$}");
    let (integer, fraction) = formatted
        .split_once('.')
        .map_or((formatted.as_str(), None), |(integer, fraction)| {
            (integer, Some(fraction))
        });
    let mut result = group_integer_digits(integer);
    if let Some(fraction) = fraction {
        result.push(',');
        result.push_str(fraction);
        while result.ends_with('0') {
            result.pop();
        }
        if result.ends_with(',') {
            result.pop();
        }
    }
    result
}

fn format_local_signed(value: f64) -> String {
    let sign = if value >= 0.0 { '+' } else { '-' };
    format!("{sign}{}", format_local_currency(value.abs(), 2))
}

fn format_signed_percentage(value: f64) -> String {
    let mut result = format!("{value:+.2}");
    while result.ends_with('0') {
        result.pop();
    }
    if result.ends_with('.') {
        result.pop();
    }
    result
}

fn spread_route(
    label: &'static str,
    sell_price: f64,
    official: f64,
    details: Vec<RuloDetail>,
) -> RuloRoute {
    let difference = sell_price - official;
    let percentage = if official == 0.0 {
        0.0
    } else {
        (difference / official) * 100.0
    };
    RuloRoute {
        label,
        sell_price: format_local_currency(sell_price, 2),
        difference: format_local_signed(difference),
        percentage: format_signed_percentage(percentage),
        details,
    }
}

fn best_ask(quotes: &[ExchangeQuote]) -> Option<(&ExchangeQuote, f64)> {
    let mut best: Option<(&ExchangeQuote, f64)> = None;
    for quote in quotes {
        let Some(price) = quote.price.filter(|price| *price > 0.0) else {
            continue;
        };
        if best.is_none_or(|(_, best_price)| price < best_price) {
            best = Some((quote, price));
        }
    }
    best
}

fn best_bid(quotes: &[ExchangeQuote]) -> Option<(&ExchangeQuote, f64)> {
    let mut best: Option<(&ExchangeQuote, f64)> = None;
    for quote in quotes {
        let Some(price) = quote.price.filter(|price| *price > 0.0) else {
            continue;
        };
        if best.is_none_or(|(_, best_price)| price > best_price) {
            best = Some((quote, price));
        }
    }
    best
}

#[must_use]
pub fn evaluate_rulo(input: &RuloInput) -> RuloEvaluation {
    let Some(official) = input.official.filter(|price| *price > 0.0) else {
        return RuloEvaluation::OfficialError;
    };
    let official_cost = official * input.usd_amount;
    let base_usd = format_local_currency(input.usd_amount, 0);
    let base_ars = format_local_currency(official_cost, 2);
    let mut routes = Vec::new();

    if let Some(mep) = input.mep.filter(|price| *price != 0.0) {
        let final_ars = mep * input.usd_amount;
        routes.push(spread_route(
            "MEP (AL30 CI)",
            mep,
            official,
            vec![
                RuloDetail::Result(format!(
                    "{base_usd} USD → {} ARS",
                    format_local_currency(final_ars, 2)
                )),
                RuloDetail::Profit(format_local_signed(final_ars - official_cost)),
            ],
        ));
    }
    if let Some(blue) = input.blue.filter(|price| *price != 0.0) {
        let final_ars = blue * input.usd_amount;
        routes.push(spread_route(
            "Blue",
            blue,
            official,
            vec![
                RuloDetail::Result(format!(
                    "{base_usd} USD → {} ARS",
                    format_local_currency(final_ars, 2)
                )),
                RuloDetail::Profit(format_local_signed(final_ars - official_cost)),
            ],
        ));
    }
    if let (Some((ask, ask_price)), Some((bid, bid_price))) =
        (best_ask(&input.usd_to_usdt), best_bid(&input.usdt_to_ars))
    {
        let usdt = input.usd_amount / ask_price;
        let ars = usdt * bid_price;
        routes.push(spread_route(
            "USDT",
            ars / input.usd_amount,
            official,
            vec![
                RuloDetail::Steps(format!(
                    "USD→USDT {}, USDT→ARS {}",
                    ask.exchange.to_uppercase(),
                    bid.exchange.to_uppercase()
                )),
                RuloDetail::Result(format!(
                    "{base_usd} USD → {} USDT → {} ARS",
                    format_local_currency(usdt, 2),
                    format_local_currency(ars, 2)
                )),
                RuloDetail::Profit(format_local_signed(ars - official_cost)),
            ],
        ));
    }
    RuloEvaluation::Routes(RuloPlan {
        official: format_local_currency(official, 2),
        base_usd,
        base_ars,
        routes,
    })
}

#[must_use]
pub fn render_rulo(evaluation: &RuloEvaluation, locale: Locale) -> String {
    let RuloEvaluation::Routes(plan) = evaluation else {
        return match locale {
            Locale::Es => {
                "No pude conseguir el dólar oficial para armar los rulos. Probá más tarde"
                    .to_owned()
            }
            Locale::En => {
                "I could not load the official rate for the arbitrage routes. Try again later"
                    .to_owned()
            }
        };
    };
    if plan.routes.is_empty() {
        return match locale {
            Locale::Es => "Ahora no hay ningún rulo que cierre".to_owned(),
            Locale::En => "There is no viable arbitrage route right now".to_owned(),
        };
    }
    let mut lines = match locale {
        Locale::Es => vec![
            "🔁 Rulos desde el oficial".to_owned(),
            format!(
                "💵 Oficial: ${}\nInvertís {} USD = ${} ARS",
                plan.official, plan.base_usd, plan.base_ars
            ),
        ],
        Locale::En => vec![
            "🔁 Arbitrage from the official rate".to_owned(),
            format!(
                "💵 Official: ${}\nInvesting {} USD = ${} ARS",
                plan.official, plan.base_usd, plan.base_ars
            ),
        ],
    };
    for route in &plan.routes {
        let marker = if route.difference.starts_with('-') {
            "🔴"
        } else {
            "🟢"
        };
        let percentage = match locale {
            Locale::Es => route.percentage.replace('.', ","),
            Locale::En => route.percentage.clone(),
        };
        lines.push(String::new());
        lines.push(format!("{marker} {}: {percentage}%", route.label));
        lines.push(match locale {
            Locale::Es => format!(
                "Vendés a ${} ({} vs oficial)",
                route.sell_price, route.difference
            ),
            Locale::En => format!(
                "Sell at ${} ({} vs official)",
                route.sell_price, route.difference
            ),
        });
        for detail in &route.details {
            lines.push(match (detail, locale) {
                (RuloDetail::Steps(text), Locale::Es) => format!("Ruta: {text}"),
                (RuloDetail::Steps(text), Locale::En) => format!("Route: {text}"),
                (RuloDetail::Result(text), _) => text.clone(),
                (RuloDetail::Profit(text), Locale::Es) => format!("Ganancia: {text} ARS"),
                (RuloDetail::Profit(text), Locale::En) => format!("Profit: {text} ARS"),
            });
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{ExchangeQuote, RuloEvaluation, RuloInput, evaluate_rulo, render_rulo};
    use crate::locale::Locale;

    fn complete_input() -> RuloInput {
        RuloInput {
            official: Some(1440.0),
            mep: Some(1459.73),
            blue: Some(1430.0),
            usd_to_usdt: vec![
                ExchangeQuote {
                    exchange: "buenbit".to_owned(),
                    price: Some(1.031),
                },
                ExchangeQuote {
                    exchange: "xapo".to_owned(),
                    price: None,
                },
            ],
            usdt_to_ars: vec![ExchangeQuote {
                exchange: "buenbit".to_owned(),
                price: Some(1458.44),
            }],
            usd_amount: 1000.0,
        }
    }

    #[test]
    fn evaluates_all_routes_with_legacy_number_formatting() {
        let RuloEvaluation::Routes(plan) = evaluate_rulo(&complete_input()) else {
            return;
        };
        assert_eq!(plan.official, "1.440");
        assert_eq!(plan.base_ars, "1.440.000");
        assert_eq!(plan.routes.len(), 3);
        assert_eq!(plan.routes[0].difference, "+19,73");
        assert_eq!(
            plan.routes[0].details[1],
            super::RuloDetail::Profit("+19.730".to_owned())
        );
        assert_eq!(plan.routes[2].label, "USDT");
    }

    #[test]
    fn reports_missing_official_and_empty_routes() {
        let mut input = complete_input();
        input.official = Some(0.0);
        assert_eq!(evaluate_rulo(&input), RuloEvaluation::OfficialError);
        input.official = Some(1440.0);
        input.mep = None;
        input.blue = None;
        input.usd_to_usdt.clear();
        input.usdt_to_ars.clear();
        assert_eq!(
            evaluate_rulo(&input),
            RuloEvaluation::Routes(super::RuloPlan {
                official: "1.440".to_owned(),
                base_usd: "1.000".to_owned(),
                base_ars: "1.440.000".to_owned(),
                routes: Vec::new(),
            })
        );
    }

    #[test]
    fn renders_exact_bilingual_routes_and_guard_messages() {
        let evaluation = evaluate_rulo(&complete_input());
        let spanish = render_rulo(&evaluation, Locale::Es);
        assert!(spanish.starts_with(
            "🔁 Rulos desde el oficial\n💵 Oficial: $1.440\nInvertís 1.000 USD = $1.440.000 ARS"
        ));
        assert!(spanish.contains("Ganancia: +19.730 ARS"));
        assert!(spanish.contains("Ruta: USD→USDT BUENBIT, USDT→ARS BUENBIT"));
        let english = render_rulo(&evaluation, Locale::En);
        assert!(english.starts_with("🔁 Arbitrage from the official rate"));
        assert!(english.contains("Profit: -10.000 ARS"));
        assert!(spanish.contains("🟢 MEP (AL30 CI): +1,37%"));
        assert!(spanish.contains("🔴 Blue"));

        assert_eq!(
            render_rulo(&RuloEvaluation::OfficialError, Locale::Es),
            "No pude conseguir el dólar oficial para armar los rulos. Probá más tarde"
        );
        let mut input = complete_input();
        input.mep = None;
        input.blue = None;
        input.usd_to_usdt.clear();
        input.usdt_to_ars.clear();
        assert_eq!(
            render_rulo(&evaluate_rulo(&input), Locale::En),
            "There is no viable arbitrage route right now"
        );
    }
}
