//! BCRA market-variable normalization and deterministic command formatting.

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::locale::Locale;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BcraVariable {
    pub description: String,
    pub value: String,
    pub date: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BcraBands {
    pub lower: f64,
    pub upper: f64,
    pub date: String,
    pub lower_change: Option<f64>,
    pub upper_change: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ItcrmDetails {
    pub value: f64,
    pub date: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CountryRisk {
    pub value_bps: f64,
    pub delta_one_day: Option<f64>,
    pub valuation_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BcraSnapshot {
    pub variables: Vec<BcraVariable>,
    pub bands: Option<BcraBands>,
    pub itcrm: Option<ItcrmDetails>,
    pub country_risk: Option<CountryRisk>,
    pub stale: bool,
}

#[must_use]
pub fn classify_bcra_command(command: &str) -> bool {
    matches!(command, "/bcra" | "/variables")
}

fn normalized(value: &str) -> String {
    value
        .nfkd()
        .filter(char::is_ascii)
        .collect::<String>()
        .to_lowercase()
}

fn matches(value: &str, pattern: &str) -> bool {
    Regex::new(pattern).is_ok_and(|regex| regex.is_match(&normalized(value)))
}

fn trimmed(value: f64, decimals: usize) -> String {
    let formatted = format!("{value:.decimals$}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn format_bcra_value(value: &str, percentage: bool) -> String {
    let normalized = if percentage {
        value.replace(',', ".")
    } else {
        value.replace('.', "").replace(',', ".")
    };
    let Ok(number) = normalized.parse::<f64>() else {
        return if percentage {
            format!("{value}%")
        } else {
            value.to_owned()
        };
    };
    if percentage {
        return if number >= 10.0 {
            format!("{number:.1}%")
        } else {
            format!("{number:.2}%")
        };
    }
    if number >= 1_000_000.0 {
        return grouped(number / 1_000.0, 0);
    }
    if number >= 1_000.0 {
        return grouped(number, 0);
    }
    format!("{number:.2}")
}

/// The BCRA publishes values as `1.250,75`; show them as `1,250.75`.
fn source_number(value: &str) -> String {
    crate::output_format::readable_number(&value.replace('.', "").replace(',', "."))
}

fn grouped(value: f64, decimals: usize) -> String {
    let rendered = format!("{value:.decimals$}");
    let (whole, fractional) = rendered.split_once('.').unwrap_or((&rendered, ""));
    let negative = whole.starts_with('-');
    let digits = whole.trim_start_matches('-');
    let mut output = String::new();
    if negative {
        output.push('-');
    }
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    if !fractional.is_empty() {
        output.push('.');
        output.push_str(fractional);
    }
    output
}

/// Sources write the same day as `24/09/26` or `24/09/2026`.
fn same_day(date: &str, common_date: Option<&str>) -> bool {
    common_date.is_some_and(|common| short_year(date) == short_year(common))
}

/// Shorten `dd/mm/yyyy` dates to `dd/mm/yy` to keep lines compact.
fn short_year(date: &str) -> String {
    match date.rsplit_once('/') {
        Some((prefix, year)) if year.len() == 4 && year.starts_with("20") => {
            format!("{prefix}/{}", &year[2..])
        }
        _ => date.to_owned(),
    }
}

fn variable_line(
    variable: &BcraVariable,
    locale: Locale,
    common_date: Option<&str>,
) -> Option<String> {
    let description = variable.description.as_str();
    let value = variable.value.as_str();
    let line = if matches(description, r"base\s*monetaria") {
        match locale {
            Locale::Es => format!(
                "Base monetaria: ${} mill. pesos",
                format_bcra_value(value, false)
            ),
            Locale::En => format!(
                "Monetary base: ${} million pesos",
                format_bcra_value(value, false)
            ),
        }
    } else if matches(
        description,
        r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual",
    ) {
        match locale {
            Locale::Es => format!("Inflación mensual: {}", format_bcra_value(value, true)),
            Locale::En => format!("Monthly inflation: {}", format_bcra_value(value, true)),
        }
    } else if matches(
        description,
        r"mediana.*variacion.*interanual.*(12|doce).*meses.*(relevamiento.*expectativas.*mercado|rem)|inflacion.*esperada",
    ) {
        match locale {
            Locale::Es => format!("Inflación esperada: {}", format_bcra_value(value, true)),
            Locale::En => format!("Expected inflation: {}", format_bcra_value(value, true)),
        }
    } else if matches(
        description,
        r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual",
    ) {
        match locale {
            Locale::Es => format!("Inflación interanual: {}", format_bcra_value(value, true)),
            Locale::En => format!("Yearly inflation: {}", format_bcra_value(value, true)),
        }
    } else if matches(description, "tamar") {
        format!("TAMAR: {}", format_bcra_value(value, true))
    } else if matches(description, "badlar") {
        format!("BADLAR: {}", format_bcra_value(value, true))
    } else if matches(
        description,
        r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor",
    ) {
        match locale {
            Locale::Es => format!("Dólar minorista: ${}", source_number(value)),
            Locale::En => format!("Retail dollar: ${}", source_number(value)),
        }
    } else if matches(description, r"tipo.*cambio.*mayorista") {
        match locale {
            Locale::Es => format!("Dólar mayorista: ${}", source_number(value)),
            Locale::En => format!("Wholesale dollar: ${}", source_number(value)),
        }
    } else if matches(description, r"unidad.*valor.*adquisitivo|\buva\b") {
        format!("UVA: ${}", source_number(value))
    } else if matches(
        description,
        r"coeficiente.*estabilizacion.*referencia|\bcer\b",
    ) {
        format!("CER: {}", source_number(value))
    } else if matches(description, r"reservas.*internacionales") {
        match locale {
            Locale::Es => format!("Reservas: USD {} millones", format_bcra_value(value, false)),
            Locale::En => format!("Reserves: USD {} million", format_bcra_value(value, false)),
        }
    } else {
        return None;
    };
    if variable.date.is_empty()
        || variable.date == variable.value
        || same_day(&variable.date, common_date)
    {
        Some(line)
    } else {
        Some(format!("{line} ({})", short_year(&variable.date)))
    }
}

fn parse_date(value: &str) -> Option<(i64, i64, i64)> {
    let first = value.split_whitespace().next()?;
    if let Some((year, rest)) = first.split_once('-') {
        let (month, day) = rest.split_once('-')?;
        return Some((year.parse().ok()?, month.parse().ok()?, day.parse().ok()?));
    }
    let mut parts = first.split('/');
    let day = parts.next()?.parse().ok()?;
    let month = parts.next()?.parse().ok()?;
    let raw_year: i64 = parts.next()?.parse().ok()?;
    let year = if raw_year < 100 {
        2_000 + raw_year
    } else {
        raw_year
    };
    Some((year, month, day))
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[must_use]
pub fn render_bcra(snapshot: &BcraSnapshot, locale: Locale, today_days: i64) -> String {
    if snapshot.variables.is_empty() {
        return match locale {
            Locale::Es => "No pude conseguir las variables del BCRA. Probá más tarde".to_owned(),
            Locale::En => "I could not load the BCRA variables. Try again later".to_owned(),
        };
    }
    let mut latest_days = None;
    const PATTERNS: [&str; 11] = [
        r"base\s*monetaria",
        r"variacion.*mensual.*indice.*precios.*consumidor|inflacion.*mensual",
        r"variacion.*interanual.*indice.*precios.*consumidor|inflacion.*interanual",
        r"mediana.*variacion.*interanual.*(12|doce).*meses.*(relevamiento.*expectativas.*mercado|rem)|inflacion.*esperada",
        "tamar",
        "badlar",
        r"tipo.*cambio.*minorista|minorista.*promedio.*vendedor",
        r"tipo.*cambio.*mayorista",
        r"unidad.*valor.*adquisitivo|\buva\b",
        r"coeficiente.*estabilizacion.*referencia|\bcer\b",
        r"reservas.*internacionales",
    ];
    // A series can match more than one pattern (the REM expectation also reads
    // as year-over-year inflation); list each series once.
    // Match the specific expectation pattern (index 3) before the broader
    // year-over-year one (index 2), then show lines in the usual order.
    let mut matched = Vec::<(usize, &BcraVariable)>::new();
    for index in [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10] {
        if let Some(variable) = snapshot.variables.iter().find(|variable| {
            matches(&variable.description, PATTERNS[index])
                && !matched
                    .iter()
                    .any(|(_, known)| std::ptr::eq(*known, *variable))
        }) {
            matched.push((index, variable));
        }
    }
    matched.sort_by_key(|(index, _)| *index);
    let shown = matched
        .into_iter()
        .map(|(_, variable)| variable)
        .collect::<Vec<_>>();
    // Most indicators share one publication date; state it once in the title.
    let common_date = shown
        .iter()
        .map(|variable| variable.date.as_str())
        .filter(|date| !date.is_empty())
        .fold(Vec::<(&str, usize)>::new(), |mut counts, date| {
            match counts.iter_mut().find(|(known, _)| *known == date) {
                Some((_, count)) => *count += 1,
                None => counts.push((date, 1)),
            }
            counts
        })
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .max_by_key(|(_, count)| *count)
        .map(|(date, _)| date);
    let mut lines = vec![
        match (locale, common_date) {
            (Locale::Es, Some(date)) => format!("Indicadores del BCRA al {}", short_year(date)),
            (Locale::En, Some(date)) => format!("BCRA indicators as of {}", short_year(date)),
            (Locale::Es, None) => "Indicadores del BCRA".to_owned(),
            (Locale::En, None) => "BCRA indicators".to_owned(),
        },
        String::new(),
    ];
    for variable in shown {
        if let Some(line) = variable_line(variable, locale, common_date) {
            lines.push(line);
        }
        if let Some((year, month, day)) = parse_date(&variable.date) {
            let candidate = days_from_civil(year, month, day);
            latest_days = Some(latest_days.map_or(candidate, |latest: i64| latest.max(candidate)));
        }
    }
    if let Some(risk) = &snapshot.country_risk {
        let decimals = usize::from(risk.value_bps.abs() < 100.0);
        let value = trimmed(risk.value_bps, decimals);
        let mut details = risk.valuation_label.iter().cloned().collect::<Vec<_>>();
        if let Some(delta) = risk.delta_one_day.filter(|delta| delta.abs() >= 0.05) {
            let decimals = usize::from(delta.abs() < 100.0);
            let sign = if delta > 0.0 { "+" } else { "-" };
            let change = format!("{sign}{}", trimmed(delta.abs(), decimals));
            details.push(match locale {
                Locale::Es => format!("{change} bps vs ayer"),
                Locale::En => format!("{change} bps from yesterday"),
            });
        }
        let mut line = match locale {
            Locale::Es => format!("Riesgo país: {value} bps"),
            Locale::En => format!("Country risk: {value} bps"),
        };
        if !details.is_empty() {
            line.push_str(&format!(" ({})", details.join(" | ")));
        }
        lines.push(line);
    }
    if let Some(bands) = &snapshot.bands {
        let lower = crate::output_format::readable_number(&trimmed(bands.lower, 2));
        let upper = crate::output_format::readable_number(&trimmed(bands.upper, 2));
        let mut line = match locale {
            Locale::Es => format!("Bandas cambiarias: piso ${lower} / techo ${upper}"),
            Locale::En => format!("Exchange-rate bands: floor ${lower} / ceiling ${upper}"),
        };
        if !bands.date.is_empty() && !same_day(&bands.date, common_date) {
            line.push_str(&format!(" ({})", bands.date));
        }
        lines.push(line);
    }
    if let Some(itcrm) = &snapshot.itcrm {
        let suffix = if itcrm.date.is_empty() || same_day(&itcrm.date, common_date) {
            String::new()
        } else {
            format!(" ({})", itcrm.date)
        };
        lines.push(format!(
            "TCRM: {}{suffix}",
            crate::output_format::readable_number(&trimmed(itcrm.value, 2))
        ));
    }
    let notes_start = lines.len();
    if snapshot.stale {
        lines.push(
            match locale {
                Locale::Es => "No hay actualización nueva del BCRA; te muestro lo último que tengo",
                Locale::En => "There is no new BCRA update; showing the latest data",
            }
            .to_owned(),
        );
    }
    if let Some(latest_days) = latest_days {
        let age = today_days - latest_days;
        if age >= 3 {
            lines.push(match locale {
                Locale::Es => {
                    format!("Los datos del BCRA tienen {age} días de atraso. Chequeá más tarde")
                }
                Locale::En => format!("BCRA data is {age} days old. Check again later"),
            });
        }
    }
    if lines.len() > notes_start {
        lines.insert(notes_start, String::new());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{
        BcraBands, BcraSnapshot, BcraVariable, CountryRisk, ItcrmDetails, classify_bcra_command,
        days_from_civil, render_bcra,
    };
    use crate::locale::Locale;

    #[test]
    fn recognizes_public_aliases() {
        assert!(classify_bcra_command("/bcra"));
        assert!(classify_bcra_command("/variables"));
        assert!(!classify_bcra_command("/variable"));
    }

    #[test]
    fn renders_every_indicator_family_enrichment_and_staleness() {
        let variables = [
            ("Base monetaria", "5.000.000,50"),
            ("Inflación mensual", "5,2"),
            ("Inflación interanual", "150,5"),
            ("Inflación esperada", "3,1"),
            ("Tasa TAMAR", "45,0"),
            ("Tasa BADLAR", "40,5"),
            ("Tipo de cambio minorista promedio vendedor", "1.250,75"),
            ("Tipo de cambio mayorista de referencia", "1.180,25"),
            ("Unidad de Valor Adquisitivo UVA", "500,75"),
            ("Coeficiente de Estabilización de Referencia CER", "0,45"),
            ("Reservas internacionales", "25.000"),
        ]
        .into_iter()
        .map(|(description, value)| BcraVariable {
            description: description.to_owned(),
            value: value.to_owned(),
            // Monthly inflation is published on a different day than the rest.
            date: if description == "Inflación mensual" {
                "31/12/2024".to_owned()
            } else {
                "15/01/2025".to_owned()
            },
        })
        .collect();
        let snapshot = BcraSnapshot {
            variables,
            bands: Some(BcraBands {
                lower: 950.12,
                upper: 1_460.34,
                date: "15/09/25".to_owned(),
                lower_change: Some(1.0),
                upper_change: Some(2.0),
            }),
            itcrm: Some(ItcrmDetails {
                value: 123.45,
                date: "01/02/25".to_owned(),
            }),
            country_risk: Some(CountryRisk {
                value_bps: 685.21,
                delta_one_day: Some(-12.3),
                valuation_label: Some("29/10 12:34".to_owned()),
            }),
            stale: true,
        };
        let text = render_bcra(&snapshot, Locale::Es, days_from_civil(2025, 1, 20));
        for expected in [
            "Indicadores del BCRA al 15/01/25\n\n",
            "Base monetaria: $5,000 mill. pesos\n",
            "Inflación mensual: 5.20% (31/12/24)",
            "Inflación interanual: 150.5%",
            "Inflación esperada: 3.10%",
            "TAMAR: 45.0%",
            "Dólar minorista: $1,250.75",
            "Reservas: USD 25,000 millones",
            "Riesgo país: 685 bps (29/10 12:34 | -12.3 bps vs ayer)",
            "Bandas cambiarias: piso $950.12 / techo $1,460.34 (15/09/25)",
            "TCRM: 123.45 (01/02/25)",
            "\n\nNo hay actualización nueva del BCRA",
            "Los datos del BCRA tienen 5 días de atraso",
        ] {
            assert!(text.contains(expected), "missing {expected} in {text}");
        }
    }

    #[test]
    fn lists_each_series_once_even_when_it_matches_two_patterns() {
        let variable = |description: &str, value: &str| BcraVariable {
            description: description.to_owned(),
            value: value.to_owned(),
            date: "19/09/25".to_owned(),
        };
        let snapshot = BcraSnapshot {
            variables: vec![
                variable(
                    "Mediana de la variación interanual esperada del índice de precios al consumidor para los próximos 12 meses (REM)",
                    "21,8",
                ),
                variable(
                    "Variación interanual del índice de precios al consumidor",
                    "33,6",
                ),
            ],
            bands: None,
            itcrm: None,
            country_risk: None,
            stale: false,
        };
        let text = render_bcra(&snapshot, Locale::Es, 0);
        assert_eq!(text.matches("Inflación esperada: 21.8%").count(), 1);
        assert!(text.contains("Inflación interanual: 33.6%"));
    }

    #[test]
    fn shortens_any_four_digit_year() {
        assert_eq!(super::short_year("15/01/2026"), "15/01/26");
        assert_eq!(super::short_year("15/01/26"), "15/01/26");
        assert_eq!(super::short_year("2026-01-15"), "2026-01-15");
        assert!(super::same_day("24/09/26", Some("24/09/2026")));
        assert!(!super::same_day("25/09/26", Some("24/09/2026")));
        assert!(!super::same_day("24/09/26", None));
    }

    #[test]
    fn empty_and_english_snapshots_are_localized() {
        let empty = BcraSnapshot {
            variables: Vec::new(),
            bands: None,
            itcrm: None,
            country_risk: None,
            stale: false,
        };
        assert_eq!(
            render_bcra(&empty, Locale::En, 0),
            "I could not load the BCRA variables. Try again later"
        );
        let snapshot = BcraSnapshot {
            variables: vec![BcraVariable {
                description: "Reservas internacionales".to_owned(),
                value: "25000".to_owned(),
                date: String::new(),
            }],
            bands: None,
            itcrm: None,
            country_risk: Some(CountryRisk {
                value_bps: 55.5,
                delta_one_day: Some(0.01),
                valuation_label: None,
            }),
            stale: false,
        };
        let text = render_bcra(&snapshot, Locale::En, 0);
        assert!(text.contains("Reserves: USD 25,000 million"));
        assert!(text.contains("Country risk: 55.5 bps"));
        assert!(!text.contains("yesterday"));
    }
}
