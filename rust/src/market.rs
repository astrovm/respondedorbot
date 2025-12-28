use crate::http_cache::cached_get_json;
use crate::bcra;

const TTL_PRICE: u64 = 300;
const TTL_DOLLAR: u64 = 300;

pub async fn get_prices(
    http: &reqwest::Client,
    redis: &redis::Client,
    msg_text: &str,
) -> Option<String> {
    let mut convert_to = "USD".to_string();
    let mut convert_param = "USD".to_string();

    let mut text = msg_text.trim().to_string();
    if text.to_uppercase().contains(" IN ") {
        let upper = text.to_uppercase();
        let parts: Vec<&str> = upper.split_whitespace().collect();
        if let Some(last) = parts.last() {
            let currency = last.to_string();
            let supported = supported_currencies();
            if supported.contains(&currency.as_str()) {
                if currency == "SATS" {
                    convert_param = "BTC".to_string();
                } else {
                    convert_param = currency.clone();
                }
                convert_to = currency;
                text = text
                    .to_uppercase()
                    .replace(&format!("IN {}", convert_to), "")
                    .trim()
                    .to_string();
            } else {
                return Some(format!("no laburo con {} gordo", currency));
            }
        }
    }

    let params = vec![
        ("start", "1".to_string()),
        ("limit", "100".to_string()),
        ("convert", convert_param.clone()),
    ];

    let headers = vec![
        (
            "X-CMC_PRO_API_KEY",
            std::env::var("COINMARKETCAP_KEY").unwrap_or_default(),
        ),
        ("Accepts", "application/json".to_string()),
    ];

    let data = cached_get_json(
        http,
        redis,
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
        Some(&params),
        Some(&headers),
        TTL_PRICE,
    )
    .await?;

    let mut coins = data.get("data")?.as_array()?.clone();
    if coins.is_empty() {
        return Some("Error getting crypto prices".to_string());
    }

    let mut limit = 0usize;
    if !text.is_empty() {
        for part in text.split(',') {
            if let Ok(num) = part.trim().parse::<f64>() {
                let num = num.floor() as usize;
                if num > limit {
                    limit = num;
                }
            }
        }
    }

    let mut filtered = Vec::new();
    if !text.is_empty() {
        let mut requested: Vec<String> = text
            .replace(' ', "")
            .split(',')
            .map(|v| v.to_uppercase())
            .collect();

        if requested.iter().any(|v| v == "STABLES" || v == "STABLECOINS") {
            requested.extend(stablecoin_list().iter().map(|v| v.to_string()));
        }

        for coin in &coins {
            let symbol = coin
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_uppercase();
            let name = coin
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_uppercase();
            if requested.iter().any(|v| v == &symbol || v == &name) {
                filtered.push(coin.clone());
            }
        }

        if filtered.is_empty() && limit > 0 {
            filtered = coins.iter().take(limit).cloned().collect();
        }
    }

    if !filtered.is_empty() {
        coins = filtered;
    }

    if limit == 0 {
        limit = 10;
    }

    let mut lines = Vec::new();
    for coin in coins.iter().take(limit) {
        let symbol = coin
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let quote = coin.get("quote")?.get(&convert_param)?;
        let price = quote.get("price")?.as_f64()?;
        let pct = quote
            .get("percent_change_24h")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let mut price_value = price;
        if convert_to == "SATS" {
            price_value *= 100_000_000.0;
        }

        let price_text = trim_float(price_value, 6);
        let pct_text = format_signed(pct, 2);
        lines.push(format!("{}: {} {} ({}% 24hs)", symbol, price_text, convert_to, pct_text));
    }

    Some(lines.join("\n"))
}

pub async fn get_dollar_rates(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<String> {
    let data = cached_get_json(
        http,
        redis,
        "https://criptoya.com/api/dolar",
        None,
        None,
        TTL_DOLLAR,
    )
    .await?;

    let data = data.get("data").cloned().unwrap_or(data);

    let mut rates = vec![
        rate(&data, "Mayorista", &["mayorista", "price"], &["mayorista", "variation"]),
        rate(&data, "Oficial", &["oficial", "price"], &["oficial", "variation"]),
        rate(&data, "Tarjeta", &["tarjeta", "price"], &["tarjeta", "variation"]),
        rate(&data, "MEP", &["mep", "al30", "ci", "price"], &["mep", "al30", "ci", "variation"]),
        rate(&data, "CCL", &["ccl", "al30", "ci", "price"], &["ccl", "al30", "ci", "variation"]),
        rate(&data, "Blue", &["blue", "ask"], &["blue", "variation"]),
        rate(&data, "Bitcoin", &["cripto", "ccb", "ask"], &["cripto", "ccb", "variation"]),
        rate(&data, "USDC", &["cripto", "usdc", "ask"], &["cripto", "usdc", "variation"]),
        rate(&data, "USDT", &["cripto", "usdt", "ask"], &["cripto", "usdt", "variation"]),
    ];

    let mut bands_date: Option<String> = None;
    if let Some(bands) = bcra::get_currency_band_limits(http, redis).await {
        bands_date = bands.date.clone();
        if let Some(lower) = bands.lower {
            rates.push(DollarRate {
                name: "Banda piso".to_string(),
                price: lower,
                history: bands.lower_change_pct,
            });
        }
        if let Some(upper) = bands.upper {
            rates.push(DollarRate {
                name: "Banda techo".to_string(),
                price: upper,
                history: bands.upper_change_pct,
            });
        }
    }

    rates.retain(|r| r.price > 0.0);
    rates.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));

    let mut lines = Vec::new();
    for rate in rates {
        let price_text = trim_float(rate.price, 2);
        let mut line = format!("{}: {}", rate.name, price_text);
        if let Some(hist) = rate.history {
            line.push_str(&format!(" ({}% 24hs)", format_signed(hist, 2)));
        }
        lines.push(line);
    }

    if let Some(date) = bands_date {
        lines.push(format!("Bandas al {}", date));
    }

    Some(lines.join("\n"))
}

pub async fn get_market_context(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(btc) = get_btc_price(http, redis).await {
        parts.push(format!("BTC: {}", trim_float(btc, 2)));
    }
    if let Some(dollar) = get_dollar_rates(http, redis).await {
        parts.push(dollar);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

pub async fn get_devo(
    http: &reqwest::Client,
    redis: &redis::Client,
    msg_text: &str,
) -> String {
    let mut fee = 0.0;
    let mut compra = 0.0;

    let text = msg_text.trim();
    if text.contains(',') {
        let parts: Vec<&str> = text.split(',').collect();
        if let Some(first) = parts.get(0) {
            fee = first.trim().parse::<f64>().unwrap_or(0.0) / 100.0;
        }
        if let Some(second) = parts.get(1) {
            compra = second.trim().parse::<f64>().unwrap_or(0.0);
        }
    } else if !text.is_empty() {
        fee = text.parse::<f64>().unwrap_or(0.0) / 100.0;
    }

    if fee.is_nan() || fee > 1.0 || compra.is_nan() || compra < 0.0 {
        return "Invalid input. Fee should be between 0 and 100, and purchase amount should be a positive number.".to_string();
    }

    let dollars = cached_get_json(
        http,
        redis,
        "https://criptoya.com/api/dolar",
        None,
        None,
        TTL_DOLLAR,
    )
    .await;
    let Some(dollars) = dollars else {
        return "Error getting dollar rates".to_string();
    };
    let data = dollars.get("data").cloned().unwrap_or(dollars);

    let usdt_ask = data["cripto"]["usdt"]["ask"].as_f64().unwrap_or(0.0);
    let usdt_bid = data["cripto"]["usdt"]["bid"].as_f64().unwrap_or(0.0);
    let usdt = (usdt_ask + usdt_bid) / 2.0;
    let oficial = data["oficial"]["price"].as_f64().unwrap_or(0.0);
    let tarjeta = data["tarjeta"]["price"].as_f64().unwrap_or(0.0);

    let profit = -(fee * usdt + oficial - usdt) / tarjeta;

    let mut msg = format!(
        "Profit: {:.2}%\n\nFee: {:.2}%\nOficial: {}\nUSDT: {}\nTarjeta: {}",
        profit * 100.0,
        fee * 100.0,
        trim_float(oficial, 2),
        trim_float(usdt, 2),
        trim_float(tarjeta, 2)
    );

    if compra > 0.0 {
        let compra_ars = compra * tarjeta;
        let compra_usdt = compra_ars / usdt;
        let ganancia_ars = compra_ars * profit;
        let ganancia_usdt = ganancia_ars / usdt;
        msg = format!(
            "{} USD Tarjeta = {} ARS = {} USDT\nGanarias {} ARS / {} USDT\nTotal: {} ARS / {} USDT\n\n{}",
            trim_float(compra, 2),
            trim_float(compra_ars, 2),
            trim_float(compra_usdt, 2),
            trim_float(ganancia_ars, 2),
            trim_float(ganancia_usdt, 2),
            trim_float(compra_ars + ganancia_ars, 2),
            trim_float(compra_usdt + ganancia_usdt, 2),
            msg
        );
    }

    msg
}

pub async fn get_rulo(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> String {
    let usd_amount: f64 = 1000.0;
    let amount_param = if usd_amount.fract() == 0.0 {
        format!("{}", usd_amount as i64)
    } else {
        usd_amount.to_string()
    };

    let dollars = cached_get_json(
        http,
        redis,
        "https://criptoya.com/api/dolar",
        None,
        None,
        TTL_DOLLAR,
    )
    .await;

    let Some(dollars) = dollars else {
        return "Error consiguiendo cotizaciones del dólar".to_string();
    };

    let data = dollars.get("data").cloned().unwrap_or(dollars);
    let oficial_price = safe_float(data.get("oficial").and_then(|v| v.get("price")));
    if oficial_price <= 0.0 {
        return "No pude conseguir el oficial para armar el rulo".to_string();
    }

    let oficial_cost_ars = oficial_price * usd_amount;
    let base_usd = format_local_currency(usd_amount, 0);
    let base_ars = format_local_currency(oficial_cost_ars, 2);

    let mut lines = vec![
        format!(
            "Rulos desde Oficial (precio oficial: {} ARS/USD)",
            format_local_currency(oficial_price, 2)
        ),
        format!("Inversión base: {} USD → {} ARS", base_usd, base_ars),
        "".to_string(),
    ];

    let mep_price = safe_float(data.get("mep").and_then(|v| v.get("al30")).and_then(|v| v.get("ci")).and_then(|v| v.get("price")));
    if mep_price > 0.0 {
        let mep_final = mep_price * usd_amount;
        let mep_profit = mep_final - oficial_cost_ars;
        let extra = vec![
            format!("Resultado: {} USD → {} ARS", base_usd, format_local_currency(mep_final, 2)),
            format!("Ganancia: {} ARS", format_local_signed(mep_profit, 2)),
        ];
        lines.push(format_spread_line("MEP (AL30 CI)", mep_price, oficial_price, &extra));
    }

    let blue_price = safe_float(data.get("blue").and_then(|v| v.get("bid")).or_else(|| data.get("blue").and_then(|v| v.get("price"))));
    if blue_price > 0.0 {
        let blue_final = blue_price * usd_amount;
        let blue_profit = blue_final - oficial_cost_ars;
        let extra = vec![
            format!("Resultado: {} USD → {} ARS", base_usd, format_local_currency(blue_final, 2)),
            format!("Ganancia: {} ARS", format_local_signed(blue_profit, 2)),
        ];
        lines.push(format_spread_line("Blue", blue_price, oficial_price, &extra));
    }

    let usd_usdt = cached_get_json(
        http,
        redis,
        &format!("https://criptoya.com/api/USDT/USD/{}", amount_param),
        None,
        None,
        TTL_DOLLAR,
    )
    .await;
    let usdt_ars = cached_get_json(
        http,
        redis,
        &format!("https://criptoya.com/api/USDT/ARS/{}", amount_param),
        None,
        None,
        TTL_DOLLAR,
    )
    .await;

    let excluded_usd_to_usdt = ["banexcoin", "xapo", "x4t"];
    let excluded_usdt_to_ars = ["okexp2p"];

    let mut best_usd_to_usdt: Option<(String, f64)> = None;
    if let Some(usd_usdt) = usd_usdt {
        if let Some(map) = usd_usdt.get("data").and_then(|v| v.as_object()) {
            for (exchange, quote) in map {
                if excluded_usd_to_usdt.iter().any(|e| e.eq_ignore_ascii_case(exchange)) {
                    continue;
                }
                let ask = safe_float(quote.get("totalAsk").or_else(|| quote.get("ask")));
                if ask <= 0.0 {
                    continue;
                }
                if best_usd_to_usdt.as_ref().map(|(_, v)| ask < *v).unwrap_or(true) {
                    best_usd_to_usdt = Some((exchange.to_string(), ask));
                }
            }
        }
    }

    let mut best_usdt_to_ars: Option<(String, f64)> = None;
    if let Some(usdt_ars) = usdt_ars {
        if let Some(map) = usdt_ars.get("data").and_then(|v| v.as_object()) {
            for (exchange, quote) in map {
                if excluded_usdt_to_ars.iter().any(|e| e.eq_ignore_ascii_case(exchange)) {
                    continue;
                }
                let bid = safe_float(quote.get("totalBid").or_else(|| quote.get("bid")));
                if bid <= 0.0 {
                    continue;
                }
                if best_usdt_to_ars.as_ref().map(|(_, v)| bid > *v).unwrap_or(true) {
                    best_usdt_to_ars = Some((exchange.to_string(), bid));
                }
            }
        }
    }

    if let (Some(best_usd_to_usdt), Some(best_usdt_to_ars)) = (best_usd_to_usdt, best_usdt_to_ars) {
        let usd_to_usdt_rate = best_usd_to_usdt.1;
        let usdt_to_ars_rate = best_usdt_to_ars.1;
        let usdt_obtained = usd_amount / usd_to_usdt_rate;
        let ars_obtained = usdt_obtained * usdt_to_ars_rate;
        let final_price = ars_obtained / usd_amount;
        let profit = ars_obtained - oficial_cost_ars;
        let extra = vec![
            format!("Tramos: USD→USDT {}, USDT→ARS {}", best_usd_to_usdt.0.to_uppercase(), best_usdt_to_ars.0.to_uppercase()),
            format!("Resultado: {} USD → {} USDT → {} ARS", base_usd, format_local_currency(usdt_obtained, 2), format_local_currency(ars_obtained, 2)),
            format!("Ganancia: {} ARS", format_local_signed(profit, 2)),
        ];
        lines.push(format_spread_line("USDT", final_price, oficial_price, &extra));
    }

    if lines.len() <= 2 {
        return "No encontré ningún rulo potable".to_string();
    }

    lines.join("\n")
}

pub async fn powerlaw(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> String {
    let today = chrono::Utc::now();
    let since = chrono::DateTime::parse_from_rfc3339("2009-01-04T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let days_since = (today - since).num_days();
    let value = 1.0117e-17 * (days_since as f64).powf(5.82);

    let price = get_btc_price(http, redis).await.unwrap_or(0.0);
    if price <= 0.0 {
        return "Error getting BTC price for power law calculation".to_string();
    }

    let percentage = ((price - value) / value) * 100.0;
    let percentage_txt = if percentage > 0.0 {
        format!("{percentage:.2}% caro boludo")
    } else {
        format!("{:.2}% regalado gordo", percentage.abs())
    };

    format!("segun power law btc deberia estar en {:.2} usd ({})", value, percentage_txt)
}

pub async fn rainbow(
    http: &reqwest::Client,
    redis: &redis::Client,
) -> String {
    let today = chrono::Utc::now();
    let since = chrono::DateTime::parse_from_rfc3339("2009-01-09T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let days_since = (today - since).num_days();
    let value = 10f64.powf(2.66167155005961 * (days_since as f64).ln() - 17.9183761889864);

    let price = get_btc_price(http, redis).await.unwrap_or(0.0);
    if price <= 0.0 {
        return "Error getting BTC price for rainbow calculation".to_string();
    }

    let percentage = ((price - value) / value) * 100.0;
    let percentage_txt = if percentage > 0.0 {
        format!("{percentage:.2}% caro boludo")
    } else {
        format!("{:.2}% regalado gordo", percentage.abs())
    };

    format!("segun rainbow chart btc deberia estar en {:.2} usd ({})", value, percentage_txt)
}

pub fn convert_base(msg_text: &str) -> String {
    let input_parts: Vec<&str> = msg_text.split(',').collect();
    if input_parts.len() != 3 {
        return "capo mandate algo como /convertbase 101, 2, 10 y te paso de binario a decimal".to_string();
    }
    let number_str = input_parts[0].trim();
    let base_from: i32 = input_parts[1].trim().parse().unwrap_or(0);
    let base_to: i32 = input_parts[2].trim().parse().unwrap_or(0);

    if !number_str.chars().all(|c| c.is_ascii_alphanumeric()) {
        return "el numero tiene que ser alfanumerico boludo".to_string();
    }
    if !(2..=36).contains(&base_from) {
        return format!("base origen '{}' tiene que ser entre 2 y 36 gordo", input_parts[1].trim());
    }
    if !(2..=36).contains(&base_to) {
        return format!("base destino '{}' tiene que ser entre 2 y 36 boludo", input_parts[2].trim());
    }

    let mut value: i128 = 0;
    for ch in number_str.chars() {
        let digit = if ch.is_ascii_digit() {
            ch.to_digit(10).unwrap_or(0) as i128
        } else {
            (ch.to_ascii_uppercase() as i128) - ('A' as i128) + 10
        };
        value = value * base_from as i128 + digit;
    }

    if value == 0 {
        return format!("ahi tenes boludo, {} en base {} es 0 en base {}", number_str, base_from, base_to);
    }

    let mut digits = Vec::new();
    let mut current = value;
    while current > 0 {
        let digit_value = (current % base_to as i128) as i32;
        let digit = if digit_value >= 10 {
            ((digit_value - 10) as u8 + b'A') as char
        } else {
            char::from_digit(digit_value as u32, 10).unwrap()
        };
        digits.push(digit);
        current /= base_to as i128;
    }
    digits.reverse();
    let result: String = digits.into_iter().collect();
    format!(
        "ahi tenes boludo, {} en base {} es {} en base {}",
        number_str, base_from, result, base_to
    )
}

async fn get_btc_price(http: &reqwest::Client, redis: &redis::Client) -> Option<f64> {
    let params = vec![
        ("start", "1".to_string()),
        ("limit", "1".to_string()),
        ("convert", "USD".to_string()),
    ];
    let headers = vec![
        (
            "X-CMC_PRO_API_KEY",
            std::env::var("COINMARKETCAP_KEY").unwrap_or_default(),
        ),
        ("Accepts", "application/json".to_string()),
    ];
    let data = cached_get_json(
        http,
        redis,
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
        Some(&params),
        Some(&headers),
        TTL_PRICE,
    )
    .await?;
    let first = data.get("data")?.as_array()?.get(0)?.clone();
    let quote = first.get("quote")?.get("USD")?;
    quote.get("price").and_then(|v| v.as_f64())
}

#[derive(Debug, Clone)]
struct DollarRate {
    name: String,
    price: f64,
    history: Option<f64>,
}

fn rate(data: &serde_json::Value, name: &str, path: &[&str], history_path: &[&str]) -> DollarRate {
    let price = dig(data, path).unwrap_or(0.0);
    let history = dig(data, history_path);
    DollarRate {
        name: name.to_string(),
        price,
        history,
    }
}

fn dig(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_f64()
}

fn trim_float(value: f64, decimals: usize) -> String {
    let formatted = format!("{:.*}", decimals, value);
    formatted.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn format_signed(value: f64, decimals: usize) -> String {
    let sign = if value >= 0.0 { "+" } else { "-" };
    format!("{}{:.*}", sign, decimals, value.abs())
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

fn supported_currencies() -> Vec<&'static str> {
    vec![
        "ARS", "AUD", "BRL", "BTC", "BUSD", "CAD", "CHF", "CLP", "CNY", "COP", "CZK",
        "DAI", "DKK", "ETH", "EUR", "GBP", "HKD", "ILS", "INR", "ISK", "JPY", "KRW",
        "MXN", "NZD", "PEN", "SATS", "SEK", "SGD", "TWD", "USD", "USDC", "USDT", "UYU",
        "XAU", "XMR",
    ]
}

fn stablecoin_list() -> Vec<&'static str> {
    vec![
        "BUSD", "DAI", "DOC", "EURT", "FDUSD", "FRAX", "GHO", "GUSD", "LUSD", "MAI",
        "MIM", "MIMATIC", "NUARS", "PAXG", "PYUSD", "RAI", "SUSD", "TUSD", "USDC", "USDD",
        "USDM", "USDP", "USDT", "UXD", "XAUT", "XSGD",
    ]
}

fn safe_float(value: Option<&serde_json::Value>) -> f64 {
    value.and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn format_local_currency(value: f64, decimals: usize) -> String {
    let mut s = format!("{:.*}", decimals, value);
    if decimals > 0 {
        s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    s.replace('.', ",")
}

fn format_local_signed(value: f64, decimals: usize) -> String {
    let sign = if value >= 0.0 { "+" } else { "-" };
    format!("{}{:.*}", sign, decimals, value.abs()).replace('.', ",")
}

fn format_spread_line(label: &str, sell_price: f64, oficial_price: f64, details: &[String]) -> String {
    let diff = sell_price - oficial_price;
    let pct = if oficial_price != 0.0 { diff / oficial_price * 100.0 } else { 0.0 };
    let mut lines = vec![
        format!("- {label}"),
        format!("  • Precio venta: {} ARS/USD", format_local_currency(sell_price, 2)),
        format!(
            "  • Spread vs oficial: {} ARS ({}%)",
            format_local_signed(diff, 2),
            format_signed(pct, 2)
        ),
    ];
    for detail in details {
        lines.push(format!("  • {detail}"));
    }
    lines.join("\n")
}
