use rand::Rng;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

pub fn parse_command(message_text: &str, bot_username: Option<&str>) -> Option<(String, String)> {
    let trimmed = message_text.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let mut parts = trimmed.splitn(2, ' ');
    let raw_command = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim().to_string();

    let mut command = raw_command.to_string();
    if let Some((name, target)) = raw_command.split_once('@') {
        if let Some(expected) = bot_username {
            if !target.eq_ignore_ascii_case(expected) {
                return None;
            }
        }
        command = name.to_string();
    }

    Some((command, args))
}

pub fn help_text() -> String {
    "comandos disponibles boludo:\n\n- /ask, /pregunta, /che, /gordo: te contesto cualquier gilada\n\n- /bcra, /variables: te tiro las variables economicas del bcra\n\n- /comando, /command algo: te lo convierto en comando de telegram\n\n- /convertbase 101, 2, 10: te paso numeros entre bases (ej: binario 101 a decimal)\n\n- /buscar algo: te busco en la web\n\n- /agent: te muestro lo ultimo que penso el agente autonomo\n\n- /eleccion: odds actuales de Polymarket para Diputados 2025\n\n- /devo 0.5, 100: te calculo el arbitraje entre tarjeta y crypto (fee%, monto opcional)\n\n- /rulo: te armo los rulos desde el oficial\n\n- /dolar, /dollar, /usd: te tiro la posta del blue y todos los dolares\n\n- /instance: te digo donde estoy corriendo\n\n- /config: cambiá la config del gordo, link fixer y demases\n\n- /prices, /precio, /precios, /presio, /presios: top 10 cryptos en usd\n- /prices in btc: top 10 en btc\n- /prices 20: top 20 en usd\n- /prices 100 in eur: top 100 en eur\n- /prices btc, eth, xmr: bitcoin, ethereum y monero en usd\n- /prices dai in sats: dai en satoshis\n- /prices stables: stablecoins en usd\n\n- /random pizza, carne, sushi: elijo por vos\n- /random 1-10: numero random del 1 al 10\n\n- /powerlaw: te tiro el precio justo de btc segun power law y si esta caro o barato\n- /rainbow: idem pero con el rainbow chart\n\n- /satoshi, /sat, /sats: te digo cuanto vale un satoshi\n\n- /time: timestamp unix actual\n\n- /transcribe: te transcribo audio o describo imagen (responde a un mensaje)".to_string()
}

pub fn select_random(msg_text: &str) -> String {
    let trimmed = msg_text.trim();
    if trimmed.is_empty() {
        return "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo".to_string();
    }

    let mut rng = rand::rng();

    if trimmed.contains(',') {
        let values: Vec<&str> = trimmed.split(',').map(|v| v.trim()).filter(|v| !v.is_empty()).collect();
        if values.len() >= 2 {
            let idx = rng.random_range(0..values.len());
            return values[idx].to_string();
        }
    }

    if trimmed.contains('-') {
        let parts: Vec<&str> = trimmed.split('-').collect();
        if parts.len() == 2 {
            if let (Ok(start), Ok(end)) = (parts[0].trim().parse::<i64>(), parts[1].trim().parse::<i64>()) {
                if start < end {
                    let value = rng.random_range(start..=end);
                    return value.to_string();
                }
            }
        }
    }

    "mandate algo como 'pizza, carne, sushi' o '1-10' boludo, no me hagas laburar al pedo".to_string()
}

pub fn convert_to_command(msg_text: &str) -> String {
    if msg_text.trim().is_empty() {
        return "y que queres que convierta boludo? mandate texto".to_string();
    }

    let emoji_text = demojize_es(msg_text);
    let upper = emoji_text.to_uppercase();

    let re_n_enie = Regex::new(r"\bÑ\b").unwrap();
    let replaced_ni = re_n_enie.replace_all(&upper, "ENIE").to_string();
    let replaced_ni = replaced_ni.replace('Ñ', "NI");

    let normalized = replaced_ni
        .nfd()
        .filter(|c| c.is_ascii())
        .collect::<String>();

    let single_spaced = Regex::new(r"\s+").unwrap().replace_all(&normalized, " ");
    let mut translated = single_spaced.replace("...", "_PUNTOSSUSPENSIVOS_");
    translated = translated.replace([' ', '\n'], "_")
        .replace('?', "_SIGNODEPREGUNTA_")
        .replace('!', "_SIGNODEEXCLAMACION_")
        .replace('.', "_PUNTO_");

    let underscored = Regex::new(r"_+").unwrap().replace_all(&translated, "_");
    let cleaned = Regex::new(r"[^A-Za-z0-9_]")
        .unwrap()
        .replace_all(&underscored, "");
    let cleaned = Regex::new(r"^_+|_+$").unwrap().replace_all(&cleaned, "");

    let cleaned = cleaned.to_string();
    if cleaned.is_empty() {
        return "no me mandes giladas boludo, tiene que tener letras o numeros".to_string();
    }

    format!("/{}", cleaned)
}

fn demojize_es(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let replacement = match ch {
            '😄' => "_CARA_SONRIENDO_CON_OJOS_SONRIENTES_",
            _ => "",
        };
        if !replacement.is_empty() {
            out.push_str(replacement);
        } else if is_emoji(ch) {
            out.push_str("_EMOJI_");
        } else {
            out.push(ch);
        }
    }
    out
}

fn is_emoji(ch: char) -> bool {
    let code = ch as u32;
    (0x1F300..=0x1FAFF).contains(&code)
        || (0x2600..=0x26FF).contains(&code)
        || (0x2700..=0x27BF).contains(&code)
}
