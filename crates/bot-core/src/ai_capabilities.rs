//! Authoritative localized capability catalog for the AI tool.

use crate::locale::Locale;

const CAPABILITIES_ES: &str = r#"CAPACIDADES DEL BOT:
- si el usuario pregunta qué podés hacer, respondé desde esta lista
- no inventes comandos; /buscar y /search no existen
- si existe comando exacto para algo, sugerilo con el comando exacto
- /ask, /pregunta, /che, /gordo: te contesto mensajes normales; en grupos respondo si me mencionan, me responden, usan trigger random o mandan comando ia
- búsqueda web nativa: en mensajes normales puedo buscar en internet cuando hace falta
- /p, /prices, /price, /precios, /precio, /presios, /presio, /bresio, /bresios, /brecio, /brecios: precios crypto, acciones y otros activos
- /c, /cripto, /criptos, /crypto, /cryptos: precios solo de crypto
- /clima, /weather: clima actual para cualquier ciudad o ubicación
- token cards: si el mensaje completo es un address Solana/EVM o un $ticker, uso el mismo buscador de precios; un activo lleva gráfico y los tokens llevan card con stats, socials y links
- /dolar, /dollar, /usd: cotizaciones del dólar y variaciones por ventana
- /s, /accion, /acciones, /stock, /stocks: precios de acciones por símbolo o empresa desde Yahoo Finance
- /petroleo, /oil: precio Brent y WTI
- /bcra, /variables: variables económicas del BCRA
- /eleccion, /elecciones, /election, /elections: top 10 de elecciones globales en Polymarket por liquidez
- /rulo, /devo, /powerlaw, /rainbow, /satoshi, /sat, /sats: rulo desde oficial, arbitraje tarjeta/crypto, power law, rainbow chart y sats
- /transcribe, /transcript, /describe: transcribo voice/audio/video/video_note o subtítulos de YouTube y describo fotos, stickers o GIF respondiendo al mensaje; también puedo procesar media cuando me hablan
- links: arreglo links de X/Twitter, Bluesky, Instagram y Reddit según config; leo metadata, tweets y transcripts de YouTube como contexto
- /tarea, /tareas, /task, /tasks: agendo recordatorios y tareas recurrentes por lenguaje natural; cualquiera de los comandos lista sin texto y crea con texto
- /resumen, /summary, /tldr: resumo el chat, guardo resumen acumulado y recupero mensajes relevantes para responder con contexto
- /convertbase, /random, /time, /comando, /command, /instance: random, conversión de bases, comandos Telegram, timestamp e instancia
- /gm, /gn: gif random de buenos días o buenas noches
- /config, /configs, /settings: config por chat: idioma, links, followups, timezone, random replies y límite gratis por usuario/hora
- /language, /idioma: cambiá entre español e inglés
- /topup, /balance, /charges, /history, /gastos, /transfer: saldo, historial de gastos, topup con Telegram Stars y transferencia de créditos personales al grupo
- /printcredits, /creditlog (solo admin): mint y log de créditos, solo admin
- /help: muestro comandos y features"#;

const CAPABILITIES_EN: &str = r#"BOT CAPABILITIES:
- if the user asks what you can do, answer from this list
- do not invent commands; /buscar and /search do not exist
- when an exact command exists, suggest that exact command
- /ask, /pregunta, /che, /gordo: I answer normal messages; in groups I respond to mentions, replies, random triggers, and AI commands
- web search: I can search the web when a current answer needs it
- /p, /prices, /price, /precios, /precio, /presios, /presio, /bresio, /bresios, /brecio, /brecios: crypto, stock, and other asset prices
- /c, /cripto, /criptos, /crypto, /cryptos: crypto-only prices
- /clima, /weather: current weather for any city or location
- token cards: send a Solana or EVM address, or a $ticker; canonical assets resolve first, single assets get charts, and unknown symbols use exact token lookup
- /dolar, /dollar, /usd: dollar exchange rates and changes by time window
- /s, /accion, /acciones, /stock, /stocks: stock prices by symbol or company from Yahoo Finance
- /petroleo, /oil: Brent and WTI oil prices
- /bcra, /variables: economic variables from Argentina's central bank
- /eleccion, /elecciones, /election, /elections: top global election markets on Polymarket by liquidity
- /rulo, /devo, /powerlaw, /rainbow, /satoshi, /sat, /sats: official-rate, card/crypto, power-law, rainbow-chart, and satoshi tools
- /transcribe, /transcript, /describe: transcribe audio, video, or YouTube captions; describe images, stickers, or GIFs
- links: fix supported social links and read linked content as context
- /tarea, /tareas, /task, /tasks: create reminders and recurring tasks with natural language
- /resumen, /summary, /tldr: summarize chats and retrieve relevant prior messages
- /convertbase, /random, /time, /comando, /command, /instance: random selection, base conversion, Telegram commands, timestamps, and instance info
- /gm, /gn: random good-morning and good-night GIFs
- /config, /configs, /settings: all chat settings, including language, links, timezone, replies, and group limits
- /language, /idioma: switch between Spanish and English
- /topup, /balance, /charges, /history, /gastos, /transfer: balance, expense history, Telegram Stars top-ups, and group transfers
- /printcredits, /creditlog (admin only): mint and inspect credits; admin only
- /help: show commands and features"#;

#[must_use]
pub const fn render_ai_capabilities(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => CAPABILITIES_ES,
        Locale::En => CAPABILITIES_EN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_bilingual_authoritative_and_includes_admin_restrictions() {
        let spanish = render_ai_capabilities(Locale::Es);
        assert!(spanish.starts_with("CAPACIDADES DEL BOT:"));
        assert!(spanish.contains("/buscar y /search no existen"));
        assert!(spanish.contains("/printcredits, /creditlog (solo admin)"));
        let english = render_ai_capabilities(Locale::En);
        assert!(english.starts_with("BOT CAPABILITIES:"));
        assert!(english.contains("do not invent commands"));
        assert!(english.contains("/printcredits, /creditlog (admin only)"));
    }
}
