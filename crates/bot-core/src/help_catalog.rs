//! Localized, dependency-free help catalog used by `/help`.

use crate::locale::Locale;

const HELP_ES: &str = r#"esto es lo que sé hacer, boludo:

ia:
- /ask, /pregunta, /che, /gordo: te contesto mensajes normales; en grupos respondo si me mencionan, me responden, usan trigger random o mandan comando ia
  ejemplo: /gordo explicame esto
- búsqueda web nativa: en mensajes normales puedo buscar en internet cuando hace falta
  ejemplo: buscá qué pasó con...

mercado:
- /prices, /price, /precios, /precio, /presios, /presio, /bresio, /bresios, /brecio, /brecios, /c, /crypto, /criptos: precios crypto, acciones y otros activos; /crypto limita la búsqueda a crypto
  ejemplo: /c nvda
  ejemplo: /precio btc nvda
  ejemplo: /precio stock:META
  ejemplo: /prices btc eth xmr
  ejemplo: /prices 20
  ejemplo: /prices 100 in eur
  ejemplo: /prices btc 7d
  ejemplo: /prices stables

general:
- /clima, /weather: clima actual para cualquier ciudad o ubicación
  ejemplo: /clima Córdoba, Argentina

mercado:
- token cards: si el mensaje completo es un address Solana/EVM o un $ticker, mando card con chart/imagen, stats, socials, links y botones; si el $ticker no es un token, busco su precio de mercado
  ejemplo: J8PS...pump
  ejemplo: $GLORP
- /dolar, /dollar, /usd: cotizaciones del dólar y variaciones por ventana
  ejemplo: /usd 1h
- /acciones, /stocks: precios de acciones por símbolo o empresa desde Yahoo Finance
  ejemplo: /acciones aapl tsla
  ejemplo: /acciones Mercado Libre
- /petroleo, /oil: precio Brent y WTI
- /bcra, /variables: variables económicas del BCRA
- /eleccion, /elecciones, /election, /elections: top 10 de elecciones globales en Polymarket por liquidez
- /rulo, /devo, /powerlaw, /rainbow, /satoshi, /sat, /sats: rulo desde oficial, arbitraje tarjeta/crypto, power law, rainbow chart y sats
  ejemplo: /devo 0.5, 100

media:
- /transcribe, /describe: transcribo voice/audio/video/video_note y describo fotos o stickers respondiendo al mensaje; también puedo procesar media cuando me hablan

links:
- links: arreglo links de X/Twitter, Bluesky, Instagram y Reddit según config; leo metadata, tweets y transcripts de YouTube como contexto

productividad:
- /tarea, /tareas, /task, /tasks: agendo recordatorios y tareas recurrentes por lenguaje natural; cualquiera de los comandos lista sin texto y crea con texto
  ejemplo: /tarea mañana recordame pagar el alquiler
  ejemplo: /tasks

memoria:
- /resumen, /summary, /tldr: resumo el chat, guardo resumen acumulado y recupero mensajes relevantes para responder con contexto
  ejemplo: /resumen focus en crypto

utilidades:
- /convertbase, /random, /time, /comando, /command, /instance: random, conversión de bases, comandos Telegram, timestamp e instancia
  ejemplo: /random pizza, carne, sushi
  ejemplo: /convertbase 101, 2, 10
- /gm, /gn: gif random de buenos días o buenas noches

config:
- /config, /configs, /settings: config por chat: idioma, links, followups, timezone, random replies y límite gratis por usuario/hora
- /language, /idioma: cambiá entre español e inglés

créditos:
- /topup, /balance, /charges, /history, /gastos, /transfer: saldo, historial de gastos, topup con Telegram Stars y transferencia de créditos personales al grupo
  ejemplo: /charges 10
  ejemplo: /transfer 1.5

utilidades:
- /help: muestro comandos y features"#;

const HELP_EN: &str = r#"what I can do:

AI:
- /ask, /pregunta, /che, /gordo: I answer normal messages; in groups I respond to mentions, replies, random triggers, and AI commands
  example: /gordo explain this
- web search: I can search the web when a current answer needs it
  example: search what happened with...

markets:
- /prices, /price, /precios, /precio, /presios, /presio, /bresio, /bresios, /brecio, /brecios, /c, /crypto, /criptos: crypto, stock, and other asset prices; /crypto limits lookup to crypto
  example: /c nvda
  example: /price btc nvda
  example: /price stock:META
  example: /prices btc eth xmr
  example: /prices 20
  example: /prices 100 in eur
  example: /prices btc 7d
  example: /prices stables

general:
- /clima, /weather: current weather for any city or location
  example: /weather London

markets:
- token cards: send a Solana or EVM address, or a $ticker, for a market card; unknown token cashtags fall back to market prices
  example: J8PS...pump
  example: $GLORP
- /dolar, /dollar, /usd: dollar exchange rates and changes by time window
  example: /usd 1h
- /acciones, /stocks: stock prices by symbol or company from Yahoo Finance
  example: /stocks aapl tsla
  example: /stocks Mercado Libre
- /petroleo, /oil: Brent and WTI oil prices
- /bcra, /variables: economic variables from Argentina's central bank
- /eleccion, /elecciones, /election, /elections: top global election markets on Polymarket by liquidity
- /rulo, /devo, /powerlaw, /rainbow, /satoshi, /sat, /sats: official-rate, card/crypto, power-law, rainbow-chart, and satoshi tools
  example: /devo 0.5, 100

media:
- /transcribe, /describe: transcribe audio and video or describe images and stickers

links:
- links: fix supported social links and read linked content as context

productivity:
- /tarea, /tareas, /task, /tasks: create reminders and recurring tasks with natural language
  example: /task tomorrow remind me to pay rent
  example: /tasks

memory:
- /resumen, /summary, /tldr: summarize chats and retrieve relevant prior messages
  example: /summary focus on crypto

utilities:
- /convertbase, /random, /time, /comando, /command, /instance: random selection, base conversion, Telegram commands, timestamps, and instance info
  example: /random pizza, steak, sushi
  example: /convertbase 101, 2, 10
- /gm, /gn: random good-morning and good-night GIFs

settings:
- /config, /configs, /settings: all chat settings, including language, links, timezone, replies, and group limits
- /language, /idioma: switch between Spanish and English

credits:
- /topup, /balance, /charges, /history, /gastos, /transfer: balance, expense history, Telegram Stars top-ups, and group transfers
  example: /charges 10
  example: /transfer 1.5

utilities:
- /help: show commands and features"#;

#[must_use]
pub const fn render_help_text(locale: Locale) -> &'static str {
    match locale {
        Locale::Es => HELP_ES,
        Locale::En => HELP_EN,
    }
}

#[cfg(test)]
mod tests {
    use super::render_help_text;
    use crate::{locale::Locale, telegram_actions::MAX_TELEGRAM_TEXT_LENGTH};

    #[test]
    fn catalogs_preserve_all_public_feature_groups_and_fit_one_message() {
        for (locale, expected_header, expected_example) in [
            (
                Locale::Es,
                "esto es lo que sé hacer, boludo:",
                "/charges 10",
            ),
            (Locale::En, "what I can do:", "/summary focus on crypto"),
        ] {
            let help = render_help_text(locale);
            assert!(help.starts_with(expected_header));
            assert!(help.contains(expected_example));
            assert!(help.contains("/help"));
            assert!(help.chars().count() <= MAX_TELEGRAM_TEXT_LENGTH);
        }
    }
}
