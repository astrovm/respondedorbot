//! Bilingual Telegram command-menu catalog.

use serde::Serialize;

use crate::locale::Locale;
use crate::telegram_actions::TelegramAction;

#[derive(Debug, Clone, Copy)]
struct CommandGroup {
    aliases: &'static [&'static str],
    description_es: &'static str,
    description_en: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TelegramCommand {
    pub command: &'static str,
    pub description: &'static str,
}

const COMMAND_GROUPS: &[CommandGroup] = &[
    CommandGroup {
        aliases: &["ask", "pregunta", "che", "gordo"],
        description_es: "te contesto cualquier gilada",
        description_en: "ask me anything",
    },
    CommandGroup {
        aliases: &["config", "configs", "settings"],
        description_es: "tocás la config del gordo y de los links",
        description_en: "open all bot settings",
    },
    CommandGroup {
        aliases: &["language", "idioma"],
        description_es: "cambiás el idioma del bot [es|en]",
        description_en: "change the bot language [es|en]",
    },
    CommandGroup {
        aliases: &["convertbase"],
        description_es: "te paso números entre bases",
        description_en: "convert numbers between bases",
    },
    CommandGroup {
        aliases: &["random"],
        description_es: "elijo por vos entre opciones o números",
        description_en: "pick an option or number for you",
    },
    CommandGroup {
        aliases: &[
            "prices", "price", "precios", "precio", "presios", "presio", "bresio", "bresios",
            "brecio", "brecios", "p",
        ],
        description_es: "precios de crypto, acciones y otros activos [símbolo o empresa]",
        description_en: "crypto, stock, and other asset prices [symbol or company]",
    },
    CommandGroup {
        aliases: &["c", "cripto", "criptos", "crypto", "cryptos"],
        description_es: "precios crypto por símbolo [moneda] [1h/24h/7d/30d]",
        description_en: "crypto prices by symbol [currency] [1h/24h/7d/30d]",
    },
    CommandGroup {
        aliases: &["clima", "weather"],
        description_es: "clima actual [ciudad o ubicación]",
        description_en: "current weather [city or location]",
    },
    CommandGroup {
        aliases: &["dolar", "dollar", "usd"],
        description_es: "cotizaciones del dolar [1h/6h/12h/24h/48h]",
        description_en: "dollar exchange rates [1h/6h/12h/24h/48h]",
    },
    CommandGroup {
        aliases: &["petroleo", "oil"],
        description_es: "te paso el precio del Brent y del WTI",
        description_en: "Brent and WTI oil prices",
    },
    CommandGroup {
        aliases: &["accion", "acciones", "s", "stock", "stocks"],
        description_es: "precios por símbolo o empresa [aapl tsla]",
        description_en: "stock prices by symbol or company [aapl tsla]",
    },
    CommandGroup {
        aliases: &["eleccion", "elecciones", "election", "elections"],
        description_es: "top 10 de elecciones globales en Polymarket por liquidez",
        description_en: "top global Polymarket elections by liquidity",
    },
    CommandGroup {
        aliases: &["rulo"],
        description_es: "te armo los rulos desde el oficial",
        description_en: "calculate arbitrage from the official exchange rate",
    },
    CommandGroup {
        aliases: &["devo"],
        description_es: "te calculo el arbitraje entre tarjeta y crypto",
        description_en: "calculate card and crypto arbitrage",
    },
    CommandGroup {
        aliases: &["powerlaw"],
        description_es: "te tiro el precio justo de btc según power law",
        description_en: "Bitcoin power-law fair price",
    },
    CommandGroup {
        aliases: &["rainbow"],
        description_es: "te tiro el precio justo de btc según rainbow chart",
        description_en: "Bitcoin rainbow-chart fair price",
    },
    CommandGroup {
        aliases: &["satoshi", "sat", "sats"],
        description_es: "te digo cuánto vale un satoshi",
        description_en: "current value of one satoshi",
    },
    CommandGroup {
        aliases: &["time"],
        description_es: "timestamp unix actual",
        description_en: "current Unix timestamp",
    },
    CommandGroup {
        aliases: &["comando", "command"],
        description_es: "te lo convierto en comando de telegram",
        description_en: "convert text into a Telegram command",
    },
    CommandGroup {
        aliases: &["instance"],
        description_es: "nombre de esta instancia del bot",
        description_en: "name of this bot instance",
    },
    CommandGroup {
        aliases: &["help"],
        description_es: "te muestro todos los comandos",
        description_en: "show all commands",
    },
    CommandGroup {
        aliases: &["transcribe", "transcript", "describe"],
        description_es: "transcribo audio o YouTube y describo imágenes o GIF",
        description_en: "transcribe audio or YouTube and describe images or GIFs",
    },
    CommandGroup {
        aliases: &["bcra", "variables"],
        description_es: "te tiro las variables económicas del bcra",
        description_en: "show BCRA economic variables",
    },
    CommandGroup {
        aliases: &["topup"],
        description_es: "cargás créditos IA con Telegram Stars por privado",
        description_en: "add AI credits with Telegram Stars in private",
    },
    CommandGroup {
        aliases: &["balance"],
        description_es: "te muestro tu saldo IA",
        description_en: "show your AI balance",
    },
    CommandGroup {
        aliases: &["charges", "history", "gastos"],
        description_es: "te muestro cuánto pagaste por cada uso de IA [cantidad]",
        description_en: "show what each AI use cost [count]",
    },
    CommandGroup {
        aliases: &["transfer"],
        description_es: "le pasás créditos tuyos al grupo",
        description_en: "move your credits to the group",
    },
    CommandGroup {
        aliases: &["gm"],
        description_es: "gif de buenos días",
        description_en: "random good-morning GIF",
    },
    CommandGroup {
        aliases: &["gn"],
        description_es: "gif de buenas noches",
        description_en: "random good-night GIF",
    },
    CommandGroup {
        aliases: &["tarea", "tareas", "task", "tasks"],
        description_es: "creá una tarea con texto o listá las existentes",
        description_en: "create a task from text or list existing tasks",
    },
    CommandGroup {
        aliases: &["resumen", "summary", "tldr"],
        description_es: "resumí la conversación [enfoque opcional]",
        description_en: "summarize the conversation [optional focus]",
    },
];

#[must_use]
pub fn telegram_commands(locale: Locale) -> Vec<TelegramCommand> {
    let mut commands = COMMAND_GROUPS
        .iter()
        .flat_map(|group| {
            let description = match locale {
                Locale::Es => group.description_es,
                Locale::En => group.description_en,
            };
            group.aliases.iter().map(move |command| TelegramCommand {
                command,
                description,
            })
        })
        .collect::<Vec<_>>();
    commands.sort_unstable_by_key(|command| command.command);
    commands
}

#[must_use]
pub fn command_publication_actions() -> Vec<TelegramAction> {
    [
        (None, Locale::Es),
        (Some("es"), Locale::Es),
        (Some("en"), Locale::En),
    ]
    .into_iter()
    .map(|(language_code, locale)| TelegramAction::SetCommands {
        commands: telegram_commands(locale),
        language_code: language_code.map(ToOwned::to_owned),
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use sha2::{Digest, Sha256};

    use super::{command_publication_actions, telegram_commands};
    use crate::locale::Locale;

    fn sha256_hex(value: &str) -> String {
        let mut encoded = String::with_capacity(64);
        for byte in Sha256::digest(value) {
            assert!(write!(&mut encoded, "{byte:02x}").is_ok());
        }
        encoded
    }

    #[test]
    fn menus_match_exact_catalog_hashes() {
        for (locale, expected_hash) in [
            (
                Locale::Es,
                "10b4affa0ff7788639e40830864167163a3a2b0d2898039f5f784752508450a7",
            ),
            (
                Locale::En,
                "85fe6c0b368a678df15bb41ec2d659ab9005a0ec4362ddbd15b817b1c5300d98",
            ),
        ] {
            let commands = telegram_commands(locale);
            assert_eq!(commands.len(), 75);
            let encoded = serde_json::to_string(&commands);
            assert!(encoded.is_ok());
            let digest = encoded.map(|value| sha256_hex(&value));
            assert_eq!(digest.ok().as_deref(), Some(expected_hash));
        }
    }

    #[test]
    fn menus_are_sorted_unique_and_exclude_hidden_admin_commands() {
        let commands = telegram_commands(Locale::Es);
        assert!(
            commands
                .windows(2)
                .all(|pair| pair[0].command < pair[1].command)
        );
        assert!(!commands.iter().any(|entry| {
            matches!(
                entry.command,
                "printcredits" | "creditlog" | "buscar" | "search"
            )
        }));
        assert!(commands.iter().any(|entry| entry.command == "tldr"));
    }

    #[test]
    fn publication_plans_default_spanish_and_english_menus() {
        let actions = command_publication_actions();
        assert_eq!(actions.len(), 3);
        let languages = actions
            .iter()
            .filter_map(|action| match action {
                crate::telegram_actions::TelegramAction::SetCommands {
                    commands,
                    language_code,
                } => Some((commands.len(), language_code.as_deref())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(languages, [(75, None), (75, Some("es")), (75, Some("en"))]);
    }
}
