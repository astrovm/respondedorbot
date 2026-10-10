//! Bilingual Telegram command-menu catalog.

use serde::Serialize;

use crate::locale::Locale;
use crate::telegram_actions::{CommandScope, TelegramAction};
use crate::telegram_input::ChatId;

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
        description_es: "preguntame lo que quieras",
        description_en: "ask me anything",
    },
    CommandGroup {
        aliases: &["config", "configs", "settings"],
        description_es: "configuración del chat y de los links",
        description_en: "chat and link settings",
    },
    CommandGroup {
        aliases: &["language", "idioma"],
        description_es: "cambiar el idioma [es|en]",
        description_en: "change the language [es|en]",
    },
    CommandGroup {
        aliases: &["convertbase"],
        description_es: "convertir números entre bases",
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
        description_es: "precios de crypto, acciones y más [símbolo o empresa]",
        description_en: "crypto, stock and other prices [symbol or company]",
    },
    CommandGroup {
        aliases: &["c", "cripto", "criptos", "crypto", "cryptos"],
        description_es: "precios de crypto [moneda] [1h/24h/7d/30d]",
        description_en: "crypto prices [currency] [1h/24h/7d/30d]",
    },
    CommandGroup {
        aliases: &["clima", "weather"],
        description_es: "clima actual [ciudad]",
        description_en: "current weather [city]",
    },
    CommandGroup {
        aliases: &["dolar", "dollar", "usd"],
        description_es: "cotizaciones del dólar [1h/6h/12h/24h/48h]",
        description_en: "dollar exchange rates [1h/6h/12h/24h/48h]",
    },
    CommandGroup {
        aliases: &["petroleo", "oil"],
        description_es: "precio del Brent y del WTI",
        description_en: "Brent and WTI oil prices",
    },
    CommandGroup {
        aliases: &["accion", "acciones", "s", "stock", "stocks"],
        description_es: "precios de acciones [símbolo o empresa]",
        description_en: "stock prices [symbol or company]",
    },
    CommandGroup {
        aliases: &["eleccion", "elecciones", "election", "elections"],
        description_es: "elecciones más líquidas en Polymarket",
        description_en: "top elections on Polymarket by liquidity",
    },
    CommandGroup {
        aliases: &["rulo"],
        description_es: "rulos de arbitraje desde el oficial",
        description_en: "arbitrage routes from the official rate",
    },
    CommandGroup {
        aliases: &["devo"],
        description_es: "arbitraje entre tarjeta y crypto [fee, monto]",
        description_en: "card and crypto arbitrage [fee, amount]",
    },
    CommandGroup {
        aliases: &["powerlaw"],
        description_es: "precio justo de BTC según power law",
        description_en: "Bitcoin power-law fair price",
    },
    CommandGroup {
        aliases: &["rainbow"],
        description_es: "precio justo de BTC según rainbow chart",
        description_en: "Bitcoin rainbow-chart fair price",
    },
    CommandGroup {
        aliases: &["satoshi", "sat", "sats"],
        description_es: "cuánto vale un satoshi",
        description_en: "current value of one satoshi",
    },
    CommandGroup {
        aliases: &["time"],
        description_es: "timestamp Unix actual",
        description_en: "current Unix timestamp",
    },
    CommandGroup {
        aliases: &["comando", "command"],
        description_es: "convertir texto en /comando",
        description_en: "turn text into a /command",
    },
    CommandGroup {
        aliases: &["instance"],
        description_es: "qué instancia del bot responde",
        description_en: "which bot instance is answering",
    },
    CommandGroup {
        aliases: &["help"],
        description_es: "ayuda y lista de comandos",
        description_en: "help and command list",
    },
    CommandGroup {
        aliases: &["transcribe", "transcript", "describe"],
        description_es: "transcribir audio/YouTube o describir imágenes",
        description_en: "transcribe audio/YouTube or describe images",
    },
    CommandGroup {
        aliases: &["bcra", "variables"],
        description_es: "indicadores económicos del BCRA",
        description_en: "BCRA economic indicators",
    },
    CommandGroup {
        aliases: &["topup"],
        description_es: "cargar créditos con Telegram Stars",
        description_en: "add credits with Telegram Stars",
    },
    CommandGroup {
        aliases: &["balance"],
        description_es: "ver tu saldo de créditos",
        description_en: "check your credit balance",
    },
    CommandGroup {
        aliases: &["charges", "history", "gastos"],
        description_es: "historial de gastos de IA [cantidad]",
        description_en: "AI spending history [count]",
    },
    CommandGroup {
        aliases: &["transfer"],
        description_es: "pasar créditos tuyos al grupo [monto]",
        description_en: "move your credits to the group [amount]",
    },
    CommandGroup {
        aliases: &["ignore", "ignorar", "vetar", "ban"],
        description_es: "admins: que ignore a alguien en el grupo",
        description_en: "admins: make me ignore someone in the group",
    },
    CommandGroup {
        aliases: &["unignore", "designorar", "desvetar", "unban"],
        description_es: "admins: que deje de ignorar a alguien",
        description_en: "admins: stop ignoring someone",
    },
    CommandGroup {
        aliases: &["ignored", "ignorados", "vetados", "bans", "banned"],
        description_es: "ver a quién ignoro en el grupo",
        description_en: "see who I ignore in the group",
    },
    CommandGroup {
        aliases: &["limitar", "limit"],
        description_es: "admins: cuántos mensajes por hora le paga el grupo a alguien",
        description_en: "admins: how many messages per hour the group pays for someone",
    },
    CommandGroup {
        aliases: &["limitados", "limited"],
        description_es: "ver quién tiene límite propio en el grupo",
        description_en: "see who has their own limit in the group",
    },
    CommandGroup {
        aliases: &["gastosgrupo", "groupcharges"],
        description_es: "admins: quién gastó créditos del grupo [días]",
        description_en: "admins: who spent the group's credits [days]",
    },
    CommandGroup {
        aliases: &["gm"],
        description_es: "GIF de buenos días",
        description_en: "good-morning GIF",
    },
    CommandGroup {
        aliases: &["gn"],
        description_es: "GIF de buenas noches",
        description_en: "good-night GIF",
    },
    CommandGroup {
        aliases: &["tarea", "tareas", "task", "tasks"],
        description_es: "crear o ver tareas programadas",
        description_en: "create or view scheduled tasks",
    },
    CommandGroup {
        aliases: &["resumen", "summary", "tldr"],
        description_es: "resumir la conversación [enfoque opcional]",
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

/// The `/` picker shows one entry per command, most useful first. Names are
/// the same in every language (mostly English, Spanish for the Argentine
/// ones) so the menu never changes; only the descriptions are translated.
/// Every alias in [`telegram_commands`] still works.
const MENU: &[&str] = &[
    "ask",
    "p",
    "c",
    "s",
    "dolar",
    "weather",
    "tarea",
    "resumen",
    "transcribe",
    "balance",
    "topup",
    "charges",
    "transfer",
    "config",
    "language",
    "ignore",
    "unignore",
    "ignored",
    "limit",
    "limited",
    "groupcharges",
    "help",
    "bcra",
    "rulo",
    "devo",
    "oil",
    "elections",
    "powerlaw",
    "rainbow",
    "satoshi",
    "random",
    "convertbase",
    "command",
    "time",
    "gm",
    "gn",
    "instance",
];

#[must_use]
pub fn primary_telegram_commands(locale: Locale) -> Vec<TelegramCommand> {
    MENU.iter()
        .filter_map(|command| {
            COMMAND_GROUPS
                .iter()
                .find(|group| group.aliases.contains(command))
                .map(|group| TelegramCommand {
                    command,
                    description: match locale {
                        Locale::Es => group.description_es,
                        Locale::En => group.description_en,
                    },
                })
        })
        .collect()
}

/// Startup menus: the same commands everywhere, described in Spanish unless
/// the member's Telegram app is in English.
#[must_use]
pub fn command_publication_actions() -> Vec<TelegramAction> {
    [
        (None, Locale::Es, CommandScope::Default),
        (Some("es"), Locale::Es, CommandScope::Default),
        (Some("en"), Locale::En, CommandScope::Default),
        (None, Locale::Es, CommandScope::AllGroupChats),
        (Some("en"), Locale::En, CommandScope::AllGroupChats),
    ]
    .into_iter()
    .map(
        |(language_code, locale, scope)| TelegramAction::SetCommands {
            commands: primary_telegram_commands(locale),
            language_code: language_code.map(ToOwned::to_owned),
            scope,
        },
    )
    .collect()
}

/// Menu for one chat with descriptions in its configured language, so they
/// match the replies even when a member's Telegram app uses another language.
#[must_use]
pub fn chat_command_menu_action(chat_id: ChatId, locale: Locale) -> TelegramAction {
    TelegramAction::SetCommands {
        commands: primary_telegram_commands(locale),
        language_code: None,
        scope: CommandScope::Chat(chat_id),
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use sha2::{Digest, Sha256};

    use super::{command_publication_actions, telegram_commands};
    use crate::locale::Locale;
    use crate::telegram_input::ChatId;

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
                "f8a967d606a2cee730019d136cdd65c2a2901d1af370b0c632c7e86f72df7508",
            ),
            (
                Locale::En,
                "95879fe87904db115fb4e0f3a8c55453105f37805e794f0c48d4b93123bb331b",
            ),
        ] {
            let commands = telegram_commands(locale);
            assert_eq!(commands.len(), 94);
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
        for hidden in ["printcredits", "creditlog", "buscar", "search"] {
            assert!(!commands.iter().any(|entry| entry.command == hidden));
        }
        assert!(commands.iter().any(|entry| entry.command == "tldr"));
    }

    #[test]
    fn primary_menu_names_are_the_same_in_every_language() {
        let spanish = super::primary_telegram_commands(Locale::Es);
        let english = super::primary_telegram_commands(Locale::En);
        let names = |menu: &[super::TelegramCommand]| {
            menu.iter().map(|entry| entry.command).collect::<Vec<_>>()
        };
        assert_eq!(names(&spanish), names(&english));
        for name in [
            "dolar",
            "weather",
            "tarea",
            "resumen",
            "charges",
            "language",
            "ignore",
            "unignore",
            "ignored",
            "oil",
            "elections",
            "command",
        ] {
            assert!(spanish.iter().any(|entry| entry.command == name), "{name}");
        }
        let description = |menu: &[super::TelegramCommand], name: &str| {
            menu.iter()
                .find(|entry| entry.command == name)
                .map(|entry| entry.description)
        };
        assert_eq!(
            description(&spanish, "ignored"),
            Some("ver a quién ignoro en el grupo")
        );
        assert_eq!(
            description(&english, "ignored"),
            Some("see who I ignore in the group")
        );
    }

    #[test]
    fn primary_menu_keeps_aliases_out_of_picker_only() {
        let commands = super::primary_telegram_commands(Locale::En);
        for alias in [
            "prices",
            "precios",
            "settings",
            "tldr",
            "dollar",
            "clima",
            "task",
            "summary",
            "gastos",
            "idioma",
            "vetar",
            "desvetar",
            "vetados",
            "banned",
            "ban",
            "unban",
            "bans",
            "ignorar",
            "gastosgrupo",
            "petroleo",
            "elecciones",
            "comando",
        ] {
            assert!(!commands.iter().any(|c| c.command == alias));
            assert!(
                telegram_commands(Locale::En)
                    .iter()
                    .any(|c| c.command == alias)
            );
        }
    }

    #[test]
    fn primary_menu_lists_every_command_group_once_in_curated_order() {
        for locale in [Locale::Es, Locale::En] {
            let commands = super::primary_telegram_commands(locale);
            assert_eq!(commands.len(), super::COMMAND_GROUPS.len());
            assert_eq!(commands[0].command, "ask");
            for group in super::COMMAND_GROUPS {
                assert_eq!(
                    commands
                        .iter()
                        .filter(|entry| group.aliases.contains(&entry.command))
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn publication_plans_spanish_and_english_descriptions_for_private_and_group_chats() {
        use crate::telegram_actions::{CommandScope, TelegramAction};
        let actions = command_publication_actions();
        // Unrelated actions are ignored by the filter below.
        let unrelated = TelegramAction::DeleteMessage {
            chat_id: ChatId(1),
            message_id: crate::telegram_input::MessageId(1),
        };
        let plans = actions
            .iter()
            .chain(std::iter::once(&unrelated))
            .filter_map(|action| match action {
                TelegramAction::SetCommands {
                    commands,
                    language_code,
                    scope,
                } => Some((commands[0].description, language_code.as_deref(), *scope)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let es = "preguntame lo que quieras";
        let en = "ask me anything";
        assert_eq!(
            plans,
            [
                (es, None, CommandScope::Default),
                (es, Some("es"), CommandScope::Default),
                (en, Some("en"), CommandScope::Default),
                (es, None, CommandScope::AllGroupChats),
                (en, Some("en"), CommandScope::AllGroupChats),
            ]
        );
    }

    #[test]
    fn chat_menu_uses_the_chat_scope_and_its_language() {
        use crate::telegram_actions::{CommandScope, TelegramAction};
        for (locale, expected) in [
            (Locale::Es, "preguntame lo que quieras"),
            (Locale::En, "ask me anything"),
        ] {
            let action = super::chat_command_menu_action(ChatId(-42), locale);
            assert!(matches!(
                &action,
                TelegramAction::SetCommands {
                    commands,
                    language_code: None,
                    scope: CommandScope::Chat(ChatId(-42)),
                } if commands[0].command == "ask" && commands[0].description == expected
            ));
        }
    }
}
