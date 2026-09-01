//! Typed, deterministic AI system and conversation prompt construction.

use crate::locale::Locale;
use crate::message_state::truncate_text;

const PROMPT_TEXT_LIMIT: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptContent {
    Text(String),
    TextParts(Vec<String>),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptToolCall {
    pub id: String,
    pub call_type: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub content: PromptContent,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<PromptToolCall>,
}

impl PromptMessage {
    #[must_use]
    pub fn text(role: PromptRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: PromptContent::Text(content.into()),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }
    }

    #[must_use]
    pub fn assistant_tool_calls(content: Option<&str>, tool_calls: Vec<PromptToolCall>) -> Self {
        Self {
            role: PromptRole::Assistant,
            content: content.map_or(PromptContent::Empty, |text| {
                PromptContent::Text(text.to_owned())
            }),
            tool_call_id: None,
            tool_calls,
        }
    }

    #[must_use]
    pub fn tool_result(tool_call_id: &str, content: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Tool,
            content: PromptContent::Text(content.into()),
            tool_call_id: Some(tool_call_id.to_owned()),
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryMessage {
    pub role: PromptRole,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievedMessage {
    pub role: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationPromptInput {
    pub locale: Locale,
    pub chat_type: String,
    pub chat_title: String,
    pub first_name: String,
    pub username: String,
    pub formatted_time: String,
    pub message_text: String,
    pub reply_context: Option<String>,
    pub link_context: Option<String>,
    pub enable_web_search: bool,
    pub summary: Option<String>,
    pub history: Vec<HistoryMessage>,
    pub retrieved: Vec<RetrievedMessage>,
}

#[must_use]
pub fn build_system_prompt(
    persona: &str,
    locale: Locale,
    formatted_date: &str,
    tools_active: bool,
    task_mode: bool,
) -> String {
    let task_prefix = if task_mode {
        match locale {
            Locale::Es => {
                "EJECUTANDO TAREA PROGRAMADA:\nRespondé la siguiente instrucción y nada más.\nNo hagas preguntas, no ofrezcas seguimientos, no pidas confirmación.\nGenerá tu respuesta y terminá.\n\n"
            }
            Locale::En => {
                "RUNNING SCHEDULED TASK:\nAnswer the following instruction and nothing else.\nDo not ask questions, offer follow-ups, or request confirmation.\nGenerate the answer and finish.\n\n"
            }
        }
    } else {
        ""
    };
    let tool_instruction = if tools_active {
        match locale {
            Locale::Es => {
                "\n\nHERRAMIENTAS:\nLlamalas directamente, sin pedir permiso ni narrar antes.\nNo expliques qué vas a hacer antes de usar una herramienta simple.\nSi el usuario pide una herramienta disponible por nombre, usala.\nUsá calculate para toda aritmética; no calcules mentalmente.\nUsá las demás herramientas cuando necesites datos actuales o externos.\n"
            }
            Locale::En => {
                "\n\nTOOLS:\nCall them directly without asking permission or narrating first.\nDo not explain what you will do before using a simple tool.\nIf the user requests an available tool by name, use it.\nUse calculate for all arithmetic; do not calculate mentally.\nUse the other tools when you need current or external data.\n"
            }
        }
    } else {
        ""
    };
    let (date_header, language_header, language) = match locale {
        Locale::Es => (
            "FECHA ACTUAL:",
            "IDIOMA DE RESPUESTA:",
            "Respondé en español por defecto. Si el usuario pide explícitamente otro idioma o una traducción, seguí ese pedido.",
        ),
        Locale::En => (
            "CURRENT DATE:",
            "RESPONSE LANGUAGE:",
            "Reply in English by default. If the user explicitly requests another language or a translation, follow that request.",
        ),
    };
    format!(
        "{task_prefix}{persona}\n{tool_instruction}\n\n{date_header}\n{formatted_date}\n\n{language_header}\n{language}\n"
    )
}

#[must_use]
pub fn build_conversation_prompt(input: &ConversationPromptInput) -> Vec<PromptMessage> {
    let mut messages = Vec::new();
    if let Some(summary) = input.summary.as_deref().filter(|value| !value.is_empty()) {
        let header = match input.locale {
            Locale::Es => "RESUMEN ACUMULADO DEL CHAT:",
            Locale::En => "ACCUMULATED CHAT SUMMARY:",
        };
        messages.push(PromptMessage::text(
            PromptRole::System,
            format!("{header}\n{summary}"),
        ));
    }
    let retrieved = input
        .retrieved
        .iter()
        .filter(|message| !message.text.is_empty())
        .collect::<Vec<_>>();
    if !retrieved.is_empty() {
        let header = match input.locale {
            Locale::Es => "MENSAJES ANTERIORES RELEVANTES:",
            Locale::En => "RELEVANT EARLIER MESSAGES:",
        };
        let body = retrieved
            .iter()
            .map(|message| format!("- {}: {}", message.role, message.text))
            .collect::<Vec<_>>()
            .join("\n");
        messages.push(PromptMessage::text(
            PromptRole::System,
            format!("{header}\n{body}"),
        ));
    }
    messages.extend(input.history.iter().map(|message| PromptMessage {
        role: message.role,
        content: PromptContent::TextParts(vec![message.text.clone()]),
        tool_call_id: None,
        tool_calls: Vec::new(),
    }));

    let (
        context_header,
        chat_label,
        user_label,
        time_label,
        anonymous,
        reply_header,
        message_header,
    ) = match input.locale {
        Locale::Es => (
            "CONTEXTO:",
            "Chat",
            "Usuario",
            "Hora",
            "Usuario",
            "MENSAJE AL QUE RESPONDE:",
            "MENSAJE:",
        ),
        Locale::En => (
            "CONTEXT:",
            "Chat",
            "User",
            "Time",
            "User",
            "MESSAGE BEING REPLIED TO:",
            "MESSAGE:",
        ),
    };
    let title = if input.chat_type == "private" || input.chat_title.is_empty() {
        String::new()
    } else {
        format!(" ({})", input.chat_title)
    };
    let first_name = if input.first_name.is_empty() {
        anonymous
    } else {
        &input.first_name
    };
    let username = if input.username.is_empty() {
        String::new()
    } else {
        format!(" ({})", input.username)
    };
    let mut parts = vec![
        context_header.to_owned(),
        format!("- {chat_label}: {}{title}", input.chat_type),
        format!("- {user_label}: {first_name}{username}"),
        format!("- {time_label}: {}", input.formatted_time),
    ];
    let last_is_assistant = messages
        .last()
        .is_some_and(|message| message.role == PromptRole::Assistant);
    if !last_is_assistant
        && let Some(reply) = input
            .reply_context
            .as_deref()
            .filter(|value| !value.is_empty())
    {
        parts.extend([
            String::new(),
            reply_header.to_owned(),
            truncate_text(Some(reply), PROMPT_TEXT_LIMIT),
        ]);
    }
    if let Some(links) = input
        .link_context
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        parts.extend([String::new(), links.to_owned()]);
    }
    let instructions = match input.locale {
        Locale::Es => [
            "INSTRUCCIONES:",
            "- mantené el personaje del gordo",
            "- usá lenguaje coloquial argentino",
            "- respondé en minúsculas, sin emojis, sin punto final",
            "- respondé en una sola frase salvo que sea necesario explicar algo complejo",
        ],
        Locale::En => [
            "INSTRUCTIONS:",
            "- stay in the gordo character",
            "- use casual English",
            "- respond without emojis or a final period",
            "- use one sentence unless a complex explanation needs more",
        ],
    };
    parts.extend([
        String::new(),
        message_header.to_owned(),
        truncate_text(Some(&input.message_text), PROMPT_TEXT_LIMIT),
        String::new(),
    ]);
    parts.extend(instructions.into_iter().map(str::to_owned));
    if input.enable_web_search {
        parts.push(
            match input.locale {
                Locale::Es => "- si no estás seguro de un dato actual, podés buscarlo en internet",
                Locale::En => "- use web search when you are unsure about a current fact",
            }
            .to_owned(),
        );
    }
    messages.push(PromptMessage::text(PromptRole::User, parts.join("\n")));
    messages
}

#[cfg(test)]
mod tests {
    use crate::locale::Locale;

    use super::{
        ConversationPromptInput, HistoryMessage, PromptContent, PromptMessage, PromptRole,
        RetrievedMessage, build_conversation_prompt, build_system_prompt,
    };

    fn input(locale: Locale) -> ConversationPromptInput {
        ConversationPromptInput {
            locale,
            chat_type: "group".to_owned(),
            chat_title: "Synthetic Chat".to_owned(),
            first_name: "Synthetic".to_owned(),
            username: "tester".to_owned(),
            formatted_time: "12:34".to_owned(),
            message_text: "what happened?".to_owned(),
            reply_context: Some("earlier message".to_owned()),
            link_context: Some("LINKS:\n- https://example.test".to_owned()),
            enable_web_search: true,
            summary: Some("prior summary".to_owned()),
            history: vec![HistoryMessage {
                role: PromptRole::User,
                text: "recent history".to_owned(),
            }],
            retrieved: vec![RetrievedMessage {
                role: "assistant".to_owned(),
                text: "older answer".to_owned(),
            }],
        }
    }

    #[test]
    fn system_prompt_matches_bilingual_task_tool_date_and_language_contract() {
        let spanish = build_system_prompt(
            "synthetic persona",
            Locale::Es,
            "Monday 01/01/2024",
            true,
            true,
        );
        assert!(spanish.starts_with("EJECUTANDO TAREA PROGRAMADA:"));
        assert!(spanish.contains("synthetic persona\n\n\nHERRAMIENTAS:"));
        assert!(spanish.contains("Usá calculate para toda aritmética"));
        assert!(spanish.contains("FECHA ACTUAL:\nMonday 01/01/2024"));
        assert!(spanish.contains("Respondé en español por defecto"));

        let english = build_system_prompt(
            "synthetic persona",
            Locale::En,
            "Tuesday 02/01/2024",
            false,
            false,
        );
        assert!(english.starts_with("synthetic persona\n"));
        assert!(!english.contains("TOOLS:"));
        assert!(english.contains("CURRENT DATE:\nTuesday 02/01/2024"));
        assert!(english.contains("Reply in English by default"));
    }

    #[test]
    fn conversation_prompt_preserves_summary_retrieval_history_and_context_order() {
        let messages = build_conversation_prompt(&input(Locale::En));
        assert_eq!(
            messages[..3],
            [
                PromptMessage::text(
                    PromptRole::System,
                    "ACCUMULATED CHAT SUMMARY:\nprior summary"
                ),
                PromptMessage::text(
                    PromptRole::System,
                    "RELEVANT EARLIER MESSAGES:\n- assistant: older answer"
                ),
                PromptMessage {
                    role: PromptRole::User,
                    content: PromptContent::TextParts(vec!["recent history".to_owned()]),
                    tool_call_id: None,
                    tool_calls: Vec::new(),
                }
            ]
        );
        let PromptContent::Text(final_prompt) = &messages[3].content else {
            return;
        };
        assert!(final_prompt.starts_with("CONTEXT:\n- Chat: group (Synthetic Chat)"));
        assert!(final_prompt.contains("- User: Synthetic (tester)\n- Time: 12:34"));
        assert!(final_prompt.contains("MESSAGE BEING REPLIED TO:\nearlier message"));
        assert!(final_prompt.contains("LINKS:\n- https://example.test"));
        assert!(final_prompt.contains("MESSAGE:\nwhat happened?\n\nINSTRUCTIONS:"));
        assert!(
            final_prompt.ends_with("- use web search when you are unsure about a current fact")
        );
    }

    #[test]
    fn assistant_tail_suppresses_duplicate_reply_context_and_private_title() {
        let mut value = input(Locale::Es);
        value.chat_type = "private".to_owned();
        value.history = vec![HistoryMessage {
            role: PromptRole::Assistant,
            text: "last answer".to_owned(),
        }];
        value.summary = None;
        value.retrieved.clear();
        value.link_context = None;
        value.enable_web_search = false;
        let messages = build_conversation_prompt(&value);
        let PromptContent::Text(prompt) = &messages[1].content else {
            return;
        };
        assert!(prompt.contains("- Chat: private\n"));
        assert!(!prompt.contains("Synthetic Chat"));
        assert!(!prompt.contains("MENSAJE AL QUE RESPONDE"));
        assert!(!prompt.contains("buscarlo en internet"));
    }

    #[test]
    fn user_controlled_prompt_fragments_use_the_legacy_unicode_safe_limit() {
        let mut value = input(Locale::En);
        value.message_text = "á".repeat(4_100);
        value.reply_context = Some("β".repeat(4_100));
        let messages = build_conversation_prompt(&value);
        let PromptContent::Text(prompt) = &messages[3].content else {
            return;
        };
        assert!(prompt.contains(&format!("{}...", "á".repeat(4_093))));
        assert!(prompt.contains(&format!("{}...", "β".repeat(4_093))));
    }
}
