use serde_json::json;
use std::time::{Duration, Instant};

use crate::tools;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct AiContext {
    pub system_prompt: String,
    pub time_label: String,
    pub market_info: Option<String>,
    pub weather_info: Option<String>,
    pub hn_info: Option<String>,
    pub agent_memory: Option<String>,
}

pub async fn ask_ai(
    http: &reqwest::Client,
    redis: &redis::Client,
    context: &AiContext,
    messages: Vec<ChatMessage>,
) -> Option<String> {
    let system_message = build_system_message(context);
    let mut conversation = messages;

    let mut attempts = 0;
    let mut current = complete_with_providers(http, &system_message, &conversation).await?;

    loop {
        if let Some((tool_name, tool_args)) = tools::parse_tool_call(&current) {
            let tool_output = tools::execute_tool(http, redis, &tool_name, &tool_args).await;
            let tool_context = json!({
                "tool": tool_name,
                "args": tool_args,
                "result": tool_output,
            });

            conversation.push(ChatMessage {
                role: "assistant".to_string(),
                content: sanitize_tool_artifacts(&current),
            });
            conversation.push(ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "RESULTADO DE HERRAMIENTA:\n{}",
                    tool_context.to_string().chars().take(4000).collect::<String>()
                ),
            });

            attempts += 1;
            if attempts >= 3 {
                return Some(format!("Resultado de herramienta:\n{}", tool_output));
            }

            if let Some(next) = complete_with_providers(http, &system_message, &conversation).await
            {
                current = next;
                continue;
            }
            return Some(format!("Resultado de herramienta:\n{}", tool_output));
        }
        return Some(current);
    }
}

fn build_system_message(context: &AiContext) -> String {
    let mut blocks = Vec::new();
    blocks.push(context.system_prompt.clone());
    blocks.push(format!("FECHA ACTUAL:\n{}", context.time_label));
    if let Some(market) = &context.market_info {
        blocks.push(format!("CONTEXTO DEL MERCADO:\n{}", market));
    }
    if let Some(weather) = &context.weather_info {
        blocks.push(format!("CLIMA EN BUENOS AIRES:\n{}", weather));
    }
    if let Some(hn) = &context.hn_info {
        blocks.push(format!("NOTICIAS DE HACKER NEWS:\n{}", hn));
    }
    blocks.push("CONTEXTO POLITICO:\n- Javier Milei (alias miller, javo, javito, javeto) es el presidente de Argentina desde el 10/12/2023 hasta el 10/12/2027".to_string());
    if let Some(memory) = &context.agent_memory {
        blocks.push(memory.clone());
    }
    blocks.push("HERRAMIENTAS DISPONIBLES:\n- web_search: buscador web actual (devuelve hasta 10 resultados).\n- fetch_url: trae el texto plano de una URL http/https para citar fragmentos.\n\nCÓMO LLAMAR HERRAMIENTAS:\n[TOOL] <nombre> {JSON}\nEjemplos:\n  [TOOL] web_search {\"query\": \"inflación argentina hoy\"}\n  [TOOL] fetch_url {\"url\": \"https://example.com/noticia\"}\n".to_string());

    blocks.join("\n\n")
}

async fn complete_with_providers(
    http: &reqwest::Client,
    system_prompt: &str,
    messages: &[ChatMessage],
) -> Option<String> {
    let start = Instant::now();
    if let Some(result) = openrouter_request(http, system_prompt, messages).await {
        return Some(result);
    }
    if let Some(result) = groq_request(http, system_prompt, messages).await {
        return Some(result);
    }
    if let Some(result) = cloudflare_request(http, system_prompt, messages).await {
        return Some(result);
    }
    if start.elapsed() > Duration::from_secs(10) {
        return None;
    }
    None
}

async fn openrouter_request(
    http: &reqwest::Client,
    system_prompt: &str,
    messages: &[ChatMessage],
) -> Option<String> {
    let api_key = std::env::var("OPENROUTER_API_KEY").ok()?;
    let url = "https://openrouter.ai/api/v1/chat/completions";

    let models = [
        "moonshotai/kimi-k2:free",
        "x-ai/grok-4-fast:free",
        "deepseek/deepseek-chat-v3.1:free",
    ];

    let payload = json!({
        "model": models[0],
        "messages": build_messages(system_prompt, messages),
        "max_tokens": 1024,
        "extra_body": {"models": models[1..].to_vec()},
    });

    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .ok()?;
    let body = resp.json::<serde_json::Value>().await.ok()?;
    body.get("choices")
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

async fn groq_request(
    http: &reqwest::Client,
    system_prompt: &str,
    messages: &[ChatMessage],
) -> Option<String> {
    let api_key = std::env::var("GROQ_API_KEY").ok()?;
    let url = "https://api.groq.com/openai/v1/chat/completions";

    let payload = json!({
        "model": "moonshotai/kimi-k2-instruct-0905",
        "messages": build_messages(system_prompt, messages),
        "max_tokens": 1024,
    });

    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .ok()?;
    let body = resp.json::<serde_json::Value>().await.ok()?;
    body.get("choices")
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

async fn cloudflare_request(
    http: &reqwest::Client,
    system_prompt: &str,
    messages: &[ChatMessage],
) -> Option<String> {
    let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").ok()?;
    let api_key = std::env::var("CLOUDFLARE_API_KEY").ok()?;
    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/mistralai/mistral-small-3.1-24b-instruct"
    );

    let payload = json!({
        "messages": build_messages(system_prompt, messages),
    });

    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .ok()?;
    let body = resp.json::<serde_json::Value>().await.ok()?;
    body.get("result")
        .and_then(|v| v.get("response"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn build_messages(system_prompt: &str, messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(json!({"role": "system", "content": system_prompt}));
    for msg in messages {
        out.push(json!({"role": msg.role, "content": msg.content}));
    }
    out
}

fn sanitize_tool_artifacts(text: &str) -> String {
    let mut output = text.to_string();
    if let Some(pos) = output.find("[TOOL]") {
        output.truncate(pos);
    }
    output.trim().to_string()
}
