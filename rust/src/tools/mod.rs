use html_escape::decode_html_entities;
use regex::Regex;
use serde_json::json;
use url::Url;
use urlencoding::decode;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: Option<String>,
}

pub fn parse_tool_call(text: &str) -> Option<(String, serde_json::Value)> {
    let line = text.lines().find(|line| line.trim_start().starts_with("[TOOL]"))?;
    let trimmed = line.trim();
    let rest = trimmed.strip_prefix("[TOOL]")?.trim();
    let mut parts = rest.splitn(2, ' ');
    let name = parts.next()?.trim().to_string();
    let args_raw = parts.next().unwrap_or("{}").trim();
    let args = serde_json::from_str::<serde_json::Value>(args_raw).ok()?;
    Some((name, args))
}

pub async fn execute_tool(
    http: &reqwest::Client,
    name: &str,
    args: &serde_json::Value,
) -> String {
    match name {
        "web_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as usize;
            let results = web_search(http, query, limit).await;
            let payload = json!({
                "query": query,
                "results": results.iter().map(|r| {
                    json!({
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                    })
                }).collect::<Vec<_>>()
            });
            payload.to_string()
        }
        "fetch_url" => {
            let url = args.get("url").and_then(|v| v.as_str()).unwrap_or("");
            let payload = fetch_url(http, url).await;
            payload.to_string()
        }
        _ => json!({"error": "unknown tool"}).to_string(),
    }
}

pub async fn web_search(http: &reqwest::Client, query: &str, limit: usize) -> Vec<SearchResult> {
    if query.trim().is_empty() {
        return vec![];
    }
    let encoded = urlencoding::encode(query);
    let url = format!("https://duckduckgo.com/html/?q={}", encoded);

    let response = match http.get(url).send().await {
        Ok(resp) => resp,
        Err(_) => return vec![],
    };

    let body = match response.text().await {
        Ok(text) => text,
        Err(_) => return vec![],
    };

    let link_re = Regex::new(r#"<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>"#).ok();
    let snippet_re = Regex::new(r#"<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>"#).ok();

    let mut results = Vec::new();

    if let Some(link_re) = link_re {
        for caps in link_re.captures_iter(&body) {
            let raw_url = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let mut final_url = raw_url.to_string();
            if raw_url.contains("duckduckgo.com/l/?") {
                if let Ok(parsed) = Url::parse(raw_url) {
                    if let Some(uddg) = parsed.query_pairs().find(|(k, _)| k == "uddg") {
                        if let Ok(decoded) = decode(uddg.1.as_ref()) {
                            final_url = decoded.to_string();
                        } else {
                            final_url = uddg.1.to_string();
                        }
                    }
                }
            }
            let title = decode_html_entities(caps.get(2).map(|m| m.as_str()).unwrap_or(""))
                .to_string();

            let snippet = snippet_re
                .as_ref()
                .and_then(|re| re.captures_iter(&body).next())
                .and_then(|caps| caps.get(1))
                .map(|m| decode_html_entities(m.as_str()).to_string());

            results.push(SearchResult {
                title,
                url: final_url,
                snippet,
            });

            if results.len() >= limit {
                break;
            }
        }
    }

    results
}

pub fn format_search_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        if query.trim().is_empty() {
            return "No encontré resultados.".to_string();
        }
        return format!("No encontré resultados para: {}", query);
    }

    let mut lines = Vec::new();
    if !query.trim().is_empty() {
        lines.push(format!("Resultados para: {}", query));
        lines.push("".to_string());
    }
    for (idx, result) in results.iter().enumerate() {
        lines.push(format!("{}. {}", idx + 1, result.title));
        lines.push(result.url.clone());
        if let Some(snippet) = &result.snippet {
            if !snippet.is_empty() {
                lines.push(snippet.clone());
            }
        }
        lines.push("".to_string());
    }
    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

pub async fn fetch_url(http: &reqwest::Client, url: &str) -> serde_json::Value {
    if url.trim().is_empty() {
        return json!({"url": url, "error": "missing url"});
    }

    let response = match http.get(url).send().await {
        Ok(resp) => resp,
        Err(err) => {
            return json!({"url": url, "error": err.to_string()});
        }
    };

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => {
            return json!({"url": url, "error": err.to_string()});
        }
    };

    let mut text = String::new();
    if content_type.contains("text/") || content_type.contains("html") {
        text = String::from_utf8_lossy(&bytes).to_string();
    }

    let title = extract_title(&text);
    let content = strip_html(&text);
    let truncated = content.len() > 4000;
    let content = if truncated { content[..4000].to_string() } else { content };

    json!({
        "url": url,
        "title": title,
        "content": content,
        "truncated": truncated,
    })
}

fn strip_html(input: &str) -> String {
    let script_re = Regex::new(r"(?is)<script.*?>.*?</script>").ok();
    let style_re = Regex::new(r"(?is)<style.*?>.*?</style>").ok();
    let tag_re = Regex::new(r"(?is)<[^>]+>").ok();
    let mut text = input.to_string();
    if let Some(re) = script_re {
        text = re.replace_all(&text, " ").to_string();
    }
    if let Some(re) = style_re {
        text = re.replace_all(&text, " ").to_string();
    }
    if let Some(re) = tag_re {
        text = re.replace_all(&text, " ").to_string();
    }
    decode_html_entities(&text).to_string()
}

fn extract_title(input: &str) -> String {
    let title_re = Regex::new(r"(?is)<title>(.*?)</title>").ok();
    if let Some(re) = title_re {
        if let Some(cap) = re.captures(input) {
            return decode_html_entities(cap.get(1).map(|m| m.as_str()).unwrap_or(""))
                .to_string();
        }
    }
    "".to_string()
}
