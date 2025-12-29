use regex::Regex;

use crate::http::HttpClient;

const HN_RSS_URL: &str = "https://hnrss.org/best";

#[derive(Debug, Clone)]
pub struct HnItem {
    pub title: String,
    pub link: String,
}

pub async fn get_hn_context(http: &HttpClient) -> Vec<HnItem> {
    let response = match http.get(HN_RSS_URL).send().await {
        Ok(resp) => resp,
        Err(_) => return vec![],
    };
    let body = match response.text().await {
        Ok(text) => text,
        Err(_) => return vec![],
    };

    let item_re = Regex::new(r"(?is)<item>(.*?)</item>").ok();
    let title_re = Regex::new(r"(?is)<title>(.*?)</title>").ok();
    let link_re = Regex::new(r"(?is)<link>(.*?)</link>").ok();

    let mut items = Vec::new();
    if let Some(item_re) = item_re {
        for cap in item_re.captures_iter(&body) {
            let item = cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let title = title_re
                .as_ref()
                .and_then(|re| re.captures(item))
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            let link = link_re
                .as_ref()
                .and_then(|re| re.captures(item))
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            if !title.is_empty() && !link.is_empty() {
                items.push(HnItem { title, link });
            }
            if items.len() >= 5 {
                break;
            }
        }
    }

    items
}

pub fn format_hn_items(items: &[HnItem]) -> String {
    if items.is_empty() {
        return "Sin noticias relevantes ahora".to_string();
    }
    let mut lines = Vec::new();
    for item in items {
        lines.push(format!("- {} ({})", item.title, item.link));
    }
    lines.join("\n")
}
