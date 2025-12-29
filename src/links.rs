use regex::Regex;
use std::time::Duration;
use url::{ParseError, Url};

use crate::http::{self, HttpClient, CONTENT_TYPE};

const ALTERNATIVE_FRONTENDS: &[&str] = &[
    "fxtwitter.com",
    "fixupx.com",
    "fxbsky.app",
    "kkinstagram.com",
    "rxddit.com",
    "vxtiktok.com",
];

const ORIGINAL_FRONTENDS: &[&str] = &[
    "twitter.com",
    "x.com",
    "xcancel.com",
    "bsky.app",
    "instagram.com",
    "reddit.com",
    "tiktok.com",
];

pub async fn replace_links(http: &HttpClient, text: &str) -> (String, bool, Vec<String>) {
    let wrapped = |url: String| {
        let http = http.clone();
        async move { can_embed_url(&http, &url).await }
    };
    replace_links_with_checker(text, wrapped).await
}

pub async fn replace_links_with_checker<F, Fut>(
    text: &str,
    checker: F,
) -> (String, bool, Vec<String>)
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let patterns = [
        (
            r"(https?://)(?:www\.)?twitter\.com([^\s]*)",
            r"${1}fxtwitter.com${2}",
        ),
        (
            r"(https?://)(?:www\.)?x\.com([^\s]*)",
            r"${1}fixupx.com${2}",
        ),
        (
            r"(https?://)(?:www\.)?xcancel\.com([^\s]*)",
            r"${1}fixupx.com${2}",
        ),
        (
            r"(https?://)(?:www\.)?bsky\.app([^\s]*)",
            r"${1}fxbsky.app${2}",
        ),
        (
            r"(https?://)(?:www\.)?instagram\.com([^\s]*)",
            r"${1}kkinstagram.com${2}",
        ),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)reddit\.com([^\s]*)",
            r"${1}${2}rxddit.com${3}",
        ),
        (
            r"(https?://)((?:[a-zA-Z0-9-]+\.)?)tiktok\.com([^\s]*)",
            r"${1}${2}vxtiktok.com${3}",
        ),
    ];

    let mut changed = false;
    let mut original_links = Vec::new();
    let mut new_text = text.to_string();

    for (pattern, replacement) in patterns.iter() {
        let re = Regex::new(pattern).unwrap();
        let mut out = String::new();
        let mut last = 0;
        for m in re.find_iter(&new_text) {
            out.push_str(&new_text[last..m.start()]);
            let original = m.as_str();
            let replaced = re.replace(original, *replacement).to_string();
            let replaced = strip_tracking(&replaced);

            if is_twitter_user_profile(original) {
                out.push_str(original);
                last = m.end();
                continue;
            }

            if checker(replaced.clone()).await {
                changed = true;
                original_links.push(strip_tracking(original));
                out.push_str(&replaced);
            } else {
                out.push_str(original);
            }
            last = m.end();
        }
        out.push_str(&new_text[last..]);
        new_text = out;
    }

    let url_re = Regex::new(r"(https?://[^\s]+)").unwrap();
    let mut cleaned = String::new();
    let mut last = 0;
    for m in url_re.find_iter(&new_text) {
        cleaned.push_str(&new_text[last..m.start()]);
        let url = strip_tracking(m.as_str());
        cleaned.push_str(&url);
        last = m.end();
    }
    cleaned.push_str(&new_text[last..]);

    (cleaned, changed, original_links)
}

pub fn is_social_frontend(host: &str) -> bool {
    let host = host.to_lowercase();
    ALTERNATIVE_FRONTENDS
        .iter()
        .chain(ORIGINAL_FRONTENDS.iter())
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

async fn can_embed_url(http: &HttpClient, url: &str) -> bool {
    let response = match http.get(url).timeout(Duration::from_secs(5)).send().await {
        Ok(resp) => resp,
        Err(_) => match http::get_with_ssl_fallback(http, url).await {
            Ok(resp) => resp,
            Err(_) => return false,
        },
    };

    let content_type = response
        .header(CONTENT_TYPE)
        .unwrap_or_default()
        .to_lowercase();

    if content_type.starts_with("image/")
        || content_type.starts_with("video/")
        || content_type.starts_with("audio/")
    {
        return true;
    }

    if !content_type.contains("text/html") {
        return false;
    }

    let body = match response.text().await {
        Ok(text) => text,
        Err(_) => return false,
    };

    let sample = &body[..body.len().min(20000)];
    let meta_re = Regex::new(
        r#"(?is)<meta[^>]+(?:property|name)=['"]?(og:[^'"\s>]+|twitter:[^'"\s>]+)['"]?[^>]*>"#,
    )
    .unwrap();
    meta_re.is_match(sample)
}

pub fn strip_tracking(url: &str) -> String {
    if let Ok(parsed) = Url::parse(url) {
        if is_social_frontend(parsed.host_str().unwrap_or("")) {
            let mut clean = parsed.clone();
            clean.set_query(None);
            clean.set_fragment(None);
            return clean.to_string();
        }
    }
    url.to_string()
}

fn is_twitter_user_profile(url: &str) -> bool {
    let parsed = match Url::parse(url) {
        Ok(parsed) => parsed,
        Err(ParseError::RelativeUrlWithoutBase) => return false,
        Err(_) => return false,
    };
    let host = parsed.host_str().unwrap_or("").trim_start_matches("www.");
    if !matches!(host, "twitter.com" | "x.com" | "xcancel.com") {
        return false;
    }
    let path = parsed.path().trim_matches('/');
    if path.is_empty() {
        return false;
    }
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return false;
    }
    if segments.contains(&"status") {
        return false;
    }
    let reserved = [
        "home",
        "share",
        "intent",
        "i",
        "search",
        "explore",
        "notifications",
        "messages",
        "settings",
        "compose",
        "privacy",
        "tos",
    ];
    let first = segments[0].trim_start_matches('@');
    if reserved.contains(&first) {
        return false;
    }
    if segments.len() == 1 {
        return true;
    }
    let profile_subpages = ["with_replies", "media", "likes"];
    segments.len() == 2 && profile_subpages.contains(&segments[1])
}
