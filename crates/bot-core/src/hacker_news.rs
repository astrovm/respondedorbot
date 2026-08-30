//! Hacker News feed normalization and deterministic list formatting.

use regex::Regex;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HackerNewsItem {
    pub title: String,
    pub url: String,
    pub points: Option<i64>,
    pub comments: Option<i64>,
    pub comments_url: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HackerNewsRenderItem {
    pub title: String,
    pub url: String,
    pub points: Option<i64>,
    pub comments: Option<i64>,
    pub comments_url: String,
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum FeedItemError {
    #[error("feed metadata integer is outside the supported range")]
    IntegerRange,
}

fn extract_integer(pattern: &str, text: &str) -> Result<Option<i64>, FeedItemError> {
    let Some(captures) = Regex::new(pattern)
        .ok()
        .and_then(|regex| regex.captures(text))
    else {
        return Ok(None);
    };
    let Some(value) = captures.get(1) else {
        return Ok(None);
    };
    value
        .as_str()
        .parse::<i64>()
        .map(Some)
        .map_err(|_| FeedItemError::IntegerRange)
}

fn extract_text(pattern: &str, text: &str) -> String {
    Regex::new(pattern)
        .ok()
        .and_then(|regex| regex.captures(text))
        .and_then(|captures| {
            captures
                .get(1)
                .map(|value| value.as_str().trim().to_owned())
        })
        .unwrap_or_default()
}

/// Normalize one XML-adapter item and extract the metadata embedded in its
/// description. The adapter decodes HTML entities before calling this function.
pub fn normalize_feed_item(
    title: &str,
    url: &str,
    description: &str,
) -> Result<Option<HackerNewsItem>, FeedItemError> {
    let title = title.trim();
    if title.is_empty() {
        return Ok(None);
    }
    Ok(Some(HackerNewsItem {
        title: title.to_owned(),
        url: url.trim().to_owned(),
        points: extract_integer(r"Points:\s*(\d+)", description)?,
        comments: extract_integer(r"# Comments:\s*(\d+)", description)?,
        comments_url: extract_text(r#"Comments URL: <a href="([^"]+)""#, description),
    }))
}

/// Format normalized items for the AI context and Hacker News tool.
#[must_use]
pub fn format_items(
    items: &[HackerNewsRenderItem],
    include_discussion: bool,
    no_data: &str,
    comments_label: &str,
) -> String {
    if items.is_empty() {
        return format!("- {no_data}");
    }

    let lines = items
        .iter()
        .map(|item| {
            let mut stats = Vec::new();
            if let Some(points) = item.points {
                stats.push(format!("{points} pts"));
            }
            if let Some(comments) = item.comments {
                stats.push(format!("{comments} {comments_label}"));
            }
            let stats = if stats.is_empty() {
                String::new()
            } else {
                format!(" ({})", stats.join(", "))
            };
            let mut line = format!("- {}{stats}", item.title);
            if !item.url.is_empty() {
                line.push_str(" → ");
                line.push_str(&item.url);
            }
            if include_discussion && !item.comments_url.is_empty() {
                line.push_str(" (HN: ");
                line.push_str(&item.comments_url);
                line.push(')');
            }
            line
        })
        .collect::<Vec<_>>();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{
        FeedItemError, HackerNewsItem, HackerNewsRenderItem, format_items, normalize_feed_item,
    };

    #[test]
    fn normalizes_feed_metadata_and_ignores_blank_titles() {
        assert_eq!(
            normalize_feed_item(
                "  Synthetic story  ",
                " https://example.test/story ",
                r#"Points: 123<br># Comments: 45<br>Comments URL: <a href="https://news.ycombinator.com/item?id=1">comments</a>"#,
            ),
            Ok(Some(HackerNewsItem {
                title: "Synthetic story".to_owned(),
                url: "https://example.test/story".to_owned(),
                points: Some(123),
                comments: Some(45),
                comments_url: "https://news.ycombinator.com/item?id=1".to_owned(),
            }))
        );
        assert_eq!(normalize_feed_item("  ", "url", "Points: 1"), Ok(None));
    }

    #[test]
    fn reports_metadata_that_cannot_fit_the_typed_boundary() {
        assert_eq!(
            normalize_feed_item("story", "", "Points: 999999999999999999999999"),
            Err(FeedItemError::IntegerRange)
        );
    }

    #[test]
    fn formats_optional_stats_links_and_discussions() {
        let items = [HackerNewsRenderItem {
            title: "Synthetic story".to_owned(),
            url: "https://example.test/story".to_owned(),
            points: Some(99),
            comments: Some(12),
            comments_url: "https://news.ycombinator.com/item?id=2".to_owned(),
        }];
        assert_eq!(
            format_items(&items, true, "no data", "comments"),
            "- Synthetic story (99 pts, 12 comments) → https://example.test/story (HN: https://news.ycombinator.com/item?id=2)"
        );
        assert_eq!(
            format_items(&items, false, "no data", "comments"),
            "- Synthetic story (99 pts, 12 comments) → https://example.test/story"
        );
        assert_eq!(format_items(&[], true, "no data", "comments"), "- no data");
    }
}
