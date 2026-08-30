//! Provider-emitted pseudo tool-call parsing.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PseudoToolCall {
    pub id: String,
    pub name: String,
    pub url: String,
}

#[must_use]
pub fn parse_pseudo_web_fetch(
    text: &str,
    round_index: usize,
    advertised_tool_names: &[String],
    web_fetch_registered: bool,
) -> Option<PseudoToolCall> {
    if !web_fetch_registered || !advertised_tool_names.iter().any(|name| name == "web_fetch") {
        return None;
    }
    let (name, url) = parse_dsml_call(text).or_else(|| parse_last_line_call(text))?;
    if name != "web_fetch" || !is_http_url(&url) {
        return None;
    }
    Some(PseudoToolCall {
        id: format!("pseudo_call_{}", round_index.saturating_add(1)),
        name,
        url,
    })
}

fn parse_dsml_call(text: &str) -> Option<(String, String)> {
    const INVOKE_PREFIX: &str = "<｜｜DSML｜｜invoke";
    const NAME_ATTRIBUTE: &str = "name=\"";
    const URL_PARAMETER_PREFIX: &str = "<｜｜DSML｜｜parameter name=\"url\" string=\"true\">";
    const URL_PARAMETER_END: &str = "</｜｜DSML｜｜parameter>";
    const INVOKE_END: &str = "</｜｜DSML｜｜invoke>";

    let invoke_start = text.find(INVOKE_PREFIX)?;
    let invoke = &text[invoke_start + INVOKE_PREFIX.len()..];
    let name_start = invoke.find(NAME_ATTRIBUTE)? + NAME_ATTRIBUTE.len();
    let name_tail = &invoke[name_start..];
    let name_end = name_tail.find('"')?;
    let name = &name_tail[..name_end];
    if !is_identifier(name) {
        return None;
    }
    let parameter_start = invoke.find(URL_PARAMETER_PREFIX)? + URL_PARAMETER_PREFIX.len();
    let parameter_tail = &invoke[parameter_start..];
    let parameter_end = parameter_tail.find(URL_PARAMETER_END)?;
    let url = &parameter_tail[..parameter_end];
    if url.is_empty() || url.chars().any(char::is_whitespace) {
        return None;
    }
    let after_parameter = &parameter_tail[parameter_end + URL_PARAMETER_END.len()..];
    if !after_parameter.trim_start().starts_with(INVOKE_END) {
        return None;
    }
    Some((name.to_owned(), url.to_owned()))
}

fn parse_last_line_call(text: &str) -> Option<(String, String)> {
    let candidate = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())?
        .trim();
    let opening = candidate.find('(')?;
    if !candidate.ends_with(')') {
        return None;
    }
    let name = candidate[..opening].trim();
    if !is_identifier(name) {
        return None;
    }
    let arguments = candidate[opening + 1..candidate.len() - 1].trim();
    let url = parse_url_argument(arguments)?;
    Some((name.to_owned(), url))
}

fn parse_url_argument(arguments: &str) -> Option<String> {
    if arguments.starts_with('"') {
        return serde_json::from_str::<String>(arguments).ok();
    }
    if arguments.starts_with('\'') {
        return parse_single_quoted_string(arguments);
    }
    let parameters: Value = serde_json::from_str(arguments).ok()?;
    parameters
        .as_object()?
        .get("url")?
        .as_str()
        .map(str::to_owned)
}

fn parse_single_quoted_string(value: &str) -> Option<String> {
    if value.len() < 2 || !value.ends_with('\'') {
        return None;
    }
    let mut result = String::new();
    let mut characters = value[1..value.len() - 1].chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            result.push(character);
            continue;
        }
        let escaped = characters.next()?;
        match escaped {
            '\\' => result.push('\\'),
            '\'' => result.push('\''),
            '"' => result.push('"'),
            'n' => result.push('\n'),
            'r' => result.push('\r'),
            't' => result.push('\t'),
            _ => return None,
        }
    }
    Some(result)
}

fn is_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn is_http_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

#[cfg(test)]
mod tests {
    use super::{PseudoToolCall, parse_pseudo_web_fetch};

    fn advertised() -> Vec<String> {
        vec!["web_fetch".to_owned()]
    }

    #[test]
    fn parses_standalone_last_line_json_and_single_quote_calls() {
        for (text, expected_url) in [
            (
                "web_fetch(\"https://example.com/a\")",
                "https://example.com/a",
            ),
            ("checking now\nweb_fetch({'url': 'invalid'})", ""),
            (
                "checking now\nweb_fetch({\"url\":\"https://example.com/b\"})",
                "https://example.com/b",
            ),
            (
                "web_fetch('https://example.com/c')",
                "https://example.com/c",
            ),
        ] {
            let actual = parse_pseudo_web_fetch(text, 2, &advertised(), true);
            if expected_url.is_empty() {
                assert_eq!(actual, None);
            } else {
                assert_eq!(
                    actual,
                    Some(PseudoToolCall {
                        id: "pseudo_call_3".to_owned(),
                        name: "web_fetch".to_owned(),
                        url: expected_url.to_owned(),
                    })
                );
            }
        }
    }

    #[test]
    fn parses_dsml_and_rejects_prose_unknown_unadvertised_or_unregistered_calls() {
        let dsml = concat!(
            "<｜｜DSML｜｜tool_calls>\n",
            "<｜｜DSML｜｜invoke name=\"web_fetch\">\n",
            "<｜｜DSML｜｜parameter name=\"url\" string=\"true\">",
            "https://example.com/post",
            "</｜｜DSML｜｜parameter>\n",
            "</｜｜DSML｜｜invoke>\n",
            "</｜｜DSML｜｜tool_calls>"
        );
        assert_eq!(
            parse_pseudo_web_fetch(dsml, 0, &advertised(), true),
            Some(PseudoToolCall {
                id: "pseudo_call_1".to_owned(),
                name: "web_fetch".to_owned(),
                url: "https://example.com/post".to_owned(),
            })
        );
        for (text, tools, registered) in [
            (
                "I might call web_fetch(\"https://example.com\") later",
                advertised(),
                true,
            ),
            ("calculate(\"https://example.com\")", advertised(), true),
            ("web_fetch(\"ftp://example.com\")", advertised(), true),
            (
                "web_fetch(\"https://example.com\")",
                vec!["calculate".to_owned()],
                true,
            ),
            ("web_fetch(\"https://example.com\")", advertised(), false),
        ] {
            assert_eq!(parse_pseudo_web_fetch(text, 0, &tools, registered), None);
        }
    }
}
