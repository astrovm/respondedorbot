use respondedorbot::{commands, links, market, tools};
use futures::executor::block_on;

#[test]
fn test_parse_command_basic() {
    let result = commands::parse_command("/help", None);
    assert!(result.is_some());
    let (cmd, args) = result.unwrap();
    assert_eq!(cmd, "/help");
    assert_eq!(args, "");
}

#[test]
fn test_parse_command_with_username_match() {
    let result = commands::parse_command("/help@mybot", Some("mybot"));
    let (cmd, args) = result.expect("command should be parsed for matching bot");
    assert_eq!(cmd, "/help");
    assert_eq!(args, "");
}

#[test]
fn test_parse_command_ignores_other_bot() {
    let result = commands::parse_command("/help@otherbot", Some("mybot"));
    assert!(result.is_none());
}

#[test]
fn test_help_text_lists_main_features() {
    let text = commands::help_text();
    for cmd in ["/transcribe", "/agent", "/eleccion", "/prices", "/random", "/powerlaw", "/rainbow", "/convertbase"] {
        assert!(
            text.contains(cmd),
            "help text should mention {cmd}"
        );
    }
}

#[test]
fn test_convert_base_ok() {
    let result = market::convert_base("101, 2, 10");
    assert!(result.contains("101"));
    assert!(result.contains("5"));
}

#[test]
fn test_convert_base_invalid() {
    let result = market::convert_base("101");
    assert!(result.contains("/convertbase"));
}

#[test]
fn test_select_random_range() {
    let result = commands::select_random("1-3");
    let value: i64 = result.parse().unwrap();
    assert!((1..=3).contains(&value));
}

#[test]
fn test_select_random_invalid_hint() {
    let result = commands::select_random("   ");
    assert!(result.contains("mandate algo"));
}

#[test]
fn test_convert_to_command_basic() {
    let result = commands::convert_to_command("hola ñandú ñ");
    assert_eq!(result, "/HOLA_NIANDU_ENIE");
}

#[test]
fn test_convert_to_command_emoji() {
    let result = commands::convert_to_command("😄hello 😄 world");
    assert!(result.contains("CARA_SONRIENDO_CON_OJOS_SONRIENTES"));
}

#[test]
fn test_format_search_results_empty() {
    let results: Vec<tools::SearchResult> = vec![];
    let formatted = tools::format_search_results("consulta", &results);
    assert!(formatted.contains("No encontré"));
}

#[test]
fn test_replace_links_twitter() {
    let text = "mirá esto https://twitter.com/user/status/123";
    let (out, changed, _) =
        block_on(links::replace_links_with_checker(text, |_| async { true }));
    assert!(changed);
    assert!(out.contains("fxtwitter.com"));
}

#[test]
fn test_replace_links_profile_ignored() {
    let text = "perfil https://twitter.com/usuario";
    let (out, changed, _) =
        block_on(links::replace_links_with_checker(text, |_| async { true }));
    assert!(!changed);
    assert!(out.contains("twitter.com/usuario"));
}
