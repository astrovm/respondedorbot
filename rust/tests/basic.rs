use respondedorbot::{commands, market, tools};

#[test]
fn test_parse_command_basic() {
    let result = commands::parse_command("/help", None);
    assert!(result.is_some());
    let (cmd, args) = result.unwrap();
    assert_eq!(cmd, "/help");
    assert_eq!(args, "");
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
fn test_format_search_results_empty() {
    let results: Vec<tools::SearchResult> = vec![];
    let formatted = tools::format_search_results("consulta", &results);
    assert!(formatted.contains("No encontré"));
}
