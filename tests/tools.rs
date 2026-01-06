use wasm_bindgen_test::wasm_bindgen_test;
use respondedorbot::tools::{
    extract_title, format_search_results, parse_cached_results, parse_tool_call, strip_html,
    SearchResult,
};

#[wasm_bindgen_test]
fn test_extract_title() {
    let html = "<html><head><title>Hola</title></head><body>ok</body></html>";
    let title = extract_title(html);
    assert_eq!(title, "Hola");
}

#[wasm_bindgen_test]
fn test_strip_html() {
    let html = "<p>hola</p><script>bad</script><style>nope</style>";
    let cleaned = strip_html(html);
    assert!(cleaned.contains("hola"));
    assert!(!cleaned.contains("bad"));
}

#[wasm_bindgen_test]
fn test_parse_tool_call() {
    let input = "[TOOL] web_search {\"query\":\"hola\"}";
    let parsed = parse_tool_call(input).unwrap();
    assert_eq!(parsed.0, "web_search");
    assert_eq!(parsed.1.get("query").and_then(|v| v.as_str()), Some("hola"));
}

#[wasm_bindgen_test]
fn test_format_search_results_with_snippet() {
    let results = vec![SearchResult {
        title: "Titulo".to_string(),
        url: "https://example.com".to_string(),
        snippet: Some("Snippet".to_string()),
    }];
    let formatted = format_search_results("consulta", &results);
    assert!(formatted.contains("Resultados para: consulta"));
    assert!(formatted.contains("Titulo"));
    assert!(formatted.contains("https://example.com"));
    assert!(formatted.contains("Snippet"));
}

#[wasm_bindgen_test]
fn test_parse_cached_results() {
    let value = serde_json::json!({
        "results": [
            {"title": "Titulo", "url": "https://example.com", "snippet": "Snippet"}
        ]
    });
    let parsed = parse_cached_results(&value).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].title, "Titulo");
}
