use respondedorbot::ai::{build_system_message, sanitize_tool_artifacts, AiContext};

#[test]
fn test_build_system_message_includes_context() {
    let ctx = AiContext {
        system_prompt: "Prompt".to_string(),
        time_label: "Hoy".to_string(),
        market_info: Some("BTC".to_string()),
        weather_info: Some("Soleado".to_string()),
        hn_info: Some("HN".to_string()),
        agent_memory: Some("MEM".to_string()),
    };
    let msg = build_system_message(&ctx);
    assert!(msg.contains("Prompt"));
    assert!(msg.contains("FECHA ACTUAL"));
    assert!(msg.contains("CONTEXTO DEL MERCADO"));
    assert!(msg.contains("CLIMA EN BUENOS AIRES"));
    assert!(msg.contains("NOTICIAS DE HACKER NEWS"));
    assert!(msg.contains("MEM"));
    assert!(msg.contains("HERRAMIENTAS DISPONIBLES"));
}

#[test]
fn test_sanitize_tool_artifacts() {
    let raw = "hola\n[TOOL] web_search {}";
    let cleaned = sanitize_tool_artifacts(raw);
    assert_eq!(cleaned, "hola");
}
