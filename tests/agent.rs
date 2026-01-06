use wasm_bindgen_test::wasm_bindgen_test;
use rand::rngs::StdRng;
use rand::SeedableRng;
use respondedorbot::agent::{
    agent_sections_are_valid, build_agent_fallback_entry, build_agent_retry_prompt_with_rng,
    extract_agent_section_content, find_repetitive_recent_thought, normalize_agent_text,
};

#[wasm_bindgen_test]
fn test_normalize_agent_text_strips_accents() {
    let raw = "Próximo PASO: dólar!";
    let normalized = normalize_agent_text(raw);
    assert_eq!(normalized, "proximo paso dolar");
}

#[wasm_bindgen_test]
fn test_extract_agent_section_content() {
    let text = "HALLAZGOS: uno\ndos\nPRÓXIMO PASO: siguiente";
    let hallazgos = extract_agent_section_content(text, "HALLAZGOS", &["PRÓXIMO PASO"]);
    let paso = extract_agent_section_content(text, "PRÓXIMO PASO", &["HALLAZGOS"]);
    assert_eq!(hallazgos.unwrap(), "uno\ndos");
    assert_eq!(paso.unwrap(), "siguiente");
}

#[wasm_bindgen_test]
fn test_agent_sections_are_valid() {
    let text = "HALLAZGOS: dato\nPRÓXIMO PASO: accion";
    assert!(agent_sections_are_valid(text));
    assert!(!agent_sections_are_valid("HALLAZGOS: solo"));
}

#[wasm_bindgen_test]
fn test_find_repetitive_recent_thought_matches() {
    let recent = vec!["HALLAZGOS: hola mundo\nPRÓXIMO PASO: nada".to_string()];
    let found = find_repetitive_recent_thought(
        "HALLAZGOS: hola mundo\nPRÓXIMO PASO: nada",
        &recent,
    );
    assert!(found.is_some());
}

#[wasm_bindgen_test]
fn test_build_agent_fallback_entry_contains_prefix() {
    let entry = build_agent_fallback_entry("HALLAZGOS: foo\nPRÓXIMO PASO: bar");
    assert!(entry.starts_with("HALLAZGOS: registré que estaba en un loop repitiendo"));
}

#[wasm_bindgen_test]
fn test_build_agent_retry_prompt_with_rng_includes_preview() {
    let mut rng = StdRng::seed_from_u64(42);
    let previous = "HALLAZGOS: foo\nPRÓXIMO PASO: bar";
    let prompt = build_agent_retry_prompt_with_rng(Some(previous), &mut rng);
    assert!(prompt.contains("HALLAZGOS"));
    assert!(prompt.contains("PRÓXIMO PASO"));
    assert!(prompt.contains("foo"));
}
