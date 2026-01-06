use wasm_bindgen_test::wasm_bindgen_test;
use respondedorbot::chat_config::{build_config_keyboard, build_config_text, ChatConfig};

#[wasm_bindgen_test]
fn test_build_config_text() {
    let config = ChatConfig {
        link_mode: "reply".to_string(),
        ai_random_replies: true,
        ai_command_followups: false,
    };
    let text = build_config_text(&config);
    assert!(text.contains("Link fixer"));
    assert!(text.contains("Reply"));
    assert!(text.contains("Random AI replies"));
}

#[wasm_bindgen_test]
fn test_build_config_keyboard() {
    let config = ChatConfig::default();
    let keyboard = build_config_keyboard(&config);
    let inline = keyboard.get("inline_keyboard").unwrap();
    assert!(inline.is_array());
}
