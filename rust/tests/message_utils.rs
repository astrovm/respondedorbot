use respondedorbot::chat_config::ChatConfig;
use respondedorbot::message_utils::{
    build_ai_messages, build_reply_context_text, clean_duplicate_response, contains_url,
    format_user_identity, gen_random, post_process_ai_response, remove_gordo_prefix,
    should_gordo_respond_core, strip_leading_context, strip_user_identity_prefix, ChatHistoryEntry,
};
use respondedorbot::models::{Chat, FileRef, Message, User};

fn base_user(username: &str) -> User {
    User {
        id: 1,
        first_name: "Pepe".to_string(),
        last_name: Some("Gomez".to_string()),
        username: Some(username.to_string()),
    }
}

fn base_message() -> Message {
    Message {
        message_id: 10,
        chat: Chat {
            id: 99,
            kind: "group".to_string(),
            title: Some("Grupo".to_string()),
        },
        from: Some(base_user("pepe")),
        text: Some("hola".to_string()),
        reply_to_message: None,
        photo: None,
        voice: None,
        audio: None,
        sticker: None,
        video: None,
        document: None,
    }
}

#[test]
fn test_remove_gordo_prefix() {
    let raw = "gordo: hola\nGORDO: che";
    let cleaned = remove_gordo_prefix(raw);
    assert_eq!(cleaned, "hola\nche");
}

#[test]
fn test_strip_user_identity_prefix() {
    let response = "Pepe Gomez (pepe): hola";
    let cleaned = strip_user_identity_prefix(response, Some("Pepe Gomez (pepe)"));
    assert_eq!(cleaned, "hola");
}

#[test]
fn test_clean_duplicate_response_lines_and_sentences() {
    let raw = "hola\nhola\nchau\nchau\n\nhola. hola. chau. chau.";
    let cleaned = clean_duplicate_response(raw);
    assert!(cleaned.contains("hola\nchau"));
    assert!(cleaned.contains("hola. chau"));
}

#[test]
fn test_strip_leading_context() {
    let response = "Contexto largo: hola boludo";
    let cleaned = strip_leading_context(
        response,
        &[Some("Contexto largo".to_string()), Some("otra".to_string())],
    );
    assert_eq!(cleaned, "hola boludo");
}

#[test]
fn test_post_process_ai_response_combined() {
    let raw = "gordo: Pepe: hola\nhola";
    let cleaned =
        post_process_ai_response(raw, &[Some("Pepe: hola".to_string())], Some("Pepe"));
    assert_eq!(cleaned, "hola");
}

#[test]
fn test_format_user_identity() {
    let user = base_user("pepe");
    let formatted = format_user_identity(Some(&user));
    assert_eq!(formatted, "Pepe Gomez (pepe)");
}

#[test]
fn test_build_reply_context_text_video_document() {
    let mut reply = base_message();
    reply.text = None;
    reply.video = Some(FileRef {
        file_id: "vid".to_string(),
    });
    let mut msg = base_message();
    msg.reply_to_message = Some(Box::new(reply));
    let context = build_reply_context_text(&msg).unwrap();
    assert!(context.contains("un video"));

    let mut reply = base_message();
    reply.text = None;
    reply.document = Some(FileRef {
        file_id: "doc".to_string(),
    });
    let mut msg = base_message();
    msg.reply_to_message = Some(Box::new(reply));
    let context = build_reply_context_text(&msg).unwrap();
    assert!(context.contains("un archivo adjunto"));
}

#[test]
fn test_contains_url() {
    assert!(contains_url("mirá https://example.com"));
    assert!(!contains_url("nada aca"));
}

#[test]
fn test_gen_random_variants() {
    let mut observed_ok = true;
    for _ in 0..50 {
        let msg = gen_random("Pepe");
        let valid = ["si", "no", "si boludo", "no boludo", "si Pepe", "no Pepe"];
        if !valid.contains(&msg.as_str()) {
            observed_ok = false;
            break;
        }
    }
    assert!(observed_ok);
}

#[test]
fn test_build_ai_messages_includes_context() {
    let msg = base_message();
    let history: Vec<ChatHistoryEntry> = Vec::new();
    let messages = build_ai_messages(&msg, &history, "hola", Some("pepe: hola"));
    let last = messages.last().unwrap().content.clone();
    assert!(last.contains("CONTEXTO:"));
    assert!(last.contains("MENSAJE AL QUE RESPONDE:"));
    assert!(last.contains("MENSAJE:"));
}

#[test]
fn test_should_gordo_respond_block_followups() {
    let mut reply = base_message();
    reply.from = Some(base_user("bot"));
    let mut msg = base_message();
    msg.reply_to_message = Some(Box::new(reply));
    let chat_config = ChatConfig {
        link_mode: "off".to_string(),
        ai_random_replies: false,
        ai_command_followups: false,
    };
    let metadata = serde_json::json!({"type":"command","uses_ai":false});
    let should = should_gordo_respond_core(
        "bot",
        &msg,
        "hola",
        &chat_config,
        Some(&metadata),
        &["bot".to_string()],
        0.05,
    );
    assert!(!should);
}

#[test]
fn test_should_gordo_respond_block_link_fix() {
    let mut reply = base_message();
    reply.from = Some(base_user("bot"));
    reply.text = Some("fxtwitter.com".to_string());
    let mut msg = base_message();
    msg.reply_to_message = Some(Box::new(reply));
    let chat_config = ChatConfig::default();
    let metadata = serde_json::json!({"type":"link_fix"});
    let should = should_gordo_respond_core(
        "bot",
        &msg,
        "hola",
        &chat_config,
        Some(&metadata),
        &["bot".to_string()],
        0.05,
    );
    assert!(!should);
}

#[test]
fn test_should_gordo_respond_trigger() {
    let msg = base_message();
    let chat_config = ChatConfig::default();
    let should = should_gordo_respond_core(
        "bot",
        &msg,
        "hola bot",
        &chat_config,
        None,
        &["bot".to_string()],
        0.05,
    );
    assert!(should);
}
