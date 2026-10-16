//! OpenRouter replies, both whole and streamed in arbitrary chunks.

#![no_main]

use std::collections::BTreeMap;

use bot_adapters::openrouter_chat::{
    ChatCompletionRequest, HttpRequest, HttpResponse, OpenRouterChatError,
    OpenRouterStreamTransport, parse_chat_completion, stream_with,
};
use libfuzzer_sys::fuzz_target;

struct Chunks<'a>(&'a [u8], usize);

impl OpenRouterStreamTransport for Chunks<'_> {
    fn post_stream(
        &self,
        _request: &HttpRequest,
        on_bytes: &mut dyn FnMut(&[u8]) -> Result<(), OpenRouterChatError>,
    ) -> Result<(), OpenRouterChatError> {
        for chunk in self.0.chunks(self.1.max(1)) {
            on_bytes(chunk)?;
        }
        Ok(())
    }
}

fuzz_target!(|input: (u8, &[u8])| {
    let (chunk_size, bytes) = input;
    let body = String::from_utf8_lossy(bytes).into_owned();
    let _ = parse_chat_completion(
        HttpResponse {
            status_code: 200,
            body,
            headers: BTreeMap::new(),
        },
        "synthetic/model",
    );
    let request = ChatCompletionRequest {
        model: "synthetic/model".to_owned(),
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        max_tokens: None,
        temperature: None,
        provider: None,
        reasoning: None,
        stream: true,
    };
    let _ = stream_with(
        &Chunks(bytes, usize::from(chunk_size)),
        "synthetic-key",
        "https://openrouter.example.test/api/v1",
        &request,
        |_event| Ok(()),
    );
});
