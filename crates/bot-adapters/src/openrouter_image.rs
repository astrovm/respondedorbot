//! Picture generation through OpenRouter's chat completions API. The picture
//! comes back as a base64 `data:` URL in `choices[0].message.images`.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use bot_core::provider_pricing::ImageModel;
use serde_json::{Map, Value, json};

use crate::openrouter_chat::{
    HttpRequest, OpenRouterChatError, OpenRouterTransport, completion_url, http_error,
    invalid_json, request_json_error,
};

/// A picture the model starts from, for edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceImage<'a> {
    pub content_type: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRequest<'a> {
    pub model: ImageModel,
    pub prompt: &'a str,
    pub aspect_ratio: &'a str,
    pub source: Option<SourceImage<'a>>,
}

/// What came back. A model that declines gives no picture, only text, and
/// the call can still cost money, so the usage is kept either way.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedImage {
    pub picture: Option<Picture>,
    pub text: String,
    pub generation_id: Option<String>,
    pub model: String,
    pub upstream_provider: Option<String>,
    pub usage: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Picture {
    pub content_type: String,
    pub bytes: Vec<u8>,
}

pub fn generate_image_with<T: OpenRouterTransport>(
    transport: &T,
    api_key: &str,
    base_url: &str,
    request: &ImageRequest<'_>,
) -> Result<GeneratedImage, OpenRouterChatError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(OpenRouterChatError::MissingApiKey);
    }
    let mut content = vec![json!({"type": "text", "text": request.prompt})];
    if let Some(source) = request.source {
        content.push(json!({
            "type": "image_url",
            "image_url": {"url": format!(
                "data:{};base64,{}",
                source.content_type,
                BASE64_STANDARD.encode(source.bytes)
            )},
        }));
    }
    let body = serde_json::to_string(&json!({
        "model": request.model.id,
        "messages": [{"role": "user", "content": content}],
        "modalities": request.model.modalities,
        "image_config": {"aspect_ratio": request.aspect_ratio},
        "usage": {"include": true},
    }))
    .map_err(request_json_error)?;
    let response = transport.post(&HttpRequest {
        url: completion_url(base_url)?,
        bearer_token: api_key.to_owned(),
        body,
    })?;
    if response.status_code >= 400 {
        return Err(http_error(
            response.status_code,
            &response.body,
            &response.headers,
        ));
    }
    let envelope = serde_json::from_str::<Value>(&response.body).map_err(invalid_json)?;
    let message = envelope
        .pointer("/choices/0/message")
        .ok_or(OpenRouterChatError::MalformedResponse)?;
    let text = |value: Option<&Value>| {
        value
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    Ok(GeneratedImage {
        picture: message
            .get("images")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|image| image.pointer("/image_url/url").and_then(Value::as_str))
            .find_map(decode_data_url),
        text: text(message.get("content"))
            .or_else(|| text(message.get("refusal")))
            .unwrap_or_default(),
        generation_id: text(envelope.get("id")),
        model: text(envelope.get("model")).unwrap_or_else(|| request.model.id.to_owned()),
        upstream_provider: text(envelope.get("provider")),
        usage: envelope
            .get("usage")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default(),
    })
}

fn decode_data_url(url: &str) -> Option<Picture> {
    let (content_type, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
    let bytes = BASE64_STANDARD.decode(data.trim()).ok()?;
    (!bytes.is_empty()).then(|| Picture {
        content_type: content_type.trim().to_ascii_lowercase(),
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use bot_core::provider_pricing::{BEST_IMAGE_MODEL, FAST_IMAGE_MODEL};

    use super::*;
    use crate::openrouter_chat::HttpResponse;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct Transport {
        reply: Result<HttpResponse, OpenRouterChatError>,
        sent: RefCell<Vec<HttpRequest>>,
    }

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.sent.borrow_mut().push(request.clone());
            self.reply.clone()
        }
    }

    fn replying(status_code: u16, body: &str) -> Transport {
        Transport {
            reply: Ok(HttpResponse {
                status_code,
                body: body.to_owned(),
                headers: BTreeMap::new(),
            }),
            sent: RefCell::new(Vec::new()),
        }
    }

    fn request(source: Option<SourceImage<'_>>) -> ImageRequest<'_> {
        ImageRequest {
            model: FAST_IMAGE_MODEL,
            prompt: "a red cube on a white table",
            aspect_ratio: "16:9",
            source,
        }
    }

    /// The shape of a real reply from `recraft/recraft-v4.1-flash-v2:free`
    /// on 2026-11-23, with a tiny picture in place of the 900 KB WebP.
    fn real_reply(picture: &str) -> String {
        json!({
            "id": "gen-1795487088-07OXTAFY3w5xHnTsYmYK",
            "object": "chat.completion",
            "model": "recraft/recraft-v4.1-flash-v2:free",
            "provider": "Recraft",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": null,
                    "images": [{"type": "image_url", "image_url": {"url": picture}}],
                },
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4175, "cost": 0.02},
        })
        .to_string()
    }

    #[test]
    fn a_picture_comes_back_decoded_with_its_usage() -> TestResult {
        let transport = replying(200, &real_reply("data:image/WEBP;base64,UklGRgAAAABXRUJQ"));
        let image = generate_image_with(
            &transport,
            " key ",
            "https://openrouter.test/api/v1",
            &request(None),
        )?;
        assert_eq!(
            image.picture,
            Some(Picture {
                content_type: "image/webp".to_owned(),
                bytes: b"RIFF\0\0\0\0WEBP".to_vec(),
            })
        );
        assert_eq!(image.text, "");
        assert_eq!(
            image.generation_id.as_deref(),
            Some("gen-1795487088-07OXTAFY3w5xHnTsYmYK")
        );
        assert_eq!(image.model, "recraft/recraft-v4.1-flash-v2:free");
        assert_eq!(image.upstream_provider.as_deref(), Some("Recraft"));
        assert_eq!(image.usage.get("cost"), Some(&json!(0.02)));
        let sent = transport.sent.borrow();
        assert_eq!(
            sent[0].url,
            "https://openrouter.test/api/v1/chat/completions"
        );
        assert_eq!(sent[0].bearer_token, "key");
        let body = serde_json::from_str::<Value>(&sent[0].body)?;
        assert_eq!(
            body,
            json!({
                "model": "bytedance-seed/seedream-5-5",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "a red cube on a white table"},
                ]}],
                "modalities": ["image"],
                "image_config": {"aspect_ratio": "16:9"},
                "usage": {"include": true},
            })
        );
        Ok(())
    }

    #[test]
    fn an_edit_sends_the_starting_picture_as_a_data_url() -> TestResult {
        let transport = replying(200, &real_reply("data:image/png;base64,iVBORw=="));
        let source = SourceImage {
            content_type: "image/jpeg",
            bytes: b"\xFF\xD8\xFF",
        };
        let image = generate_image_with(
            &transport,
            "key",
            "https://openrouter.test/api/v1",
            &ImageRequest {
                model: BEST_IMAGE_MODEL,
                ..request(Some(source))
            },
        )?;
        assert!(image.picture.is_some());
        let body = serde_json::from_str::<Value>(&transport.sent.borrow()[0].body)?;
        assert_eq!(body["model"], "openai/gpt-image-2.6-sunburst");
        assert_eq!(
            body["messages"][0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}})
        );
        Ok(())
    }

    #[test]
    fn a_declined_picture_keeps_the_models_words_and_usage() -> TestResult {
        let cases = [
            (
                json!({"content": " I can't draw that. "}),
                "I can't draw that.",
            ),
            (json!({"content": null, "refusal": "policy"}), "policy"),
            (
                json!({"images": [
                    {"image_url": {"url": "https://cdn.example/a.png"}},
                    {"image_url": {"url": "data:image/png;base64,***"}},
                    {"image_url": {"url": "data:image/png;base64,"}},
                    {"image_url": {"url": "data:image/png,plain"}},
                ]}),
                "",
            ),
        ];
        for (message, text) in cases {
            let body = json!({"choices": [{"message": message}], "usage": {"cost": 0.01}});
            let image = generate_image_with(
                &replying(200, &body.to_string()),
                "key",
                "https://openrouter.test/api/v1",
                &request(None),
            )?;
            assert_eq!(image.picture, None);
            assert_eq!(image.text, text);
            assert_eq!(image.model, "bytedance-seed/seedream-5-5");
            assert_eq!(image.generation_id, None);
            assert_eq!(image.usage.get("cost"), Some(&json!(0.01)));
        }
        let no_usage = generate_image_with(
            &replying(200, r#"{"choices":[{"message":{}}]}"#),
            "key",
            "https://openrouter.test/api/v1",
            &request(None),
        )?;
        assert!(no_usage.usage.is_empty());
        Ok(())
    }

    #[test]
    fn bad_keys_urls_replies_and_errors_are_reported() {
        let call = |transport: &Transport, key: &str, base: &str| {
            generate_image_with(transport, key, base, &request(None))
        };
        let ok = replying(200, "{}");
        assert_eq!(
            call(&ok, " ", "https://openrouter.test"),
            Err(OpenRouterChatError::MissingApiKey)
        );
        assert_eq!(
            call(&ok, "key", "not a url"),
            Err(OpenRouterChatError::InvalidBaseUrl)
        );
        assert_eq!(
            call(&ok, "key", "https://openrouter.test"),
            Err(OpenRouterChatError::MalformedResponse)
        );
        assert!(matches!(
            call(&replying(200, "not json"), "key", "https://openrouter.test"),
            Err(OpenRouterChatError::InvalidJson(_))
        ));
        assert!(matches!(
            call(
                &replying(404, r#"{"error":{"message":"No endpoints found"}}"#),
                "key",
                "https://openrouter.test"
            ),
            Err(OpenRouterChatError::Http { status_code: 404, message }) if message.contains("No endpoints")
        ));
        let failing = Transport {
            reply: Err(OpenRouterChatError::Transport("reset".to_owned())),
            sent: RefCell::new(Vec::new()),
        };
        assert_eq!(
            call(&failing, "key", "https://openrouter.test"),
            Err(OpenRouterChatError::Transport("reset".to_owned()))
        );
    }
}
