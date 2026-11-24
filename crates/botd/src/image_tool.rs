//! The `generate_image` tool: makes a picture, or edits the photo in the
//! message, with an OpenRouter image model and sends it to the chat.
//!
//! The picture is paid like the rest of the reply. The tool holds the
//! model's highest price from the payer's balance before the call, and the
//! reply's settlement charges what OpenRouter reports and gives back the rest.

use bot_adapters::openrouter_chat::{OpenRouterChatError, ReqwestOpenRouterTransport};
use bot_adapters::openrouter_image::{
    GeneratedImage, ImageRequest, SourceImage, generate_image_with,
};
use bot_core::ai_reserve::credit_units_from_usd_micros;
use bot_core::locale::Locale;
use bot_core::media_reply::classify_media;
use bot_core::provider_pricing::{BEST_IMAGE_MODEL, FAST_IMAGE_MODEL, ImageModel};
use serde_json::{Map, Value, json};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::conversation::CreditHold;
use crate::media::MediaFileSource;
use crate::media_tools::{
    Delivery, MediaBudget, MediaSender, MediaToolContext, budget_spent_message, deliver,
    rejection_message,
};
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

const TOOL: &str = "generate_image";
/// The reservation kind and the billing segment kind of a picture.
const USAGE_KIND: &str = "image_generation";
/// The formats the image models take as a starting picture.
const EDITABLE_TYPES: &[&str] = &["image/jpeg", "image/png", "image/webp"];

pub trait ImageGenerator {
    fn generate(&self, request: &ImageRequest<'_>) -> Result<GeneratedImage, OpenRouterChatError>;
}

pub struct OpenRouterImageGenerator<Transport = ReqwestOpenRouterTransport> {
    transport: Transport,
    api_key: String,
    base_url: String,
}

impl<Transport> OpenRouterImageGenerator<Transport> {
    #[must_use]
    pub fn new(transport: Transport, api_key: &str, base_url: &str) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
        }
    }
}

impl<Transport: bot_adapters::openrouter_chat::OpenRouterTransport> ImageGenerator
    for OpenRouterImageGenerator<Transport>
{
    fn generate(&self, request: &ImageRequest<'_>) -> Result<GeneratedImage, OpenRouterChatError> {
        generate_image_with(&self.transport, &self.api_key, &self.base_url, request)
    }
}

/// What one `generate_image` call asks for.
struct ImageAsk<'a> {
    prompt: &'a str,
    best: bool,
    aspect_ratio: &'a str,
    edit_photo: bool,
    caption: &'a str,
}

pub struct ImageTool<Generator, Photos, Credits, Sender> {
    generator: Generator,
    photos: Photos,
    /// The photo, sticker or GIF in the message or the message it replies to.
    photo_file_id: Option<String>,
    credits: Credits,
    sender: Sender,
    context: MediaToolContext,
    budget: MediaBudget,
}

impl<Generator, Photos, Credits, Sender> ImageTool<Generator, Photos, Credits, Sender> {
    /// `photo` is where to download the picture to edit, and its file id.
    #[must_use]
    pub fn new(
        generator: Generator,
        photo: (Photos, Option<String>),
        credits: Credits,
        sender: Sender,
        context: MediaToolContext,
        budget: MediaBudget,
    ) -> Self {
        let (photos, photo_file_id) = photo;
        Self {
            generator,
            photos,
            photo_file_id,
            credits,
            sender,
            context,
            budget,
        }
    }
}

impl<Generator, Photos, Credits, Sender> ImageTool<Generator, Photos, Credits, Sender>
where
    Generator: ImageGenerator,
    Photos: MediaFileSource,
    Credits: CreditHold,
    Sender: MediaSender,
{
    fn generate(&mut self, ask: &ImageAsk<'_>) -> ToolExecutionResult {
        let locale = self.context.locale;
        // A picture that cannot be sent would still be paid for.
        if self.budget.spent() {
            return ToolExecutionResult::output(budget_spent_message(locale));
        }
        let model = if ask.best {
            BEST_IMAGE_MODEL
        } else {
            FAST_IMAGE_MODEL
        };
        let source = if ask.edit_photo {
            match self.source_photo() {
                Ok(source) => Some(source),
                Err(result) => return result,
            }
        } else {
            None
        };
        if let Some(refusal) = self.hold_credits(model) {
            return refusal;
        }
        let generated = match self.generator.generate(&ImageRequest {
            model,
            prompt: ask.prompt,
            aspect_ratio: ask.aspect_ratio,
            source: source.as_ref().map(|(content_type, bytes)| SourceImage {
                content_type,
                bytes,
            }),
        }) {
            Ok(generated) => generated,
            Err(error) => {
                return ToolExecutionResult::with_diagnostics(
                    match (locale, &error) {
                        (Locale::Es, OpenRouterChatError::Http { message, .. }) => {
                            format!("el modelo de imágenes no hizo la imagen: {message}")
                        }
                        (Locale::En, OpenRouterChatError::Http { message, .. }) => {
                            format!("the image model did not make the picture: {message}")
                        }
                        (Locale::Es, _) => "no pude hacer la imagen".to_owned(),
                        (Locale::En, _) => "could not make the picture".to_owned(),
                    },
                    vec![format!("{TOOL} with {} failed: {error:?}", model.id)],
                );
            }
        };
        let segment = billing_segment(model, &generated);
        let mut result = match generated.picture {
            None => ToolExecutionResult::output(match (locale, generated.text.as_str()) {
                (Locale::Es, "") => "el modelo no devolvió ninguna imagen".to_owned(),
                (Locale::En, "") => "the model returned no picture".to_owned(),
                (Locale::Es, text) => format!("el modelo no hizo la imagen y dijo: {text}"),
                (Locale::En, text) => format!("the model made no picture and said: {text}"),
            }),
            Some(picture) => {
                match classify_media(&picture.content_type, &picture.bytes, "/image") {
                    Ok(media) => deliver(
                        &mut self.sender,
                        self.context,
                        &self.budget,
                        Delivery {
                            tool: TOOL,
                            file: picture.bytes.into(),
                            media,
                            caption: ask.caption,
                        },
                    ),
                    Err(rejection) => {
                        ToolExecutionResult::output(rejection_message(locale, rejection))
                    }
                }
            }
        };
        result.billing_segment = Some(segment);
        result
    }

    /// The picture to edit, with its content type, or the reply that says
    /// why there is none.
    fn source_photo(&mut self) -> Result<(String, Vec<u8>), ToolExecutionResult> {
        let locale = self.context.locale;
        let Some(file_id) = self.photo_file_id.as_deref() else {
            return Err(ToolExecutionResult::output(match locale {
                Locale::Es => {
                    "no hay foto para editar: tiene que venir en el mensaje o en el mensaje al que responde"
                }
                Locale::En => {
                    "there is no photo to edit: it must be in the message or in the message it replies to"
                }
            }));
        };
        let unreadable = match locale {
            Locale::Es => "no pude bajar la foto para editarla",
            Locale::En => "could not download the photo to edit",
        };
        let bytes = match self.photos.download(file_id) {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return Err(ToolExecutionResult::output(unreadable)),
            Err(error) => {
                return Err(ToolExecutionResult::with_diagnostics(
                    unreadable,
                    vec![format!("{TOOL} photo download failed: {error}")],
                ));
            }
        };
        match classify_media("", &bytes, "") {
            Ok(media) if EDITABLE_TYPES.contains(&media.content_type.as_str()) => {
                Ok((media.content_type, bytes))
            }
            _ => Err(ToolExecutionResult::output(match locale {
                Locale::Es => {
                    "solo puedo editar fotos JPEG, PNG o WebP, no GIFs ni stickers animados"
                }
                Locale::En => {
                    "I can only edit JPEG, PNG or WebP photos, not GIFs or animated stickers"
                }
            })),
        }
    }

    /// Holds the model's highest price, or returns the reply that says why
    /// the picture cannot be paid for.
    fn hold_credits(&mut self, model: ImageModel) -> Option<ToolExecutionResult> {
        let locale = self.context.locale;
        // The prices are small constants. If one ever overflowed, holding
        // the most possible makes billing refuse it instead of holding less.
        let amount = credit_units_from_usd_micros(model.max_usd_micros).unwrap_or(i64::MAX);
        let held = self.credits.hold(USAGE_KIND, amount);
        match held {
            Ok(decision) if decision.authorized => None,
            Ok(_) => Some(ToolExecutionResult::output(
                match (locale, model == BEST_IMAGE_MODEL) {
                    (Locale::Es, true) => {
                        "no alcanzan los créditos para una imagen de la mejor calidad; probá con quality fast o cargá con /topup"
                    }
                    (Locale::En, true) => {
                        "there are not enough credits for a best-quality picture; try quality fast or top up with /topup"
                    }
                    (Locale::Es, false) => {
                        "no alcanzan los créditos para hacer la imagen; cargá con /topup"
                    }
                    (Locale::En, false) => {
                        "there are not enough credits to make the picture; top up with /topup"
                    }
                },
            )),
            Err(error) => Some(ToolExecutionResult::with_diagnostics(
                tool_output::failed(locale, TOOL),
                vec![format!("{TOOL} credit hold failed: {error}")],
            )),
        }
    }
}

/// The picture's cost for the reply's settlement, in the shape of a chat
/// round from OpenRouter. Without a reported cost, the reconciler reads it
/// later from the generation id.
fn billing_segment(model: ImageModel, generated: &GeneratedImage) -> Value {
    let mut metadata = Map::from_iter([("provider".to_owned(), json!("openrouter"))]);
    if generated.model != model.id {
        metadata.insert("requested_model".to_owned(), json!(model.id));
    }
    if let Some(id) = &generated.generation_id {
        metadata.insert("provider_generation_id".to_owned(), json!(id));
    }
    if let Some(provider) = &generated.upstream_provider {
        metadata.insert("upstream_provider".to_owned(), json!(provider));
    }
    if generated.usage.is_empty() {
        metadata.insert("provider_usage_pending".to_owned(), json!(true));
    }
    json!({
        "kind": USAGE_KIND,
        "model": generated.model,
        "usage": generated.usage,
        "source": "openrouter",
        "metadata": metadata,
    })
}

impl<Generator, Photos, Credits, Sender> ExternalToolExecutor
    for ImageTool<Generator, Photos, Credits, Sender>
where
    Generator: ImageGenerator,
    Photos: MediaFileSource,
    Credits: CreditHold,
    Sender: MediaSender,
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        match request {
            ExternalToolRequest::GenerateImage {
                prompt,
                best,
                aspect_ratio,
                edit_photo,
                caption,
            } => self.generate(&ImageAsk {
                prompt: &prompt,
                best,
                aspect_ratio: &aspect_ratio,
                edit_photo,
                caption: &caption,
            }),
            _ => ToolExecutionResult::output(tool_output::incompatible(self.context.locale, TOOL)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::{BTreeMap, VecDeque};

    use bot_adapters::openrouter_chat::{HttpRequest, HttpResponse, OpenRouterTransport};
    use bot_adapters::openrouter_image::Picture;
    use bot_adapters::telegram_actions::{ActionError, ActionOutcome};
    use bot_core::media_reply::MediaKind;
    use bot_core::telegram_actions::TelegramAction;
    use bot_core::telegram_input::{ChatId, MessageId};

    use super::*;
    use crate::conversation::ReserveDecision;

    type TestResult = Result<(), Box<dyn std::error::Error>>;
    type Asked = (String, String, String, Option<(String, Vec<u8>)>);
    /// The photo's file id, its download, the English and Spanish answers,
    /// and whether the failure is logged.
    type PhotoCase<'a> = (
        Option<&'a str>,
        Result<Option<Vec<u8>>, String>,
        &'a str,
        &'a str,
        bool,
    );

    const WEBP: &[u8] = b"RIFF\0\0\0\0WEBPVP8 ";
    const JPEG: &[u8] = b"\xFF\xD8\xFF\xE0photo";

    struct Generator {
        reply: Result<GeneratedImage, OpenRouterChatError>,
        asked: RefCell<Vec<Asked>>,
    }

    impl ImageGenerator for Generator {
        fn generate(
            &self,
            request: &ImageRequest<'_>,
        ) -> Result<GeneratedImage, OpenRouterChatError> {
            self.asked.borrow_mut().push((
                request.model.id.to_owned(),
                request.prompt.to_owned(),
                request.aspect_ratio.to_owned(),
                request
                    .source
                    .map(|source| (source.content_type.to_owned(), source.bytes.to_vec())),
            ));
            self.reply.clone()
        }
    }

    struct Photos {
        reply: Result<Option<Vec<u8>>, String>,
        asked: Vec<String>,
    }

    impl MediaFileSource for Photos {
        fn download(&mut self, file_id: &str) -> Result<Option<Vec<u8>>, String> {
            self.asked.push(file_id.to_owned());
            self.reply.clone()
        }
    }

    struct Credits {
        reply: Result<bool, String>,
        held: Vec<(String, i64)>,
    }

    impl CreditHold for Credits {
        fn hold(&mut self, kind: &str, amount: i64) -> Result<ReserveDecision, String> {
            self.held.push((kind.to_owned(), amount));
            self.reply.clone().map(|authorized| ReserveDecision {
                authorized,
                user_balance: 0,
                chat_balance: 0,
                source: None,
                denial: None,
            })
        }
    }

    #[derive(Default)]
    struct Sender {
        replies: VecDeque<Result<ActionOutcome, ActionError>>,
        sent: Vec<TelegramAction>,
    }

    impl MediaSender for Sender {
        fn send(&mut self, action: TelegramAction) -> Result<ActionOutcome, ActionError> {
            self.sent.push(action);
            self.replies
                .pop_front()
                .unwrap_or(Ok(ActionOutcome::Completed {
                    message_id: Some(9),
                }))
        }
    }

    type Tool = ImageTool<Generator, Photos, Credits, Sender>;

    fn made(picture: Option<(&str, &[u8])>, text: &str) -> GeneratedImage {
        GeneratedImage {
            picture: picture.map(|(content_type, bytes)| Picture {
                content_type: content_type.to_owned(),
                bytes: bytes.to_vec(),
            }),
            text: text.to_owned(),
            generation_id: Some("gen-1".to_owned()),
            model: FAST_IMAGE_MODEL.id.to_owned(),
            upstream_provider: Some("ByteDance".to_owned()),
            usage: Map::from_iter([("cost".to_owned(), json!(0.02))]),
        }
    }

    fn tool(
        reply: Result<GeneratedImage, OpenRouterChatError>,
        photo: (Option<&str>, Result<Option<Vec<u8>>, String>),
        credits: Result<bool, String>,
        locale: Locale,
    ) -> Tool {
        ImageTool::new(
            Generator {
                reply,
                asked: RefCell::new(Vec::new()),
            },
            (
                Photos {
                    reply: photo.1,
                    asked: Vec::new(),
                },
                photo.0.map(str::to_owned),
            ),
            Credits {
                reply: credits,
                held: Vec::new(),
            },
            Sender::default(),
            MediaToolContext {
                chat_id: -100,
                reply_to_message_id: Some(7),
                locale,
            },
            MediaBudget::default(),
        )
    }

    fn new_picture(reply: Result<GeneratedImage, OpenRouterChatError>, locale: Locale) -> Tool {
        tool(reply, (None, Ok(None)), Ok(true), locale)
    }

    fn ask(best: bool, edit_photo: bool) -> ExternalToolRequest {
        ExternalToolRequest::GenerateImage {
            prompt: "a red cube".to_owned(),
            best,
            aspect_ratio: "16:9".to_owned(),
            edit_photo,
            caption: "Cube".to_owned(),
        }
    }

    #[test]
    fn a_new_picture_is_paid_for_made_and_sent_with_its_cost() {
        let mut tool = new_picture(Ok(made(Some(("image/webp", WEBP)), "")), Locale::En);
        let result = tool.execute(ask(false, false), "call");
        assert_eq!(
            result.output,
            "Sent the image to the chat. People can see it: do not paste the link or describe it again unless asked"
        );
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("Here is the image.")
        );
        assert_eq!(
            result.billing_segment,
            Some(json!({
                "kind": "image_generation",
                "model": "bytedance-seed/seedream-5-5",
                "usage": {"cost": 0.02},
                "source": "openrouter",
                "metadata": {
                    "provider": "openrouter",
                    "provider_generation_id": "gen-1",
                    "upstream_provider": "ByteDance",
                },
            }))
        );
        assert_eq!(tool.credits.held, [("image_generation".to_owned(), 1_000)]);
        assert_eq!(
            *tool.generator.asked.borrow(),
            [(
                "bytedance-seed/seedream-5-5".to_owned(),
                "a red cube".to_owned(),
                "16:9".to_owned(),
                None
            )]
        );
        assert!(tool.photos.asked.is_empty());
        assert!(matches!(
            &tool.sender.sent[..],
            [TelegramAction::SendMedia {
                chat_id: ChatId(-100),
                kind: MediaKind::Photo,
                file,
                file_name,
                content_type,
                reply_to_message_id: Some(MessageId(7)),
                caption,
            }] if **file == *WEBP
                && file_name == "image.webp"
                && content_type == "image/webp"
                && caption == "Cube"
        ));
    }

    #[test]
    fn an_edit_sends_the_photo_from_the_message_to_the_best_model() {
        // Seedream labels its PNG pictures image/jpeg, so the bytes decide.
        let mut generated = made(Some(("image/jpeg", b"\x89PNG\r\n\x1A\nimage")), "");
        generated.model = "openai/gpt-image-2.6-sunburst-20261101".to_owned();
        generated.generation_id = None;
        generated.upstream_provider = None;
        generated.usage = Map::new();
        let mut tool = tool(
            Ok(generated),
            (Some("photo-1"), Ok(Some(JPEG.to_vec()))),
            Ok(true),
            Locale::Es,
        );
        let result = tool.execute(ask(true, true), "call");
        assert!(result.output.starts_with("Mandé la imagen al chat"));
        assert_eq!(
            result.billing_segment,
            Some(json!({
                "kind": "image_generation",
                "model": "openai/gpt-image-2.6-sunburst-20261101",
                "usage": {},
                "source": "openrouter",
                "metadata": {
                    "provider": "openrouter",
                    "requested_model": "openai/gpt-image-2.6-sunburst",
                    "provider_usage_pending": true,
                },
            }))
        );
        assert_eq!(tool.photos.asked, ["photo-1"]);
        assert_eq!(tool.credits.held, [("image_generation".to_owned(), 8_000)]);
        assert_eq!(
            tool.generator.asked.borrow()[0].3,
            Some(("image/jpeg".to_owned(), JPEG.to_vec()))
        );
        assert_eq!(tool.generator.asked.borrow()[0].0, BEST_IMAGE_MODEL.id);
        assert!(matches!(
            &tool.sender.sent[..],
            [TelegramAction::SendMedia { content_type, file_name, .. }]
                if content_type == "image/png" && file_name == "image.png"
        ));
    }

    #[test]
    fn an_edit_needs_a_photo_it_can_read() {
        let cases: [PhotoCase<'_>; 4] = [
            (
                None,
                Ok(None),
                "there is no photo to edit: it must be in the message or in the message it replies to",
                "no hay foto para editar: tiene que venir en el mensaje o en el mensaje al que responde",
                false,
            ),
            (
                Some("photo-1"),
                Ok(None),
                "could not download the photo to edit",
                "no pude bajar la foto para editarla",
                false,
            ),
            (
                Some("photo-1"),
                Err("getFile failed".to_owned()),
                "could not download the photo to edit",
                "no pude bajar la foto para editarla",
                true,
            ),
            (
                Some("sticker-1"),
                Ok(Some(b"GIF89a....".to_vec())),
                "I can only edit JPEG, PNG or WebP photos, not GIFs or animated stickers",
                "solo puedo editar fotos JPEG, PNG o WebP, no GIFs ni stickers animados",
                false,
            ),
        ];
        for (file_id, download, english, spanish, logged) in cases {
            for (locale, expected) in [(Locale::En, english), (Locale::Es, spanish)] {
                let mut tool = tool(
                    Ok(made(Some(("image/webp", WEBP)), "")),
                    (file_id, download.clone()),
                    Ok(true),
                    locale,
                );
                let result = tool.execute(ask(false, true), "call");
                assert_eq!(result.output, expected);
                assert_eq!(result.diagnostics.len(), usize::from(logged));
                assert!(tool.credits.held.is_empty());
                assert!(tool.generator.asked.borrow().is_empty());
                assert!(tool.sender.sent.is_empty());
            }
        }
    }

    #[test]
    fn nothing_is_made_without_credits_or_room_in_the_reply() {
        let cases = [
            (
                Locale::En,
                true,
                "there are not enough credits for a best-quality picture; try quality fast or top up with /topup",
            ),
            (
                Locale::Es,
                true,
                "no alcanzan los créditos para una imagen de la mejor calidad; probá con quality fast o cargá con /topup",
            ),
            (
                Locale::En,
                false,
                "there are not enough credits to make the picture; top up with /topup",
            ),
            (
                Locale::Es,
                false,
                "no alcanzan los créditos para hacer la imagen; cargá con /topup",
            ),
        ];
        for (locale, best, expected) in cases {
            let mut tool = tool(
                Ok(made(Some(("image/webp", WEBP)), "")),
                (None, Ok(None)),
                Ok(false),
                locale,
            );
            assert_eq!(tool.execute(ask(best, false), "call").output, expected);
            assert!(tool.generator.asked.borrow().is_empty());
        }

        let mut broken = tool(
            Ok(made(Some(("image/webp", WEBP)), "")),
            (None, Ok(None)),
            Err("database down".to_owned()),
            Locale::En,
        );
        let result = broken.execute(ask(false, false), "call");
        assert_eq!(result.output, "tool 'generate_image' failed");
        assert_eq!(
            result.diagnostics,
            ["generate_image credit hold failed: database down"]
        );
        assert!(broken.generator.asked.borrow().is_empty());

        let mut tool = new_picture(Ok(made(Some(("image/webp", WEBP)), "")), Locale::En);
        for _ in 0..crate::media_tools::MAX_MEDIA_PER_REPLY {
            tool.execute(ask(false, false), "call");
        }
        let result = tool.execute(ask(false, false), "call");
        assert_eq!(
            result.output,
            "already sent 10 files in this reply; no more can be sent"
        );
        assert_eq!(result.billing_segment, None);
        assert_eq!(tool.credits.held.len(), 10);
        assert_eq!(tool.generator.asked.borrow().len(), 10);
    }

    #[test]
    fn failed_and_declined_pictures_say_what_happened() {
        let refused = OpenRouterChatError::Http {
            status_code: 400,
            message: "prompt flagged".to_owned(),
        };
        let reset = OpenRouterChatError::Transport("reset".to_owned());
        let failures = [
            (
                Locale::En,
                refused.clone(),
                "the image model did not make the picture: prompt flagged",
            ),
            (
                Locale::Es,
                refused,
                "el modelo de imágenes no hizo la imagen: prompt flagged",
            ),
            (Locale::En, reset.clone(), "could not make the picture"),
            (Locale::Es, reset, "no pude hacer la imagen"),
        ];
        for (locale, error, expected) in failures {
            let result = new_picture(Err(error), locale).execute(ask(false, false), "call");
            assert_eq!(result.output, expected);
            assert_eq!(result.billing_segment, None);
            assert!(
                result.diagnostics[0]
                    .starts_with("generate_image with bytedance-seed/seedream-5-5 failed")
            );
        }

        let declines = [
            (Locale::En, "", "the model returned no picture"),
            (Locale::Es, "", "el modelo no devolvió ninguna imagen"),
            (Locale::En, "No.", "the model made no picture and said: No."),
            (Locale::Es, "No.", "el modelo no hizo la imagen y dijo: No."),
        ];
        for (locale, text, expected) in declines {
            let mut tool = new_picture(Ok(made(None, text)), locale);
            let result = tool.execute(ask(false, false), "call");
            assert_eq!(result.output, expected);
            // The call can still cost money, so the reply pays for it.
            assert!(result.billing_segment.is_some());
            assert!(tool.sender.sent.is_empty());
        }

        let mut page = new_picture(Ok(made(Some(("text/html", b"<html>")), "")), Locale::En);
        let result = page.execute(ask(false, false), "call");
        assert!(result.output.starts_with("that link is a web page"));
        assert!(result.billing_segment.is_some());
        assert!(page.sender.sent.is_empty());

        let mut unsent = new_picture(Ok(made(Some(("image/webp", WEBP)), "")), Locale::En);
        unsent.sender.replies.push_back(Ok(ActionOutcome::Failed {
            status_code: Some(403),
            description: "bot was blocked".to_owned(),
        }));
        let result = unsent.execute(ask(false, false), "call");
        assert_eq!(
            result.output,
            "Telegram did not accept the file: bot was blocked"
        );
        assert!(result.billing_segment.is_some());
    }

    #[test]
    fn other_requests_are_refused() {
        let result = new_picture(Ok(made(None, "")), Locale::En)
            .execute(ExternalToolRequest::BotCapabilities, "call");
        assert_eq!(
            result.output,
            "tool 'generate_image' received an incompatible request"
        );
    }

    struct Transport(RefCell<Vec<HttpRequest>>);

    impl OpenRouterTransport for Transport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.0.borrow_mut().push(request.clone());
            Ok(HttpResponse {
                status_code: 200,
                body: r#"{"choices":[{"message":{"images":[{"image_url":{"url":"data:image/png;base64,iVBORw=="}}]}}]}"#.to_owned(),
                headers: BTreeMap::new(),
            })
        }
    }

    #[test]
    fn the_openrouter_generator_posts_with_its_key_to_its_base_url() -> TestResult {
        let generator = OpenRouterImageGenerator::new(
            Transport(RefCell::new(Vec::new())),
            "key",
            "https://openrouter.test/api/v1",
        );
        let image = generator.generate(&ImageRequest {
            model: FAST_IMAGE_MODEL,
            prompt: "a red cube",
            aspect_ratio: "1:1",
            source: None,
        })?;
        assert!(image.picture.is_some());
        let sent = generator.transport.0.borrow();
        assert_eq!(
            sent[0].url,
            "https://openrouter.test/api/v1/chat/completions"
        );
        assert_eq!(sent[0].bearer_token, "key");
        Ok(())
    }
}
