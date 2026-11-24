//! Native AI tools that put files in the chat: a picture, GIF, video, audio
//! track or any other file from a link, and a screenshot of a web page.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use bot_adapters::browser::PageCapturer;
use bot_adapters::telegram_actions::{ActionError, ActionOutcome, execute_with};
use bot_adapters::telegram_http::{
    ReqwestTelegramTransport, TelegramTransport, TransportFailureKind,
};
use bot_adapters::web_fetch::{
    HostResolver, PublicFetchError, PublicFile, WebFetchTransport, fetch_public_media,
};
use bot_core::locale::Locale;
use bot_core::media_reply::{MediaFile, MediaKind, MediaRejection, classify_media};
use bot_core::telegram_actions::TelegramAction;
use bot_core::telegram_input::{ChatId, MessageId};

use crate::chat_tool_loop::ToolExecutionResult;
use crate::tool_output;
use crate::tool_requests::{ExternalToolExecutor, ExternalToolRequest};

/// The most files one reply may send, so a request cannot flood the chat.
pub const MAX_MEDIA_PER_REPLY: usize = 10;

pub type SharedCapturer = Arc<dyn PageCapturer + Send + Sync>;

/// Counts the files one reply has sent, across both media tools.
#[derive(Debug, Clone, Default)]
pub struct MediaBudget(Arc<AtomicUsize>);

impl MediaBudget {
    /// Takes one send from the budget, or returns false when it is spent.
    fn take(&self) -> bool {
        self.0
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |sent| {
                (sent < MAX_MEDIA_PER_REPLY).then_some(sent + 1)
            })
            .is_ok()
    }

    /// Gives back a send that did not happen.
    fn refund(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }

    /// True when the reply has no sends left, so a tool can stop before it
    /// pays for a file it could not send.
    pub(crate) fn spent(&self) -> bool {
        self.0.load(Ordering::Relaxed) >= MAX_MEDIA_PER_REPLY
    }
}

/// What a media tool says when the reply has sent all the files it may.
pub(crate) fn budget_spent_message(locale: Locale) -> String {
    match locale {
        Locale::Es => {
            format!("ya mandé {MAX_MEDIA_PER_REPLY} archivos en esta respuesta, no mando más")
        }
        Locale::En => {
            format!("already sent {MAX_MEDIA_PER_REPLY} files in this reply; no more can be sent")
        }
    }
}

pub trait MediaFetcher {
    fn fetch(&self, url: &str) -> Result<PublicFile, PublicFetchError>;
}

/// Downloads through the public-address checks of `web_fetch`.
pub struct PublicMediaFetcher<Transport, Resolver> {
    transport: Transport,
    resolver: Resolver,
}

impl<Transport, Resolver> PublicMediaFetcher<Transport, Resolver> {
    #[must_use]
    pub const fn new(transport: Transport, resolver: Resolver) -> Self {
        Self {
            transport,
            resolver,
        }
    }
}

impl<Transport: WebFetchTransport, Resolver: HostResolver> MediaFetcher
    for PublicMediaFetcher<Transport, Resolver>
{
    fn fetch(&self, url: &str) -> Result<PublicFile, PublicFetchError> {
        fetch_public_media(&self.transport, &self.resolver, url)
    }
}

pub trait MediaSender {
    fn send(&mut self, action: TelegramAction) -> Result<ActionOutcome, ActionError>;
}

pub struct TelegramMediaSender<Transport = ReqwestTelegramTransport> {
    transport: Transport,
    token: String,
}

impl TelegramMediaSender {
    pub fn new(token: &str) -> Result<Self, TransportFailureKind> {
        Ok(Self::with_transport(
            ReqwestTelegramTransport::new()?,
            token,
        ))
    }
}

impl<Transport> TelegramMediaSender<Transport> {
    #[must_use]
    pub fn with_transport(transport: Transport, token: &str) -> Self {
        Self {
            transport,
            token: token.to_owned(),
        }
    }
}

impl<Transport: TelegramTransport> MediaSender for TelegramMediaSender<Transport> {
    fn send(&mut self, action: TelegramAction) -> Result<ActionOutcome, ActionError> {
        execute_with(&self.transport, &self.token, action)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MediaToolContext {
    pub chat_id: i64,
    pub reply_to_message_id: Option<i64>,
    pub locale: Locale,
}

/// Handles both `send_media` and `page_screenshot`.
pub struct MediaTool<Fetcher, Sender> {
    fetcher: Fetcher,
    capturer: Option<SharedCapturer>,
    sender: Sender,
    context: MediaToolContext,
    budget: MediaBudget,
}

impl<Fetcher, Sender> MediaTool<Fetcher, Sender> {
    #[must_use]
    pub const fn new(
        fetcher: Fetcher,
        capturer: Option<SharedCapturer>,
        sender: Sender,
        context: MediaToolContext,
        budget: MediaBudget,
    ) -> Self {
        Self {
            fetcher,
            capturer,
            sender,
            context,
            budget,
        }
    }
}

impl<Fetcher: MediaFetcher, Sender: MediaSender> MediaTool<Fetcher, Sender> {
    fn send_link(&mut self, url: &str, caption: &str) -> ToolExecutionResult {
        let locale = self.context.locale;
        let file = match self.fetcher.fetch(url) {
            Ok(file) => file,
            Err(error) => {
                let detail = match &error {
                    PublicFetchError::Blocked { .. } => None,
                    PublicFetchError::Request { detail, .. } => Some(detail.as_str()),
                };
                return ToolExecutionResult::with_diagnostics(
                    detail.map_or_else(
                        || error.public_message(locale).to_owned(),
                        |detail| format!("{}: {detail}", error.public_message(locale)),
                    ),
                    vec![format!(
                        "send_media fetch of {} failed: {error:?}",
                        error.url()
                    )],
                );
            }
        };
        let path = url::Url::parse(&file.url)
            .map(|url| url.path().to_owned())
            .unwrap_or_default();
        match classify_media(&file.content_type, &file.body, &path) {
            Ok(media) => deliver(
                &mut self.sender,
                self.context,
                &self.budget,
                Delivery {
                    tool: "send_media",
                    file: file.body.into(),
                    media,
                    caption,
                },
            ),
            Err(rejection) => ToolExecutionResult::output(rejection_message(locale, rejection)),
        }
    }

    fn send_screenshot(
        &mut self,
        url: &str,
        full_page: bool,
        caption: &str,
    ) -> ToolExecutionResult {
        let locale = self.context.locale;
        let Some(capturer) = self.capturer.as_ref() else {
            return ToolExecutionResult::output(tool_output::unavailable(
                locale,
                "page_screenshot",
            ));
        };
        match capturer.screenshot(url, full_page) {
            Ok(image) => {
                let media = classify_media("image/jpeg", &image, "/screenshot.jpg");
                match media {
                    Ok(media) => deliver(
                        &mut self.sender,
                        self.context,
                        &self.budget,
                        Delivery {
                            tool: "page_screenshot",
                            file: image.into(),
                            media,
                            caption,
                        },
                    ),
                    Err(rejection) => {
                        ToolExecutionResult::output(rejection_message(locale, rejection))
                    }
                }
            }
            Err(error) => ToolExecutionResult::with_diagnostics(
                match locale {
                    Locale::Es => "no pude sacar la captura de esa página",
                    Locale::En => "could not take a screenshot of that page",
                },
                vec![format!("page_screenshot of {url} failed: {error}")],
            ),
        }
    }
}

/// One file for [`deliver`], and the tool that sends it.
pub(crate) struct Delivery<'a> {
    pub tool: &'a str,
    pub file: Arc<[u8]>,
    pub media: MediaFile,
    pub caption: &'a str,
}

/// Sends the file as its kind. Telegram refuses some photos and videos
/// (odd sizes or encodings) that it takes as plain documents, so a
/// refused media file is sent again as a document.
pub(crate) fn deliver<Sender: MediaSender>(
    sender: &mut Sender,
    context: MediaToolContext,
    budget: &MediaBudget,
    delivery: Delivery<'_>,
) -> ToolExecutionResult {
    let Delivery {
        tool,
        file,
        media,
        caption,
    } = delivery;
    let locale = context.locale;
    if !budget.take() {
        return ToolExecutionResult::output(budget_spent_message(locale));
    }
    let mut kind = media.kind;
    let mut diagnostics = Vec::new();
    loop {
        let action = TelegramAction::SendMedia {
            chat_id: ChatId(context.chat_id),
            kind,
            file: Arc::clone(&file),
            file_name: media.file_name.clone(),
            content_type: media.content_type.clone(),
            reply_to_message_id: context.reply_to_message_id.map(MessageId),
            caption: caption.to_owned(),
        };
        match sender.send(action) {
            Ok(ActionOutcome::Completed { .. }) => {
                let mut result =
                    ToolExecutionResult::with_diagnostics(sent_message(locale, kind), diagnostics);
                result.failure_fallback = Some(sent_fallback(locale, kind).to_owned());
                return result;
            }
            Ok(ActionOutcome::Failed {
                status_code: Some(400),
                description,
            }) if kind != MediaKind::Document => {
                diagnostics.push(format!(
                    "{} refused, sending as a document: {description}",
                    kind.method()
                ));
                kind = MediaKind::Document;
            }
            // Telegram may have posted the file before the reply went
            // wrong, so the slot stays used and the model must not retry.
            reply @ (Ok(
                ActionOutcome::TransportFailed(_)
                | ActionOutcome::Failed {
                    status_code: Some(500..=599),
                    ..
                },
            )
            | Err(ActionError::InvalidResponse)) => {
                diagnostics.push(format!("{} result unknown: {reply:?}", kind.method()));
                return ToolExecutionResult::with_diagnostics(
                    match locale {
                        Locale::Es => {
                            "no sé si el archivo llegó al chat; no lo mandes de nuevo sin preguntar"
                        }
                        Locale::En => {
                            "I do not know if the file reached the chat; do not send it again without asking"
                        }
                    },
                    diagnostics,
                );
            }
            Ok(ActionOutcome::Failed { description, .. }) => {
                budget.refund();
                return ToolExecutionResult::with_diagnostics(
                    match locale {
                        Locale::Es => format!("Telegram no aceptó el archivo: {description}"),
                        Locale::En => {
                            format!("Telegram did not accept the file: {description}")
                        }
                    },
                    diagnostics,
                );
            }
            Ok(ActionOutcome::RateLimited {
                retry_after_seconds,
            }) => {
                budget.refund();
                diagnostics.push(format!(
                    "{} rate limited; retry after {retry_after_seconds:?} seconds",
                    kind.method()
                ));
                return ToolExecutionResult::with_diagnostics(
                    match locale {
                        Locale::Es => {
                            "Telegram está limitando los envíos; no mandes más archivos ahora"
                        }
                        Locale::En => "Telegram is limiting sends; do not send more files now",
                    },
                    diagnostics,
                );
            }
            Err(error) => {
                budget.refund();
                diagnostics.push(format!("{} failed: {error}", kind.method()));
                return ToolExecutionResult::with_diagnostics(
                    tool_output::failed(locale, tool),
                    diagnostics,
                );
            }
        }
    }
}

const fn kind_name(locale: Locale, kind: MediaKind) -> &'static str {
    match (locale, kind) {
        (Locale::Es, MediaKind::Photo) => "la imagen",
        (Locale::Es, MediaKind::Animation) => "el GIF",
        (Locale::Es, MediaKind::Video) => "el video",
        (Locale::Es, MediaKind::Audio) => "el audio",
        (Locale::Es, MediaKind::Document) => "el archivo",
        (Locale::En, MediaKind::Photo) => "the image",
        (Locale::En, MediaKind::Animation) => "the GIF",
        (Locale::En, MediaKind::Video) => "the video",
        (Locale::En, MediaKind::Audio) => "the audio",
        (Locale::En, MediaKind::Document) => "the file",
    }
}

fn sent_message(locale: Locale, kind: MediaKind) -> String {
    let name = kind_name(locale, kind);
    match locale {
        Locale::Es => format!(
            "Mandé {name} al chat. Ya se ve: no pegues el link ni lo describas de nuevo salvo que te lo pidan"
        ),
        Locale::En => format!(
            "Sent {name} to the chat. People can see it: do not paste the link or describe it again unless asked"
        ),
    }
}

/// The reply shown when the AI fails after the file was sent.
const fn sent_fallback(locale: Locale, kind: MediaKind) -> &'static str {
    match (locale, kind) {
        (Locale::Es, MediaKind::Photo) => "Ahí va la imagen.",
        (Locale::Es, MediaKind::Animation) => "Ahí va el GIF.",
        (Locale::Es, MediaKind::Video) => "Ahí va el video.",
        (Locale::Es, MediaKind::Audio) => "Ahí va el audio.",
        (Locale::Es, MediaKind::Document) => "Ahí va el archivo.",
        (Locale::En, MediaKind::Photo) => "Here is the image.",
        (Locale::En, MediaKind::Animation) => "Here is the GIF.",
        (Locale::En, MediaKind::Video) => "Here is the video.",
        (Locale::En, MediaKind::Audio) => "Here is the audio.",
        (Locale::En, MediaKind::Document) => "Here is the file.",
    }
}

pub(crate) const fn rejection_message(locale: Locale, rejection: MediaRejection) -> &'static str {
    match (locale, rejection) {
        (Locale::Es, MediaRejection::WebPage) => {
            "ese link es una página sin imagen principal; usá page_screenshot para mostrarla o buscá el link directo al archivo"
        }
        (Locale::En, MediaRejection::WebPage) => {
            "that link is a web page with no main image; use page_screenshot to show it or find a direct link to the file"
        }
        (Locale::Es, MediaRejection::Empty) => "el archivo está vacío",
        (Locale::En, MediaRejection::Empty) => "the file is empty",
        (Locale::Es, MediaRejection::TooLarge) => "el archivo pasa el límite de 50 MB de Telegram",
        (Locale::En, MediaRejection::TooLarge) => "the file is over Telegram's 50 MB limit",
    }
}

impl<Fetcher: MediaFetcher, Sender: MediaSender> ExternalToolExecutor
    for MediaTool<Fetcher, Sender>
{
    fn execute(
        &mut self,
        request: ExternalToolRequest,
        _tool_call_id: &str,
    ) -> ToolExecutionResult {
        match request {
            ExternalToolRequest::SendMedia { url, caption } => self.send_link(&url, &caption),
            ExternalToolRequest::PageScreenshot {
                url,
                full_page,
                caption,
            } => self.send_screenshot(&url, full_page, &caption),
            _ => ToolExecutionResult::output(tool_output::incompatible(
                self.context.locale,
                "send_media",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use bot_adapters::browser::RenderError;
    use bot_adapters::web_fetch::{WebFetchResponse, WebFetchTransportError};

    use super::*;

    const JPEG: &[u8] = b"\xFF\xD8\xFF\xE0synthetic";

    struct Fetcher(Result<PublicFile, PublicFetchError>);

    impl MediaFetcher for Fetcher {
        fn fetch(&self, _url: &str) -> Result<PublicFile, PublicFetchError> {
            self.0.clone()
        }
    }

    fn file(url: &str, content_type: &str, body: &[u8]) -> Fetcher {
        Fetcher(Ok(PublicFile {
            url: url.to_owned(),
            content_type: content_type.to_owned(),
            body: body.to_vec(),
        }))
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

    fn sender(replies: Vec<Result<ActionOutcome, ActionError>>) -> Sender {
        Sender {
            replies: replies.into(),
            sent: Vec::new(),
        }
    }

    struct Capturer {
        reply: Result<Vec<u8>, RenderError>,
        calls: Mutex<Vec<(String, bool)>>,
    }

    impl PageCapturer for Capturer {
        fn screenshot(&self, url: &str, full_page: bool) -> Result<Vec<u8>, RenderError> {
            let _ = self
                .calls
                .lock()
                .map(|mut calls| calls.push((url.to_owned(), full_page)));
            self.reply.clone()
        }
    }

    fn media_tool(
        fetcher: Fetcher,
        capturer: Option<SharedCapturer>,
        sender: Sender,
        locale: Locale,
    ) -> MediaTool<Fetcher, Sender> {
        MediaTool::new(
            fetcher,
            capturer,
            sender,
            MediaToolContext {
                chat_id: -100,
                reply_to_message_id: Some(7),
                locale,
            },
            MediaBudget::default(),
        )
    }

    fn send_media(url: &str, caption: &str) -> ExternalToolRequest {
        ExternalToolRequest::SendMedia {
            url: url.to_owned(),
            caption: caption.to_owned(),
        }
    }

    fn sent_kinds(tool: &MediaTool<Fetcher, Sender>) -> Vec<(MediaKind, String, String)> {
        tool.sender
            .sent
            .iter()
            .filter_map(|action| match action {
                TelegramAction::SendMedia {
                    kind,
                    file_name,
                    content_type,
                    ..
                } => Some((*kind, file_name.clone(), content_type.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_linked_image_is_sent_as_a_photo_reply_with_its_caption() {
        let mut tool = media_tool(
            file("https://cdn.example/cats/cat.jpg", "image/jpeg", JPEG),
            None,
            Sender::default(),
            Locale::En,
        );
        let result = tool.execute(send_media("https://example.com/cat", "A cat"), "call");
        assert_eq!(
            result.output,
            "Sent the image to the chat. People can see it: do not paste the link or describe it again unless asked"
        );
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("Here is the image.")
        );
        assert_eq!(
            tool.sender.sent,
            [TelegramAction::SendMedia {
                chat_id: ChatId(-100),
                kind: MediaKind::Photo,
                file: Arc::from(JPEG),
                file_name: "cat.jpg".to_owned(),
                content_type: "image/jpeg".to_owned(),
                reply_to_message_id: Some(MessageId(7)),
                caption: "A cat".to_owned(),
            }]
        );
    }

    #[test]
    fn a_refused_media_file_is_sent_again_as_a_document() {
        let refused = || {
            Ok(ActionOutcome::Failed {
                status_code: Some(400),
                description: "PHOTO_INVALID_DIMENSIONS".to_owned(),
            })
        };
        let mut tool = media_tool(
            file(
                "https://cdn.example/tall.png",
                "image/png",
                b"\x89PNG\r\n\x1A\nx",
            ),
            None,
            sender(vec![refused()]),
            Locale::Es,
        );
        let result = tool.execute(send_media("https://cdn.example/tall.png", ""), "call");
        assert_eq!(
            sent_kinds(&tool),
            [
                (
                    MediaKind::Photo,
                    "tall.png".to_owned(),
                    "image/png".to_owned()
                ),
                (
                    MediaKind::Document,
                    "tall.png".to_owned(),
                    "image/png".to_owned()
                ),
            ]
        );
        assert!(result.output.starts_with("Mandé el archivo al chat."));
        assert_eq!(
            result.failure_fallback.as_deref(),
            Some("Ahí va el archivo.")
        );
        assert_eq!(
            result.diagnostics,
            ["sendPhoto refused, sending as a document: PHOTO_INVALID_DIMENSIONS"]
        );

        // A document that Telegram refuses is not retried.
        let mut tool = media_tool(
            file("https://cdn.example/a.pdf", "application/pdf", b"%PDF-1.7"),
            None,
            sender(vec![refused()]),
            Locale::En,
        );
        let result = tool.execute(send_media("https://cdn.example/a.pdf", ""), "call");
        assert_eq!(
            result.output,
            "Telegram did not accept the file: PHOTO_INVALID_DIMENSIONS"
        );
        assert_eq!(tool.sender.sent.len(), 1);
    }

    #[test]
    fn send_failures_tell_the_model_what_happened() {
        let cases = [
            (
                Ok(ActionOutcome::Failed {
                    status_code: Some(403),
                    description: "bot was kicked".to_owned(),
                }),
                Locale::Es,
                "Telegram no aceptó el archivo: bot was kicked",
            ),
            (
                Ok(ActionOutcome::RateLimited {
                    retry_after_seconds: Some(3),
                }),
                Locale::Es,
                "Telegram está limitando los envíos; no mandes más archivos ahora",
            ),
            (
                Ok(ActionOutcome::RateLimited {
                    retry_after_seconds: None,
                }),
                Locale::En,
                "Telegram is limiting sends; do not send more files now",
            ),
            (
                Ok(ActionOutcome::TransportFailed(
                    TransportFailureKind::Timeout,
                )),
                Locale::Es,
                "no sé si el archivo llegó al chat; no lo mandes de nuevo sin preguntar",
            ),
            (
                Ok(ActionOutcome::TransportFailed(
                    TransportFailureKind::Timeout,
                )),
                Locale::En,
                "I do not know if the file reached the chat; do not send it again without asking",
            ),
            (
                Ok(ActionOutcome::Failed {
                    status_code: Some(502),
                    description: "Bad Gateway".to_owned(),
                }),
                Locale::En,
                "I do not know if the file reached the chat; do not send it again without asking",
            ),
            (
                Err(ActionError::InvalidResponse),
                Locale::En,
                "I do not know if the file reached the chat; do not send it again without asking",
            ),
            (
                Err(ActionError::InvalidAction),
                Locale::En,
                "tool 'send_media' failed",
            ),
        ];
        for (reply, locale, output) in cases {
            let mut tool = media_tool(
                file("https://cdn.example/a.gif", "image/gif", b"GIF89a"),
                None,
                sender(vec![reply]),
                locale,
            );
            let result = tool.execute(send_media("https://cdn.example/a.gif", ""), "call");
            assert_eq!(result.output, output);
            assert_eq!(result.failure_fallback, None);
            assert_eq!(
                result.diagnostics.len(),
                usize::from(!output.starts_with("Telegram no"))
            );
        }
    }

    #[test]
    fn download_problems_and_pages_without_a_picture_send_nothing() {
        let cases = [
            (
                Fetcher(Err(PublicFetchError::Blocked {
                    url: "http://10.0.0.1/".to_owned(),
                })),
                Locale::En,
                "URL is not allowed",
            ),
            (
                Fetcher(Err(PublicFetchError::Request {
                    url: "https://cdn.example/huge.mp4".to_owned(),
                    detail: "file is over the upload limit".to_owned(),
                })),
                Locale::Es,
                "no se pudo obtener la URL: file is over the upload limit",
            ),
            (
                file("https://example.com/", "text/html", b"<html>"),
                Locale::Es,
                "ese link es una página sin imagen principal; usá page_screenshot para mostrarla o buscá el link directo al archivo",
            ),
            (
                file("https://example.com/", "text/html", b"<html>"),
                Locale::En,
                "that link is a web page with no main image; use page_screenshot to show it or find a direct link to the file",
            ),
            (
                file("https://cdn.example/a", "image/png", b""),
                Locale::Es,
                "el archivo está vacío",
            ),
            (
                file("https://cdn.example/a", "image/png", b""),
                Locale::En,
                "the file is empty",
            ),
            (
                file("not a url", "image/png", b""),
                Locale::En,
                "the file is empty",
            ),
        ];
        for (fetcher, locale, output) in cases {
            let mut tool = media_tool(fetcher, None, Sender::default(), locale);
            let result = tool.execute(send_media("https://example.com/", ""), "call");
            assert_eq!(result.output, output);
            assert!(tool.sender.sent.is_empty());
        }
        assert_eq!(
            rejection_message(Locale::Es, MediaRejection::TooLarge),
            "el archivo pasa el límite de 50 MB de Telegram"
        );
        assert_eq!(
            rejection_message(Locale::En, MediaRejection::TooLarge),
            "the file is over Telegram's 50 MB limit"
        );
    }

    #[test]
    fn each_kind_has_its_own_words() {
        let kinds = [
            MediaKind::Photo,
            MediaKind::Animation,
            MediaKind::Video,
            MediaKind::Audio,
            MediaKind::Document,
        ];
        assert_eq!(
            kinds.map(|kind| sent_fallback(Locale::Es, kind)),
            [
                "Ahí va la imagen.",
                "Ahí va el GIF.",
                "Ahí va el video.",
                "Ahí va el audio.",
                "Ahí va el archivo."
            ]
        );
        assert_eq!(
            kinds.map(|kind| sent_fallback(Locale::En, kind)),
            [
                "Here is the image.",
                "Here is the GIF.",
                "Here is the video.",
                "Here is the audio.",
                "Here is the file."
            ]
        );
        assert_eq!(
            kinds.map(|kind| kind_name(Locale::Es, kind)),
            ["la imagen", "el GIF", "el video", "el audio", "el archivo"]
        );
        assert_eq!(
            kinds.map(|kind| kind_name(Locale::En, kind)),
            ["the image", "the GIF", "the video", "the audio", "the file"]
        );
    }

    #[test]
    fn one_reply_sends_at_most_ten_files_across_both_tools() {
        let budget = MediaBudget::default();
        let context = MediaToolContext {
            chat_id: 1,
            reply_to_message_id: None,
            locale: Locale::En,
        };
        let capturer: SharedCapturer = Arc::new(Capturer {
            reply: Ok(JPEG.to_vec()),
            calls: Mutex::new(Vec::new()),
        });
        let mut links = MediaTool::new(
            file("https://cdn.example/a.jpg", "image/jpeg", JPEG),
            None,
            Sender::default(),
            context,
            budget.clone(),
        );
        let mut screenshots = MediaTool::new(
            file("https://unused.example/", "", b""),
            Some(capturer),
            Sender::default(),
            context,
            budget.clone(),
        );
        // A failed send does not use up the budget.
        links
            .sender
            .replies
            .push_back(Err(ActionError::InvalidAction));
        links.execute(send_media("https://cdn.example/a.jpg", ""), "call");
        for _ in 0..5 {
            assert!(!budget.spent());
            links.execute(send_media("https://cdn.example/a.jpg", ""), "call");
            screenshots.execute(
                ExternalToolRequest::PageScreenshot {
                    url: "https://example.com/".to_owned(),
                    full_page: false,
                    caption: String::new(),
                },
                "call",
            );
        }
        assert!(budget.spent());
        let refused = links.execute(send_media("https://cdn.example/a.jpg", ""), "call");
        assert_eq!(
            refused.output,
            "already sent 10 files in this reply; no more can be sent"
        );
        assert_eq!(links.sender.sent.len() + screenshots.sender.sent.len(), 11);
        links.context.locale = Locale::Es;
        assert_eq!(
            links
                .execute(send_media("https://cdn.example/a.jpg", ""), "call")
                .output,
            "ya mandé 10 archivos en esta respuesta, no mando más"
        );
    }

    #[test]
    fn screenshots_are_sent_as_photos_and_report_browser_failures() {
        let capturer = Arc::new(Capturer {
            reply: Ok(JPEG.to_vec()),
            calls: Mutex::new(Vec::new()),
        });
        let mut tool = media_tool(
            file("https://unused.example/", "", b""),
            Some(Arc::clone(&capturer) as SharedCapturer),
            Sender::default(),
            Locale::En,
        );
        let request = |full_page| ExternalToolRequest::PageScreenshot {
            url: "https://example.com/".to_owned(),
            full_page,
            caption: "The site".to_owned(),
        };
        let result = tool.execute(request(true), "call");
        assert!(result.output.starts_with("Sent the image to the chat."));
        // Only file sends are listed.
        tool.sender
            .sent
            .push(TelegramAction::SendTyping { chat_id: ChatId(1) });
        assert_eq!(
            sent_kinds(&tool),
            [(
                MediaKind::Photo,
                "screenshot.jpg".to_owned(),
                "image/jpeg".to_owned()
            )]
        );
        assert_eq!(
            capturer
                .calls
                .lock()
                .map(|calls| calls.clone())
                .unwrap_or_default(),
            [("https://example.com/".to_owned(), true)]
        );

        for (reply, locale, output) in [
            (
                Err(RenderError::Failed("navigation failed".to_owned())),
                Locale::Es,
                "no pude sacar la captura de esa página",
            ),
            (
                Err(RenderError::Start("missing".to_owned())),
                Locale::En,
                "could not take a screenshot of that page",
            ),
            (Ok(Vec::new()), Locale::En, "the file is empty"),
        ] {
            let mut tool = media_tool(
                file("https://unused.example/", "", b""),
                Some(Arc::new(Capturer {
                    reply,
                    calls: Mutex::new(Vec::new()),
                })),
                Sender::default(),
                locale,
            );
            assert_eq!(tool.execute(request(false), "call").output, output);
            assert!(tool.sender.sent.is_empty());
        }

        let mut without_browser = media_tool(
            file("https://unused.example/", "", b""),
            None,
            Sender::default(),
            Locale::En,
        );
        assert_eq!(
            without_browser.execute(request(false), "call").output,
            "tool 'page_screenshot' is unavailable"
        );
        assert_eq!(
            without_browser
                .execute(ExternalToolRequest::GetPolls, "call")
                .output,
            "tool 'send_media' received an incompatible request"
        );
    }

    struct Transport {
        responses: RefCell<VecDeque<WebFetchResponse>>,
    }

    impl WebFetchTransport for Transport {
        fn get(&self, _url: &str) -> Result<WebFetchResponse, WebFetchTransportError> {
            self.responses
                .borrow_mut()
                .pop_front()
                .ok_or(WebFetchTransportError::Connection)
        }
    }

    struct Resolver;

    impl HostResolver for Resolver {
        fn addresses(&self, _hostname: &str, _port: u16) -> Result<Vec<std::net::IpAddr>, String> {
            Ok(vec![std::net::IpAddr::from([93, 184, 216, 34])])
        }
    }

    #[test]
    fn the_public_fetcher_downloads_through_the_web_fetch_checks() {
        let fetcher = PublicMediaFetcher::new(
            Transport {
                responses: RefCell::new(VecDeque::from([WebFetchResponse {
                    status_code: 200,
                    content_type: "image/gif".to_owned(),
                    location: None,
                    body: b"GIF89a".to_vec(),
                    truncated: false,
                }])),
            },
            Resolver,
        );
        assert_eq!(
            fetcher.fetch("https://example.com/a.gif"),
            Ok(PublicFile {
                url: "https://example.com/a.gif".to_owned(),
                content_type: "image/gif".to_owned(),
                body: b"GIF89a".to_vec(),
            })
        );
        assert!(matches!(
            fetcher.fetch("http://localhost/a.gif"),
            Err(PublicFetchError::Blocked { .. })
        ));
    }

    struct Telegram {
        requests: RefCell<Vec<bot_adapters::telegram_http::TelegramMultipartRequest>>,
    }

    impl TelegramTransport for Telegram {
        fn send(
            &self,
            _request: &bot_adapters::telegram_http::TelegramRequest,
        ) -> Result<bot_adapters::telegram_http::HttpResponse, TransportFailureKind> {
            Err(TransportFailureKind::Connection)
        }

        fn send_action_multipart(
            &self,
            request: &bot_adapters::telegram_http::TelegramMultipartRequest,
        ) -> Result<bot_adapters::telegram_http::HttpResponse, TransportFailureKind> {
            self.requests.borrow_mut().push(request.clone());
            Ok(bot_adapters::telegram_http::HttpResponse {
                status_code: 200,
                body: r#"{"ok":true,"result":{"message_id":5}}"#.to_owned(),
            })
        }
    }

    #[test]
    fn the_telegram_sender_uploads_with_the_bot_token() {
        let mut sender = TelegramMediaSender::with_transport(
            Telegram {
                requests: RefCell::new(Vec::new()),
            },
            "synthetic-token",
        );
        let outcome = sender.send(TelegramAction::SendMedia {
            chat_id: ChatId(1),
            kind: MediaKind::Audio,
            file: Arc::from(b"ID3".as_slice()),
            file_name: "song.mp3".to_owned(),
            content_type: "audio/mpeg".to_owned(),
            reply_to_message_id: None,
            caption: String::new(),
        });
        assert_eq!(
            outcome,
            Ok(ActionOutcome::Completed {
                message_id: Some(5)
            })
        );
        // Actions without a file go through the plain request path.
        assert_eq!(
            sender.send(TelegramAction::SendTyping { chat_id: ChatId(1) }),
            Ok(ActionOutcome::TransportFailed(
                TransportFailureKind::Connection
            ))
        );
        let requests = sender.transport.requests.borrow();
        assert_eq!(requests[0].token, "synthetic-token");
        assert_eq!(requests[0].endpoint, "sendAudio");
        assert!(TelegramMediaSender::new("synthetic-token").is_ok());
    }
}
