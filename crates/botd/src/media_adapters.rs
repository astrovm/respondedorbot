//! Production adapters for the native media pipeline.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use bot_adapters::media_provider::{
    GroqTranscriptionTransport, MediaProviderError, MediaProviderResult,
    ReqwestGroqTranscriptionTransport, VisionRequest, describe_image_with,
    transcribe_audio_groq_with, transcribe_audio_openrouter_with,
};
use bot_adapters::openrouter_chat::{OpenRouterTransport, ReqwestOpenRouterTransport};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_media_cache::{cache_media, get_cached_media};
use bot_adapters::telegram_http::{
    TELEGRAM_FILE_MAX_BYTES, TelegramFileOutcome, TelegramFileTransport, TelegramHttpOutcome,
    TelegramTransport, download_file_with, request_with,
};
use bot_core::provider_errors::{ProviderErrorFacts, classify_provider_error};
use serde_json::{Value, json};

use crate::media::{
    MediaCache, MediaFileSource, MediaProcessor, PreparedAudio, PreparedImage,
    TranscriptionProvider, VisionProvider,
};

const MEDIA_CACHE_TTL_SECONDS: i64 = 7 * 24 * 60 * 60;
const TELEGRAM_FILE_TIMEOUT_SECONDS: u64 = 30;
const MEDIA_PROCESS_TIMEOUT: Duration = Duration::from_secs(60);
const MEDIA_PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const MEDIA_PROCESS_OUTPUT_MAX_BYTES: u64 = 20_000_000;

pub struct TelegramMediaFiles<Transport> {
    transport: Transport,
    token: String,
}

impl<Transport> TelegramMediaFiles<Transport> {
    #[must_use]
    pub fn new(transport: Transport, token: &str) -> Self {
        Self {
            transport,
            token: token.to_owned(),
        }
    }
}

impl<Transport> MediaFileSource for TelegramMediaFiles<Transport>
where
    Transport: TelegramTransport + TelegramFileTransport,
{
    fn download(&mut self, file_id: &str) -> Result<Option<Vec<u8>>, String> {
        let outcome = request_with(
            &self.transport,
            &self.token,
            "getFile",
            "GET",
            Some(json!({"file_id": file_id})),
            None,
            TELEGRAM_FILE_TIMEOUT_SECONDS,
        )
        .map_err(|error| error.to_string())?;
        let TelegramHttpOutcome::Response { status_code, body } = outcome else {
            return Ok(None);
        };
        if !(200..300).contains(&status_code) {
            return Ok(None);
        }
        let payload = serde_json::from_str::<Value>(&body).map_err(|error| error.to_string())?;
        let file_path = payload
            .get("result")
            .and_then(|result| result.get("file_path"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let Some(file_path) = file_path else {
            return Ok(None);
        };
        match download_file_with(
            &self.transport,
            &self.token,
            file_path,
            TELEGRAM_FILE_TIMEOUT_SECONDS,
        )
        .map_err(|error| error.to_string())?
        {
            TelegramFileOutcome::Downloaded(bytes) => Ok(Some(bytes)),
            TelegramFileOutcome::HttpError { .. } | TelegramFileOutcome::TransportError { .. } => {
                Ok(None)
            }
        }
    }
}

pub struct RedisMediaCache {
    endpoint: RedisEndpoint,
}

impl RedisMediaCache {
    #[must_use]
    pub const fn new(endpoint: RedisEndpoint) -> Self {
        Self { endpoint }
    }
}

impl MediaCache for RedisMediaCache {
    fn get(&mut self, prefix: &str, file_id: &str) -> Result<Option<String>, String> {
        get_cached_media(&self.endpoint, prefix, file_id).map_err(|error| error.to_string())
    }

    fn set(&mut self, prefix: &str, file_id: &str, text: &str) -> Result<(), String> {
        cache_media(
            &self.endpoint,
            prefix,
            file_id,
            text,
            MEDIA_CACHE_TTL_SECONDS,
        )
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct FfmpegMediaProcessor {
    ffmpeg: String,
    ffprobe: String,
    max_image_size: u32,
}

impl Default for FfmpegMediaProcessor {
    fn default() -> Self {
        Self {
            ffmpeg: "ffmpeg".to_owned(),
            ffprobe: "ffprobe".to_owned(),
            max_image_size: 512,
        }
    }
}

impl FfmpegMediaProcessor {
    fn run(program: &str, arguments: &[String], input: &[u8]) -> Result<Vec<u8>, String> {
        Self::run_bounded(
            program,
            arguments,
            input,
            MEDIA_PROCESS_TIMEOUT,
            TELEGRAM_FILE_MAX_BYTES,
            MEDIA_PROCESS_OUTPUT_MAX_BYTES,
        )
    }

    fn run_bounded(
        program: &str,
        arguments: &[String],
        input: &[u8],
        timeout: Duration,
        max_input_bytes: u64,
        max_output_bytes: u64,
    ) -> Result<Vec<u8>, String> {
        if input.len() as u64 > max_input_bytes {
            return Err("media input exceeds the size limit".to_owned());
        }
        if timeout.is_zero() {
            return Err("media process timeout must be positive".to_owned());
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| "media process timeout is too large".to_owned())?;
        let mut child = Command::new(program)
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| error.to_string())?;
        let Some(mut stdin) = child.stdin.take() else {
            Self::terminate(&mut child);
            return Err("media process did not expose stdin".to_owned());
        };
        let Some(stdout) = child.stdout.take() else {
            Self::terminate(&mut child);
            return Err("media process did not expose stdout".to_owned());
        };
        thread::scope(|scope| {
            let writer = scope.spawn(move || {
                let result = stdin.write_all(input).map_err(|error| error.to_string());
                drop(stdin);
                result
            });
            let reader = scope.spawn(move || {
                let mut output = Vec::new();
                stdout
                    .take(max_output_bytes.saturating_add(1))
                    .read_to_end(&mut output)
                    .map_err(|error| error.to_string())?;
                if output.len() as u64 > max_output_bytes {
                    return Err("media process output exceeds the size limit".to_owned());
                }
                Ok(output)
            });

            let status = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) if Instant::now() < deadline => {
                        thread::sleep(MEDIA_PROCESS_POLL_INTERVAL);
                    }
                    Ok(None) => {
                        Self::terminate(&mut child);
                        break Err(format!("{program} timed out while processing media"));
                    }
                    Err(error) => {
                        Self::terminate(&mut child);
                        break Err(error.to_string());
                    }
                }
            };

            let write_result = writer
                .join()
                .map_err(|_| "media input writer panicked".to_owned())?;
            let output = reader
                .join()
                .map_err(|_| "media output reader panicked".to_owned())??;
            let status = status?;
            write_result?;
            if status.success() && !output.is_empty() {
                Ok(output)
            } else {
                Err(format!("{program} could not process media"))
            }
        })
    }

    fn terminate(child: &mut std::process::Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    fn duration(&self, input: &[u8]) -> Option<f64> {
        let format_duration = Self::run(
            &self.ffprobe,
            &[
                "-v".to_owned(),
                "error".to_owned(),
                "-show_entries".to_owned(),
                "format=duration".to_owned(),
                "-of".to_owned(),
                "default=noprint_wrappers=1:nokey=1".to_owned(),
                "pipe:0".to_owned(),
            ],
            input,
        )
        .ok()
        .and_then(|output| String::from_utf8(output).ok())
        .and_then(|output| output.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
        if format_duration.is_some() {
            return format_duration;
        }
        let packet_durations = Self::run(
            &self.ffprobe,
            &[
                "-v".to_owned(),
                "error".to_owned(),
                "-select_streams".to_owned(),
                "a:0".to_owned(),
                "-show_entries".to_owned(),
                "packet=duration_time".to_owned(),
                "-of".to_owned(),
                "csv=p=0".to_owned(),
                "pipe:0".to_owned(),
            ],
            input,
        )
        .ok()?;
        let duration = String::from_utf8(packet_durations)
            .ok()?
            .lines()
            .filter_map(|line| line.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        (duration.is_finite() && duration > 0.0).then_some(duration)
    }

    fn prepare_audio_bounded(
        &self,
        input: &[u8],
        duration_hint_seconds: Option<f64>,
        max_input_bytes: u64,
    ) -> Result<Option<PreparedAudio>, String> {
        let extracted = Self::run_bounded(
            &self.ffmpeg,
            &[
                "-loglevel".to_owned(),
                "error".to_owned(),
                "-i".to_owned(),
                "pipe:0".to_owned(),
                "-vn".to_owned(),
                "-ac".to_owned(),
                "1".to_owned(),
                "-c:a".to_owned(),
                "libopus".to_owned(),
                "-f".to_owned(),
                "webm".to_owned(),
                "pipe:1".to_owned(),
            ],
            input,
            MEDIA_PROCESS_TIMEOUT,
            max_input_bytes,
            MEDIA_PROCESS_OUTPUT_MAX_BYTES,
        )
        .unwrap_or_else(|_| input.to_vec());
        let hinted = duration_hint_seconds.filter(|value| value.is_finite() && *value > 0.0);
        let duration_seconds = hinted
            .or_else(|| self.duration(&extracted))
            .or_else(|| self.duration(input));
        Ok(duration_seconds.map(|duration_seconds| PreparedAudio {
            bytes: extracted,
            duration_seconds,
        }))
    }
}

impl MediaProcessor for FfmpegMediaProcessor {
    fn prepare_image(&mut self, input: &[u8]) -> Result<Option<PreparedImage>, String> {
        let size = self.max_image_size;
        let filter = format!(
            "thumbnail=30,scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease"
        );
        let output = Self::run(
            &self.ffmpeg,
            &[
                "-loglevel".to_owned(),
                "error".to_owned(),
                "-i".to_owned(),
                "pipe:0".to_owned(),
                "-vf".to_owned(),
                filter,
                "-f".to_owned(),
                "image2pipe".to_owned(),
                "-vcodec".to_owned(),
                "webp".to_owned(),
                "-frames:v".to_owned(),
                "1".to_owned(),
                "pipe:1".to_owned(),
            ],
            input,
        );
        Ok(output.ok().map(|bytes| PreparedImage {
            bytes,
            mime: "image/webp".to_owned(),
        }))
    }

    fn prepare_audio(
        &mut self,
        input: &[u8],
        duration_hint_seconds: Option<f64>,
    ) -> Result<Option<PreparedAudio>, String> {
        self.prepare_audio_bounded(input, duration_hint_seconds, TELEGRAM_FILE_MAX_BYTES)
    }
}

pub struct OpenRouterVisionProvider<Transport> {
    transport: Transport,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u64,
}

impl<Transport> OpenRouterVisionProvider<Transport> {
    #[must_use]
    pub fn new(
        transport: Transport,
        api_key: &str,
        base_url: &str,
        model: &str,
        max_tokens: u64,
    ) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            model: model.to_owned(),
            max_tokens,
        }
    }
}

impl<Transport: OpenRouterTransport> VisionProvider for OpenRouterVisionProvider<Transport> {
    fn describe(
        &mut self,
        image: &PreparedImage,
        prompt: &str,
        file_id: &str,
    ) -> Result<Option<MediaProviderResult>, String> {
        let system_prompt = if prompt.is_ascii() {
            "respond in English without emojis or markdown."
        } else {
            "respondé siempre en minúsculas, sin emojis, sin markdown y en lenguaje coloquial argentino."
        };
        describe_image_with(
            &self.transport,
            VisionRequest {
                api_key: &self.api_key,
                base_url: &self.base_url,
                model: &self.model,
                system_prompt,
                user_prompt: prompt,
                image_bytes: &image.bytes,
                image_mime: &image.mime,
                max_tokens: self.max_tokens,
                file_id: Some(file_id),
            },
        )
        .map(Some)
        .map_err(|error| error.to_string())
    }
}

pub struct FallbackTranscriptionProvider<Groq, OpenRouter> {
    groq: Groq,
    openrouter: OpenRouter,
    groq_accounts: Vec<(String, String)>,
    openrouter_api_key: Option<String>,
    openrouter_base_url: String,
    groq_model: String,
    openrouter_model: String,
    default_backoff_seconds: u64,
    cooldowns: HashMap<String, Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionProviderConfig {
    pub groq_accounts: Vec<(String, String)>,
    pub openrouter_api_key: Option<String>,
    pub openrouter_base_url: String,
    pub groq_model: String,
    pub openrouter_model: String,
    pub default_backoff_seconds: u64,
}

impl<Groq, OpenRouter> FallbackTranscriptionProvider<Groq, OpenRouter> {
    #[must_use]
    pub fn new(groq: Groq, openrouter: OpenRouter, config: TranscriptionProviderConfig) -> Self {
        Self {
            groq,
            openrouter,
            groq_accounts: config.groq_accounts,
            openrouter_api_key: config.openrouter_api_key,
            openrouter_base_url: config.openrouter_base_url,
            groq_model: config.groq_model,
            openrouter_model: config.openrouter_model,
            default_backoff_seconds: config.default_backoff_seconds,
            cooldowns: HashMap::new(),
        }
    }

    fn cooling_down(&self, account: &str) -> bool {
        self.cooldowns
            .get(account)
            .is_some_and(|deadline| *deadline > Instant::now())
    }

    fn mark_cooldown(&mut self, account: &str, seconds: u64) {
        if let Some(deadline) = Instant::now().checked_add(Duration::from_secs(seconds)) {
            self.cooldowns.insert(account.to_owned(), deadline);
        }
    }
}

impl<Groq, OpenRouter> TranscriptionProvider for FallbackTranscriptionProvider<Groq, OpenRouter>
where
    Groq: GroqTranscriptionTransport,
    OpenRouter: OpenRouterTransport,
{
    fn transcribe(
        &mut self,
        audio: &PreparedAudio,
        file_id: &str,
    ) -> Result<Option<MediaProviderResult>, String> {
        for (account, api_key) in self.groq_accounts.clone() {
            if self.cooling_down(&account) {
                continue;
            }
            match transcribe_audio_groq_with(
                &self.groq,
                &api_key,
                &self.groq_model,
                &audio.bytes,
                Some(file_id),
                audio.duration_seconds,
                &account,
            ) {
                Ok(result) => return Ok(Some(result)),
                Err(error) => {
                    let policy = classify_media_error(&error);
                    if policy.rate_limited {
                        self.mark_cooldown(
                            &account,
                            error
                                .retry_after_seconds()
                                .unwrap_or(self.default_backoff_seconds),
                        );
                        continue;
                    }
                    if policy.try_next_groq_account {
                        continue;
                    }
                    break;
                }
            }
        }
        let Some(api_key) = self.openrouter_api_key.as_deref() else {
            return Ok(None);
        };
        transcribe_audio_openrouter_with(
            &self.openrouter,
            api_key,
            &self.openrouter_base_url,
            &self.openrouter_model,
            &audio.bytes,
            Some(file_id),
        )
        .map(Some)
        .map_err(|error| error.to_string())
    }
}

fn classify_media_error(
    error: &MediaProviderError,
) -> bot_core::provider_errors::ProviderErrorPolicy {
    classify_provider_error(ProviderErrorFacts {
        status_code: error.status_code().map(i64::from),
        status: None,
        code: error.code(),
        message: &error.to_string(),
    })
}

pub type ProductionVisionProvider = OpenRouterVisionProvider<ReqwestOpenRouterTransport>;
pub type ProductionTranscriptionProvider =
    FallbackTranscriptionProvider<ReqwestGroqTranscriptionTransport, ReqwestOpenRouterTransport>;

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use bot_adapters::media_provider::{GroqTranscriptionRequest, GroqTranscriptionResponse};
    use bot_adapters::openrouter_chat::{HttpRequest, HttpResponse, OpenRouterChatError};
    use bot_adapters::telegram_http::{
        BinaryHttpResponse, HttpResponse as TelegramResponse, TelegramFileRequest, TelegramRequest,
        TransportFailureKind,
    };

    use super::*;

    struct TelegramMediaTransport {
        metadata: RefCell<Option<Result<TelegramResponse, TransportFailureKind>>>,
        file: RefCell<Option<Result<BinaryHttpResponse, TransportFailureKind>>>,
    }

    impl TelegramTransport for TelegramMediaTransport {
        fn send(&self, _: &TelegramRequest) -> Result<TelegramResponse, TransportFailureKind> {
            self.metadata
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    impl TelegramFileTransport for TelegramMediaTransport {
        fn download(
            &self,
            _: &TelegramFileRequest,
        ) -> Result<BinaryHttpResponse, TransportFailureKind> {
            self.file
                .borrow_mut()
                .take()
                .unwrap_or(Err(TransportFailureKind::Request))
        }
    }

    fn telegram_media(
        metadata: Result<TelegramResponse, TransportFailureKind>,
        file: Result<BinaryHttpResponse, TransportFailureKind>,
    ) -> TelegramMediaFiles<TelegramMediaTransport> {
        TelegramMediaFiles::new(
            TelegramMediaTransport {
                metadata: RefCell::new(Some(metadata)),
                file: RefCell::new(Some(file)),
            },
            "synthetic-token",
        )
    }

    #[test]
    fn telegram_media_source_requires_valid_metadata_and_successful_download() {
        let metadata = || TelegramResponse {
            status_code: 200,
            body: json!({"result":{"file_path":"media/synthetic.bin"}}).to_string(),
        };
        let mut success = telegram_media(
            Ok(metadata()),
            Ok(BinaryHttpResponse {
                status_code: 200,
                body: vec![1, 2, 3],
            }),
        );
        assert_eq!(success.download("synthetic-file"), Ok(Some(vec![1, 2, 3])));

        let mut invalid_json = telegram_media(
            Ok(TelegramResponse {
                status_code: 200,
                body: "invalid".to_owned(),
            }),
            Err(TransportFailureKind::Request),
        );
        assert!(invalid_json.download("synthetic-file").is_err());

        for metadata in [
            Ok(TelegramResponse {
                status_code: 404,
                body: String::new(),
            }),
            Err(TransportFailureKind::Timeout),
        ] {
            let mut source = telegram_media(metadata, Err(TransportFailureKind::Request));
            assert_eq!(source.download("synthetic-file"), Ok(None));
        }

        let mut failed_download =
            telegram_media(Ok(metadata()), Err(TransportFailureKind::Connection));
        assert_eq!(failed_download.download("synthetic-file"), Ok(None));
    }

    #[test]
    fn redis_media_cache_round_trips_text_against_local_redis() -> Result<(), String> {
        let Some(port) = std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
        else {
            return Ok(());
        };
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned()),
            port,
            password: std::env::var("TEST_REDIS_PASSWORD")
                .ok()
                .filter(|value| !value.is_empty()),
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let file_id = format!("synthetic-media-{nonce}");
        let mut cache = RedisMediaCache::new(endpoint);
        assert_eq!(cache.get("synthetic", &file_id)?, None);
        cache.set("synthetic", &file_id, "synthetic transcript")?;
        assert_eq!(
            cache.get("synthetic", &file_id)?,
            Some("synthetic transcript".to_owned())
        );
        Ok(())
    }

    fn pcm_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = u32::try_from(samples.len() * 2).unwrap_or(0);
        let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    struct Groq {
        responses: RefCell<Vec<Result<GroqTranscriptionResponse, MediaProviderError>>>,
        accounts: RefCell<Vec<String>>,
    }

    impl GroqTranscriptionTransport for Groq {
        fn transcribe(
            &self,
            request: &GroqTranscriptionRequest,
        ) -> Result<GroqTranscriptionResponse, MediaProviderError> {
            self.accounts
                .borrow_mut()
                .push(request.bearer_token.clone());
            self.responses.borrow_mut().remove(0)
        }
    }

    struct OpenRouter {
        calls: RefCell<usize>,
    }

    impl OpenRouterTransport for OpenRouter {
        fn post(&self, _request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            *self.calls.borrow_mut() += 1;
            Ok(HttpResponse {
                status_code: 200,
                headers: BTreeMap::new(),
                body: json!({
                    "choices": [{"message": {"content": "fallback transcript"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 2}
                })
                .to_string(),
            })
        }
    }

    fn groq_response(status_code: u16, body: Value) -> GroqTranscriptionResponse {
        GroqTranscriptionResponse {
            status_code,
            headers: BTreeMap::new(),
            body: body.to_string(),
        }
    }

    struct VisionTransport {
        requests: RefCell<Vec<HttpRequest>>,
        responses: RefCell<Vec<Result<HttpResponse, OpenRouterChatError>>>,
    }

    impl OpenRouterTransport for VisionTransport {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            self.responses.borrow_mut().remove(0)
        }
    }

    fn vision_response(text: &str) -> HttpResponse {
        HttpResponse {
            status_code: 200,
            headers: BTreeMap::new(),
            body: json!({
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}
            })
            .to_string(),
        }
    }

    #[test]
    fn vision_provider_selects_the_prompt_language_and_preserves_media_metadata() {
        for (prompt, expected_system_prompt) in [
            (
                "Describe the synthetic image",
                "respond in English without emojis or markdown.",
            ),
            (
                "describí la imagen sintética",
                "respondé siempre en minúsculas, sin emojis, sin markdown y en lenguaje coloquial argentino.",
            ),
        ] {
            let mut provider = OpenRouterVisionProvider::new(
                VisionTransport {
                    requests: RefCell::new(Vec::new()),
                    responses: RefCell::new(vec![Ok(vision_response("synthetic description"))]),
                },
                "synthetic-key",
                "https://example.test/api/v1",
                "synthetic/vision-model",
                321,
            );
            let result = provider
                .describe(
                    &PreparedImage {
                        bytes: vec![1, 2, 3],
                        mime: "image/png".to_owned(),
                    },
                    prompt,
                    "synthetic-file",
                )
                .unwrap_or_else(|_| unreachable!())
                .unwrap_or_else(|| unreachable!());
            assert_eq!(result.text, "synthetic description");

            let requests = provider.transport.requests.borrow();
            let payload = serde_json::from_str::<Value>(&requests[0].body).unwrap_or(Value::Null);
            assert_eq!(payload["messages"][0]["content"], expected_system_prompt);
            assert_eq!(payload["messages"][1]["content"][0]["text"], prompt);
            assert_eq!(payload["max_tokens"], 321);
            assert!(
                payload["messages"][1]["content"][1]["image_url"]["url"]
                    .as_str()
                    .is_some_and(|url| url.starts_with("data:image/png;base64,"))
            );
            assert_eq!(
                result.billing_segment["metadata"]["file_id"],
                "synthetic-file"
            );
        }
    }

    #[test]
    fn vision_provider_reports_transport_failures() {
        let mut provider = OpenRouterVisionProvider::new(
            VisionTransport {
                requests: RefCell::new(Vec::new()),
                responses: RefCell::new(vec![Err(OpenRouterChatError::Transport(
                    "synthetic transport failure".to_owned(),
                ))]),
            },
            "synthetic-key",
            "https://example.test/api/v1",
            "synthetic/vision-model",
            321,
        );
        let result = provider.describe(
            &PreparedImage {
                bytes: vec![1, 2, 3],
                mime: "image/png".to_owned(),
            },
            "Describe the synthetic image",
            "synthetic-file",
        );
        assert!(matches!(result, Err(ref error) if error.contains("synthetic transport failure")));
    }

    #[test]
    fn groq_rate_limit_tries_next_account_before_openrouter() {
        let groq = Groq {
            responses: RefCell::new(vec![
                Ok(groq_response(
                    429,
                    json!({"error": {"message": "rate limit"}}),
                )),
                Ok(groq_response(
                    200,
                    json!({"text": "second account transcript"}),
                )),
            ]),
            accounts: RefCell::new(Vec::new()),
        };
        let openrouter = OpenRouter {
            calls: RefCell::new(0),
        };
        let mut provider = FallbackTranscriptionProvider::new(
            groq,
            openrouter,
            TranscriptionProviderConfig {
                groq_accounts: vec![
                    ("free".to_owned(), "free-key".to_owned()),
                    ("paid".to_owned(), "paid-key".to_owned()),
                ],
                openrouter_api_key: Some("openrouter-key".to_owned()),
                openrouter_base_url: "https://synthetic.invalid".to_owned(),
                groq_model: "whisper-large-v3".to_owned(),
                openrouter_model: "google/gemini".to_owned(),
                default_backoff_seconds: 60,
            },
        );
        let result = provider.transcribe(
            &PreparedAudio {
                bytes: b"audio".to_vec(),
                duration_seconds: 3.0,
            },
            "file-1",
        );
        assert!(matches!(
            result,
            Ok(Some(MediaProviderResult { ref text, .. })) if text == "second account transcript"
        ));
        assert_eq!(
            provider.groq.accounts.borrow().as_slice(),
            ["free-key", "paid-key"]
        );
        assert_eq!(*provider.openrouter.calls.borrow(), 0);
    }

    #[test]
    fn unrecoverable_groq_failure_uses_openrouter_fallback() {
        let groq = Groq {
            responses: RefCell::new(vec![Ok(groq_response(
                500,
                json!({"error": {"message": "server failed"}}),
            ))]),
            accounts: RefCell::new(Vec::new()),
        };
        let openrouter = OpenRouter {
            calls: RefCell::new(0),
        };
        let mut provider = FallbackTranscriptionProvider::new(
            groq,
            openrouter,
            TranscriptionProviderConfig {
                groq_accounts: vec![("free".to_owned(), "free-key".to_owned())],
                openrouter_api_key: Some("openrouter-key".to_owned()),
                openrouter_base_url: "https://synthetic.invalid".to_owned(),
                groq_model: "whisper-large-v3".to_owned(),
                openrouter_model: "google/gemini".to_owned(),
                default_backoff_seconds: 60,
            },
        );
        let result = provider.transcribe(
            &PreparedAudio {
                bytes: b"audio".to_vec(),
                duration_seconds: 3.0,
            },
            "file-1",
        );
        assert!(matches!(
            result,
            Ok(Some(MediaProviderResult { ref text, .. })) if text == "fallback transcript"
        ));
        assert_eq!(*provider.openrouter.calls.borrow(), 1);
    }

    #[test]
    fn duration_parser_rejects_empty_and_non_finite_values() {
        let processor = FfmpegMediaProcessor::default();
        assert_eq!(processor.duration(b"not media"), None);
    }

    #[cfg(unix)]
    #[test]
    fn media_process_drains_output_while_writing_input() -> Result<(), String> {
        let size = 2 * 1024 * 1024;
        let output = FfmpegMediaProcessor::run_bounded(
            "sh",
            &[
                "-c".to_owned(),
                format!("head -c {size} /dev/zero; cat >/dev/null"),
            ],
            &vec![1; size],
            Duration::from_secs(5),
            size as u64,
            size as u64,
        )?;
        assert_eq!(output.len(), size);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn media_process_kills_timed_out_children() {
        let started = Instant::now();
        let result = FfmpegMediaProcessor::run_bounded(
            "sh",
            &["-c".to_owned(), "while :; do :; done".to_owned()],
            &[],
            Duration::from_millis(50),
            1,
            1,
        );
        assert!(matches!(result, Err(ref error) if error.contains("timed out")));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[cfg(unix)]
    #[test]
    fn media_process_rejects_oversized_input_and_output() {
        assert_eq!(
            FfmpegMediaProcessor::run_bounded(
                "unused",
                &[],
                &[0; 17],
                Duration::from_secs(1),
                16,
                16,
            ),
            Err("media input exceeds the size limit".to_owned())
        );
        assert_eq!(
            FfmpegMediaProcessor::run_bounded(
                "sh",
                &["-c".to_owned(), "head -c 1024 /dev/zero".to_owned()],
                &[],
                Duration::from_secs(1),
                16,
                16,
            ),
            Err("media process output exceeds the size limit".to_owned())
        );
    }

    #[test]
    fn installed_ffmpeg_normalizes_real_image_and_audio_payloads() -> Result<(), String> {
        let version = Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map_err(|error| format!("ffmpeg must be installed for media tests: {error}"))?;
        assert!(version.status.success());
        let mut processor = FfmpegMediaProcessor::default();
        let image = processor
            .prepare_image(b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00")
            .unwrap_or_else(|_| unreachable!())
            .unwrap_or_else(|| unreachable!());
        assert_eq!(image.mime, "image/webp");
        assert!(image.bytes.starts_with(b"RIFF"));
        assert_eq!(image.bytes.get(8..12), Some(b"WEBP".as_slice()));

        let gif = b"GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xff\xff\xff!\xf9\x04\x01\0\0\0\0,\0\0\0\0\x01\0\x01\0\0\x02\x02D\x01\0;";
        let gif_frame = processor
            .prepare_image(gif)
            .unwrap_or_else(|_| unreachable!())
            .unwrap_or_else(|| unreachable!());
        assert_eq!(gif_frame.mime, "image/webp");
        assert!(gif_frame.bytes.starts_with(b"RIFF"));
        assert_eq!(gif_frame.bytes.get(8..12), Some(b"WEBP".as_slice()));

        let wav = pcm_wav(8_000, &[0; 800]);
        let audio = processor
            .prepare_audio(&wav, None)
            .unwrap_or_else(|_| unreachable!())
            .unwrap_or_else(|| unreachable!());
        assert!(audio.bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]));
        assert!((0.08..=0.2).contains(&audio.duration_seconds));
        Ok(())
    }
}
