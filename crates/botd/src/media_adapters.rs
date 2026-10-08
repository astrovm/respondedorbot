//! Production adapters for the native media pipeline.

use std::io::{self, ErrorKind, Read, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bot_adapters::media_provider::{
    MediaProviderError, MediaProviderResult, VisionRequest, describe_image_with,
    transcribe_audio_openrouter_with,
};
use bot_adapters::openrouter_chat::{
    OpenRouterChatError, OpenRouterPricingCache, OpenRouterTransport, ReqwestOpenRouterTransport,
};
use bot_adapters::redis_connection::RedisEndpoint;
use bot_adapters::redis_media_cache::{cache_media, get_cached_media};
use bot_adapters::telegram_http::{
    TELEGRAM_FILE_MAX_BYTES, TelegramFileOutcome, TelegramFileTransport, TelegramHttpOutcome,
    TelegramTransport, download_file_with, request_with,
};
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
        // request_with cannot fail here: the timeout is a positive constant and
        // the payload is always valid JSON, so transport problems surface as
        // TelegramHttpOutcome::TransportError below.
        let outcome = request_with(
            &self.transport,
            &self.token,
            "getFile",
            "GET",
            Some(json!({"file_id": file_id})),
            None,
            TELEGRAM_FILE_TIMEOUT_SECONDS,
        )
        .map_err(error_text)?;
        let TelegramHttpOutcome::Response { status_code, body } = outcome else {
            eprintln!(
                "Media trace: {}",
                json!({
                    "event": "telegram_file_metadata_empty",
                    "file_id": file_id,
                    "stage": "getFile",
                })
            );
            return Ok(None);
        };
        if !(200..300).contains(&status_code) {
            eprintln!(
                "Media trace: {}",
                json!({
                    "event": "telegram_file_metadata_failure",
                    "file_id": file_id,
                    "stage": "getFile",
                    "status_code": status_code,
                })
            );
            return Ok(None);
        }
        let payload = serde_json::from_str::<Value>(&body).map_err(|error| {
            eprintln!(
                "Media trace: {}",
                json!({
                    "event": "telegram_file_metadata_invalid",
                    "file_id": file_id,
                    "stage": "getFile",
                    "status_code": status_code,
                    "error": truncate_error(&error.to_string(), 300),
                })
            );
            error.to_string()
        })?;
        let file_path = payload
            .get("result")
            .and_then(|result| result.get("file_path"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let Some(file_path) = file_path else {
            eprintln!(
                "Media trace: {}",
                json!({
                    "event": "telegram_file_metadata_missing_path",
                    "file_id": file_id,
                    "stage": "getFile",
                    "status_code": status_code,
                })
            );
            return Ok(None);
        };
        // download_file_with cannot fail here either: the timeout is a
        // positive constant, so transport problems surface as
        // TelegramFileOutcome variants below.
        match download_file_with(
            &self.transport,
            &self.token,
            file_path,
            TELEGRAM_FILE_TIMEOUT_SECONDS,
        )
        .map_err(error_text)?
        {
            TelegramFileOutcome::Downloaded(bytes) => {
                eprintln!(
                    "Media trace: {}",
                    json!({
                        "event": "telegram_file_download_result",
                        "file_id": file_id,
                        "bytes": bytes.len(),
                    })
                );
                Ok(Some(bytes))
            }
            TelegramFileOutcome::HttpError { status_code, .. } => {
                eprintln!(
                    "Media trace: {}",
                    json!({
                        "event": "telegram_file_download_failure",
                        "file_id": file_id,
                        "stage": "download",
                        "status_code": status_code,
                    })
                );
                Ok(None)
            }
            TelegramFileOutcome::TransportError { kind } => {
                eprintln!(
                    "Media trace: {}",
                    json!({
                        "event": "telegram_file_download_failure",
                        "file_id": file_id,
                        "stage": "download",
                        "error": truncate_error(&format!("{kind:?}"), 300),
                    })
                );
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
        get_cached_media(&self.endpoint, prefix, file_id).map_err(error_text)
    }

    fn set(&mut self, prefix: &str, file_id: &str, text: &str) -> Result<(), String> {
        cache_media(
            &self.endpoint,
            prefix,
            file_id,
            text,
            MEDIA_CACHE_TTL_SECONDS,
        )
        .map_err(error_text)
    }
}

fn error_text(error: impl std::fmt::Display) -> String {
    error.to_string()
}

const MEDIA_PIPES_UNAVAILABLE: &str = "media process pipes are unavailable";

/// A temporary copy of media input, readable only by this user and removed
/// when dropped.
struct MediaInputFile {
    path: std::path::PathBuf,
}

impl MediaInputFile {
    fn create(input: &[u8]) -> io::Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The container may reuse a PID across restarts, so the start time
        // keeps names from clashing with files a crash left behind.
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "botd-media-{}-{started}-{sequence}",
            std::process::id()
        ));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let file = Self { path };
        options.open(&file.path)?.write_all(input)?;
        Ok(file)
    }
}

impl Drop for MediaInputFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Whether a RIFF/WAVE payload carries any audio samples. ffmpeg still
/// writes a valid header when the input has no decodable audio.
fn wav_has_samples(bytes: &[u8]) -> bool {
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return true;
    }
    let mut offset = 12;
    while let Some(header) = bytes.get(offset..offset + 8) {
        let size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
        if &header[..4] == b"data" {
            // Piped output can't seek back to fill in the size, so ffmpeg
            // leaves it unset; what follows the header is the real data.
            return bytes.len() > offset + 8;
        }
        offset = offset.saturating_add(8).saturating_add(size + (size & 1));
    }
    false
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
        // MP4 files often keep their index at the end, which ffmpeg can only
        // reach by seeking; from a pipe it decodes nothing. Media read from
        // `pipe:0` goes through a private temporary file instead.
        let input_file = if arguments.iter().any(|argument| argument == "pipe:0") {
            Some(MediaInputFile::create(input).map_err(error_text)?)
        } else {
            None
        };
        let arguments = arguments
            .iter()
            .map(|argument| match &input_file {
                Some(file) if argument == "pipe:0" => file.path.as_os_str().to_owned(),
                _ => argument.into(),
            })
            .collect::<Vec<std::ffi::OsString>>();
        let input = if input_file.is_some() { &[] } else { input };
        let mut child = Command::new(program)
            .args(&arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(error_text)?;
        // Both pipes are requested above, so a missing one is reported through
        // the same write and read failures as a broken pipe.
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        thread::scope(|scope| {
            let writer = scope.spawn(move || {
                stdin
                    .ok_or(io::Error::other(MEDIA_PIPES_UNAVAILABLE))
                    .and_then(|mut stdin| stdin.write_all(input))
            });
            let reader = scope.spawn(move || {
                let mut output = Vec::new();
                stdout
                    .ok_or(io::Error::other(MEDIA_PIPES_UNAVAILABLE))
                    .and_then(|stdout| {
                        stdout
                            .take(max_output_bytes.saturating_add(1))
                            .read_to_end(&mut output)
                    })
                    .map_err(error_text)?;
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
                    unfinished => {
                        let failure = unfinished.err().map(error_text);
                        let timeout = || format!("{program} timed out while processing media");
                        break Err(Self::abandon(&mut child, failure.unwrap_or_else(timeout)));
                    }
                }
            };

            let write_result = writer
                .join()
                .or(Err(String::from("media input writer panicked")))?;
            let output = reader
                .join()
                .or(Err(String::from("media output reader panicked")))??;
            let status = status?;
            if status.success() && !output.is_empty() {
                // Frame extraction can finish before the entire animation is consumed.
                match write_result {
                    Err(error) if error.kind() != ErrorKind::BrokenPipe => Err(error.to_string()),
                    _ => Ok(output),
                }
            } else {
                Err(format!("{program} could not process media"))
            }
        })
    }

    fn terminate(child: &mut std::process::Child) {
        let _ = child.kill();
        let _ = child.wait();
    }

    fn abandon(child: &mut std::process::Child, error: String) -> String {
        Self::terminate(child);
        error
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

    /// True only when ffprobe reads the media and finds no audio stream.
    fn has_no_audio_stream(&self, input: &[u8]) -> bool {
        Self::run(
            &self.ffprobe,
            &[
                "-v".to_owned(),
                "error".to_owned(),
                "-show_entries".to_owned(),
                "stream=codec_type".to_owned(),
                "-of".to_owned(),
                "csv=p=0".to_owned(),
                "pipe:0".to_owned(),
            ],
            input,
        )
        .is_ok_and(|output| {
            !String::from_utf8_lossy(&output)
                .lines()
                .any(|line| line.trim() == "audio")
        })
    }

    fn prepare_audio_bounded(
        &self,
        input: &[u8],
        duration_hint_seconds: Option<f64>,
        max_input_bytes: u64,
    ) -> Result<Option<PreparedAudio>, String> {
        let started = Instant::now();
        let extracted = Self::run_bounded(
            &self.ffmpeg,
            &[
                "-loglevel".to_owned(),
                "error".to_owned(),
                "-i".to_owned(),
                "pipe:0".to_owned(),
                "-vn".to_owned(),
                // OpenRouter speech-to-text documents wav input for
                // microsoft/mai-transcribe-2; opus-in-webm is rejected
                // with HTTP 400.
                "-ac".to_owned(),
                "1".to_owned(),
                "-ar".to_owned(),
                "16000".to_owned(),
                "-c:a".to_owned(),
                "pcm_s16le".to_owned(),
                "-f".to_owned(),
                "wav".to_owned(),
                "pipe:1".to_owned(),
            ],
            input,
            MEDIA_PROCESS_TIMEOUT,
            max_input_bytes,
            MEDIA_PROCESS_OUTPUT_MAX_BYTES,
        )
        .or_else(|error| {
            eprintln!(
                "{}",
                audio_conversion_failure_diagnostic(
                    &error,
                    &self.ffmpeg,
                    input.len(),
                    started.elapsed(),
                )
            );
            // A clip with no audio track has nothing to transcribe; anything
            // else goes to the provider as it came.
            if self.has_no_audio_stream(input) {
                Err(())
            } else {
                Ok(input.to_vec())
            }
        });
        let extracted = match extracted {
            Ok(extracted) if wav_has_samples(&extracted) => Some(extracted),
            Ok(_) | Err(()) => None,
        };
        let Some(extracted) = extracted else {
            eprintln!(
                "Media trace: {}",
                json!({
                    "event": "audio_prepare_invalid",
                    "error_kind": "NoAudio",
                    "input_bytes": input.len(),
                    "duration_hint_seconds": duration_hint_seconds,
                    "elapsed_ms": started.elapsed().as_millis(),
                })
            );
            return Ok(None);
        };
        let hinted = duration_hint_seconds.filter(|value| value.is_finite() && *value > 0.0);
        let duration_seconds = hinted
            .or_else(|| self.duration(&extracted))
            .or_else(|| self.duration(input));
        match duration_seconds {
            Some(duration_seconds) => {
                eprintln!(
                    "Media trace: {}",
                    json!({
                        "event": "audio_prepare_result",
                        "input_bytes": input.len(),
                        "output_bytes": extracted.len(),
                        "duration_seconds": duration_seconds,
                        "duration_hint_seconds": duration_hint_seconds,
                        "elapsed_ms": started.elapsed().as_millis(),
                    })
                );
                Ok(Some(PreparedAudio {
                    bytes: extracted,
                    duration_seconds,
                }))
            }
            None => {
                eprintln!(
                    "Media trace: {}",
                    json!({
                        "event": "audio_prepare_invalid",
                        "input_bytes": input.len(),
                        "output_bytes": extracted.len(),
                        "duration_hint_seconds": duration_hint_seconds,
                        "elapsed_ms": started.elapsed().as_millis(),
                    })
                );
                Ok(None)
            }
        }
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
    pricing: Option<Arc<OpenRouterPricingCache>>,
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
            pricing: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
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
        let price_ceiling = self
            .pricing
            .as_ref()
            .map(|pricing| pricing.price_ceiling(&self.model))
            .transpose()
            .map_err(error_text)?;
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
                price_ceiling,
                file_id: Some(file_id),
            },
        )
        .map(Some)
        .map_err(error_text)
    }
}

pub struct OpenRouterTranscriptionProvider<Transport> {
    transport: Transport,
    api_key: String,
    openrouter_base_url: String,
    model: String,
}

impl<Transport> OpenRouterTranscriptionProvider<Transport> {
    #[must_use]
    pub fn new(transport: Transport, api_key: &str, base_url: &str, model: &str) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            openrouter_base_url: base_url.to_owned(),
            model: model.to_owned(),
        }
    }
}

impl<Transport> TranscriptionProvider for OpenRouterTranscriptionProvider<Transport>
where
    Transport: OpenRouterTransport,
{
    fn transcribe(
        &mut self,
        audio: &PreparedAudio,
        file_id: &str,
    ) -> Result<Option<MediaProviderResult>, String> {
        let started = Instant::now();
        eprintln!(
            "{}",
            transcription_diagnostic(&self.model, audio, file_id, Duration::ZERO, None)
        );
        let result = transcribe_audio_openrouter_with(
            &self.transport,
            &self.api_key,
            &self.openrouter_base_url,
            &self.model,
            &audio.bytes,
            audio.duration_seconds,
            Some(file_id),
        );
        eprintln!(
            "{}",
            transcription_diagnostic(
                &self.model,
                audio,
                file_id,
                started.elapsed(),
                Some(&result)
            )
        );
        result.map(Some).map_err(error_text)
    }
}

fn truncate_error(message: &str, max_chars: usize) -> String {
    let mut truncated: String = message.chars().take(max_chars).collect();
    if message.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
}

fn transcription_diagnostic(
    model: &str,
    audio: &PreparedAudio,
    file_id: &str,
    elapsed: Duration,
    result: Option<&Result<MediaProviderResult, MediaProviderError>>,
) -> String {
    let event = match result {
        None => "transcription_start",
        Some(Ok(output)) if output.text.is_empty() => "transcription_empty",
        Some(Ok(_)) => "transcription_result",
        Some(Err(_)) => "transcription_failure",
    };
    let mut details = json!({
        "event": event,
        "model": model,
        "file_id": file_id,
        "bytes": audio.bytes.len(),
        "duration_seconds": audio.duration_seconds,
        "elapsed_ms": elapsed.as_millis(),
    });
    if let Some(Ok(output)) = result {
        details["text_chars"] = json!(output.text.chars().count());
    }
    if let Some(Err(error)) = result {
        details["error_kind"] = json!(media_provider_error_kind(error));
        details["status_code"] = json!(error.status_code());
        details["retry_after_seconds"] = json!(error.retry_after_seconds());
        details["code"] = json!(error.code());
        details["error"] = json!(truncate_error(&error.to_string(), 300));
    }
    format!("Media trace: {details}")
}

fn media_provider_error_kind(error: &MediaProviderError) -> &'static str {
    match error {
        MediaProviderError::MissingCredential => "MissingCredential",
        MediaProviderError::Transport(_) => "Transport",
        MediaProviderError::Http { .. } => "Http",
        MediaProviderError::InvalidJson(_) => "InvalidJson",
        MediaProviderError::MissingText => "MissingText",
        MediaProviderError::OpenRouter(error) => match error {
            OpenRouterChatError::MissingApiKey => "OpenRouter.MissingApiKey",
            OpenRouterChatError::MissingModel => "OpenRouter.MissingModel",
            OpenRouterChatError::MissingModelPricing { .. } => "OpenRouter.MissingModelPricing",
            OpenRouterChatError::InvalidBaseUrl => "OpenRouter.InvalidBaseUrl",
            OpenRouterChatError::RequestJson(_) => "OpenRouter.RequestJson",
            OpenRouterChatError::Transport(_) => "OpenRouter.Transport",
            OpenRouterChatError::RateLimited { .. } => "OpenRouter.RateLimited",
            OpenRouterChatError::Http { .. } => "OpenRouter.Http",
            OpenRouterChatError::InvalidJson(_) => "OpenRouter.InvalidJson",
            OpenRouterChatError::ResponseTooLarge => "OpenRouter.ResponseTooLarge",
            OpenRouterChatError::MalformedResponse => "OpenRouter.MalformedResponse",
            OpenRouterChatError::IncompleteStream => "OpenRouter.IncompleteStream",
            OpenRouterChatError::Stream(_) => "OpenRouter.Stream",
            OpenRouterChatError::ThinkingTimeout => "OpenRouter.ThinkingTimeout",
        },
    }
}

fn audio_conversion_failure_diagnostic(
    error: &str,
    program: &str,
    input_bytes: usize,
    elapsed: Duration,
) -> String {
    let kind = match error {
        "media input exceeds the size limit" => "InputTooLarge",
        "media process output exceeds the size limit" => "OutputTooLarge",
        "media process timeout must be positive" | "media process timeout is too large" => {
            "InvalidTimeout"
        }
        "media process did not expose stdin" | "media process did not expose stdout" => {
            "MissingPipe"
        }
        "media input writer panicked" | "media output reader panicked" => "WorkerPanicked",
        _ if error == format!("{program} timed out while processing media") => "Timeout",
        _ if error == format!("{program} could not process media") => "UnsuccessfulOrEmptyOutput",
        _ => "ProcessIo",
    };
    format!(
        "Media trace: {}",
        json!({
            "event": "audio_conversion_failure",
            "error_kind": kind,
            "bytes": input_bytes,
            "fallback": true,
            "elapsed_ms": elapsed.as_millis(),
        })
    )
}

pub type ProductionVisionProvider = OpenRouterVisionProvider<ReqwestOpenRouterTransport>;
pub type ProductionTranscriptionProvider =
    OpenRouterTranscriptionProvider<ReqwestOpenRouterTransport>;

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn telegram_media_source_reports_missing_paths_and_http_failures() {
        let metadata = || TelegramResponse {
            status_code: 200,
            body: json!({"result":{"file_path":"media/synthetic.bin"}}).to_string(),
        };
        for body in [
            json!({"result": {}}).to_string(),
            json!({"result": {"file_path": ""}}).to_string(),
            json!({}).to_string(),
        ] {
            let mut source = telegram_media(
                Ok(TelegramResponse {
                    status_code: 200,
                    body,
                }),
                Err(TransportFailureKind::Request),
            );
            assert_eq!(source.download("synthetic-file"), Ok(None));
        }

        let mut http_failure = telegram_media(
            Ok(metadata()),
            Ok(BinaryHttpResponse {
                status_code: 500,
                body: Vec::new(),
            }),
        );
        assert_eq!(http_failure.download("synthetic-file"), Ok(None));
    }

    #[test]
    fn transcription_provider_reports_transport_failures() {
        let mut provider = OpenRouterTranscriptionProvider::new(
            VisionTransport {
                requests: RefCell::new(Vec::new()),
                responses: RefCell::new(vec![Err(OpenRouterChatError::Transport(
                    "synthetic transport failure".to_owned(),
                ))]),
            },
            "synthetic-key",
            "https://example.test/api/v1",
            "synthetic/transcription-model",
        );
        let result = provider.transcribe(
            &PreparedAudio {
                bytes: b"RIFF....WAVE synthetic audio".to_vec(),
                duration_seconds: 3.0,
            },
            "file-1",
        );
        assert!(matches!(result, Err(ref error) if error.contains("synthetic transport failure")));
    }

    #[test]
    fn transcription_diagnostics_cover_every_event_and_helper() {
        let audio = PreparedAudio {
            bytes: vec![1, 2, 3],
            duration_seconds: 2.5,
        };
        let start = transcription_diagnostic("model", &audio, "file-1", Duration::ZERO, None);
        assert!(start.contains("transcription_start"));
        assert!(start.contains("file-1"));

        let ok = Ok(MediaProviderResult {
            text: "hola".to_owned(),
            billing_segment: Value::Null,
        });
        let result = transcription_diagnostic("model", &audio, "file-1", Duration::ZERO, Some(&ok));
        assert!(result.contains("transcription_result"));
        assert!(result.contains("text_chars"));

        let empty = Ok(MediaProviderResult {
            text: String::new(),
            billing_segment: Value::Null,
        });
        let empty_trace =
            transcription_diagnostic("model", &audio, "file-1", Duration::ZERO, Some(&empty));
        assert!(empty_trace.contains("transcription_empty"));

        for error in [
            MediaProviderError::MissingCredential,
            MediaProviderError::Transport("broken".to_owned()),
            MediaProviderError::Http {
                status_code: 503,
                code: "busy".to_owned(),
                message: "slow down".to_owned(),
                retry_after_seconds: Some(4),
            },
            MediaProviderError::InvalidJson("bad".to_owned()),
            MediaProviderError::MissingText,
            MediaProviderError::OpenRouter(OpenRouterChatError::MissingApiKey),
            MediaProviderError::OpenRouter(OpenRouterChatError::MissingModel),
            MediaProviderError::OpenRouter(OpenRouterChatError::MissingModelPricing {
                model: "m".to_owned(),
            }),
            MediaProviderError::OpenRouter(OpenRouterChatError::InvalidBaseUrl),
            MediaProviderError::OpenRouter(OpenRouterChatError::RequestJson("bad".to_owned())),
            MediaProviderError::OpenRouter(OpenRouterChatError::Transport("down".to_owned())),
            MediaProviderError::OpenRouter(OpenRouterChatError::RateLimited {
                retry_after_seconds: Some(2),
                message: "limited".to_owned(),
            }),
            MediaProviderError::OpenRouter(OpenRouterChatError::Http {
                status_code: 502,
                message: "upstream".to_owned(),
            }),
            MediaProviderError::OpenRouter(OpenRouterChatError::InvalidJson("bad".to_owned())),
            MediaProviderError::OpenRouter(OpenRouterChatError::ResponseTooLarge),
            MediaProviderError::OpenRouter(OpenRouterChatError::MalformedResponse),
            MediaProviderError::OpenRouter(OpenRouterChatError::IncompleteStream),
            MediaProviderError::OpenRouter(OpenRouterChatError::Stream("broken".to_owned())),
            MediaProviderError::OpenRouter(OpenRouterChatError::ThinkingTimeout),
        ] {
            let failure = Err(error);
            let trace =
                transcription_diagnostic("model", &audio, "file-1", Duration::ZERO, Some(&failure));
            assert!(trace.contains("transcription_failure"));
            assert!(trace.contains("error_kind"));
        }

        assert_eq!(truncate_error("short", 300), "short");
        let long = "x".repeat(400);
        let truncated = truncate_error(&long, 300);
        assert_eq!(truncated.chars().count(), 301);
        assert!(truncated.ends_with('…'));

        for (message, program, kind) in [
            (
                "media input exceeds the size limit",
                "ffmpeg",
                "InputTooLarge",
            ),
            (
                "media process output exceeds the size limit",
                "ffmpeg",
                "OutputTooLarge",
            ),
            (
                "media process timeout must be positive",
                "ffmpeg",
                "InvalidTimeout",
            ),
            (
                "media process timeout is too large",
                "ffmpeg",
                "InvalidTimeout",
            ),
            (
                "media process did not expose stdin",
                "ffmpeg",
                "MissingPipe",
            ),
            (
                "media process did not expose stdout",
                "ffmpeg",
                "MissingPipe",
            ),
            ("media input writer panicked", "ffmpeg", "WorkerPanicked"),
            ("media output reader panicked", "ffmpeg", "WorkerPanicked"),
            (
                "ffmpeg timed out while processing media",
                "ffmpeg",
                "Timeout",
            ),
            (
                "ffmpeg could not process media",
                "ffmpeg",
                "UnsuccessfulOrEmptyOutput",
            ),
            ("broken pipe", "ffmpeg", "ProcessIo"),
        ] {
            let trace = audio_conversion_failure_diagnostic(message, program, 7, Duration::ZERO);
            assert!(trace.contains(kind), "{message} should map to {kind}");
        }
    }

    #[test]
    fn prepare_audio_without_measurable_duration_reports_invalid() {
        let mut processor = FfmpegMediaProcessor::default();
        assert_eq!(processor.prepare_audio(b"not media", None), Ok(None));
    }

    #[test]
    fn redis_media_cache_round_trips_text_against_local_redis() -> TestResult {
        std::env::var("TEST_REDIS_PORT")
            .ok()
            .and_then(|value| value.parse().ok())
            .map_or(Ok(()), round_trip_redis_media_cache)
    }

    fn round_trip_redis_media_cache(port: u16) -> TestResult {
        let endpoint = RedisEndpoint {
            host: std::env::var("TEST_REDIS_HOST").unwrap_or(String::from("127.0.0.1")),
            port,
            // Empty passwords are ignored by the Redis client.
            password: std::env::var("TEST_REDIS_PASSWORD").ok(),
        };
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
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

    struct OpenRouter {
        requests: RefCell<Vec<HttpRequest>>,
        response: HttpResponse,
    }

    impl OpenRouterTransport for OpenRouter {
        fn post(&self, request: &HttpRequest) -> Result<HttpResponse, OpenRouterChatError> {
            self.requests.borrow_mut().push(request.clone());
            Ok(self.response.clone())
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
    fn vision_provider_selects_the_prompt_language_and_preserves_media_metadata() -> TestResult {
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
            let described = provider.describe(
                &PreparedImage {
                    bytes: vec![1, 2, 3],
                    mime: "image/png".to_owned(),
                },
                prompt,
                "synthetic-file",
            );
            let result = described?.ok_or("no media was produced")?;
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
        Ok(())
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
    fn transcription_provider_uses_openrouter_audio_endpoint() {
        let mut provider = OpenRouterTranscriptionProvider::new(
            OpenRouter {
                requests: RefCell::new(Vec::new()),
                response: HttpResponse {
                    status_code: 200,
                    headers: BTreeMap::new(),
                    body: json!({
                        "text": "synthetic transcript",
                        "usage": {"seconds": 3.0, "cost": "0.0000833"}
                    })
                    .to_string(),
                },
            },
            "synthetic-key",
            "https://synthetic.invalid/api/v1",
            "microsoft/mai-transcribe-2",
        );
        let result = provider.transcribe(
            &PreparedAudio {
                bytes: b"RIFF....WAVE synthetic audio".to_vec(),
                duration_seconds: 3.0,
            },
            "file-1",
        );
        assert!(matches!(
            result.as_ref(),
            Ok(Some(MediaProviderResult { text, .. })) if text == "synthetic transcript"
        ));
        assert_eq!(provider.transport.requests.borrow().len(), 1);
        let requests = provider.transport.requests.borrow();
        let request = &requests[0];
        let payload = serde_json::from_str::<Value>(&request.body).unwrap_or(Value::Null);
        assert_eq!(
            request.url,
            "https://synthetic.invalid/api/v1/audio/transcriptions"
        );
        assert_eq!(request.bearer_token, "synthetic-key");
        assert_eq!(payload["model"], "microsoft/mai-transcribe-2");
        assert_eq!(payload["input_audio"]["format"], "wav");
        assert_eq!(
            result
                .as_ref()
                .ok()
                .and_then(|value| value.as_ref())
                .map(|value| value.billing_segment["metadata"]["provider"].clone()),
            Some(json!("openrouter"))
        );
    }

    #[test]
    fn duration_parser_rejects_empty_and_non_finite_values() {
        let processor = FfmpegMediaProcessor::default();
        assert_eq!(processor.duration(b"not media"), None);
    }

    #[cfg(unix)]
    #[test]
    fn media_process_drains_output_while_writing_input() -> TestResult {
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
        );
        assert_eq!(output.map(|output| output.len()), Ok(size));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn early_input_closure_requires_successful_nonempty_output() {
        for script in ["printf frame; exit 7", "exit 0"] {
            let result = FfmpegMediaProcessor::run_bounded(
                "sh",
                &["-c".to_owned(), script.to_owned()],
                &vec![1; 2 * 1024 * 1024],
                Duration::from_secs(5),
                2 * 1024 * 1024,
                1024,
            );
            assert_eq!(result, Err("sh could not process media".to_owned()));
        }
    }

    #[test]
    fn media_process_rejects_a_zero_timeout_before_spawning() {
        assert_eq!(
            FfmpegMediaProcessor::run_bounded(
                "synthetic-program-that-does-not-exist",
                &[],
                &[1, 2, 3],
                Duration::ZERO,
                16,
                16,
            ),
            Err("media process timeout must be positive".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn successful_output_tolerates_a_process_that_stops_reading_input() {
        // The pipe buffer is far smaller than the input, so the writer hits a
        // closed pipe once `head` exits after reading its first byte.
        let result = FfmpegMediaProcessor::run_bounded(
            "head",
            &["-c".to_owned(), "1".to_owned()],
            &vec![7; 4 * 1024 * 1024],
            Duration::from_secs(5),
            4 * 1024 * 1024,
            16,
        );
        assert_eq!(result, Ok(vec![7]));
    }

    #[test]
    fn duration_prefers_the_container_duration_when_ffprobe_reports_one() -> TestResult {
        // A seekable output lets ffmpeg record the total length in the FLAC
        // header, which ffprobe then reports as the container duration.
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("botd-duration-{nonce}.flac"));
        let generated = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
            ])
            .arg(&path)
            .status();
        let flac = std::fs::read(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert!(generated.is_ok_and(|status| status.success()));
        assert!(flac.starts_with(b"fLaC"));
        let duration = FfmpegMediaProcessor::default().duration(&flac);
        assert!(
            duration.is_some_and(|seconds| (seconds - 2.0).abs() < 0.05),
            "{duration:?}"
        );
        Ok(())
    }

    #[test]
    fn duration_sums_packets_when_the_container_has_none() -> TestResult {
        // Written to a pipe, FLAC can't record its total length, so ffprobe
        // reports no container duration and the packets are summed instead.
        let generated = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-f",
                "flac",
                "pipe:1",
            ])
            .output()?;
        assert!(generated.status.success());
        let duration = FfmpegMediaProcessor::default().duration(&generated.stdout);
        assert!(
            duration.is_some_and(|seconds| (seconds - 2.0).abs() < 0.05),
            "{duration:?}"
        );
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
    fn extracts_one_frame_from_a_large_animation() -> TestResult {
        let generated = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=15:duration=5",
                "-f",
                "gif",
                "pipe:1",
            ])
            .output()?;
        assert!(generated.status.success());
        assert!(generated.stdout.len() > 256 * 1024);
        let frame = FfmpegMediaProcessor::default()
            .prepare_image(&generated.stdout)?
            .ok_or("valid animation did not produce a frame")?;
        assert_eq!(frame.mime, "image/webp");
        let pixels = FfmpegMediaProcessor::run(
            "ffmpeg",
            &[
                "-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            .map(str::to_owned),
            &frame.bytes,
        );
        assert_eq!(pixels.map(|pixels| pixels.len()), Ok(320 * 240 * 3));
        Ok(())
    }

    /// An MP4 with its index after the media data, as phones and Telegram
    /// often produce: ffmpeg can't decode it from a pipe.
    fn mp4_with_trailing_index() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("botd-trailing-index-{nonce}.mp4"));
        let generated = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=duration=2:size=320x240:rate=15",
                "-f",
                "lavfi",
                "-i",
                "sine=duration=2",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-b:v",
                "2M",
                "-c:a",
                "aac",
                "-shortest",
            ])
            .arg(&path)
            .status();
        let mp4 = std::fs::read(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert!(generated.is_ok_and(|status| status.success()));
        let index = mp4.windows(4).position(|window| window == b"moov");
        let data = mp4.windows(4).position(|window| window == b"mdat");
        assert!(index > data, "the index must follow the media data");
        Ok(mp4)
    }

    #[test]
    fn video_with_a_trailing_index_yields_audio_and_a_frame() -> TestResult {
        let mp4 = mp4_with_trailing_index()?;
        let mut processor = FfmpegMediaProcessor::default();
        let audio = processor
            .prepare_audio(&mp4, Some(2.0))?
            .ok_or("no audio was produced")?;
        assert_eq!(audio.bytes.get(..4), Some(b"RIFF".as_slice()));
        // Two seconds of 16 kHz mono 16-bit PCM.
        assert!(audio.bytes.len() > 60_000, "{}", audio.bytes.len());
        let frame = processor
            .prepare_image(&mp4)?
            .ok_or("no frame was produced")?;
        assert_eq!(frame.mime, "image/webp");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn media_input_files_are_private_and_removed_after_use() -> TestResult {
        use std::os::unix::fs::PermissionsExt;
        let file = MediaInputFile::create(b"synthetic media")?;
        let path = file.path.clone();
        assert_eq!(std::fs::read(&path)?, b"synthetic media");
        assert_eq!(
            std::fs::metadata(&path)?.permissions().mode() & 0o777,
            0o600
        );
        let second = MediaInputFile::create(b"")?;
        assert_ne!(second.path, path);
        drop(file);
        drop(second);
        assert!(!path.exists());

        // The process reads the file, not stdin, and the file is gone after.
        let output = FfmpegMediaProcessor::run_bounded(
            "sh",
            &[
                "-c".to_owned(),
                "cat \"$0\"; printf '\\n%s' \"$0\"; cat".to_owned(),
                "pipe:0".to_owned(),
            ],
            b"from file",
            Duration::from_secs(5),
            16,
            1024,
        )
        .unwrap_or_default();
        let output = String::from_utf8(output)?;
        let (content, used) = output.split_once('\n').ok_or("no path echoed")?;
        assert_eq!(content, "from file");
        assert!(used.contains("botd-media-"), "{used}");
        assert!(!std::path::Path::new(used).exists());
        Ok(())
    }

    #[test]
    fn media_without_audio_samples_is_invalid_audio() -> TestResult {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("botd-silent-video-{nonce}.mp4"));
        let generated = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=duration=1:size=64x64:rate=5",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
            ])
            .arg(&path)
            .status();
        let mp4 = std::fs::read(&path).unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert!(generated.is_ok_and(|status| status.success()));
        // A header-only WAV and a clip with no audio track are both
        // rejected before reaching the transcriber.
        let header_only = pcm_wav(16_000, &[]);
        assert!(!wav_has_samples(&header_only));
        let mut processor = FfmpegMediaProcessor::default();
        assert_eq!(processor.prepare_audio(&header_only, Some(54.0))?, None);
        assert!(processor.has_no_audio_stream(&mp4));
        assert_eq!(processor.prepare_audio(&mp4, Some(1.0))?, None);
        // Media ffprobe can't read keeps the raw fallback.
        assert!(!processor.has_no_audio_stream(b"not media"));
        let wav = pcm_wav(8_000, &[0; 800]);
        assert!(!processor.has_no_audio_stream(&wav));
        Ok(())
    }

    #[test]
    fn wav_sample_detection_walks_the_chunks() {
        let mut with_samples = pcm_wav(16_000, &[1, 2]);
        assert!(wav_has_samples(&with_samples));
        // ffmpeg's piped output: a LIST chunk, then a data chunk with an
        // unset size and nothing after it.
        let mut piped = b"RIFF\xff\xff\xff\xffWAVEfmt \x10\0\0\0".to_vec();
        piped.extend([0; 16]);
        piped.extend(b"LIST\x03\0\0\0abc\0");
        piped.extend(b"data\xff\xff\xff\xff");
        assert!(!wav_has_samples(&piped));
        piped.extend([0, 0]);
        assert!(wav_has_samples(&piped));
        // Not WAV at all: left to the transcriber.
        assert!(wav_has_samples(b"OggS"));
        assert!(wav_has_samples(b""));
        // A WAV whose chunks end before any data chunk.
        with_samples.truncate(20);
        assert!(!wav_has_samples(&with_samples));
        assert!(!wav_has_samples(b"RIFF\0\0\0\0WAVE"));
    }

    #[test]
    fn installed_ffmpeg_normalizes_real_image_and_audio_payloads() -> TestResult {
        let version = Command::new("ffmpeg").arg("-version").output()?;
        assert!(version.status.success());
        let mut processor = FfmpegMediaProcessor::default();
        let image = processor
            .prepare_image(b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00")?
            .ok_or("no media was produced")?;
        assert_eq!(image.mime, "image/webp");
        assert!(image.bytes.starts_with(b"RIFF"));
        assert_eq!(image.bytes.get(8..12), Some(b"WEBP".as_slice()));

        let gif = b"GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xff\xff\xff!\xf9\x04\x01\0\0\0\0,\0\0\0\0\x01\0\x01\0\0\x02\x02D\x01\0;";
        let gif_frame = processor
            .prepare_image(gif)?
            .ok_or("no media was produced")?;
        assert_eq!(gif_frame.mime, "image/webp");
        assert!(gif_frame.bytes.starts_with(b"RIFF"));
        assert_eq!(gif_frame.bytes.get(8..12), Some(b"WEBP".as_slice()));

        let wav = pcm_wav(8_000, &[0; 800]);
        let audio = processor
            .prepare_audio(&wav, None)?
            .ok_or("no media was produced")?;
        assert_eq!(audio.bytes.get(..4), Some(b"RIFF".as_slice()));
        assert_eq!(audio.bytes.get(8..12), Some(b"WAVE".as_slice()));
        assert!((0.08..=0.2).contains(&audio.duration_seconds));
        Ok(())
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn redis_media_cache_reports_unreachable_redis() {
        let mut cache = RedisMediaCache::new(RedisEndpoint {
            host: "127.0.0.1".to_owned(),
            port: 1,
            password: None,
        });
        let read = cache.get("image_description", "file-1");
        let write = cache.set("image_description", "file-1", "description");
        assert!(matches!(&read, Err(error) if !error.is_empty()), "{read:?}");
        assert!(
            matches!(&write, Err(error) if !error.is_empty()),
            "{write:?}"
        );
    }

    #[test]
    fn media_process_rejects_unrepresentable_timeouts_and_missing_programs() {
        assert_eq!(
            FfmpegMediaProcessor::run_bounded("unused", &[], &[], Duration::MAX, 16, 16),
            Err("media process timeout is too large".to_owned())
        );
        let missing = FfmpegMediaProcessor::run_bounded(
            "synthetic-program-that-does-not-exist",
            &[],
            &[],
            Duration::from_secs(1),
            16,
            16,
        );
        assert!(
            matches!(&missing, Err(error) if !error.is_empty()),
            "{missing:?}"
        );
    }

    #[test]
    fn a_valid_duration_hint_measures_audio_that_ffprobe_cannot() {
        let mut processor = FfmpegMediaProcessor::default();
        assert_eq!(
            processor.prepare_audio(b"not media", Some(2.5)),
            Ok(Some(PreparedAudio {
                bytes: b"not media".to_vec(),
                duration_seconds: 2.5,
            }))
        );
        // Unusable hints are ignored rather than trusted.
        assert_eq!(
            processor.prepare_audio(b"not media", Some(f64::NAN)),
            Ok(None)
        );
        assert_eq!(processor.prepare_audio(b"not media", Some(-1.0)), Ok(None));
    }

    #[test]
    fn vision_pricing_failures_stop_before_the_provider_request() -> TestResult {
        let pricing = Arc::new(OpenRouterPricingCache::new("synthetic-key", "not-a-url")?);
        let mut provider = OpenRouterVisionProvider::new(
            VisionTransport {
                requests: RefCell::new(Vec::new()),
                responses: RefCell::new(Vec::new()),
            },
            "synthetic-key",
            "https://example.test/api/v1",
            "synthetic/vision-model",
            321,
        )
        .with_openrouter_pricing(pricing);
        let result = provider.describe(
            &PreparedImage {
                bytes: vec![1, 2, 3],
                mime: "image/png".to_owned(),
            },
            "describe",
            "synthetic-file",
        );
        assert!(
            matches!(&result, Err(error) if !error.is_empty()),
            "{result:?}"
        );
        assert!(provider.transport.requests.borrow().is_empty());
        Ok(())
    }
}
