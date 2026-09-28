//! Native cache/download/prepare/provider pipeline for Telegram media.

use std::sync::Arc;

use bot_adapters::media_provider::MediaProviderResult;
use bot_adapters::openrouter_chat::{OpenRouterChatError, OpenRouterPricingCache};
use bot_core::ai_reserve::{
    VISION_OUTPUT_TOKEN_LIMIT, estimate_transcription_reserve_credit_units,
    estimate_vision_reserve_credit_units_with_pricing,
};
use bot_core::provider_pricing::TokenPricing;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Audio,
}

impl MediaKind {
    #[must_use]
    pub const fn cache_prefix(self) -> &'static str {
        match self {
            Self::Image => "image_description",
            Self::Audio => "audio_transcription",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreparedMedia {
    Cached {
        kind: MediaKind,
        file_id: String,
        text: String,
    },
    Image {
        file_id: String,
        bytes: Vec<u8>,
        mime: String,
        reserve_credit_units: i64,
    },
    Audio {
        file_id: String,
        bytes: Vec<u8>,
        duration_seconds: f64,
        reserve_credit_units: i64,
    },
}

impl PreparedMedia {
    #[must_use]
    pub const fn kind(&self) -> MediaKind {
        match self {
            Self::Cached { kind, .. } => *kind,
            Self::Image { .. } => MediaKind::Image,
            Self::Audio { .. } => MediaKind::Audio,
        }
    }

    #[must_use]
    pub const fn reserve_credit_units(&self) -> i64 {
        match self {
            Self::Cached { .. } => 0,
            Self::Image {
                reserve_credit_units,
                ..
            }
            | Self::Audio {
                reserve_credit_units,
                ..
            } => *reserve_credit_units,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaExecution {
    pub kind: MediaKind,
    pub file_id: String,
    pub text: String,
    pub billing_segment: Option<Value>,
    pub cached: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedImagePrompt {
    pub bytes: Arc<[u8]>,
    pub mime: String,
}

pub trait MediaRuntime {
    fn estimate_reserve_credit_units(
        &mut self,
        kind: MediaKind,
        duration_hint_seconds: Option<f64>,
    ) -> Result<i64, String>;

    fn prepare(
        &mut self,
        kind: MediaKind,
        file_id: &str,
        duration_hint_seconds: Option<f64>,
    ) -> Result<PreparedMedia, String>;

    fn prepare_image_for_prompt(
        &mut self,
        _file_id: &str,
    ) -> Result<Option<PreparedImagePrompt>, String> {
        Ok(None)
    }

    fn execute(&mut self, prepared: PreparedMedia, prompt: &str) -> Result<MediaExecution, String>;
}

pub trait MediaFileSource {
    fn download(&mut self, file_id: &str) -> Result<Option<Vec<u8>>, String>;
}

pub trait MediaCache {
    fn get(&mut self, prefix: &str, file_id: &str) -> Result<Option<String>, String>;

    fn set(&mut self, prefix: &str, file_id: &str, text: &str) -> Result<(), String>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    pub bytes: Vec<u8>,
    pub mime: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreparedAudio {
    pub bytes: Vec<u8>,
    pub duration_seconds: f64,
}

pub trait MediaProcessor {
    fn prepare_image(&mut self, input: &[u8]) -> Result<Option<PreparedImage>, String>;

    fn prepare_audio(
        &mut self,
        input: &[u8],
        duration_hint_seconds: Option<f64>,
    ) -> Result<Option<PreparedAudio>, String>;
}

pub trait VisionProvider {
    fn describe(
        &mut self,
        image: &PreparedImage,
        prompt: &str,
        file_id: &str,
    ) -> Result<Option<MediaProviderResult>, String>;
}

pub trait TranscriptionProvider {
    fn transcribe(
        &mut self,
        audio: &PreparedAudio,
        file_id: &str,
    ) -> Result<Option<MediaProviderResult>, String>;
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MediaPipelineError {
    #[error("Telegram media download failed")]
    Download,
    #[error("image could not be decoded")]
    InvalidImage,
    #[error("audio could not be decoded or measured")]
    InvalidAudio,
    #[error("media provider returned no usable result")]
    ProviderUnavailable,
    #[error("media reserve estimate failed: {0}")]
    ReserveEstimate(String),
}

pub(crate) fn estimate_standard_vision_reserve_credit_units(
    pricing: &TokenPricing,
) -> Result<i64, String> {
    estimate_vision_reserve_credit_units_with_pricing(
        "Describe what you see in this image in detail.",
        0,
        1_200,
        VISION_OUTPUT_TOKEN_LIMIT,
        pricing,
    )
    .map_err(|error| error.to_string())
}

pub struct NativeMedia<Files, Cache, Processor, Vision, Transcription> {
    files: Files,
    cache: Cache,
    processor: Processor,
    vision: Vision,
    transcription: Transcription,
    vision_model: String,
    openrouter_pricing: Option<Arc<OpenRouterPricingCache>>,
}

impl<Files, Cache, Processor, Vision, Transcription>
    NativeMedia<Files, Cache, Processor, Vision, Transcription>
{
    #[must_use]
    pub fn new(
        files: Files,
        cache: Cache,
        processor: Processor,
        vision: Vision,
        transcription: Transcription,
        vision_model: &str,
    ) -> Self {
        Self {
            files,
            cache,
            processor,
            vision,
            transcription,
            vision_model: vision_model.to_owned(),
            openrouter_pricing: None,
        }
    }

    #[must_use]
    pub fn with_openrouter_pricing(mut self, pricing: Arc<OpenRouterPricingCache>) -> Self {
        self.openrouter_pricing = Some(pricing);
        self
    }

    fn estimate_image_reserve_credit_units(&self) -> Result<i64, String> {
        let pricing = crate::native_ai::reservation_pricing_for_model(
            &self.vision_model,
            self.openrouter_pricing.as_deref(),
        )?;
        estimate_standard_vision_reserve_credit_units(&pricing)
    }

    fn estimate_audio_reserve_credit_units(&self, audio_seconds: f64) -> Result<i64, String> {
        let pricing = self
            .openrouter_pricing
            .as_deref()
            .ok_or_else(|| "OpenRouter pricing cache is unavailable".to_owned())?
            .transcription_pricing(crate::native_ai::OPENROUTER_TRANSCRIPTION_MODEL)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| {
                OpenRouterChatError::MissingModelPricing {
                    model: crate::native_ai::OPENROUTER_TRANSCRIPTION_MODEL.to_owned(),
                }
                .to_string()
            })?;
        estimate_transcription_reserve_credit_units(audio_seconds, pricing.usd_micros_per_hour)
            .map_err(|error| error.to_string())
    }
}

impl<Files, Cache, Processor, Vision, Transcription> MediaRuntime
    for NativeMedia<Files, Cache, Processor, Vision, Transcription>
where
    Files: MediaFileSource,
    Cache: MediaCache,
    Processor: MediaProcessor,
    Vision: VisionProvider,
    Transcription: TranscriptionProvider,
{
    fn estimate_reserve_credit_units(
        &mut self,
        kind: MediaKind,
        duration_hint_seconds: Option<f64>,
    ) -> Result<i64, String> {
        match kind {
            MediaKind::Image => self.estimate_image_reserve_credit_units(),
            MediaKind::Audio => {
                self.estimate_audio_reserve_credit_units(duration_hint_seconds.unwrap_or(1.0))
            }
        }
    }

    fn prepare_image_for_prompt(
        &mut self,
        file_id: &str,
    ) -> Result<Option<PreparedImagePrompt>, String> {
        let bytes = self
            .files
            .download(file_id)?
            .filter(|bytes| !bytes.is_empty())
            .ok_or_else(|| MediaPipelineError::Download.to_string())?;
        let image = self
            .processor
            .prepare_image(&bytes)?
            .ok_or_else(|| MediaPipelineError::InvalidImage.to_string())?;
        Ok(Some(PreparedImagePrompt {
            bytes: Arc::from(image.bytes),
            mime: image.mime,
        }))
    }

    fn prepare(
        &mut self,
        kind: MediaKind,
        file_id: &str,
        duration_hint_seconds: Option<f64>,
    ) -> Result<PreparedMedia, String> {
        if let Ok(Some(text)) = self.cache.get(kind.cache_prefix(), file_id) {
            eprintln!(
                "Media trace: {}",
                serde_json::json!({
                    "event": "media_cache_hit",
                    "kind": format!("{kind:?}"),
                    "file_id": file_id,
                    "text_chars": text.chars().count(),
                })
            );
            return Ok(PreparedMedia::Cached {
                kind,
                file_id: file_id.to_owned(),
                text,
            });
        }
        let bytes = self
            .files
            .download(file_id)?
            .filter(|bytes| !bytes.is_empty())
            .ok_or_else(|| {
                eprintln!(
                    "Media trace: {}",
                    serde_json::json!({
                        "event": "media_download_empty",
                        "kind": format!("{kind:?}"),
                        "file_id": file_id,
                        "duration_hint_seconds": duration_hint_seconds,
                    })
                );
                MediaPipelineError::Download.to_string()
            })?;
        eprintln!(
            "Media trace: {}",
            serde_json::json!({
                "event": "media_download_result",
                "kind": format!("{kind:?}"),
                "file_id": file_id,
                "bytes": bytes.len(),
                "duration_hint_seconds": duration_hint_seconds,
            })
        );
        match kind {
            MediaKind::Image => {
                let image = self
                    .processor
                    .prepare_image(&bytes)?
                    .ok_or_else(|| MediaPipelineError::InvalidImage.to_string())?;
                let reserve_credit_units = self
                    .estimate_image_reserve_credit_units()
                    .map_err(|error| MediaPipelineError::ReserveEstimate(error).to_string())?;
                Ok(PreparedMedia::Image {
                    file_id: file_id.to_owned(),
                    bytes: image.bytes,
                    mime: image.mime,
                    reserve_credit_units,
                })
            }
            MediaKind::Audio => {
                let audio = self
                    .processor
                    .prepare_audio(&bytes, duration_hint_seconds)?
                    .filter(|audio| {
                        audio.duration_seconds.is_finite() && audio.duration_seconds > 0.0
                    })
                    .ok_or_else(|| {
                        eprintln!(
                            "Media trace: {}",
                            serde_json::json!({
                                "event": "media_audio_invalid",
                                "file_id": file_id,
                                "input_bytes": bytes.len(),
                                "duration_hint_seconds": duration_hint_seconds,
                            })
                        );
                        MediaPipelineError::InvalidAudio.to_string()
                    })?;
                let reserve_credit_units = self
                    .estimate_audio_reserve_credit_units(audio.duration_seconds)
                    .map_err(|error| {
                        eprintln!(
                            "Media trace: {}",
                            serde_json::json!({
                                "event": "media_reserve_estimate_failure",
                                "kind": "Audio",
                                "file_id": file_id,
                                "duration_seconds": audio.duration_seconds,
                                "error": error.chars().take(300).collect::<String>(),
                            })
                        );
                        MediaPipelineError::ReserveEstimate(error).to_string()
                    })?;
                eprintln!(
                    "Media trace: {}",
                    serde_json::json!({
                        "event": "media_audio_prepared",
                        "file_id": file_id,
                        "bytes": audio.bytes.len(),
                        "duration_seconds": audio.duration_seconds,
                        "reserve_credit_units": reserve_credit_units,
                    })
                );
                Ok(PreparedMedia::Audio {
                    file_id: file_id.to_owned(),
                    bytes: audio.bytes,
                    duration_seconds: audio.duration_seconds,
                    reserve_credit_units,
                })
            }
        }
    }

    fn execute(&mut self, prepared: PreparedMedia, prompt: &str) -> Result<MediaExecution, String> {
        let (kind, file_id, result) = match prepared {
            PreparedMedia::Cached {
                kind,
                file_id,
                text,
            } => {
                return Ok(MediaExecution {
                    kind,
                    file_id,
                    text,
                    billing_segment: None,
                    cached: true,
                });
            }
            PreparedMedia::Image {
                file_id,
                bytes,
                mime,
                ..
            } => {
                let image = PreparedImage { bytes, mime };
                let result = self.vision.describe(&image, prompt, &file_id)?;
                (MediaKind::Image, file_id, result)
            }
            PreparedMedia::Audio {
                file_id,
                bytes,
                duration_seconds,
                ..
            } => {
                let audio = PreparedAudio {
                    bytes,
                    duration_seconds,
                };
                let result = self.transcription.transcribe(&audio, &file_id)?;
                (MediaKind::Audio, file_id, result)
            }
        };
        let result = result.ok_or_else(|| {
            eprintln!(
                "Media trace: {}",
                serde_json::json!({
                    "event": "media_provider_empty",
                    "kind": format!("{kind:?}"),
                    "file_id": file_id,
                })
            );
            MediaPipelineError::ProviderUnavailable.to_string()
        })?;
        eprintln!(
            "Media trace: {}",
            serde_json::json!({
                "event": "media_provider_result",
                "kind": format!("{kind:?}"),
                "file_id": file_id,
                "text_chars": result.text.chars().count(),
                "empty": result.text.is_empty(),
            })
        );
        if !result.text.is_empty() {
            let _cache_result = self.cache.set(kind.cache_prefix(), &file_id, &result.text);
        }
        Ok(MediaExecution {
            kind,
            file_id,
            text: result.text,
            billing_segment: Some(result.billing_segment),
            cached: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    use bot_adapters::openrouter_chat::OpenRouterPricingCache;
    use serde_json::json;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct Files(Option<Vec<u8>>);

    impl MediaFileSource for Files {
        fn download(&mut self, _file_id: &str) -> Result<Option<Vec<u8>>, String> {
            Ok(self.0.clone())
        }
    }

    #[derive(Default)]
    struct Cache {
        values: HashMap<String, String>,
    }

    impl MediaCache for Cache {
        fn get(&mut self, prefix: &str, file_id: &str) -> Result<Option<String>, String> {
            Ok(self.values.get(&format!("{prefix}:{file_id}")).cloned())
        }

        fn set(&mut self, prefix: &str, file_id: &str, text: &str) -> Result<(), String> {
            self.values
                .insert(format!("{prefix}:{file_id}"), text.to_owned());
            Ok(())
        }
    }

    /// Decodes images and audio unless told the input is unusable.
    struct Processor {
        decodes_images: bool,
        measures_audio: bool,
    }

    impl MediaProcessor for Processor {
        fn prepare_image(&mut self, input: &[u8]) -> Result<Option<PreparedImage>, String> {
            Ok(self.decodes_images.then(|| PreparedImage {
                bytes: input.to_vec(),
                mime: "image/webp".to_owned(),
            }))
        }

        fn prepare_audio(
            &mut self,
            input: &[u8],
            duration_hint_seconds: Option<f64>,
        ) -> Result<Option<PreparedAudio>, String> {
            Ok(self.measures_audio.then(|| PreparedAudio {
                bytes: input.to_vec(),
                duration_seconds: duration_hint_seconds.unwrap_or(4.5),
            }))
        }
    }

    type CatalogServer = thread::JoinHandle<std::io::Result<()>>;

    fn pricing_cache()
    -> Result<(Arc<OpenRouterPricingCache>, CatalogServer), Box<dyn std::error::Error>> {
        pricing_cache_with(json!({
            "data": [
                {
                    "id": "google/gemini-3.1-flash-lite",
                    "pricing": {"prompt": "0.000001", "completion": "0.000001"}
                },
                {
                    "id": "microsoft/mai-transcribe-2",
                    "architecture": {
                        "modality": "audio->transcription",
                        "output_modalities": ["transcription"]
                    },
                    "pricing": {"prompt": "0.1", "completion": "0"}
                }
            ]
        }))
    }

    /// Serves one catalog response to the pricing cache's first refresh.
    fn pricing_cache_with(
        catalog: Value,
    ) -> Result<(Arc<OpenRouterPricingCache>, CatalogServer), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut request = [0_u8; 8_192];
            let _ = stream.read(&mut request);
            let body = catalog.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes())
        });
        let cache = OpenRouterPricingCache::new("synthetic-key", &format!("http://{address}"))?;
        Ok((Arc::new(cache), server))
    }

    struct Vision;

    impl VisionProvider for Vision {
        fn describe(
            &mut self,
            _image: &PreparedImage,
            _prompt: &str,
            _file_id: &str,
        ) -> Result<Option<MediaProviderResult>, String> {
            Ok(Some(MediaProviderResult {
                text: "synthetic description".to_owned(),
                billing_segment: json!({"kind": "vision"}),
            }))
        }
    }

    /// Transcribes audio unless told the provider returns nothing.
    struct Transcription {
        replies: bool,
    }

    impl TranscriptionProvider for Transcription {
        fn transcribe(
            &mut self,
            _audio: &PreparedAudio,
            _file_id: &str,
        ) -> Result<Option<MediaProviderResult>, String> {
            Ok(self.replies.then(|| MediaProviderResult {
                text: "synthetic transcript".to_owned(),
                billing_segment: json!({"kind": "transcribe"}),
            }))
        }
    }

    type TestMedia = NativeMedia<Files, Cache, Processor, Vision, Transcription>;

    fn media_with(files: Option<Vec<u8>>, model: &str) -> TestMedia {
        NativeMedia::new(
            Files(files),
            Cache::default(),
            Processor {
                decodes_images: true,
                measures_audio: true,
            },
            Vision,
            Transcription { replies: true },
            model,
        )
    }

    fn media(cache: Cache) -> TestMedia {
        let mut media = media_with(Some(vec![1, 2, 3]), "google/gemini-3.1-flash-lite");
        media.cache = cache;
        media
    }

    #[test]
    fn cache_hits_skip_download_reserve_and_provider_billing() {
        let mut cache = Cache::default();
        cache.values.insert(
            "image_description:file-1".to_owned(),
            "cached description".to_owned(),
        );
        let mut media = media(cache);
        let prepared = media.prepare(MediaKind::Image, "file-1", None);
        assert!(matches!(
            prepared,
            Ok(PreparedMedia::Cached { ref text, .. }) if text == "cached description"
        ));
        let result = prepared.and_then(|prepared| media.execute(prepared, "describe"));
        assert!(matches!(
            result,
            Ok(MediaExecution {
                cached: true,
                billing_segment: None,
                ..
            })
        ));
    }

    #[test]
    fn image_and_audio_are_prepared_reserved_executed_and_cached() -> TestResult {
        let (pricing, server) = pricing_cache()?;
        let mut media = media(Cache::default()).with_openrouter_pricing(pricing);
        let image = media.prepare(MediaKind::Image, "image-1", None);
        assert!(matches!(
            image,
            Ok(PreparedMedia::Image {
                reserve_credit_units,
                ..
            }) if reserve_credit_units > 0
        ));
        let image_result = image.and_then(|value| media.execute(value, "describe"));
        assert!(matches!(
            image_result,
            Ok(MediaExecution {
                kind: MediaKind::Image,
                cached: false,
                ..
            })
        ));

        let audio = media.prepare(MediaKind::Audio, "audio-1", Some(4.5));
        assert!(matches!(
            audio,
            Ok(PreparedMedia::Audio {
                duration_seconds: 4.5,
                reserve_credit_units,
                ..
            }) if reserve_credit_units > 0
        ));
        let audio_result = audio.and_then(|value| media.execute(value, "ignored"));
        assert!(matches!(
            audio_result,
            Ok(MediaExecution {
                kind: MediaKind::Audio,
                cached: false,
                ..
            })
        ));
        assert_eq!(
            media
                .cache
                .values
                .get("audio_transcription:audio-1")
                .map(String::as_str),
            Some("synthetic transcript")
        );
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn direct_image_prompt_preparation_returns_processed_image_bytes() -> TestResult {
        let mut media = media(Cache::default());
        let prepared = media
            .prepare_image_for_prompt("image-1")?
            .ok_or("prompt image was not prepared")?;
        assert_eq!(prepared.bytes.as_ref(), &[1, 2, 3]);
        assert_eq!(prepared.mime, "image/webp");
        Ok(())
    }

    #[test]
    fn direct_image_prompt_preparation_rejects_missing_and_undecodable_images() {
        let mut empty = media_with(Some(Vec::new()), "model");
        assert_eq!(
            empty.prepare_image_for_prompt("image-1"),
            Err(MediaPipelineError::Download.to_string())
        );
        let mut undecodable = media(Cache::default());
        undecodable.processor.decodes_images = false;
        assert_eq!(
            undecodable.prepare_image_for_prompt("image-1"),
            Err(MediaPipelineError::InvalidImage.to_string())
        );
        assert_eq!(
            undecodable.prepare(MediaKind::Image, "image-1", None),
            Err(MediaPipelineError::InvalidImage.to_string())
        );
    }

    #[test]
    fn audio_reservation_requires_a_cached_openrouter_price() {
        let mut media = media(Cache::default());
        let result = media.prepare(MediaKind::Audio, "audio-1", Some(4.5));
        assert!(matches!(
            result,
            Err(error) if error.contains("OpenRouter pricing cache is unavailable")
        ));
    }

    #[test]
    fn download_and_decode_failures_are_explicit_without_panics() {
        let mut missing = media_with(None, "model");
        assert_eq!(
            missing.prepare(MediaKind::Image, "missing", None),
            Err(MediaPipelineError::Download.to_string())
        );
    }

    #[test]
    fn unmeasurable_audio_is_rejected_as_invalid() {
        let mut media = media_with(Some(vec![1, 2, 3]), "model");
        media.processor.measures_audio = false;
        assert_eq!(
            media.prepare(MediaKind::Audio, "audio-1", Some(4.5)),
            Err(MediaPipelineError::InvalidAudio.to_string())
        );
    }

    #[test]
    fn empty_provider_results_are_reported_as_unavailable() -> TestResult {
        let (pricing, server) = pricing_cache()?;
        let mut media = media(Cache::default()).with_openrouter_pricing(pricing);
        media.transcription.replies = false;
        let prepared = media.prepare(MediaKind::Audio, "audio-1", Some(4.5))?;
        assert_eq!(
            media.execute(prepared, "ignored"),
            Err(MediaPipelineError::ProviderUnavailable.to_string())
        );
        assert!(media.cache.values.is_empty());
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn prepared_media_reports_its_kind_and_reserve_for_every_variant() {
        let cached = PreparedMedia::Cached {
            kind: MediaKind::Audio,
            file_id: "cached".to_owned(),
            text: "cached transcript".to_owned(),
        };
        let image = PreparedMedia::Image {
            file_id: "image".to_owned(),
            bytes: vec![1],
            mime: "image/webp".to_owned(),
            reserve_credit_units: 11,
        };
        let audio = PreparedMedia::Audio {
            file_id: "audio".to_owned(),
            bytes: vec![2],
            duration_seconds: 3.0,
            reserve_credit_units: 29,
        };
        assert_eq!(cached.kind(), MediaKind::Audio);
        assert_eq!(cached.reserve_credit_units(), 0);
        assert_eq!(image.kind(), MediaKind::Image);
        assert_eq!(image.reserve_credit_units(), 11);
        assert_eq!(audio.kind(), MediaKind::Audio);
        assert_eq!(audio.reserve_credit_units(), 29);
    }

    #[test]
    fn upfront_estimates_match_prepared_reserves_and_default_audio_to_one_second() -> TestResult {
        let (pricing, server) = pricing_cache()?;
        let mut media = media(Cache::default()).with_openrouter_pricing(pricing);
        let image_estimate = media.estimate_reserve_credit_units(MediaKind::Image, None);
        let default_audio = media.estimate_reserve_credit_units(MediaKind::Audio, None);
        let one_second = media.estimate_reserve_credit_units(MediaKind::Audio, Some(1.0));
        let long_audio = media.estimate_reserve_credit_units(MediaKind::Audio, Some(3_600.0));
        assert!(matches!(image_estimate, Ok(units) if units > 0));
        assert!(matches!(default_audio, Ok(units) if units > 0));
        assert_eq!(default_audio, one_second);
        assert!(
            matches!((&long_audio, &one_second), (Ok(long), Ok(short)) if long > short),
            "{long_audio:?} {one_second:?}"
        );
        // A duration no reserve can represent is rejected rather than wrapped.
        let unbounded = media.estimate_reserve_credit_units(MediaKind::Audio, Some(f64::MAX));
        assert!(
            matches!(&unbounded, Err(error) if !error.is_empty()),
            "{unbounded:?}"
        );
        let prepared = media.prepare(MediaKind::Image, "image-1", None);
        assert_eq!(
            prepared.map(|prepared| prepared.reserve_credit_units()),
            image_estimate
        );
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn image_reserve_failures_reject_the_prepared_image() {
        let mut media = media_with(Some(vec![1, 2, 3]), "synthetic/unpriced-vision");
        assert_eq!(
            media.prepare(MediaKind::Image, "image-1", None),
            Err(MediaPipelineError::ReserveEstimate(
                "OpenRouter pricing is unavailable for synthetic/unpriced-vision".to_owned()
            )
            .to_string())
        );
        assert!(media.cache.values.is_empty());
    }

    #[test]
    fn audio_reservation_requires_a_transcription_price_in_the_catalog() -> TestResult {
        let served = pricing_cache_with(json!({
            "data": [{
                "id": "google/gemini-3.1-flash-lite",
                "pricing": {"prompt": "0.000001", "completion": "0.000001"}
            }]
        }));
        let (pricing, server) = served?;
        let mut media = media(Cache::default()).with_openrouter_pricing(pricing);
        let expected = OpenRouterChatError::MissingModelPricing {
            model: crate::native_ai::OPENROUTER_TRANSCRIPTION_MODEL.to_owned(),
        }
        .to_string();
        assert_eq!(
            media.estimate_reserve_credit_units(MediaKind::Audio, Some(4.5)),
            Err(expected.clone())
        );
        assert_eq!(
            media.prepare(MediaKind::Audio, "audio-1", Some(4.5)),
            Err(MediaPipelineError::ReserveEstimate(expected).to_string())
        );
        assert!(matches!(server.join(), Ok(Ok(()))));
        Ok(())
    }

    #[test]
    fn unreachable_pricing_catalog_blocks_audio_reservations() -> TestResult {
        let pricing = Arc::new(OpenRouterPricingCache::new("synthetic-key", "not-a-url")?);
        let mut media = media(Cache::default()).with_openrouter_pricing(pricing);
        let estimate = media.estimate_reserve_credit_units(MediaKind::Audio, Some(4.5));
        assert!(
            matches!(&estimate, Err(error) if !error.is_empty()),
            "{estimate:?}"
        );
        Ok(())
    }

    #[test]
    fn standard_vision_reserve_rejects_prices_it_cannot_represent() {
        let pricing = TokenPricing {
            input_per_million: i128::MAX,
            cached_input_per_million: None,
            cache_write_per_million: None,
            audio_input_per_million: None,
            output_per_million: i128::MAX,
        };
        let estimate = estimate_standard_vision_reserve_credit_units(&pricing);
        assert!(
            matches!(&estimate, Err(error) if !error.is_empty()),
            "{estimate:?}"
        );
    }
}
