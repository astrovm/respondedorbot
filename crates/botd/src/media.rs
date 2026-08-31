//! Native cache/download/prepare/provider pipeline for Telegram media.

use bot_adapters::media_provider::MediaProviderResult;
use bot_core::ai_reserve::{
    VISION_OUTPUT_TOKEN_LIMIT, estimate_transcription_reserve_credit_units,
    estimate_vision_reserve_credit_units,
};
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

pub trait MediaRuntime {
    fn prepare(
        &mut self,
        kind: MediaKind,
        file_id: &str,
        duration_hint_seconds: Option<f64>,
    ) -> Result<PreparedMedia, String>;

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

pub struct NativeMedia<Files, Cache, Processor, Vision, Transcription> {
    files: Files,
    cache: Cache,
    processor: Processor,
    vision: Vision,
    transcription: Transcription,
    vision_model: String,
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
        }
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
    fn prepare(
        &mut self,
        kind: MediaKind,
        file_id: &str,
        duration_hint_seconds: Option<f64>,
    ) -> Result<PreparedMedia, String> {
        if let Ok(Some(text)) = self.cache.get(kind.cache_prefix(), file_id) {
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
            .ok_or_else(|| MediaPipelineError::Download.to_string())?;
        match kind {
            MediaKind::Image => {
                let image = self
                    .processor
                    .prepare_image(&bytes)?
                    .ok_or_else(|| MediaPipelineError::InvalidImage.to_string())?;
                let reserve_credit_units = estimate_vision_reserve_credit_units(
                    "Describe what you see in this image in detail.",
                    0,
                    1_200,
                    VISION_OUTPUT_TOKEN_LIMIT,
                    &self.vision_model,
                )
                .map_err(|error| {
                    MediaPipelineError::ReserveEstimate(error.to_string()).to_string()
                })?;
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
                    .ok_or_else(|| MediaPipelineError::InvalidAudio.to_string())?;
                let reserve_credit_units =
                    estimate_transcription_reserve_credit_units(audio.duration_seconds).map_err(
                        |error| MediaPipelineError::ReserveEstimate(error.to_string()).to_string(),
                    )?;
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
        let result = result.ok_or_else(|| MediaPipelineError::ProviderUnavailable.to_string())?;
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

    use serde_json::json;

    use super::*;

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

    struct Processor;

    impl MediaProcessor for Processor {
        fn prepare_image(&mut self, input: &[u8]) -> Result<Option<PreparedImage>, String> {
            Ok(Some(PreparedImage {
                bytes: input.to_vec(),
                mime: "image/webp".to_owned(),
            }))
        }

        fn prepare_audio(
            &mut self,
            input: &[u8],
            duration_hint_seconds: Option<f64>,
        ) -> Result<Option<PreparedAudio>, String> {
            Ok(Some(PreparedAudio {
                bytes: input.to_vec(),
                duration_seconds: duration_hint_seconds.unwrap_or(4.5),
            }))
        }
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

    struct Transcription;

    impl TranscriptionProvider for Transcription {
        fn transcribe(
            &mut self,
            _audio: &PreparedAudio,
            _file_id: &str,
        ) -> Result<Option<MediaProviderResult>, String> {
            Ok(Some(MediaProviderResult {
                text: "synthetic transcript".to_owned(),
                billing_segment: json!({"kind": "transcribe"}),
            }))
        }
    }

    fn media(cache: Cache) -> NativeMedia<Files, Cache, Processor, Vision, Transcription> {
        NativeMedia::new(
            Files(Some(vec![1, 2, 3])),
            cache,
            Processor,
            Vision,
            Transcription,
            "google/gemini-3.1-flash-lite-preview",
        )
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
    fn image_and_audio_are_prepared_reserved_executed_and_cached() {
        let mut media = media(Cache::default());
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
    }

    #[test]
    fn download_and_decode_failures_are_explicit_without_panics() {
        let mut missing = NativeMedia::new(
            Files(None),
            Cache::default(),
            Processor,
            Vision,
            Transcription,
            "model",
        );
        assert_eq!(
            missing.prepare(MediaKind::Image, "missing", None),
            Err(MediaPipelineError::Download.to_string())
        );
    }
}
