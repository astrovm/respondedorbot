//! Picks how a downloaded file goes to Telegram: as a photo, an animation, a
//! video, an audio track or a plain document. The file's first bytes decide,
//! because servers often send a wrong or generic content type.

/// The largest photo the Bot API accepts. Larger images go as documents.
pub const PHOTO_MAX_BYTES: usize = 10 * 1024 * 1024;
/// The largest file a bot can upload.
pub const UPLOAD_MAX_BYTES: usize = 50 * 1024 * 1024;
/// The longest media caption Telegram shows.
pub const CAPTION_MAX_CHARS: usize = 1024;
const FILE_NAME_MAX_CHARS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Photo,
    Animation,
    Video,
    Audio,
    Document,
}

impl MediaKind {
    /// The Bot API method that sends this kind.
    #[must_use]
    pub const fn method(self) -> &'static str {
        match self {
            Self::Photo => "sendPhoto",
            Self::Animation => "sendAnimation",
            Self::Video => "sendVideo",
            Self::Audio => "sendAudio",
            Self::Document => "sendDocument",
        }
    }

    /// The multipart field that holds the file.
    #[must_use]
    pub const fn field(self) -> &'static str {
        match self {
            Self::Photo => "photo",
            Self::Animation => "animation",
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Document => "document",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaFile {
    pub kind: MediaKind,
    pub content_type: String,
    pub file_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaRejection {
    Empty,
    TooLarge,
    /// The address is a web page, not a file.
    WebPage,
}

/// A known format: its content type, its usual extensions (the first is the
/// one added to a name without one) and how Telegram shows it.
struct Format(&'static str, &'static [&'static str], MediaKind);

const JPEG: Format = Format("image/jpeg", &["jpg", "jpeg"], MediaKind::Photo);
const PNG: Format = Format("image/png", &["png"], MediaKind::Photo);
const WEBP: Format = Format("image/webp", &["webp"], MediaKind::Photo);
const GIF: Format = Format("image/gif", &["gif"], MediaKind::Animation);
const MP4: Format = Format("video/mp4", &["mp4", "m4v"], MediaKind::Video);
const QUICKTIME: Format = Format("video/quicktime", &["mov"], MediaKind::Document);
const M4A: Format = Format("audio/mp4", &["m4a"], MediaKind::Audio);
const HEIC: Format = Format("image/heic", &["heic"], MediaKind::Document);
const AVIF: Format = Format("image/avif", &["avif"], MediaKind::Document);
const WEBM: Format = Format("video/webm", &["webm", "mkv"], MediaKind::Document);
const MP3: Format = Format("audio/mpeg", &["mp3"], MediaKind::Audio);
const OGG: Format = Format("audio/ogg", &["ogg", "oga", "opus"], MediaKind::Document);
const PDF: Format = Format("application/pdf", &["pdf"], MediaKind::Document);

/// Decides how to send `body`, which was served as `content_type` from a URL
/// whose path is `url_path`.
pub fn classify_media(
    content_type: &str,
    body: &[u8],
    url_path: &str,
) -> Result<MediaFile, MediaRejection> {
    if body.is_empty() {
        return Err(MediaRejection::Empty);
    }
    if body.len() > UPLOAD_MAX_BYTES {
        return Err(MediaRejection::TooLarge);
    }
    let declared = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let Some(format) = sniff(body) else {
        if declared.contains("html") || looks_like_html(body) {
            return Err(MediaRejection::WebPage);
        }
        // The type goes into the upload's headers, so it must parse there.
        let content_type = if is_mime_type(&declared) {
            declared
        } else {
            "application/octet-stream".to_owned()
        };
        return Ok(MediaFile {
            kind: MediaKind::Document,
            content_type,
            file_name: file_name(url_path, &[]),
        });
    };
    let kind = if format.2 == MediaKind::Photo && body.len() > PHOTO_MAX_BYTES {
        MediaKind::Document
    } else {
        format.2
    };
    Ok(MediaFile {
        kind,
        content_type: format.0.to_owned(),
        file_name: file_name(url_path, format.1),
    })
}

fn sniff(body: &[u8]) -> Option<Format> {
    let starts = |prefix: &[u8]| body.starts_with(prefix);
    let at = |range: std::ops::Range<usize>| body.get(range).unwrap_or_default();
    if starts(b"\xFF\xD8\xFF") {
        Some(JPEG)
    } else if starts(b"\x89PNG\r\n\x1A\n") {
        Some(PNG)
    } else if starts(b"RIFF") && at(8..12) == b"WEBP" {
        Some(WEBP)
    } else if starts(b"GIF87a") || starts(b"GIF89a") {
        Some(GIF)
    } else if at(4..8) == b"ftyp" {
        Some(match at(8..12) {
            b"qt  " => QUICKTIME,
            b"M4A " | b"M4B " => M4A,
            b"heic" | b"heix" | b"mif1" | b"msf1" => HEIC,
            b"avif" | b"avis" => AVIF,
            _ => MP4,
        })
    } else if starts(b"\x1A\x45\xDF\xA3") {
        Some(WEBM)
    } else if starts(b"ID3") || matches!(at(0..2), [0xFF, 0xFB | 0xFA | 0xF3 | 0xF2]) {
        Some(MP3)
    } else if starts(b"OggS") {
        Some(OGG)
    } else if starts(b"%PDF-") {
        Some(PDF)
    } else {
        None
    }
}

/// A bare `type/subtype` made only of the characters a MIME token allows.
fn is_mime_type(value: &str) -> bool {
    value.split_once('/').is_some_and(|(kind, subtype)| {
        [kind, subtype].iter().all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"!#$&-^_.+".contains(&byte))
        })
    })
}

fn looks_like_html(body: &[u8]) -> bool {
    let start = String::from_utf8_lossy(body.get(..512).unwrap_or(body)).to_ascii_lowercase();
    let start = start.trim_start_matches('\u{feff}').trim_start();
    start.starts_with("<!doctype html") || start.starts_with("<html")
}

/// The last part of the URL path as a safe file name, with the format's
/// extension added when the name has none of its extensions.
fn file_name(url_path: &str, extensions: &[&str]) -> String {
    let last = url_path.rsplit('/').next().unwrap_or_default();
    let cleaned = last
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_')
        })
        .take(FILE_NAME_MAX_CHARS)
        .collect::<String>();
    let base = match cleaned.trim_matches('.') {
        "" => "file",
        base => base,
    };
    let lower = base.to_ascii_lowercase();
    match extensions.first() {
        Some(extension)
            if !extensions
                .iter()
                .any(|known| lower.ends_with(&format!(".{known}"))) =>
        {
            format!("{base}.{extension}")
        }
        _ => base.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classified(
        content_type: &str,
        body: &[u8],
        path: &str,
    ) -> Result<(MediaKind, String, String), MediaRejection> {
        classify_media(content_type, body, path)
            .map(|file| (file.kind, file.content_type, file.file_name))
    }

    fn mp4_brand(brand: &[u8; 4]) -> Vec<u8> {
        [b"\0\0\0\x18ftyp".as_slice(), brand, b"rest"].concat()
    }

    #[test]
    fn the_file_bytes_pick_the_kind_whatever_the_server_says() {
        let cases: [(&[u8], MediaKind, &str); 16] = [
            (b"\xFF\xD8\xFF\xE0jpeg", MediaKind::Photo, "image/jpeg"),
            (b"\x89PNG\r\n\x1A\nrest", MediaKind::Photo, "image/png"),
            (b"RIFF\0\0\0\0WEBPVP8 ", MediaKind::Photo, "image/webp"),
            (b"GIF89a....", MediaKind::Animation, "image/gif"),
            (b"GIF87a....", MediaKind::Animation, "image/gif"),
            (&mp4_brand(b"isom"), MediaKind::Video, "video/mp4"),
            (&mp4_brand(b"qt  "), MediaKind::Document, "video/quicktime"),
            (&mp4_brand(b"M4A "), MediaKind::Audio, "audio/mp4"),
            (&mp4_brand(b"heic"), MediaKind::Document, "image/heic"),
            (&mp4_brand(b"avif"), MediaKind::Document, "image/avif"),
            (b"\x1A\x45\xDF\xA3webm", MediaKind::Document, "video/webm"),
            (b"ID3\x04tags", MediaKind::Audio, "audio/mpeg"),
            (b"\xFF\xFB\x90frame", MediaKind::Audio, "audio/mpeg"),
            (b"\xFF\xFA\x90frame", MediaKind::Audio, "audio/mpeg"),
            (b"OggS\0rest", MediaKind::Document, "audio/ogg"),
            (b"%PDF-1.7", MediaKind::Document, "application/pdf"),
        ];
        for (body, kind, content_type) in cases {
            assert_eq!(
                classified("application/octet-stream", body, "/x")
                    .map(|(found, found_type, _)| (found, found_type)),
                Ok((kind, content_type.to_owned())),
                "{body:?}"
            );
        }
    }

    #[test]
    fn unknown_files_go_as_documents_with_the_declared_type() {
        assert_eq!(
            classified("text/csv; charset=utf-8", b"a,b\n1,2\n", "/data/report.csv"),
            Ok((
                MediaKind::Document,
                "text/csv".to_owned(),
                "report.csv".to_owned()
            ))
        );
        for broken in ["application/x-foo bar", "a/b/c", "/plain", "text/", ""] {
            assert_eq!(
                classified(broken, b"\0\x01binary", "/"),
                Ok((
                    MediaKind::Document,
                    "application/octet-stream".to_owned(),
                    "file".to_owned()
                )),
                "{broken}"
            );
        }
        // A RIFF file that is not WebP, such as WAV audio, is no picture.
        assert_eq!(
            classified("audio/wav", b"RIFF\0\0\0\0WAVEfmt ", "/a.wav"),
            Ok((
                MediaKind::Document,
                "audio/wav".to_owned(),
                "a.wav".to_owned()
            ))
        );
        assert_eq!(
            classified("", b"\0\x01binary", "/"),
            Ok((
                MediaKind::Document,
                "application/octet-stream".to_owned(),
                "file".to_owned()
            ))
        );
    }

    #[test]
    fn web_pages_empty_and_oversized_files_are_rejected() {
        assert_eq!(
            classify_media("text/html; charset=utf-8", b"<p>hi</p>", "/"),
            Err(MediaRejection::WebPage)
        );
        assert_eq!(
            classify_media(
                "application/octet-stream",
                b"\xEF\xBB\xBF  <!DOCTYPE html><html>",
                "/"
            ),
            Err(MediaRejection::WebPage)
        );
        assert_eq!(
            classify_media("text/plain", b"\n<HTML><body>", "/"),
            Err(MediaRejection::WebPage)
        );
        assert_eq!(
            classify_media("image/png", b"", "/a.png"),
            Err(MediaRejection::Empty)
        );
        let mut largest = vec![0; UPLOAD_MAX_BYTES];
        assert!(classify_media("application/octet-stream", &largest, "/a.bin").is_ok());
        largest.push(0);
        assert_eq!(
            classify_media("image/png", &largest, "/a.png"),
            Err(MediaRejection::TooLarge)
        );
    }

    #[test]
    fn the_limits_are_the_bot_apis() {
        assert_eq!(PHOTO_MAX_BYTES, 10_485_760);
        assert_eq!(UPLOAD_MAX_BYTES, 52_428_800);
    }

    #[test]
    fn photos_over_the_photo_limit_go_as_documents() {
        let mut large = b"\xFF\xD8\xFF".to_vec();
        large.resize(PHOTO_MAX_BYTES + 1, 0);
        let kind = |body: &[u8]| classified("image/jpeg", body, "/big.jpg").map(|file| file.0);
        assert_eq!(kind(&large), Ok(MediaKind::Document));
        large.truncate(PHOTO_MAX_BYTES);
        assert_eq!(kind(&large), Ok(MediaKind::Photo));
    }

    #[test]
    fn file_names_are_safe_and_keep_or_gain_the_extension() {
        let jpeg = b"\xFF\xD8\xFF";
        let name = |path: &str| {
            classified("", jpeg, path)
                .map(|file| file.2)
                .unwrap_or_default()
        };
        assert_eq!(name("/photos/cat.JPEG"), "cat.JPEG");
        assert_eq!(name("/photos/cat.jpg"), "cat.jpg");
        assert_eq!(name("/photos/cat"), "cat.jpg");
        assert_eq!(name("/photos/c%20a t$.png"), "c20at.png.jpg");
        assert_eq!(name("/photos/"), "file.jpg");
        assert_eq!(name("/.."), "file.jpg");
        assert_eq!(name(&format!("/{}", "a".repeat(100))).len(), 68);
    }

    #[test]
    fn each_kind_names_its_method_and_field() {
        let kinds = [
            MediaKind::Photo,
            MediaKind::Animation,
            MediaKind::Video,
            MediaKind::Audio,
            MediaKind::Document,
        ];
        assert_eq!(
            kinds.map(MediaKind::method),
            [
                "sendPhoto",
                "sendAnimation",
                "sendVideo",
                "sendAudio",
                "sendDocument"
            ]
        );
        assert_eq!(
            kinds.map(MediaKind::field),
            ["photo", "animation", "video", "audio", "document"]
        );
    }
}
