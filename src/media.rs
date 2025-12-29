use crate::storage::Storage;
use crate::telegram;
use image::GenericImageView;

const TTL_MEDIA_CACHE: u64 = 7 * 24 * 60 * 60;

pub async fn get_cached_transcription(
    storage: &Storage,
    file_id: &str,
) -> Option<String> {
    let key = format!("audio_transcription:{file_id}");
    storage.get_string(&key).await
}

pub async fn cache_transcription(
    storage: &Storage,
    file_id: &str,
    text: &str,
) {
    let key = format!("audio_transcription:{file_id}");
    let _ = storage
        .set_string_with_ttl(&key, TTL_MEDIA_CACHE, text)
        .await;
}

pub async fn get_cached_description(
    storage: &Storage,
    file_id: &str,
) -> Option<String> {
    let key = format!("image_description:{file_id}");
    storage.get_string(&key).await
}

pub async fn cache_description(
    storage: &Storage,
    file_id: &str,
    text: &str,
) {
    let key = format!("image_description:{file_id}");
    let _ = storage
        .set_string_with_ttl(&key, TTL_MEDIA_CACHE, text)
        .await;
}

pub async fn transcribe_file_by_id(
    http: &reqwest::Client,
    storage: &Storage,
    token: &str,
    file_id: &str,
    use_cache: bool,
) -> Result<Option<String>, String> {
    if use_cache {
        if let Some(cached) = get_cached_transcription(storage, file_id).await {
            return Ok(Some(cached));
        }
    }

    let bytes = telegram::download_file(http, token, file_id)
        .await
        .map_err(|_| "download".to_string())?;
    let text = transcribe_audio_cloudflare(http, &bytes).await;
    if let Some(text) = text.clone() {
        cache_transcription(storage, file_id, &text).await;
        return Ok(Some(text));
    }
    Err("transcribe".to_string())
}

pub async fn describe_media_by_id(
    http: &reqwest::Client,
    storage: &Storage,
    token: &str,
    file_id: &str,
    prompt: &str,
) -> Result<Option<String>, String> {
    if let Some(cached) = get_cached_description(storage, file_id).await {
        return Ok(Some(cached));
    }

    let bytes = telegram::download_file(http, token, file_id)
        .await
        .map_err(|_| "download".to_string())?;
    let resized = resize_image_if_needed(&bytes, 512);
    let text = describe_image_cloudflare(http, &resized, prompt).await;
    if let Some(text) = text.clone() {
        cache_description(storage, file_id, &text).await;
        return Ok(Some(text));
    }
    Err("describe".to_string())
}

async fn describe_image_cloudflare(
    http: &reqwest::Client,
    image_data: &[u8],
    prompt: &str,
) -> Option<String> {
    let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").ok()?;
    let api_key = std::env::var("CLOUDFLARE_API_KEY").ok()?;
    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/llava-hf/llava-1.5-7b-hf"
    );

    let image_array: Vec<u8> = image_data.to_vec();
    let payload = serde_json::json!({
        "prompt": prompt,
        "image": image_array,
        "max_tokens": 1024,
    });

    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .ok()?;

    let body = resp.json::<serde_json::Value>().await.ok()?;
    let result = body.get("result")?;
    let description = result
        .get("response")
        .or_else(|| result.get("description"))
        .and_then(|v| v.as_str())?;
    Some(description.to_string())
}

async fn transcribe_audio_cloudflare(
    http: &reqwest::Client,
    audio_data: &[u8],
) -> Option<String> {
    let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID").ok()?;
    let api_key = std::env::var("CLOUDFLARE_API_KEY").ok()?;
    let url = format!(
        "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper-large-v3-turbo"
    );

    let resp = http
        .post(url)
        .bearer_auth(api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        .body(audio_data.to_vec())
        .send()
        .await
        .ok()?;

    let body = resp.json::<serde_json::Value>().await.ok()?;
    if body.get("success").and_then(|v| v.as_bool()) != Some(true) {
        return None;
    }
    body.get("result")
        .and_then(|v| v.get("text"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn resize_image_if_needed(image_data: &[u8], max_size: u32) -> Vec<u8> {
    let img = match image::load_from_memory(image_data) {
        Ok(img) => img,
        Err(_) => return image_data.to_vec(),
    };

    let (width, height) = img.dimensions();
    if width <= max_size && height <= max_size {
        return image_data.to_vec();
    }

    let resized = img.thumbnail(max_size, max_size);
    let mut buf = Vec::new();
    if resized
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .is_ok()
    {
        buf
    } else {
        image_data.to_vec()
    }
}
