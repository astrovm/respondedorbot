"""Turn Telegram media into text that the rest of the bot can understand.

The main flow is: check the cache, download the file, call a media provider,
and return both the text and the information needed to bill that provider call.
"""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from api.ai_pricing import AIUsageResult


def execute_groq_request_with_fallback(
    scope: str,
    *,
    label: str,
    token_count: int,
    audio_seconds: float,
    attempt: Callable[[str], AIUsageResult | None],
    get_accounts: Callable[[], list[str]],
    invoke_provider: Callable[..., Any],
    get_backoff_key: Callable[[str, str], str],
    log_result: Callable[..., None],
    should_try_next_account: Callable[[Exception], bool],
    is_backoff_active: Callable[[str], bool],
    default_backoff_seconds: int,
) -> AIUsageResult | None:
    """Try each configured Groq account until one succeeds or must stop.

    ProviderRuntime owns cooldowns and rate-limit handling. This helper only
    decides whether a recoverable failure should move to the next account.
    """

    configured_accounts = list(get_accounts())
    if not configured_accounts:
        print("Groq API key not configured")
        return None

    for account in configured_accounts:
        last_error: Exception | None = None

        def wrapped_attempt() -> AIUsageResult | None:
            nonlocal last_error
            try:
                return attempt(account)
            except Exception as error:
                last_error = error
                raise

        request_started_at = time.monotonic()
        result = cast(
            AIUsageResult | None,
            invoke_provider(
                "groq",
                attempt=wrapped_attempt,
                rate_limit_backoff=default_backoff_seconds,
                label=f"{label} ({account})",
                backoff_key=get_backoff_key(account, scope),
            ),
        )
        elapsed = max(0.0, time.monotonic() - request_started_at)
        if result is not None:
            result.metadata.setdefault("request_elapsed_seconds", elapsed)
        log_result(
            label=label,
            scope=scope,
            account=account,
            token_count=token_count,
            audio_seconds=audio_seconds,
            result=result,
        )
        if result:
            result.metadata.setdefault("groq_account", account)
            return result
        if last_error and should_try_next_account(last_error):
            print(
                f"{label} retrying with next account after recoverable "
                f"error on account={account}"
            )
            continue
        if is_backoff_active(get_backoff_key(account, scope)):
            continue
        break
    return None


def describe_image_result(
    image_data: bytes,
    user_text: str,
    file_id: str | None,
    *,
    use_cache: bool,
    get_cached_description: Callable[[str], str | None],
    get_client: Callable[[], Any],
    prepare_image: Callable[[bytes], tuple[bytes, str] | None],
    encode_image: Callable[[bytes], str],
    increment_request_count: Callable[[], None],
    build_usage_result: Callable[..., AIUsageResult],
    cache_description: Callable[[str, str], None],
    admin_report: Callable[..., None],
    logger: Any,
    model: str,
    max_tokens: int,
    no_markdown_prompt: str,
) -> AIUsageResult | None:
    """Describe one image, reusing a cached description when possible."""

    if file_id and use_cache:
        cached = get_cached_description(file_id)
        if cached:
            return AIUsageResult(
                kind="vision",
                text=str(cached),
                model=model,
                cached=True,
                metadata={"file_id": file_id, "cache_hit": True},
            )

    client = get_client()
    if client is None:
        return None

    prepared_image = prepare_image(image_data)
    if prepared_image is None:
        logger.info(
            "vision image decode failed file_id=%s bytes=%d",
            file_id,
            len(image_data),
        )
        return None
    image_bytes, image_mime = prepared_image

    print(f"Describing image with {model}...")
    increment_request_count()
    image_url = f"data:{image_mime};base64,{encode_image(image_bytes)}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=cast(
                Any,
                [
                    {
                        "role": "system",
                        "content": (
                            "respondé siempre en minúsculas, "
                            "sin emojis, sin markdown, en lenguaje coloquial "
                            f"argentino. {no_markdown_prompt}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    },
                ],
            ),
            max_tokens=max_tokens,
        )
    except Exception as error:
        admin_report(
            f"Vision error model={model}",
            error,
            {"model": model},
        )
        return None

    if response and hasattr(response, "choices") and response.choices:
        content = getattr(response.choices[0].message, "content", None)
        if content:
            description = str(content)
            logger.info(
                "image description success preview='%s'",
                description[:100],
            )
            result = build_usage_result(
                kind="vision",
                text=description,
                model=model,
                response=response,
                metadata={"file_id": file_id, "provider": "openrouter"},
            )
            if file_id:
                cache_description(file_id, description)
            return result
    return None


def transcribe_audio_openrouter_result(
    audio_data: bytes,
    file_id: str | None,
    *,
    get_client: Callable[[], Any],
    increment_request_count: Callable[[], None],
    build_usage_result: Callable[..., AIUsageResult],
    model: str,
) -> AIUsageResult | None:
    """Transcribe audio with OpenRouter when the preferred Groq path fails."""

    client = get_client()
    if client is None:
        print("OpenRouter transcription: no client available")
        return None

    audio_format = "webm"
    if audio_data.startswith((b"\x1aE\xdf\xa3", b"ID3")):
        audio_format = "mp3"
    elif audio_data.startswith(b"OggS"):
        audio_format = "ogg"

    print(f"Transcribing audio with OpenRouter Gemini using model={model}...")
    increment_request_count()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=cast(
                Any,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "format": audio_format,
                                    "data": base64.b64encode(audio_data).decode(
                                        "utf-8"
                                    ),
                                },
                            },
                            {
                                "type": "text",
                                "text": "Transcribe this audio exactly as spoken.",
                            },
                        ],
                    }
                ],
            ),
            max_tokens=4096,
        )
    except Exception as error:
        print(f"OpenRouter transcription error: {error}")
        return None

    if response and hasattr(response, "choices") and response.choices:
        content = getattr(response.choices[0].message, "content", None)
        if content:
            print(f"Audio transcribed successfully: {content[:100]}...")
            return build_usage_result(
                kind="transcribe",
                text=str(content),
                model=model,
                response=response,
                audio_seconds=0.0,
                metadata={
                    "file_id": file_id,
                    "cache_hit": False,
                    "provider": "openrouter",
                },
            )
    return None


def transcribe_audio_result(
    audio_data: bytes,
    file_id: str | None,
    *,
    use_cache: bool,
    get_cached_transcription: Callable[[str], str | None],
    measure_duration: Callable[[bytes], float | None],
    get_native_client: Callable[[str], Any],
    build_usage_result: Callable[..., AIUsageResult],
    execute_with_fallback: Callable[..., AIUsageResult | None],
    openrouter_fallback: Callable[[bytes, str | None], AIUsageResult | None],
    cache_transcription: Callable[[str, str], None],
    model: str,
) -> AIUsageResult | None:
    """Transcribe audio through cache -> Groq accounts -> OpenRouter fallback."""

    if file_id and use_cache:
        cached = get_cached_transcription(file_id)
        if cached:
            return AIUsageResult(
                kind="transcribe",
                text=str(cached),
                model=model,
                cached=True,
                metadata={"file_id": file_id, "cache_hit": True},
            )

    measured_audio_seconds = measure_duration(audio_data)

    def attempt(account: str) -> AIUsageResult | None:
        native_client = get_native_client(account)
        if native_client is None:
            return None
        print(
            "Transcribing audio with Groq Whisper "
            f"using account={account}..."
        )
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.webm"
        response = native_client.audio.transcriptions.create(
            model=model,
            file=audio_file,
        )
        transcription = (
            response.get("text")
            if isinstance(response, dict)
            else getattr(response, "text", None)
        )
        if transcription:
            print(f"Audio transcribed successfully: {transcription[:100]}...")
            return build_usage_result(
                kind="transcribe",
                text=str(transcription),
                model=model,
                response=response,
                audio_seconds=measured_audio_seconds,
                metadata={
                    "file_id": file_id,
                    "cache_hit": False,
                    "groq_account": account,
                },
            )
        return None

    result = execute_with_fallback(
        attempt=attempt,
        scope="transcribe",
        label="Groq Whisper",
        audio_seconds=measured_audio_seconds or 0.0,
    )
    if result is None:
        print("Groq transcription failed, trying OpenRouter fallback...")
        result = openrouter_fallback(audio_data, file_id)
    if result and result.text and file_id:
        cache_transcription(file_id, result.text)
    return result


def process_media_with_cache(
    *,
    file_id: str,
    use_cache: bool,
    cache_lookup: Callable[[str], str | None] | None,
    processor: Callable[[bytes], AIUsageResult | None],
    downloader: Callable[[str], bytes | None],
    measure_duration: Callable[[bytes], float | None],
    failure_code: str,
    logger: Any,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Run the common cache/download/process flow for Telegram media.

    The tuple contains successful text, a short failure code, and an optional
    billing segment. Exactly one of text or failure code is normally present.
    """

    try:
        if use_cache and cache_lookup:
            cached_value = cache_lookup(file_id)
            if cached_value:
                return str(cached_value), None, None
        media_bytes = downloader(file_id)
        if not media_bytes:
            return None, "download", None
        result = processor(media_bytes)
        if result:
            if not result.audio_seconds and result.kind == "transcribe":
                result.audio_seconds = measure_duration(media_bytes)
            return result.text, None, result.billing_segment()
        return None, failure_code, None
    except Exception:
        logger.exception("Error processing media %s", file_id)
        return None, failure_code, None


def transcribe_file_by_id(
    file_id: str,
    use_cache: bool,
    *,
    get_cached_transcription: Callable[[str], str | None],
    download_file: Callable[[str], bytes | None],
    measure_duration: Callable[[bytes], float | None],
    extract_audio: Callable[[bytes], bytes | None],
    transcribe: Callable[..., AIUsageResult | None],
    process_media: Callable[..., tuple[
        str | None, str | None, dict[str, Any] | None
    ]],
    logger: Any,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """Download and transcribe a Telegram audio or video file."""

    def processor(media_bytes: bytes) -> AIUsageResult | None:
        duration_seconds = measure_duration(media_bytes)
        if duration_seconds is None:
            # Telegram videos need their audio track extracted before Whisper
            # can measure and transcribe them.
            extracted = extract_audio(media_bytes)
            if extracted:
                logger.info("Extracted audio from video for transcription")
                media_bytes = extracted
                duration_seconds = measure_duration(media_bytes)
            if duration_seconds is None:
                return None
        result = transcribe(media_bytes, file_id, use_cache=use_cache)
        if result and not result.audio_seconds:
            result.audio_seconds = duration_seconds
        return result

    return process_media(
        file_id=file_id,
        use_cache=use_cache,
        cache_lookup=get_cached_transcription,
        processor=processor,
        downloader=download_file,
        failure_code="transcribe",
    )


def describe_media_by_id(
    file_id: str,
    prompt: str,
    *,
    get_cached_description: Callable[[str], str | None],
    download_file: Callable[[str], bytes | None],
    describe_image: Callable[..., AIUsageResult | None],
    process_media: Callable[..., tuple[
        str | None, str | None, dict[str, Any] | None
    ]],
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    def processor(media: bytes) -> AIUsageResult | None:
        return describe_image(media, prompt, file_id)

    return process_media(
        file_id=file_id,
        use_cache=True,
        cache_lookup=get_cached_description,
        processor=processor,
        downloader=download_file,
        failure_code="describe",
    )


@dataclass
class MediaServiceDeps:
    """External tools used by MediaService, collected in one explicit bundle."""

    cache: Any
    telegram: Any
    images: Any
    get_openrouter_client: Callable[[], Any]
    get_groq_native_client: Callable[[str], Any]
    get_groq_accounts: Callable[[], list[str]]
    invoke_provider: Callable[..., Any]
    get_backoff_key: Callable[[str, str], str]
    log_result: Callable[..., None]
    should_try_next_account: Callable[[Exception], bool]
    is_backoff_active: Callable[[str], bool]
    increment_request_count: Callable[[], None]
    build_usage_result: Callable[..., AIUsageResult]
    admin_report: Callable[..., None]
    measure_duration: Callable[[bytes], float | None]
    extract_audio: Callable[[bytes], bytes | None]
    logger: Any
    vision_model: str
    vision_max_tokens: int
    transcribe_model: str
    openrouter_transcribe_model: str
    no_markdown_prompt: str
    default_backoff_seconds: int


class MediaService:
    """Public media API used by handlers and the application runtime."""

    def __init__(self, deps: MediaServiceDeps) -> None:
        self._deps = deps

    def execute_groq_request(
        self,
        scope: str,
        *,
        label: str,
        token_count: int = 0,
        audio_seconds: float = 0.0,
        attempt: Callable[[str], AIUsageResult | None],
    ) -> AIUsageResult | None:
        return execute_groq_request_with_fallback(
            scope,
            label=label,
            token_count=token_count,
            audio_seconds=audio_seconds,
            attempt=attempt,
            get_accounts=self._deps.get_groq_accounts,
            invoke_provider=self._deps.invoke_provider,
            get_backoff_key=self._deps.get_backoff_key,
            log_result=self._deps.log_result,
            should_try_next_account=self._deps.should_try_next_account,
            is_backoff_active=self._deps.is_backoff_active,
            default_backoff_seconds=self._deps.default_backoff_seconds,
        )

    def describe_image_result(
        self,
        image_data: bytes,
        user_text: str = "¿Qué ves en esta imagen?",
        file_id: str | None = None,
        *,
        use_cache: bool = True,
    ) -> AIUsageResult | None:
        return describe_image_result(
            image_data,
            user_text,
            file_id,
            use_cache=use_cache,
            get_cached_description=self._deps.cache.get_description,
            get_client=self._deps.get_openrouter_client,
            prepare_image=self._deps.images.prepare,
            encode_image=self._deps.images.encode,
            increment_request_count=self._deps.increment_request_count,
            build_usage_result=self._deps.build_usage_result,
            cache_description=self._deps.cache.cache_description,
            admin_report=self._deps.admin_report,
            logger=self._deps.logger,
            model=self._deps.vision_model,
            max_tokens=self._deps.vision_max_tokens,
            no_markdown_prompt=self._deps.no_markdown_prompt,
        )

    def describe_image(
        self,
        image_data: bytes,
        user_text: str = "¿Qué ves en esta imagen?",
        file_id: str | None = None,
        *,
        use_cache: bool = True,
    ) -> str | None:
        result = self.describe_image_result(
            image_data,
            user_text,
            file_id,
            use_cache=use_cache,
        )
        return result.text if result else None

    def transcribe_audio_openrouter_result(
        self,
        audio_data: bytes,
        file_id: str | None = None,
    ) -> AIUsageResult | None:
        return transcribe_audio_openrouter_result(
            audio_data,
            file_id,
            get_client=self._deps.get_openrouter_client,
            increment_request_count=self._deps.increment_request_count,
            build_usage_result=self._deps.build_usage_result,
            model=self._deps.openrouter_transcribe_model,
        )

    def transcribe_audio_result(
        self,
        audio_data: bytes,
        file_id: str | None = None,
        *,
        use_cache: bool = True,
    ) -> AIUsageResult | None:
        return transcribe_audio_result(
            audio_data,
            file_id,
            use_cache=use_cache,
            get_cached_transcription=self._deps.cache.get_transcription,
            measure_duration=self._deps.measure_duration,
            get_native_client=self._deps.get_groq_native_client,
            build_usage_result=self._deps.build_usage_result,
            execute_with_fallback=self.execute_groq_request,
            openrouter_fallback=self.transcribe_audio_openrouter_result,
            cache_transcription=self._deps.cache.cache_transcription,
            model=self._deps.transcribe_model,
        )

    def transcribe_audio(
        self,
        audio_data: bytes,
        file_id: str | None = None,
        *,
        use_cache: bool = True,
    ) -> str | None:
        result = self.transcribe_audio_result(
            audio_data,
            file_id,
            use_cache=use_cache,
        )
        return result.text if result else None

    def process_with_cache(
        self,
        *,
        file_id: str,
        use_cache: bool,
        cache_lookup: Callable[[str], str | None] | None,
        processor: Callable[[bytes], AIUsageResult | None],
        downloader: Callable[[str], bytes | None] | None = None,
        failure_code: str,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        return process_media_with_cache(
            file_id=file_id,
            use_cache=use_cache,
            cache_lookup=cache_lookup,
            processor=processor,
            downloader=downloader or self._deps.telegram.download_file,
            measure_duration=self._deps.measure_duration,
            failure_code=failure_code,
            logger=self._deps.logger,
        )

    def transcribe_file(
        self,
        file_id: str,
        use_cache: bool = True,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        return transcribe_file_by_id(
            file_id,
            use_cache,
            get_cached_transcription=self._deps.cache.get_transcription,
            download_file=self._deps.telegram.download_file,
            measure_duration=self._deps.measure_duration,
            extract_audio=self._deps.extract_audio,
            transcribe=self.transcribe_audio_result,
            process_media=self.process_with_cache,
            logger=self._deps.logger,
        )

    def describe_media(
        self,
        file_id: str,
        prompt: str,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        return describe_media_by_id(
            file_id,
            prompt,
            get_cached_description=self._deps.cache.get_description,
            download_file=self._deps.telegram.download_file,
            describe_image=self.describe_image_result,
            process_media=self.process_with_cache,
        )
