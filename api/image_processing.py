from __future__ import annotations

import base64
import io
from logging import Logger
from typing import Any


def resize_image_if_needed(
    image_data: bytes,
    max_size: int,
    *,
    image_module: Any,
) -> bytes:
    try:
        image = image_module.open(io.BytesIO(image_data))
        if max(image.size) > max_size:
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (
                int(image.size[0] * ratio),
                int(image.size[1] * ratio),
            )
            image = image.resize(new_size, image_module.Resampling.LANCZOS)

        output_buffer = io.BytesIO()
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        image.save(output_buffer, format="WEBP", quality=85, optimize=True)
        return output_buffer.getvalue()
    except ImportError:
        print("WARNING: PIL not available, cannot resize image")
    except Exception as error:
        print(f"ERROR: Failed to resize image: {error}")
    return image_data


def prepare_vision_image(
    image_data: bytes,
    max_size: int,
    *,
    image_module: Any,
    logger: Logger,
) -> tuple[bytes, str] | None:
    try:
        image = image_module.open(io.BytesIO(image_data))
        image.load()
        if max(image.size) > max_size:
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (
                int(image.size[0] * ratio),
                int(image.size[1] * ratio),
            )
            image = image.resize(new_size, image_module.Resampling.LANCZOS)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        output_buffer = io.BytesIO()
        image.save(output_buffer, format="WEBP", quality=85, optimize=True)
        return output_buffer.getvalue(), "image/webp"
    except Exception as error:
        logger.info("vision image prepare failed: %s", error)
        return None


def encode_image_to_base64(image_data: bytes) -> str:
    return base64.b64encode(image_data).decode("utf-8")


class ImageService:
    def __init__(
        self,
        *,
        image_module: Any,
        logger: Logger,
        max_size: int = 512,
    ) -> None:
        self.image_module = image_module
        self._logger = logger
        self._max_size = max_size

    def resize(self, image_data: bytes, max_size: int | None = None) -> bytes:
        return resize_image_if_needed(
            image_data,
            max_size or self._max_size,
            image_module=self.image_module,
        )

    def prepare(
        self,
        image_data: bytes,
        max_size: int | None = None,
    ) -> tuple[bytes, str] | None:
        return prepare_vision_image(
            image_data,
            max_size or self._max_size,
            image_module=self.image_module,
            logger=self._logger,
        )

    def encode(self, image_data: bytes) -> str:
        return encode_image_to_base64(image_data)


__all__ = ["ImageService"]
