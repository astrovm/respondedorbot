from __future__ import annotations

import json
import logging
import random
import re
import time
import unicodedata
from os import environ
from typing import Any, Callable, Mapping, Optional, Protocol, cast

import emoji
from pykakasi import kakasi

from api.core.rust_bridge import load_rust_bridge
from api.i18n import current_locale, tr

KakasiFactory = Callable[[], Any]
_kakasi = cast(KakasiFactory, kakasi)
logger = logging.getLogger(__name__)


class _RustBaseConverter(Protocol):
    def convert_base(self, message_text: str) -> str: ...


class _RustRandomSelectionParser(Protocol):
    def parse_random_selection(self, message_text: str) -> str: ...


class _RustRandomReplyEvaluator(Protocol):
    def evaluate_random_reply(self, response_sample: int, suffix_sample: int) -> tuple[str, str]: ...


def _load_rust_base_converter() -> Optional[_RustBaseConverter]:
    module = load_rust_bridge("RUST_BASE_CONVERSION_ENABLED")
    if module is None:
        return None
    return cast(_RustBaseConverter, module)


def _load_rust_random_selection_parser() -> Optional[_RustRandomSelectionParser]:
    module = load_rust_bridge("RUST_RANDOM_SELECTION_ENABLED")
    if module is None:
        return None
    return cast(_RustRandomSelectionParser, module)


def _load_rust_random_reply_evaluator() -> Optional[_RustRandomReplyEvaluator]:
    module = load_rust_bridge("RUST_RANDOM_REPLY_ENABLED")
    if module is None:
        return None
    return cast(_RustRandomReplyEvaluator, module)


def _gen_random_python(name: str, rand_res: int, rand_name: int) -> str:
    if rand_res:
        msg = tr("random.yes")
    else:
        msg = tr("random.no")

    if rand_name == 1:
        msg = f"{msg} {tr('random.address')}"
    elif rand_name == 2:
        msg = f"{msg} {name}"

    return msg


def _random_reply_from_rust(name: str, outcome: tuple[str, str]) -> str:
    answer, suffix = outcome
    if answer not in {"yes", "no"} or suffix not in {"none", "address", "name"}:
        raise ValueError("Rust random reply result is invalid")
    message = tr(f"random.{answer}")
    if suffix == "address":
        return f"{message} {tr('random.address')}"
    if suffix == "name":
        return f"{message} {name}"
    return message


def gen_random(name: str) -> str:
    rand_res = random.randint(0, 1)
    rand_name = random.randint(0, 2)
    rust = _load_rust_random_reply_evaluator()
    if rust is not None:
        try:
            return _random_reply_from_rust(
                name,
                rust.evaluate_random_reply(rand_res, rand_name),
            )
        except Exception as error:
            logger.warning(
                "Rust random reply evaluation failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _gen_random_python(name, rand_res, rand_name)


def _select_random_python(msg_text: str) -> str:
    values = [v.strip() for v in msg_text.split(",")]
    if len(values) >= 2:
        return random.choice(values)

    try:
        start, end = [int(v.strip()) for v in msg_text.split("-")]
        if start < end:
            return str(random.randint(start, end))
    except ValueError:
        return tr("random.invalid")

    return tr("random.invalid")


def _random_selection_from_rust(raw_result: str) -> str:
    result = json.loads(raw_result)
    if not isinstance(result, Mapping):
        raise ValueError("Rust random selection result is not a mapping")
    kind = result.get("kind")
    if kind == "choices":
        values = result.get("values")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("Rust random choices are invalid")
        return cast(str, random.choice(values))
    if kind == "range":
        return str(random.randint(int(str(result["start"])), int(str(result["end"]))))
    if kind == "invalid":
        return tr("random.invalid")
    raise ValueError("Rust random selection result has an unknown kind")


def select_random(msg_text: str) -> str:
    rust = _load_rust_random_selection_parser()
    if rust is not None:
        try:
            return _random_selection_from_rust(rust.parse_random_selection(msg_text))
        except Exception as error:
            logger.warning(
                "Rust random selection failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _select_random_python(msg_text)


def _convert_base_python(msg_text: str) -> str:
    try:
        input_parts = msg_text.split(",")
        if len(input_parts) != 3:
            return tr("convert_base.usage")
        number_str, base_from_str, base_to_str = map(str.strip, input_parts)
        base_from, base_to = map(int, (base_from_str, base_to_str))

        if not all(c.isalnum() for c in number_str):
            return tr("convert_base.alphanumeric")
        if not 2 <= base_from <= 36:
            return tr("convert_base.source_range", base=base_from_str)
        if not 2 <= base_to <= 36:
            return tr("convert_base.target_range", base=base_to_str)

        digits = []
        value = 0
        for digit in number_str:
            if digit.isdigit():
                digit_value = int(digit)
            else:
                digit_value = ord(digit.upper()) - ord("A") + 10
            value = value * base_from + digit_value
        while value > 0:
            digit_value = value % base_to
            if digit_value >= 10:
                digit = chr(digit_value - 10 + ord("A"))
            else:
                digit = str(digit_value)
            digits.append(digit)
            value //= base_to
        result = "".join(reversed(digits))

        return tr(
            "convert_base.success",
            number=number_str,
            source=base_from,
            result=result,
            target=base_to,
        )
    except ValueError:
        return tr("convert_base.numbers")


def _base_conversion_from_rust(raw_result: str) -> str:
    result = json.loads(raw_result)
    if not isinstance(result, Mapping):
        raise ValueError("Rust base conversion result is not a mapping")
    kind = result.get("kind")
    if kind == "success":
        return tr(
            "convert_base.success",
            number=str(result["number"]),
            source=int(result["source"]),
            result=str(result["result"]),
            target=int(result["target"]),
        )
    if kind == "usage":
        return tr("convert_base.usage")
    if kind == "alphanumeric_required":
        return tr("convert_base.alphanumeric")
    if kind == "source_range":
        return tr("convert_base.source_range", base=str(result["input"]))
    if kind == "target_range":
        return tr("convert_base.target_range", base=str(result["input"]))
    if kind == "numbers_required":
        return tr("convert_base.numbers")
    raise ValueError("Rust base conversion result has an unknown kind")


def convert_base(msg_text: str) -> str:
    rust = _load_rust_base_converter()
    if rust is not None:
        try:
            return _base_conversion_from_rust(rust.convert_base(msg_text))
        except Exception as error:
            logger.warning(
                "Rust base conversion failed; using Python fallback: error_type=%s",
                type(error).__name__,
            )
    return _convert_base_python(msg_text)


def get_timestamp() -> str:
    return f"{int(time.time())}"


JAPANESE_TEXT_RE = re.compile(
    r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u3400-\u4DBF"
    r"\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF\U0002A700-\U0002B73F"
    r"\U0002B740-\U0002B81F\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF"
    r"\U0002F800-\U0002FA1F\U00030000-\U0003134F]"
)


def romanize_japanese(text: str) -> str:
    """Convert Japanese kana/kanji text to romaji when possible."""
    segments = _kakasi().convert(text)
    return "".join(str(segment.get("hepburn") or segment.get("orig") or "") for segment in segments)


def is_japanese_text(text: str) -> bool:
    """Return True when the text includes Japanese scripts or CJK extensions."""
    return bool(JAPANESE_TEXT_RE.search(text))


def convert_to_command(msg_text: str) -> str:
    if not msg_text:
        return tr("command.empty")

    emoji_text = emoji.demojize(
        msg_text,
        delimiters=("_", "_"),
        language=current_locale(),
    )
    if is_japanese_text(emoji_text):
        romanized_text = romanize_japanese(emoji_text)
    else:
        romanized_text = emoji_text

    replaced_ni_text = re.sub(r"\bÑ\b", "ENIE", romanized_text.upper()).replace("Ñ", "NI")

    single_spaced_text = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFD", replaced_ni_text).encode("ascii", "ignore").decode("utf-8"),
    )

    punctuation_replacements: dict[str, str | int | None] = {
        " ": "_",
        "\n": "_",
        "?": "_SIGNODEPREGUNTA_",
        "!": "_SIGNODEEXCLAMACION_",
        ".": "_PUNTO_",
    }
    translated_punctuation = re.sub(r"\.{3}", "_PUNTOSSUSPENSIVOS_", single_spaced_text).translate(
        str.maketrans(punctuation_replacements)
    )

    cleaned_text = re.sub(
        r"^_+|_+$",
        "",
        re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"_+", "_", translated_punctuation)),
    )

    if not cleaned_text:
        return tr("command.invalid")

    return f"/{cleaned_text}"


def get_instance_name() -> str:
    instance = environ.get("FRIENDLY_INSTANCE_NAME")
    return tr("instance.name", instance=instance)
