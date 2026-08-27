from __future__ import annotations

import random
import re
import time
import unicodedata
from os import environ
from typing import Any, Callable, cast

import emoji
from pykakasi import kakasi

from api.core.i18n import current_locale, tr

KakasiFactory = Callable[[], Any]
_kakasi = cast(KakasiFactory, kakasi)


def gen_random(name: str) -> str:
    rand_res = random.randint(0, 1)
    rand_name = random.randint(0, 2)

    if rand_res:
        msg = tr("random.yes")
    else:
        msg = tr("random.no")

    if rand_name == 1:
        msg = f"{msg} {tr('random.address')}"
    elif rand_name == 2:
        msg = f"{msg} {name}"

    return msg


def select_random(msg_text: str) -> str:
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


def convert_base(msg_text: str) -> str:
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
