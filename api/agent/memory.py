"""Autonomous agent memory and repetition-control helpers."""

from __future__ import annotations

import json
import random
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone, timedelta, tzinfo
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, cast

from difflib import SequenceMatcher

import redis

RedisFactoryFn = Callable[..., redis.Redis]
AdminReporterFn = Callable[[str, Optional[Exception], Optional[Dict[str, Any]]], None]
TruncateFn = Callable[[Optional[str], int], str]


_redis_factory_fn: Optional[RedisFactoryFn] = None
_admin_reporter_fn: Optional[AdminReporterFn] = None
_truncate_text_fn: Optional[TruncateFn] = None
_timezone = timezone(timedelta(hours=-3))


def configure(
    *,
    redis_factory: RedisFactoryFn,
    admin_reporter: Optional[AdminReporterFn] = None,
    truncate_text: Optional[TruncateFn] = None,
    tz: Optional[tzinfo] = None,
) -> None:
    """Register shared dependencies used by the agent helpers."""

    global _redis_factory_fn, _admin_reporter_fn, _truncate_text_fn, _timezone

    _redis_factory_fn = redis_factory
    _admin_reporter_fn = admin_reporter
    _truncate_text_fn = truncate_text
    if tz is not None:
        _timezone = tz


def _get_redis_client() -> Optional[redis.Redis]:
    if _redis_factory_fn is None:
        return None
    try:
        return _redis_factory_fn()
    except Exception:
        return None


def _admin_report(message: str, error: Optional[Exception] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if _admin_reporter_fn:
        _admin_reporter_fn(message, error, extra)


def _truncate(text: Optional[str], max_length: int) -> str:
    if _truncate_text_fn:
        return _truncate_text_fn(text, max_length)
    value = (text or "").strip()
    if len(value) <= max_length:
        return value
    return value[: max_length - 1] + "…"


AGENT_THOUGHTS_KEY = "agent:thoughts"
MAX_AGENT_THOUGHTS = 10
AGENT_THOUGHT_DISPLAY_LIMIT = 5
AGENT_THOUGHT_CHAR_LIMIT = 500
AGENT_RECENT_THOUGHT_WINDOW = 5
AGENT_REQUIRED_SECTIONS = ("HALLAZGOS", "PRÓXIMO PASO")
AGENT_EMPTY_RESPONSE_FALLBACK = (
    "HALLAZGOS: no se me ocurrió nada nuevo, pintó el vacío.\n"
    "PRÓXIMO PASO: meter una búsqueda puntual para traer un dato real y salir de la fiaca."
)
AGENT_REPETITION_RETRY_LIMIT = 3
AGENT_LOOP_FALLBACK_PREFIX = "HALLAZGOS: registré que estaba en un loop repitiendo"
AGENT_REPETITION_ESCALATION_HINT = (
    "No escribas que estás trabado o en un loop. Ejecutá de inmediato una herramienta "
    "(web_search o fetch_url) con un tema distinto y registrá datos nuevos "
    "(números, titulares, precios). Si el tema anterior no se mueve, cambiá a otro interés fuerte del gordo."
)

AGENT_KEYWORD_STOPWORDS: Set[str] = {
    "ante",
    "aqui",
    "aquel",
    "aquella",
    "aquello",
    "asi",
    "busque",
    "cada",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "cuyo",
    "datos",
    "donde",
    "durante",
    "entre",
    "este",
    "esta",
    "estas",
    "esto",
    "estos",
    "gordo",
    "hallazgos",
    "hacer",
    "hice",
    "investigue",
    "investigando",
    "investigar",
    "luego",
    "mientras",
    "mismo",
    "mucha",
    "mucho",
    "nada",
    "para",
    "pendiente",
    "pero",
    "porque",
    "proximo",
    "queda",
    "seguir",
    "sigue",
    "sobre",
    "solo",
    "todas",
    "todos",
    "todavia",
    "tema",
    "teni",
    "tenemos",
    "tener",
    "tenes",
    "tenia",
    "teniendo",
    "tengo",
    "unas",
    "unos",
    "voy",
}


def get_agent_thoughts(
    redis_client: Optional[redis.Redis] = None,
) -> List[Dict[str, Any]]:
    """Return persisted autonomous agent thoughts (newest first)."""

    client = redis_client
    if client is None:
        client = _get_redis_client()
        if client is None:
            return []

    try:
        raw_items = cast(
            List[str],
            client.lrange(AGENT_THOUGHTS_KEY, 0, MAX_AGENT_THOUGHTS - 1),
        )
    except Exception as redis_error:
        _admin_report(
            "Error retrieving agent thoughts",
            redis_error,
            {"operation": "lrange", "key": AGENT_THOUGHTS_KEY},
        )
        return []

    thoughts: List[Dict[str, Any]] = []
    for raw in raw_items:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        text_value = str(payload.get("text", "")).strip()
        if not text_value:
            continue

        timestamp_value: Optional[int]
        timestamp_raw = payload.get("timestamp")
        if isinstance(timestamp_raw, (int, float)):
            timestamp_value = int(timestamp_raw)
        elif isinstance(timestamp_raw, str) and timestamp_raw.isdigit():
            timestamp_value = int(timestamp_raw)
        else:
            timestamp_value = None

        thought_entry: Dict[str, Any] = {"text": text_value}
        if timestamp_value is not None:
            thought_entry["timestamp"] = timestamp_value
        thoughts.append(thought_entry)

    return thoughts


def build_agent_thoughts_context_message(
    thoughts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a system message describing recent agent thoughts for the model."""

    lines: List[str] = []
    for thought in thoughts:
        text = str(thought.get("text", "")).strip()
        if not text:
            continue
        timestamp_value = thought.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            dt = datetime.fromtimestamp(int(timestamp_value), tz=_timezone)
            time_label = dt.strftime("%d/%m %H:%M")
            lines.append(f"- [{time_label}] {text}")
        else:
            lines.append(f"- {text}")

    if not lines:
        return None

    context_text = (
        "MEMORIA AUTÓNOMA (más reciente primero):\n"
        + "\n".join(lines)
        + "\nUsá esta memoria cuando charles con humanos o cuando generes nuevos pensamientos autónomos."
    )

    return {
        "role": "system",
        "content": [{"type": "text", "text": context_text}],
    }


def get_agent_memory_context() -> Optional[Dict[str, Any]]:
    thoughts = get_agent_thoughts()
    return build_agent_thoughts_context_message(thoughts)


def format_agent_thoughts(thoughts: List[Dict[str, Any]]) -> str:
    """Render thoughts for human consumption."""

    if not thoughts:
        return "todavía no tengo pensamientos guardados, dejame que labure un toque."

    lines = ["🧠 Pensamientos recientes del gordo autónomo:"]
    index = 1
    for thought in thoughts:
        text = str(thought.get("text", "")).strip()
        if not text:
            continue

        formatted_text = text.replace("\r\n", "\n").replace("\r", "\n")
        formatted_text = formatted_text.replace("\n", "\n   ")
        timestamp_value = thought.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            dt = datetime.fromtimestamp(int(timestamp_value), tz=_timezone)
            time_label = dt.strftime("%d/%m %H:%M")
            lines.append(f"{index}. [{time_label}] {formatted_text}")
        else:
            lines.append(f"{index}. {formatted_text}")
        index += 1

    if len(lines) == 1:
        return "todavía no tengo pensamientos guardados, dejame que labure un toque."

    return "\n".join(lines)


def normalize_agent_text(text: str) -> str:
    """Normalize agent text for similarity comparisons."""

    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"[^a-z0-9]+", " ", without_accents)
    return collapsed.strip()


@lru_cache(maxsize=1)
def _loop_fallback_marker() -> str:
    return normalize_agent_text(AGENT_LOOP_FALLBACK_PREFIX)


def is_loop_fallback_text(text: str) -> bool:
    marker = _loop_fallback_marker()
    if not marker:
        return False
    normalized_text = normalize_agent_text(text)
    if not normalized_text:
        return False
    return normalized_text.startswith(marker)


@lru_cache(maxsize=1)
def _empty_fallback_marker() -> str:
    return normalize_agent_text(AGENT_EMPTY_RESPONSE_FALLBACK)


def is_empty_agent_thought_text(text: str) -> bool:
    sanitized = str(text or "").strip()
    if not sanitized:
        return True
    marker = _empty_fallback_marker()
    if not marker:
        return False
    normalized_text = normalize_agent_text(sanitized)
    return normalized_text == marker


def _extract_keywords_from_normalized(normalized_text: str) -> Set[str]:
    keywords: Set[str] = set()
    for token in normalized_text.split():
        if len(token) < 4:
            continue
        if token in AGENT_KEYWORD_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        keywords.add(token)
    return keywords


def get_agent_text_features(text: str) -> Tuple[str, Set[str]]:
    normalized = normalize_agent_text(text)
    if not normalized:
        return "", set()
    return normalized, _extract_keywords_from_normalized(normalized)


def extract_agent_keywords(text: str) -> Set[str]:
    _, keywords = get_agent_text_features(text)
    return keywords


def _agent_keywords_are_repetitive(
    new_keywords: Set[str], previous_keywords: Set[str]
) -> bool:
    if not new_keywords or not previous_keywords:
        return False

    overlap = new_keywords & previous_keywords
    if len(overlap) >= 3:
        return True

    min_len = min(len(new_keywords), len(previous_keywords))
    if min_len <= 1:
        return False

    if len(overlap) >= 2 and min_len <= 5:
        return True

    overlap_ratio = len(overlap) / min_len
    return overlap_ratio >= 0.6


def _normalized_texts_are_repetitive(normalized_new: str, normalized_prev: str) -> bool:
    if not normalized_new or not normalized_prev:
        return False
    if normalized_new == normalized_prev:
        return True

    similarity = SequenceMatcher(None, normalized_new, normalized_prev).ratio()
    if similarity >= 0.88:
        return True

    new_tokens = set(normalized_new.split())
    prev_tokens = set(normalized_prev.split())
    if not new_tokens or not prev_tokens:
        return False
    union_len = len(new_tokens | prev_tokens)
    if union_len == 0:
        return False

    overlap = len(new_tokens & prev_tokens) / union_len
    return overlap >= 0.75


def is_repetitive_thought(new_text: str, previous_text: Optional[str]) -> bool:
    if not new_text or not previous_text:
        return False
    normalized_new, new_keywords = get_agent_text_features(new_text)
    normalized_prev, prev_keywords = get_agent_text_features(previous_text)
    if not normalized_new or not normalized_prev:
        return False
    if _normalized_texts_are_repetitive(normalized_new, normalized_prev):
        return True
    return _agent_keywords_are_repetitive(new_keywords, prev_keywords)


def find_repetitive_recent_thought(
    new_text: str, previous_texts: Iterable[str]
) -> Optional[str]:
    normalized_new, new_keywords = get_agent_text_features(new_text)
    if not normalized_new:
        return None

    for candidate in previous_texts:
        sanitized = str(candidate or "").strip()
        if not sanitized:
            continue
        normalized_prev, previous_keywords = get_agent_text_features(sanitized)
        if not normalized_prev:
            continue
        if _normalized_texts_are_repetitive(normalized_new, normalized_prev):
            return sanitized
        if _agent_keywords_are_repetitive(new_keywords, previous_keywords):
            return sanitized
    return None


def summarize_recent_agent_topics(
    thoughts: Iterable[Dict[str, Any]], limit: int = 4
) -> List[str]:
    summaries: List[str] = []
    seen: Set[str] = set()

    for thought in thoughts:
        if len(summaries) >= limit:
            break

        if isinstance(thought, dict):
            text = str(thought.get("text", "")).strip()
        else:
            text = str(thought or "").strip()

        if not text:
            continue

        section_content = extract_agent_section_content(text, "HALLAZGOS")
        snippet_source = section_content or text
        snippet_source = re.sub(r"\s+", " ", snippet_source).strip()
        if not snippet_source:
            continue

        first_sentence_parts = re.split(r"(?<=[.!?])\s+", snippet_source, maxsplit=1)
        summary = first_sentence_parts[0][:160]
        normalized_summary = summary.lower()
        if normalized_summary in seen:
            continue
        seen.add(normalized_summary)
        summaries.append(summary)

    return summaries


def _normalize_header_value(value: Optional[str]) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", stripped)
    return collapsed.strip().lower()


def extract_agent_section_content(
    text: str, header: str, *, other_headers: Iterable[str] = ()
) -> Optional[str]:
    sanitized = str(text or "")
    if not sanitized:
        return None

    all_headers = [header, *other_headers]
    normalized_headers = {
        _normalize_header_value(item): item for item in all_headers if item
    }

    target_norm = _normalize_header_value(header)
    if target_norm not in normalized_headers:
        normalized_headers[target_norm] = header

    sections: Dict[str, List[str]] = {}
    current_norm: Optional[str] = None

    lines = sanitized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            if current_norm is not None:
                sections.setdefault(current_norm, []).append("")
            continue

        candidate, remainder = (stripped.split(":", 1) + [""])[:2]
        candidate_norm = _normalize_header_value(candidate)
        if candidate_norm in normalized_headers:
            current_norm = candidate_norm
            sections.setdefault(current_norm, [])
            if remainder.strip():
                sections[current_norm].append(remainder.strip())
            continue

        if current_norm is not None:
            sections.setdefault(current_norm, []).append(stripped)

    content_lines = sections.get(target_norm)
    if not content_lines:
        return None

    content = "\n".join(content_lines).strip()
    return content or None


def agent_sections_are_valid(text: str) -> bool:
    if not text:
        return False

    required = tuple(AGENT_REQUIRED_SECTIONS)
    for header in required:
        other_headers = tuple(h for h in required if h != header)
        if not extract_agent_section_content(text, header, other_headers=other_headers):
            return False
    return True


def get_agent_retry_hint(
    previous_text: Optional[str], rng: Optional[random.Random] = None
) -> str:
    normalized_previous = normalize_agent_text(previous_text or "")
    tokens = [token for token in normalized_previous.split() if token]

    filtered_tokens: List[str] = []
    for token in tokens:
        if token in AGENT_KEYWORD_STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        filtered_tokens.append(token)

    keyword_counter = Counter(filtered_tokens)
    top_keywords = [word for word, _ in keyword_counter.most_common(3)]

    if top_keywords:
        avoided_fragment = (
            "Marcá como prohibidos estos términos ya gastados: "
            + ", ".join(top_keywords)
            + ". "
        )
    else:
        avoided_fragment = "Arrancá desde cero sin reciclar la última búsqueda. "

    rng_instance = rng if rng is not None else random

    option_count = rng_instance.randint(3, 4)
    ordinal_words = ("primera", "segunda", "tercera", "cuarta", "quinta")
    ordinal_index = rng_instance.randint(0, option_count - 1)
    ordinal_word = ordinal_words[min(ordinal_index, len(ordinal_words) - 1)]

    brainstorming_templates = (
        "Anotá {n} búsquedas frescas en temas distintos. Ejecutá la {ordinal} con web_search.",
        "Hacé una mini lluvia de ideas con {n} queries nuevos y corré la {ordinal} usando web_search.",
        "Pensá en {n} consultas posibles que sorprendan al gordo y quedate con la {ordinal} para web_search.",
    )

    follow_up_templates = (
        "Traé números, fechas y citá la fuente puntual.",
        "Resumí el dato clave con cifras concretas y quién lo publicó.",
        "Documentá resultados verificables (monto, variación, protagonista) y la fuente exacta.",
    )

    brainstorming_prompt = rng_instance.choice(brainstorming_templates).format(
        n=option_count,
        ordinal=ordinal_word,
    )
    follow_up_prompt = rng_instance.choice(follow_up_templates)

    letter_choices = tuple("abcdefghijklmnñopqrstuvwxyz")
    chosen_letter = rng_instance.choice(letter_choices).upper()
    alternate_pool = [
        value for value in letter_choices if value.upper() != chosen_letter
    ]
    fallback_letter = (
        rng_instance.choice(alternate_pool).upper() if alternate_pool else chosen_letter
    )

    numeric_floor = rng_instance.randint(8, 24)
    numeric_multiplier = rng_instance.randint(3, 9)
    numeric_target = numeric_floor * numeric_multiplier

    constraint_templates = (
        'Sumá una restricción creativa: la búsqueda tiene que incluir un protagonista cuya inicial sea "{letter}" y una cifra cerca de {value}.',
        'Obligate a que la consulta nombre algo que empiece con "{letter}" y mencione un número alrededor de {value}.',
        'Forzá el query a combinar un actor que arranque con "{letter}" más un dato numérico aproximado a {value}.',
    )

    pivot_templates = (
        'Si web_search no trae novedad, generá otra lluvia de ideas reemplazando las palabras prohibidas por categorías nuevas y probá con inicial "{fallback}".',
        'Si la ejecución devuelve humo, descartá la idea y repetí el proceso con términos distintos que comiencen con "{fallback}".',
        'Si no aparecen datos frescos, reseteá las keywords vetadas y buscá una consulta distinta arrancando por "{fallback}".',
    )

    constraint_prompt = rng_instance.choice(constraint_templates).format(
        letter=chosen_letter, value=numeric_target
    )
    pivot_prompt = rng_instance.choice(pivot_templates).format(fallback=fallback_letter)

    return (
        avoided_fragment
        + brainstorming_prompt
        + " "
        + follow_up_prompt
        + " "
        + constraint_prompt
        + " "
        + pivot_prompt
    )


def build_agent_retry_prompt(
    previous_text: Optional[str], rng: Optional[random.Random] = None
) -> str:
    preview = _truncate(previous_text or "", 160)
    preview_single_line = preview.replace("\n", " ").strip()
    base_prompt = (
        "La última nota no sirvió: te repetiste igual que la memoria anterior o no respetaste la estructura obligatoria. "
        "Antes de escribir otra vez, completá el pendiente y contá resultados concretos. "
        "Si necesitás info fresca, llamá a la herramienta web_search con un query preciso y resumí lo que encontraste. Si ya cerraste ese tema, cambiá a otro interés fuerte del gordo en vez de seguir clavado en lo mismo. "
        + (
            f'Esto fue lo último guardado o la nota fallida: "{preview_single_line}". '
            if preview_single_line
            else ""
        )
        + "Escribí ahora una nota distinta con hechos puntuales y cerrala en dos secciones claras: "
        '"HALLAZGOS:" con los datos específicos que obtuviste y "PRÓXIMO PASO:" con la siguiente acción directa.'
    )

    hint = get_agent_retry_hint(previous_text, rng=rng)
    if hint:
        return f"{base_prompt} {hint}"
    return base_prompt


def build_agent_fallback_entry(previous_text: Optional[str]) -> str:
    sanitized_previous = (previous_text or "").strip()
    normalized_previous = normalize_agent_text(sanitized_previous)
    fallback_marker = _loop_fallback_marker()

    preview = _truncate(sanitized_previous, 120)
    preview_single_line = preview.replace("\n", " ").strip()

    include_fragment = True
    if (
        normalized_previous
        and fallback_marker
        and normalized_previous.startswith(fallback_marker)
    ):
        include_fragment = False

    loop_fragment = (
        f' "{preview_single_line}"' if include_fragment and preview_single_line else ""
    )
    return (
        f"{AGENT_LOOP_FALLBACK_PREFIX}{loop_fragment} sin generar avances reales.\n"
        "PRÓXIMO PASO: hacer una búsqueda web urgente, anotar los datos específicos que salgan y recién después planear el próximo paso."
    )


def ensure_agent_response_text(text: Optional[str]) -> str:
    sanitized = str(text or "").strip()
    return sanitized or AGENT_EMPTY_RESPONSE_FALLBACK


def build_agent_retry_messages(
    base_messages: List[Dict[str, Any]],
    assistant_text: str,
    corrective_prompt: str,
) -> List[Dict[str, Any]]:
    return base_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text or ""}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": corrective_prompt}],
        },
    ]


def request_agent_response(
    generator: Callable[[List[Dict[str, Any]]], str],
    messages: List[Dict[str, Any]],
    error_context: str,
) -> str:
    try:
        response = generator(messages)
    except Exception as ai_error:
        _admin_report(error_context, ai_error)
        raise
    return ensure_agent_response_text(response)


def save_agent_thought(
    thought_text: str, redis_client: Optional[redis.Redis] = None
) -> Optional[Dict[str, Any]]:
    sanitized = (thought_text or "").strip()
    if not sanitized:
        return None

    truncated = _truncate(sanitized, AGENT_THOUGHT_CHAR_LIMIT)

    client = redis_client
    if client is None:
        client = _get_redis_client()
        if client is None:
            return None

    timestamp_value = int(time.time())
    entry = {"text": truncated, "timestamp": timestamp_value}

    try:
        payload = json.dumps(entry, ensure_ascii=False)
        pipeline = client.pipeline()
        pipeline.lpush(AGENT_THOUGHTS_KEY, payload)
        pipeline.ltrim(AGENT_THOUGHTS_KEY, 0, MAX_AGENT_THOUGHTS - 1)
        pipeline.execute()
    except Exception as redis_error:
        _admin_report(
            "Error saving agent thought",
            redis_error,
            {"thought_preview": truncated[:80]},
        )
        return None

    return entry


def show_agent_thoughts() -> str:
    thoughts = get_agent_thoughts()
    visible_thoughts = thoughts[:AGENT_THOUGHT_DISPLAY_LIMIT]
    return format_agent_thoughts(visible_thoughts)
