from tests.support import *  # noqa: F401,F403

def test_save_agent_thought_persists_and_limits():
    from api.index import (
        save_agent_thought,
        get_agent_thoughts,
        MAX_AGENT_THOUGHTS,
        AGENT_THOUGHT_CHAR_LIMIT,
    )

    from typing import cast

    class FakePipeline:
        def __init__(self, parent):
            self.parent = parent
            self.commands = []

        def lpush(self, key, value):
            self.commands.append(("lpush", key, value))
            return self

        def ltrim(self, key, start, end):
            self.commands.append(("ltrim", key, start, end))
            return self

        def execute(self):
            results = []
            for command, key, *args in self.commands:
                if command == "lpush":
                    results.append(self.parent.lpush(key, args[0]))
                elif command == "ltrim":
                    results.append(self.parent.ltrim(key, args[0], args[1]))
            self.commands = []
            return results

    class FakeRedis:
        def __init__(self):
            self.storage = {}

        def pipeline(self):
            return FakePipeline(self)

        def lpush(self, key, value):
            values = self.storage.setdefault(key, [])
            values.insert(0, value)
            return len(values)

        def ltrim(self, key, start, end):
            values = self.storage.get(key, [])
            if end == -1:
                trimmed = values[start:]
            else:
                trimmed = values[start : end + 1]
            self.storage[key] = trimmed
            return True

        def lrange(self, key, start, end):
            values = self.storage.get(key, [])
            if end == -1:
                return values[start:]
            return values[start : end + 1]

    fake_redis = cast(redis.Redis, FakeRedis())

    for i in range(MAX_AGENT_THOUGHTS + 2):
        save_agent_thought(f"pensamiento {i} " + "x" * 600, fake_redis)

    thoughts = get_agent_thoughts(fake_redis)

    assert len(thoughts) == MAX_AGENT_THOUGHTS
    assert thoughts[0]["text"].startswith(f"pensamiento {MAX_AGENT_THOUGHTS + 1}")
    assert thoughts[-1]["text"].startswith("pensamiento 2")
    assert len(thoughts[0]["text"]) <= AGENT_THOUGHT_CHAR_LIMIT
    assert all("timestamp" in thought for thought in thoughts)


def test_format_agent_thoughts_variants():
    from api.index import format_agent_thoughts, BA_TZ

    empty_output = format_agent_thoughts([])
    assert "no tengo pensamientos" in empty_output

    sample_ts = int(datetime(2024, 1, 1, 12, 30, tzinfo=BA_TZ).timestamp())
    formatted = format_agent_thoughts(
        [{"text": "me clavé un mate y anoté ideas", "timestamp": sample_ts}]
    )

    assert "me clavé un mate" in formatted
    assert "01/01" in formatted


def test_get_agent_memory_context_builds_message():
    from api.index import get_agent_memory_context

    sample = [{"text": "armé un plan", "timestamp": 1_700_000_000}]

    with patch("api.index.get_agent_thoughts", return_value=sample):
        memory_message = get_agent_memory_context()

    assert memory_message is not None
    assert memory_message["role"] == "system"
    content = memory_message["content"][0]["text"]
    assert "armé un plan" in content


def test_show_agent_thoughts_no_entries():
    from api.index import show_agent_thoughts

    with patch("api.index.get_agent_thoughts", return_value=[]):
        response = show_agent_thoughts()

    assert "no tengo pensamientos" in response


def test_show_agent_thoughts_limits_entries():
    from api.index import show_agent_thoughts

    sample = [{"text": f"nota {i}"} for i in range(1, AGENT_THOUGHT_DISPLAY_LIMIT + 3)]

    with patch("api.index.get_agent_thoughts", return_value=sample):
        response = show_agent_thoughts()

    for i in range(1, AGENT_THOUGHT_DISPLAY_LIMIT + 1):
        assert f"nota {i}" in response

    assert f"nota {AGENT_THOUGHT_DISPLAY_LIMIT + 1}" not in response
    assert f"nota {AGENT_THOUGHT_DISPLAY_LIMIT + 2}" not in response


def test_run_agent_cycle_returns_result():
    from api.index import run_agent_cycle

    structured_text = (
        "HALLAZGOS: el riesgo país cerró en 1.200 puntos.\n"
        "PRÓXIMO PASO: revisar qué bancos emitieron reportes sobre bonos argentinos."
    )

    with patch("api.index.ask_ai", return_value=structured_text) as mock_ask, patch(
        "api.index.sanitize_tool_artifacts", side_effect=lambda x: x
    ) as mock_sanitize, patch(
        "api.index.clean_duplicate_response", side_effect=lambda x: x
    ) as mock_clean, patch(
        "api.index.get_hacker_news_context", return_value=[]
    ), patch(
        "api.index.get_agent_thoughts", return_value=[]
    ), patch(
        "api.index.save_agent_thought"
    ) as mock_save:
        mock_save.return_value = {"text": structured_text, "timestamp": 1_700_000_000}

        result = run_agent_cycle()

    assert mock_ask.call_count >= 1
    assert mock_sanitize.call_count == mock_ask.call_count
    assert mock_clean.call_count == mock_ask.call_count
    mock_save.assert_called_once()
    saved_text = mock_save.call_args[0][0]
    assert saved_text == structured_text
    assert result["text"] == structured_text
    assert result["persisted"] is True
    assert result["timestamp"] == 1_700_000_000
    assert result["iso_time"].endswith("-03:00")


def test_run_agent_cycle_breaks_loop_with_retry():
    from api.index import (
        AGENT_LOOP_FALLBACK_PREFIX,
        AGENT_REPETITION_RETRY_LIMIT,
        run_agent_cycle,
    )

    repeated_text = (
        "HALLAZGOS: seguí leyendo la nota de Bonos AL30.\n"
        "PRÓXIMO PASO: repasar qué dijo el BCRA sobre los AL30."
    )
    fresh_text = (
        "HALLAZGOS: el INDEC marcó inflación de agosto en 12,4%.\n"
        "PRÓXIMO PASO: buscar reacciones políticas a ese dato."
    )

    responses = ([repeated_text] * AGENT_REPETITION_RETRY_LIMIT) + [fresh_text]

    def fake_ask(_):
        return responses.pop(0)

    previous_entries = [{"text": repeated_text, "timestamp": 1_700_000_000}]

    with patch("api.index.ask_ai", side_effect=fake_ask) as mock_ask, patch(
        "api.index.sanitize_tool_artifacts", side_effect=lambda x: x
    ), patch(
        "api.index.clean_duplicate_response", side_effect=lambda x: x
    ), patch(
        "api.index.get_hacker_news_context", return_value=[]
    ), patch(
        "api.index.get_agent_thoughts", return_value=previous_entries
    ), patch(
        "api.index.save_agent_thought"
    ) as mock_save:
        mock_save.return_value = {"text": fresh_text, "timestamp": 1_700_000_100}

        result = run_agent_cycle()

    assert mock_ask.call_count == AGENT_REPETITION_RETRY_LIMIT + 1
    mock_save.assert_called_once()
    saved_text = mock_save.call_args[0][0]
    assert saved_text == fresh_text
    assert not saved_text.startswith(AGENT_LOOP_FALLBACK_PREFIX)
    assert result["text"] == fresh_text
    assert result["persisted"] is True
    assert result["timestamp"] == 1_700_000_100


def test_run_agent_cycle_persists_loop_fallback_when_stuck():
    from api.index import (
        AGENT_LOOP_FALLBACK_PREFIX,
        AGENT_REPETITION_RETRY_LIMIT,
        run_agent_cycle,
    )

    repeated_text = (
        "HALLAZGOS: seguí leyendo lo mismo de siempre.\n"
        "PRÓXIMO PASO: insistir con la misma nota otra vez."
    )

    responses = [repeated_text] * (AGENT_REPETITION_RETRY_LIMIT + 1)

    def fake_ask(_):
        return responses.pop(0)

    previous_entries = [{"text": repeated_text, "timestamp": 1_700_000_000}]

    with patch("api.index.ask_ai", side_effect=fake_ask) as mock_ask, patch(
        "api.index.sanitize_tool_artifacts", side_effect=lambda x: x
    ), patch(
        "api.index.clean_duplicate_response", side_effect=lambda x: x
    ), patch(
        "api.index.get_hacker_news_context", return_value=[]
    ), patch(
        "api.index.get_agent_thoughts", return_value=previous_entries
    ), patch(
        "api.index.save_agent_thought"
    ) as mock_save:

        def capture_save(text):
            return {"text": text, "timestamp": 1_700_000_200}

        mock_save.side_effect = capture_save
        result = run_agent_cycle()

    assert mock_ask.call_count == AGENT_REPETITION_RETRY_LIMIT + 1
    mock_save.assert_called_once()
    saved_text = mock_save.call_args[0][0]
    assert isinstance(saved_text, str)
    assert saved_text.startswith(AGENT_LOOP_FALLBACK_PREFIX)
    assert "pintó el vacío" not in saved_text
    assert result["text"] == saved_text
    assert result["persisted"] is True
    assert result["timestamp"] == 1_700_000_200
    assert not responses


def test_run_agent_cycle_skips_empty_fallback():
    from api.index import AGENT_EMPTY_RESPONSE_FALLBACK, run_agent_cycle

    with patch("api.index.ask_ai", return_value="") as mock_ask, patch(
        "api.index.sanitize_tool_artifacts", side_effect=lambda x: x
    ) as mock_sanitize, patch(
        "api.index.clean_duplicate_response", side_effect=lambda x: x
    ) as mock_clean, patch(
        "api.index.get_hacker_news_context", return_value=[]
    ), patch(
        "api.index.get_agent_thoughts", return_value=[]
    ), patch(
        "api.index.save_agent_thought"
    ) as mock_save:
        result = run_agent_cycle()

    mock_ask.assert_called_once()
    assert mock_sanitize.call_count == 1
    assert mock_clean.call_count == 1
    mock_save.assert_not_called()
    assert result["text"] == AGENT_EMPTY_RESPONSE_FALLBACK
    assert result["persisted"] is False
    assert "timestamp" not in result
    assert "iso_time" not in result


def test_is_repetitive_thought_detects_loop():
    previous = (
        "estaba analizando que btc rompió 116k pero eth baja, voy a buscar noticias frescas"
    )
    repeated = "Estaba analizando que BTC rompió 116k pero ETH baja, voy a buscar noticias frescas"

    assert is_repetitive_thought(repeated, previous)


def test_is_repetitive_thought_allows_new_data():
    previous = "estaba analizando que btc rompió 116k pero eth baja, voy a buscar noticias"
    updated = (
        "escaneé titulares y encontré una nota de coindesk sobre flujos asiáticos, próximo paso: revisar volúmenes"
    )

    assert not is_repetitive_thought(updated, previous)


def test_is_repetitive_thought_detects_same_topic_keywords():
    previous = (
        "seguí la rotación institucional ether bitcoin en etfs spot de estados unidos para ver flujos"
    )
    repeated = (
        "investigué rotación institucional ether bitcoin en etfs europeos y anoté que siguen los mismos flujos"
    )

    assert is_repetitive_thought(repeated, previous)


def test_summarize_recent_agent_topics_uses_hallazgos_section():
    thoughts = [
        {
            "text": "HALLAZGOS: miré la curva de bonos CER y cayó el tramo largo.\nPRÓXIMO PASO: revisar el Boncer 2026",
        },
        {
            "text": "HALLAZGOS: repasé ligas europeas de fútbol femenino.\nPRÓXIMO PASO: buscar calendario local",
        },
    ]

    summaries = summarize_recent_agent_topics(thoughts, limit=3)

    assert summaries[0].startswith("miré la curva de bonos")
    assert len(summaries) == 2


def test_build_agent_retry_prompt_mentions_previous_text():
    long_text = "a" * 200
    prompt = build_agent_retry_prompt(long_text)

    expected_preview = long_text[:157] + "..."
    assert expected_preview in prompt
    assert "web_search" in prompt


def test_build_agent_retry_prompt_adds_dynamic_guidance():
    rng = random.Random(0)
    prompt = build_agent_retry_prompt(
        "me quedé clavado mirando el dólar blue todo el día", rng=rng
    )

    assert "Marcá como prohibidos" in prompt
    assert "web_search" in prompt
    assert ("dolar" in prompt) or ("dólar" in prompt)
    assert any(ch.isdigit() for ch in prompt)
    assert any(keyword in prompt.lower() for keyword in ("inicial", "letra", "empiece"))


def test_get_agent_retry_hint_references_previous_keywords():
    rng = random.Random(1)
    hint = get_agent_retry_hint("btc bitcoin y etf spot", rng=rng)

    assert "btc" in hint or "bitcoin" in hint
    assert "web_search" in hint
    assert hint
    assert re.search(r"\d", hint)
    assert re.search(r'"[A-ZÑ]"', hint)


def test_get_agent_retry_hint_varies_with_random_seed():
    hint_a = get_agent_retry_hint("repeti el mismo tema", rng=random.Random(0))
    hint_b = get_agent_retry_hint("repeti el mismo tema", rng=random.Random(1))

    assert hint_a
    assert hint_b
    assert hint_a != hint_b


def test_build_agent_fallback_entry_mentions_loop_and_previous():
    previous = "divergencia btc eth"
    fallback = build_agent_fallback_entry(previous)

    assert "loop" in fallback
    assert previous in fallback


def test_build_agent_fallback_entry_avoids_recursive_quote():
    previous = (
        "HALLAZGOS: registré que estaba en un loop repitiendo \"algo viejo\" sin generar avances reales.\n"
        "PRÓXIMO PASO: hacer una búsqueda web urgente, anotar los datos específicos que salgan y recién después planear el próximo paso."
    )

    fallback = build_agent_fallback_entry(previous)

    assert "loop" in fallback
    assert previous not in fallback


def test_agent_sections_are_valid_accepts_accented_headers():
    text = (
        "HALLAZGOS: repasé ligas europeas de fútbol femenino.\n"
        "PRÓXIMO PASO: buscar calendario local"
    )

    assert agent_sections_are_valid(text)


def test_agent_sections_are_valid_accepts_unaccented_headers():
    text = (
        "HALLAZGOS: repasé ligas europeas de fútbol femenino.\n"
        "PROXIMO PASO: buscar calendario local"
    )

    assert agent_sections_are_valid(text)
