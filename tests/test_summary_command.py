from tests.support import *


EXHAUSTIVE_SUMMARY_PROMPT = (
    "resumí la siguiente conversación de forma exhaustiva. "
    "incluí todos los temas tratados, quién dijo qué, conclusiones, decisiones pendientes y datos relevantes. "
    "no omitas nada.\n\n"
    "REGLAS DE FORMATO OBLIGATORIAS:\n"
    "- SIEMPRE respondé en MINÚSCULAS\n"
    "- NUNCA uses emojis\n"
    "- NUNCA uses punto final\n"
    "- NUNCA uses markdown: no # headers, no ## subheaders, no tablas, no bullets con - o *\n"
    "- NUNCA respondas de forma formal o corporativa\n"
    "- SIEMPRE usá lenguaje coloquial argentino\n"
    "- SIEMPRE sé directo, crudo y honesto\n"
    "- cuando sea posible, contá la historia en UNA SOLA FRASE por tema\n"
    "- nunca rompas el personaje de bot argentino informal"
)


def test_handle_summary_command_returns_early_when_no_history(monkeypatch):
    from api.index import handle_summary_command

    monkeypatch.setattr("api.index._state_get_user_chat_summary", lambda *_: "old")
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: "m1")
    monkeypatch.setattr("api.index.get_chat_history", lambda *_: [])

    result = handle_summary_command("chat-1", MagicMock(), "foco")

    assert result.response_text == "no hay mensajes para resumir"
    assert result.pending_summary is None
    assert result.pending_marker is None
    assert result.summary_cost == 0
    assert result.billing_segments == []
    assert result.is_fallback is False


def test_handle_summary_command_uses_existing_summary_when_no_new_messages(monkeypatch):
    from api.index import IncrementalSummarySource, handle_summary_command

    monkeypatch.setattr("api.index._state_get_user_chat_summary", lambda *_: "resumen previo")
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: "m2")
    monkeypatch.setattr(
        "api.index.get_chat_history",
        lambda *_: [
            {"id": "m1", "role": "user", "content": "hola"},
            {"id": "m2", "role": "assistant", "content": "ok"},
        ],
    )
    monkeypatch.setattr(
        "api.index._build_incremental_summary_source",
        lambda *_: IncrementalSummarySource(
            prior_summary="resumen previo",
            delta_messages=[],
            is_zero_delta=True,
            next_marker=None,
        ),
    )

    def _fail_summary_model(*_args, **_kwargs):
        raise AssertionError("_call_summary_model should not run on zero delta")

    monkeypatch.setattr("api.index._call_summary_model", _fail_summary_model)

    result = handle_summary_command("chat-1", MagicMock(), "foco custom")

    assert result.response_text == "resumen previo"
    assert result.pending_summary == "resumen previo"
    assert result.pending_marker is None
    assert result.summary_cost == 0
    assert result.billing_segments == []
    assert result.is_fallback is False


def test_handle_summary_command_generates_summary_with_minimax(monkeypatch):
    from api.index import IncrementalSummarySource, handle_summary_command

    monkeypatch.setattr("api.index._state_get_user_chat_summary", lambda *_: "resumen previo")
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: "m1")
    monkeypatch.setattr(
        "api.index.get_chat_history",
        lambda *_: [
            {"id": "m1", "role": "user", "content": "hola"},
            {"id": "m2", "role": "assistant", "content": "nuevo"},
        ],
    )
    monkeypatch.setattr(
        "api.index._build_incremental_summary_source",
        lambda *_: IncrementalSummarySource(
            prior_summary="resumen previo",
            delta_messages=[{"id": "m2", "role": "assistant", "content": "nuevo"}],
            is_zero_delta=False,
            next_marker="m2",
        ),
    )
    monkeypatch.setattr("api.index.load_bot_config", lambda: {"system_prompt": "bot personality"})

    seen_summary_messages = {}

    def _fake_summary_model(messages):
        seen_summary_messages["messages"] = messages
        return "canon actualizado", 321

    monkeypatch.setattr("api.index._call_summary_model", _fake_summary_model)

    result = handle_summary_command("chat-1", MagicMock(), "instruccion")

    assert seen_summary_messages["messages"] == [
        {"role": "system", "content": "bot personality"},
        {"role": "assistant", "content": "nuevo"},
        {"role": "user", "content": "instruccion"},
    ]
    assert result.response_text == "canon actualizado"
    assert result.pending_summary == "canon actualizado"
    assert result.pending_marker == "m2"
    assert result.summary_cost == 321
    assert result.billing_segments == []
    assert result.is_fallback is False


def test_handle_summary_command_uses_custom_prompt_with_personality(monkeypatch):
    from api.index import IncrementalSummarySource, handle_summary_command

    custom_focus = "enfocate solo en riesgos y proximos pasos"

    monkeypatch.setattr("api.index._state_get_user_chat_summary", lambda *_: None)
    monkeypatch.setattr("api.index._state_get_chat_compacted_until", lambda *_: None)
    monkeypatch.setattr(
        "api.index.get_chat_history",
        lambda *_: [{"id": "m1", "role": "user", "content": "mensaje nuevo"}],
    )
    monkeypatch.setattr(
        "api.index._build_incremental_summary_source",
        lambda *_: IncrementalSummarySource(
            prior_summary=None,
            delta_messages=[{"id": "m1", "role": "user", "content": "mensaje nuevo"}],
            is_zero_delta=False,
            next_marker="m1",
        ),
    )
    monkeypatch.setattr("api.index.load_bot_config", lambda: {"system_prompt": "bot personality"})

    calls = {"summary_messages": None}

    def _fake_summary_model(messages):
        calls["summary_messages"] = messages
        return "canon", 9

    monkeypatch.setattr("api.index._call_summary_model", _fake_summary_model)

    result = handle_summary_command("chat-1", MagicMock(), custom_focus)

    assert calls["summary_messages"] is not None
    assert calls["summary_messages"][0] == {
        "role": "system",
        "content": "bot personality",
    }
    assert calls["summary_messages"][1] == {
        "role": "user",
        "content": "mensaje nuevo",
    }
    assert calls["summary_messages"][2] == {
        "role": "user",
        "content": custom_focus,
    }
    assert result.response_text == "canon"
    assert result.pending_summary == "canon"
    assert result.summary_cost == 9
    assert result.billing_segments == []
    assert result.is_fallback is False
