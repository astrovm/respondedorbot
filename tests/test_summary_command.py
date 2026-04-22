from tests.support import *


EXHAUSTIVE_SUMMARY_PROMPT = (
    "resumí la siguiente conversación de forma exhaustiva. "
    "incluí todos los temas tratados, quién dijo qué, conclusiones, "
    "decisiones pendientes y datos relevantes. no omitas nada."
)


def test_handle_summary_command_returns_early_when_no_history(monkeypatch):
    from api.index import handle_summary_command

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: "old")
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

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: "resumen previo")
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
            formatted_delta="",
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
    from api.index import IncrementalSummarySource, SUMMARY_GENERATION_PROMPT, handle_summary_command

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: "resumen previo")
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
            formatted_delta="assistant: nuevo",
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

    expected_prompt = f"bot personality\n\n{SUMMARY_GENERATION_PROMPT}"
    assert seen_summary_messages["messages"] == [
        {"role": "system", "content": expected_prompt},
        {
            "role": "user",
            "content": "resumen acumulado previo:\nresumen previo\n\nmensajes nuevos:\nassistant: nuevo",
        },
    ]
    assert result.response_text == "canon actualizado"
    assert result.pending_summary == "canon actualizado"
    assert result.pending_marker == "m2"
    assert result.summary_cost == 321
    assert result.billing_segments == []
    assert result.is_fallback is False


def test_handle_summary_command_ignores_custom_prompt_uses_generation_prompt(monkeypatch):
    from api.index import IncrementalSummarySource, SUMMARY_GENERATION_PROMPT, handle_summary_command

    custom_focus = "enfocate solo en riesgos y proximos pasos"

    monkeypatch.setattr("api.index._state_get_chat_summary", lambda *_: None)
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
            formatted_delta="user: mensaje nuevo",
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

    expected_prompt = f"bot personality\n\n{SUMMARY_GENERATION_PROMPT}"
    assert calls["summary_messages"] is not None
    assert calls["summary_messages"][0] == {
        "role": "system",
        "content": expected_prompt,
    }
    assert calls["summary_messages"][1] == {
        "role": "user",
        "content": "user: mensaje nuevo",
    }
    assert result.response_text == "canon"
    assert result.pending_summary == "canon"
    assert result.summary_cost == 9
    assert result.billing_segments == []
    assert result.is_fallback is False
