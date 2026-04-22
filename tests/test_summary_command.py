from tests.support import *


EXHAUSTIVE_SUMMARY_PROMPT = (
    "resumí la siguiente conversación de forma exhaustiva y técnica. "
    "incluí todos los temas tratados, quién dijo qué, conclusiones, "
    "decisiones pendientes y datos relevantes."
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

    def _fake_complete(system_message, messages, **kwargs):
        assert kwargs.get("response_meta") is not None
        response_meta = kwargs["response_meta"]
        response_meta["billing_segments"] = [{"usd_micros": 11}]
        assert system_message["role"] == "system"
        assert "foco custom" in system_message["content"]
        assert "resumen previo" in system_message["content"]
        assert messages == [{"role": "user", "content": "presentá el resumen"}]
        return "respuesta final"

    monkeypatch.setattr("api.index.complete_with_providers", _fake_complete)

    result = handle_summary_command("chat-1", MagicMock(), "foco custom")

    assert result.response_text == "respuesta final"
    assert result.pending_summary == "resumen previo"
    assert result.pending_marker is None
    assert result.summary_cost == 0
    assert result.billing_segments == [{"usd_micros": 11}]
    assert result.is_fallback is False


def test_handle_summary_command_updates_canonical_summary_when_new_delta_exists(monkeypatch):
    from api.index import IncrementalSummarySource, handle_summary_command

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

    seen_summary_messages = {}

    def _fake_summary_model(messages):
        seen_summary_messages["messages"] = messages
        return "canon actualizado", 321

    monkeypatch.setattr("api.index._call_summary_model", _fake_summary_model)

    def _fake_complete(system_message, messages, **kwargs):
        assert kwargs.get("response_meta") is not None
        response_meta = kwargs["response_meta"]
        response_meta["billing_segments"] = [{"usd_micros": 22}]
        assert system_message["role"] == "system"
        assert "instruccion" in system_message["content"]
        assert "canon actualizado" in system_message["content"]
        assert messages == [{"role": "user", "content": "presentá el resumen"}]
        return "render final"

    monkeypatch.setattr("api.index.complete_with_providers", _fake_complete)

    result = handle_summary_command("chat-1", MagicMock(), "instruccion")

    assert seen_summary_messages["messages"] == [
        {"role": "system", "content": EXHAUSTIVE_SUMMARY_PROMPT},
        {
            "role": "user",
            "content": "resumen acumulado previo:\nresumen previo\n\nmensajes nuevos:\nassistant: nuevo",
        },
    ]
    assert result.response_text == "render final"
    assert result.pending_summary == "canon actualizado"
    assert result.pending_marker == "m2"
    assert result.summary_cost == 321
    assert result.billing_segments == [{"usd_micros": 22}]
    assert result.is_fallback is False


def test_handle_summary_command_applies_custom_focus_only_to_render_stage(monkeypatch):
    from api.index import IncrementalSummarySource, handle_summary_command

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

    calls = {"summary_messages": None, "render_system": None}

    def _fake_summary_model(messages):
        calls["summary_messages"] = messages
        return "canon", 9

    monkeypatch.setattr("api.index._call_summary_model", _fake_summary_model)

    def _fake_complete(system_message, messages, **kwargs):
        calls["render_system"] = system_message
        assert kwargs.get("response_meta") is not None
        kwargs["response_meta"]["billing_segments"] = []
        assert messages == [{"role": "user", "content": "presentá el resumen"}]
        return "respuesta render"

    monkeypatch.setattr("api.index.complete_with_providers", _fake_complete)

    result = handle_summary_command("chat-1", MagicMock(), custom_focus)

    assert calls["summary_messages"] is not None
    assert calls["render_system"] is not None
    assert calls["summary_messages"][0] == {
        "role": "system",
        "content": EXHAUSTIVE_SUMMARY_PROMPT,
    }
    assert calls["summary_messages"][1] == {
        "role": "user",
        "content": "user: mensaje nuevo",
    }
    assert custom_focus not in calls["summary_messages"][0]["content"]
    assert calls["render_system"]["role"] == "system"
    assert custom_focus in calls["render_system"]["content"]
    assert "canon" in calls["render_system"]["content"]
    assert result.response_text == "respuesta render"
