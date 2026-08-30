from __future__ import annotations

import logging

from api.ai import pricing


class _FakeRustAiReserveEstimates:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: int, *arguments: object) -> int:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust reserve estimate failure")
        return value

    def ai_chat_output_token_limit(self, model: str) -> int:
        return self._result("limit", 91, model)

    def ai_estimate_text_tokens(self, text: str | None) -> int:
        return self._result("text", 92, text)

    def ai_estimate_nested_tokens(self, value_json: str) -> int:
        return self._result("nested", 93, value_json)

    def ai_estimate_message_tokens(self, messages_json: str) -> int:
        return self._result("messages", 94, messages_json)

    def ai_estimate_chat_reserve_credit_units(
        self,
        system_message_json: str | None,
        messages_json: str,
        max_output_tokens: int | None,
        extra_input_tokens: int,
        model: str,
    ) -> int:
        return self._result(
            "chat",
            95,
            system_message_json,
            messages_json,
            max_output_tokens,
            extra_input_tokens,
            model,
        )

    def ai_estimate_vision_reserve_credit_units(
        self,
        prompt_text: str,
        image_byte_length: int,
        extra_input_tokens: int,
        max_output_tokens: int,
        model: str,
    ) -> int:
        return self._result(
            "vision",
            96,
            prompt_text,
            image_byte_length,
            extra_input_tokens,
            max_output_tokens,
            model,
        )

    def ai_estimate_transcription_reserve_credit_units(
        self,
        audio_seconds: float,
    ) -> int:
        return self._result("transcription", 97, audio_seconds)

    def ai_estimate_firecrawl_reserve_credit_units(self) -> int:
        return self._result("firecrawl", 98)

    def ai_credit_units_from_usd_micros(self, usd_micros: int) -> int:
        return self._result("credits", 99, usd_micros)


def test_rust_ai_reserve_estimates_are_authoritative(monkeypatch) -> None:
    rust = _FakeRustAiReserveEstimates()
    monkeypatch.setattr(pricing, "_load_rust_ai_reserve_estimates", lambda: rust)

    assert pricing.chat_output_token_limit("model") == 91
    assert pricing.estimate_text_tokens("😀") == 92
    assert pricing.estimate_nested_tokens({"text": "olá"}) == 93
    assert pricing.estimate_message_tokens([{"role": "user", "content": "hi"}]) == 94
    assert (
        pricing.estimate_chat_reserve_credits(
            system_message={"role": "system", "content": "rules"},
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=5,
            extra_input_tokens=6,
            model="model",
        )
        == 95
    )
    assert (
        pricing.estimate_vision_reserve_credits(
            prompt_text="look",
            image_data=b"abc",
            extra_input_tokens=7,
            max_output_tokens=8,
            model="model",
        )
        == 96
    )
    assert pricing.estimate_transcribe_reserve_credits(9.5) == 97
    assert pricing.estimate_firecrawl_reserve_credits() == 98
    assert pricing.credit_units_from_usd_micros(10) == 99

    assert rust.calls == [
        ("limit", "model"),
        ("text", "😀"),
        ("nested", '{"text": "olá"}'),
        ("messages", '[{"role": "user", "content": "hi"}]'),
        (
            "chat",
            '{"role": "system", "content": "rules"}',
            '[{"role": "user", "content": "hi"}]',
            5,
            6,
            "model",
        ),
        ("vision", "look", 3, 7, 8, "model"),
        ("transcription", 9.5),
        ("firecrawl",),
        ("credits", 10),
    ]


def test_rust_ai_reserve_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAiReserveEstimates(fail=True)
    monkeypatch.setattr(pricing, "_load_rust_ai_reserve_estimates", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=pricing.__name__):
        assert pricing.chat_output_token_limit("other") == 1024
        assert pricing.estimate_text_tokens("abcde") == 2
        assert pricing.estimate_nested_tokens({"first": "hello", "rest": ["world", True]}) == 5
        assert (
            pricing.estimate_chat_reserve_credits(
                system_message={"role": "system", "content": "rules"},
                messages=[{"role": "user", "content": "hello"}],
            )
            == 17
        )
        assert pricing.estimate_vision_reserve_credits(prompt_text="describe") == 16
        assert pricing.estimate_transcribe_reserve_credits(1) == 7
        assert pricing.estimate_firecrawl_reserve_credits() == 34
        assert pricing.credit_units_from_usd_micros(51) == 2

    assert "using Python fallback" in caplog.text
