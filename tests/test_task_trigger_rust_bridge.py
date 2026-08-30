from __future__ import annotations

import json
from typing import Any

from api.tasks import models


class _FakeRustParser:
    def __init__(self, response: dict[str, Any] | Exception) -> None:
        self.response = response
        self.input: dict[str, Any] | None = None

    def parse_task_trigger(self, input_json: str) -> str:
        self.input = json.loads(input_json)
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def test_rust_trigger_result_becomes_typed_python_trigger(monkeypatch) -> None:
    rust = _FakeRustParser(
        {
            "trigger": {
                "kind": "cron",
                "hour": 9,
                "minute": 5,
                "weekdays": ["mon", "wed"],
                "day": None,
            },
            "error": None,
        }
    )
    monkeypatch.setattr(models, "_load_rust_task_trigger_parser", lambda: rust)

    result = models.parse_task_trigger(
        trigger_config={
            "type": "cron",
            "hour": True,
            "minute": 5,
            "day_of_week": "lun,mie",
        }
    )

    assert result == models.TriggerParseResult(
        trigger=models.CronTrigger(
            kind="cron",
            hour=9,
            minute=5,
            weekdays=("mon", "wed"),
        )
    )
    assert rust.input == {
        "delay_seconds": {"state": "missing"},
        "interval_seconds": {"state": "missing"},
        "config": {
            "kind": "cron",
            "hour": {"state": "value", "value": 1},
            "minute": {"state": "value", "value": 5},
            "weekdays": "lun,mie",
            "day": {"state": "missing"},
        },
    }


def test_rust_trigger_error_is_localized(monkeypatch) -> None:
    rust = _FakeRustParser(
        {
            "trigger": None,
            "error": {"code": "weekday", "value": "foo"},
        }
    )
    monkeypatch.setattr(models, "_load_rust_task_trigger_parser", lambda: rust)

    result = models.parse_task_trigger(
        trigger_config={
            "type": "cron",
            "hour": 9,
            "minute": 0,
            "day_of_week": "foo",
        }
    )

    assert result == models.TriggerParseResult(error="day_of_week invalido: foo")


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustParser(RuntimeError("synthetic bridge failure"))
    monkeypatch.setattr(models, "_load_rust_task_trigger_parser", lambda: rust)

    result = models.parse_task_trigger(delay_seconds=10)

    assert result == models.TriggerParseResult(
        trigger=models.DelayTrigger(kind="delay", seconds=10)
    )
    assert "using Python fallback" in caplog.text


def test_unrepresentable_integers_keep_their_error_direction(monkeypatch) -> None:
    rust = _FakeRustParser(
        {"trigger": None, "error": {"code": "delay_max"}}
    )
    monkeypatch.setattr(models, "_load_rust_task_trigger_parser", lambda: rust)

    result = models.parse_task_trigger(delay_seconds=2**100)

    assert result.error == "el maximo es 10 años"
    assert rust.input is not None
    assert rust.input["delay_seconds"] == {"state": "above_range"}
