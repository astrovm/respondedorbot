from __future__ import annotations

import pytest

from api.admin import commands


class _FakeRustAdminReports:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def parse_creditlog_limit(self, message_text: str) -> int | None:
        if self.fail:
            raise ValueError("synthetic parse failure")
        return 7 if message_text else 10

    def truncate_admin_report(
        self, text: str, max_length: int, truncated_label: str
    ) -> str:
        if self.fail:
            raise ValueError("synthetic truncation failure")
        return f"rust:{text[:1]}:{max_length}:{truncated_label}"


def test_admin_report_helpers_use_rust_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "_load_rust_admin_reports", _FakeRustAdminReports)

    assert commands._parse_creditlog_limit("12") == 7
    assert commands.truncate_creditlog_message("long report", 5) == "rust:l:5:truncado"


def test_admin_report_helpers_fall_back_after_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commands,
        "_load_rust_admin_reports",
        lambda: _FakeRustAdminReports(fail=True),
    )

    assert commands._parse_creditlog_limit("99") == 25
    assert commands._parse_creditlog_limit("invalid") is None
    assert commands.truncate_creditlog_message("abcdef", 5) == "\n\n[truncado]"
