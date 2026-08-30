from __future__ import annotations

import json

import pytest

from api.memory import background


class _FakeRustCompactionPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def _check(self) -> None:
        if self.fail:
            raise ValueError("synthetic compaction-policy failure")

    def evaluate_compaction_policy(self, *_arguments: object) -> str:
        self._check()
        return "save_and_settle"

    def compaction_job_is_due(self, next_attempt_at: float, now: float) -> bool:
        self._check()
        return next_attempt_at == 7.0 and now == 8.0

    def compaction_retry_transition(
        self,
        attempts: int,
        now: float,
        has_billing_segment: bool,
    ) -> str:
        self._check()
        assert (attempts, now, has_billing_segment) == (1, 8.0, True)
        return json.dumps(
            {
                "attempts": 9,
                "terminal": False,
                "next_attempt_at": 88.0,
                "actual_credit_units": None,
            }
        )


def test_compaction_helpers_use_rust_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        background,
        "_load_rust_compaction_policy",
        _FakeRustCompactionPolicy,
    )

    assert background._compaction_job_is_due(7.0, 8.0) is True
    assert (
        background._compaction_disposition(
            current_summary=None,
            current_marker=None,
            prior_summary=None,
            expected_marker=None,
            result_summary="result",
            target_marker="m1",
        )
        == "save_and_settle"
    )
    assert background._compaction_retry_transition(1, 8.0, True) == {
        "attempts": 9,
        "terminal": False,
        "next_attempt_at": 88.0,
        "actual_credit_units": None,
    }


def test_compaction_helpers_fall_back_after_bridge_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        background,
        "_load_rust_compaction_policy",
        lambda: _FakeRustCompactionPolicy(fail=True),
    )

    assert background._compaction_job_is_due(9.0, 8.0) is False
    assert (
        background._compaction_disposition(
            current_summary=None,
            current_marker=None,
            prior_summary=None,
            expected_marker=None,
            result_summary=None,
            target_marker="m1",
        )
        == "generate_summary"
    )
    assert background._compaction_retry_transition(0, 100.0, False) == {
        "attempts": 1,
        "terminal": False,
        "next_attempt_at": 130.0,
        "actual_credit_units": None,
    }
