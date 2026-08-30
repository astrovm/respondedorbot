from __future__ import annotations

import json
from typing import Any

import pytest

from api.memory import background


class _FakeRustCompactionJobs:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def normalize_compaction_job(self, payload: str) -> str:
        if self.fail:
            raise ValueError("synthetic compaction-job failure")
        decoded = json.loads(payload)
        decoded["schema_version"] = 1
        decoded["locale"] = "rust-locale"
        return json.dumps(decoded)


def _legacy_job() -> dict[str, Any]:
    return {
        "chat_id": "123",
        "messages": [],
        "prior_summary": None,
        "expected_marker": None,
        "target_marker": "m1",
        "reservation": {},
        "user_id": 42,
        "message_id": "99",
    }


def test_compaction_job_normalizer_uses_rust_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        background,
        "_load_rust_compaction_jobs",
        _FakeRustCompactionJobs,
    )
    legacy = _legacy_job()
    payload = json.dumps(legacy)

    normalized = background._normalize_compaction_job_payload(payload, legacy)

    assert normalized["schema_version"] == 1
    assert normalized["locale"] == "rust-locale"


def test_compaction_job_normalizer_falls_back_and_rejects_future_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        background,
        "_load_rust_compaction_jobs",
        lambda: _FakeRustCompactionJobs(fail=True),
    )
    legacy = _legacy_job()
    assert background._normalize_compaction_job_payload(json.dumps(legacy), legacy) == legacy

    future = {**legacy, "schema_version": 2}
    with pytest.raises(ValueError, match="unsupported compaction job schema version 2"):
        background._normalize_compaction_job_payload(json.dumps(future), future)


def test_new_compaction_jobs_serialize_explicit_schema_version() -> None:
    job = background.CompactionJob(**_legacy_job())

    assert job.schema_version == 1
