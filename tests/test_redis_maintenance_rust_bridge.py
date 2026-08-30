from __future__ import annotations

import json

import pytest

from api.services import maintenance


class _FakeRustMaintenance:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def run_redis_maintenance(self, *arguments: object) -> str:
        if self.fail:
            raise ValueError("synthetic Rust Redis maintenance failure")
        self.calls.append(arguments)
        return json.dumps(
            {
                "expired_keys": 3,
                "deleted_keys": 2,
                "maxmemory": "268435456",
                "maxmemory_policy": "allkeys-lru",
            }
        )


def test_maintenance_rust_path_is_the_only_redis_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMaintenance()
    monkeypatch.setattr(maintenance, "_load_rust_redis_maintenance", lambda: rust)
    monkeypatch.setattr(
        maintenance,
        "config_redis",
        lambda: pytest.fail("Python Redis owner must not run"),
    )
    monkeypatch.setattr(maintenance.credits_db, "is_configured", lambda: False)
    monkeypatch.setenv("REDIS_HOST", "redis.internal")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")

    result = maintenance.run_maintenance()

    assert rust.calls == [
        (
            "redis.internal",
            6380,
            "synthetic-password",
            maintenance.REDIS_MAXMEMORY,
            maintenance.REDIS_MAXMEMORY_POLICY,
        )
    ]
    assert result == {
        "redis": {
            "expired_keys": 3,
            "deleted_keys": 2,
            "maxmemory": "268435456",
            "maxmemory_policy": "allkeys-lru",
        },
        "ledger": {"skipped": True, "reason": "postgres not configured"},
    }


def test_maintenance_rust_error_does_not_fall_through_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMaintenance(fail=True)
    monkeypatch.setattr(maintenance, "_load_rust_redis_maintenance", lambda: rust)
    monkeypatch.setattr(
        maintenance,
        "config_redis",
        lambda: pytest.fail("Python Redis owner must not run"),
    )

    with pytest.raises(ValueError, match="synthetic Rust Redis maintenance failure"):
        maintenance.run_maintenance()


def test_maintenance_rust_path_keeps_ledger_cleanup_in_current_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _FakeRustMaintenance()
    monkeypatch.setattr(maintenance, "_load_rust_redis_maintenance", lambda: rust)
    monkeypatch.setattr(maintenance.credits_db, "is_configured", lambda: True)
    monkeypatch.setattr(
        maintenance.credits_db,
        "purge_expired_ai_ledger_events",
        lambda days: {"deleted_events": days},
    )

    result = maintenance.run_maintenance()

    assert result["ledger"] == {"deleted_events": maintenance.AI_LEDGER_RETENTION_DAYS}
