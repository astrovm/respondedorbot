import os
from concurrent.futures import Future

import pytest
import redis as redis_module

from api.core import config as config_module
from api.ai import pricing as ai_pricing_module
from api.billing import provider_usage as provider_usage_module
from api.providers import errors as provider_errors_module
from api import index as index_module
from api.providers.backoff import clear_all_cooldowns
from api.services import bcra as bcra_service
from api.services import credits_db as credits_db_service


class _FastFailRedis:
    """Redis stand-in that raises ConnectionError immediately on any call."""

    def __getattr__(self, name: str):
        def raiser(*args, **kwargs):
            raise redis_module.ConnectionError("test: Redis not available")

        return raiser


class _NoopExecutor:
    """Prevent background work from escaping an individual test."""

    def submit(self, _fn):
        future = Future()
        future.set_result(None)
        return future


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_artifacts():
    yield
    try:
        if os.path.isfile("test_api_key"):
            os.remove("test_api_key")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_caches(monkeypatch):
    bcra_service.reset_local_caches()
    monkeypatch.setattr(
        ai_pricing_module,
        "_load_rust_ai_reserve_estimates",
        lambda: None,
    )
    monkeypatch.setattr(
        ai_pricing_module,
        "_load_rust_ai_pricing",
        lambda: None,
    )
    monkeypatch.setattr(
        provider_usage_module,
        "_load_rust_ai_usage_policy",
        lambda: None,
    )
    monkeypatch.setattr(
        provider_errors_module,
        "_load_rust_provider_error_policy",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_schema",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_reads",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_balance_io",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_onboarding",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_star_payments",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_manual_credits",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_chat_ai_credits",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_ai_debt",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_ai_refunds",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_ai_charges",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_provider_usage",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_ai_settlements",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_legacy_settlements",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_audit_writes",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_audit_reads",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_reconciliation_reads",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_maintenance",
        lambda: None,
    )
    monkeypatch.setattr(
        credits_db_service,
        "_load_rust_billing_charge_history",
        lambda: None,
    )
    clear_all_cooldowns()
    index_module._chat_config_service.clear_cache()
    monkeypatch.setenv(
        "BOT_SYSTEM_PROMPT",
        "sos el gordo, un bot argentino de prueba.\n\nReglas de prueba.",
    )
    monkeypatch.setattr(
        index_module.chat_config_db_service,
        "is_configured",
        lambda: False,
    )
    monkeypatch.setattr("time.sleep", lambda *_, **__: None)
    monkeypatch.setattr(
        index_module.app_runtime.providers,
        "complete",
        lambda *_, **__: "",
    )
    monkeypatch.setattr(
        index_module.app_runtime.config,
        "redis",
        lambda *_, **__: _FastFailRedis(),
    )
    monkeypatch.setattr(index_module, "_BACKGROUND_REFRESH_EXECUTOR", _NoopExecutor())
    config_module.reset_cache()
    yield
