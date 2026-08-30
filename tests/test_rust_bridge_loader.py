from api.core import rust_bridge


def test_rust_bridge_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RUST_TEST_FEATURE_ENABLED", raising=False)
    assert rust_bridge.load_rust_bridge("RUST_TEST_FEATURE_ENABLED") is None


def test_missing_enabled_rust_bridge_warns_once(monkeypatch, caplog):
    def missing_bridge(_name):
        raise ImportError("synthetic missing bridge")

    monkeypatch.setenv("RUST_TEST_FEATURE_ENABLED", "yes")
    monkeypatch.setattr(rust_bridge.importlib, "import_module", missing_bridge)
    rust_bridge.reset_rust_bridge_cache()
    try:
        assert rust_bridge.load_rust_bridge("RUST_TEST_FEATURE_ENABLED") is None
        assert rust_bridge.load_rust_bridge("RUST_TEST_FEATURE_ENABLED") is None
    finally:
        rust_bridge.reset_rust_bridge_cache()

    assert len(caplog.records) == 1
