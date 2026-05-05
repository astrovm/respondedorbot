def test_price_refresh_loop_runs_before_sleep(monkeypatch):
    import run_polling

    calls = []

    def fake_refresh():
        calls.append("refresh")
        raise KeyboardInterrupt

    monkeypatch.setattr(run_polling, "refresh_price_caches", fake_refresh)
    monkeypatch.setattr(run_polling.time, "sleep", lambda _seconds: calls.append("sleep"))

    try:
        run_polling._price_refresh_loop()
    except KeyboardInterrupt:
        pass

    assert calls == ["refresh"]
