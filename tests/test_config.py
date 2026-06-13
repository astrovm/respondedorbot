from unittest.mock import MagicMock

from redis.exceptions import BusyLoadingError


def test_config_redis_reuses_connection_pool(monkeypatch):
    import api.core.config as config

    created_pools = []

    class DummyPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_pools.append(self)

    class DummyRedis:
        def __init__(self, connection_pool):
            self.connection_pool = connection_pool
            self.ping = MagicMock()

    monkeypatch.setattr(config.redis, "ConnectionPool", DummyPool)
    monkeypatch.setattr(config.redis, "Redis", DummyRedis)
    monkeypatch.setattr(config, "_REDIS_POOLS", {})

    first = config.config_redis(host="redis", port=6379, password="secret")
    second = config.config_redis(host="redis", port=6379, password="secret")

    assert first.connection_pool is second.connection_pool
    assert len(created_pools) == 1


def test_config_redis_retries_busy_loading_without_admin_report(monkeypatch):
    import api.core.config as config

    reports = []
    sleeps = []

    class DummyPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyRedis:
        def __init__(self, connection_pool):
            self.connection_pool = connection_pool
            self.ping = MagicMock(
                side_effect=[
                    BusyLoadingError("Redis is loading the dataset in memory"),
                    BusyLoadingError("Redis is loading the dataset in memory"),
                    True,
                ]
            )

    monkeypatch.setattr(config.redis, "ConnectionPool", DummyPool)
    monkeypatch.setattr(config.redis, "Redis", DummyRedis)
    monkeypatch.setattr(config.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(config, "_REDIS_POOLS", {})
    config.configure(admin_reporter=lambda *args: reports.append(args))

    try:
        redis_client = config.config_redis(host="redis", port=6379)
    finally:
        config.configure(admin_reporter=None)

    assert redis_client.ping.call_count == 3
    assert sleeps == [
        config._REDIS_BUSY_LOADING_RETRY_DELAY_SECONDS,
        config._REDIS_BUSY_LOADING_RETRY_DELAY_SECONDS,
    ]
    assert reports == []
