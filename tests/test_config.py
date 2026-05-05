from unittest.mock import MagicMock


def test_config_redis_reuses_connection_pool(monkeypatch):
    import api.config as config

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
