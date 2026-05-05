import json

from api.services.stale_cache import StaleCache, StaleCacheResult


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.locks = set()

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value
        return True

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.locks:
            return False
        if nx:
            self.locks.add(key)
        self.values[key] = value
        return True


def test_stale_cache_returns_fresh_value_without_refresh():
    redis_client = FakeRedis()
    cache = StaleCache(redis_client=redis_client, now=lambda: 100)
    redis_client.values["k"] = json.dumps({"timestamp": 95, "value": "fresh"})

    result = cache.get(
        key="k",
        lock_key="lock:k",
        ttl=10,
        stale_grace=60,
        refresh=lambda: "new",
        schedule_refresh=lambda fn: fn(),
    )

    assert result == StaleCacheResult(value="fresh", status="fresh")


def test_stale_cache_returns_stale_and_schedules_refresh():
    redis_client = FakeRedis()
    scheduled = []
    cache = StaleCache(redis_client=redis_client, now=lambda: 120)
    redis_client.values["k"] = json.dumps({"timestamp": 100, "value": "old"})

    result = cache.get(
        key="k",
        lock_key="lock:k",
        ttl=10,
        stale_grace=60,
        refresh=lambda: "new",
        schedule_refresh=lambda fn: scheduled.append(fn),
    )

    assert result == StaleCacheResult(value="old", status="stale")
    assert len(scheduled) == 1


def test_stale_cache_cold_miss_refreshes_inline():
    redis_client = FakeRedis()
    cache = StaleCache(redis_client=redis_client, now=lambda: 100)

    result = cache.get(
        key="k",
        lock_key="lock:k",
        ttl=10,
        stale_grace=60,
        refresh=lambda: "new",
        schedule_refresh=lambda fn: fn(),
    )

    assert result == StaleCacheResult(value="new", status="miss")
    assert json.loads(redis_client.values["k"])["value"] == "new"
