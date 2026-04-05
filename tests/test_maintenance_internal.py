from api.services.maintenance import (
    CHAT_STATE_TTL,
    GIPHY_STALE_TTL,
    prune_redis_growth,
)


class _FakeRedis:
    def __init__(self):
        self.ttls = {
            "giphy_pool_stale:gm": -1,
            "chat_history:123": -1,
            "chat_message_ids:123": -1,
            "a" * 64: -1,
            f"2026-04-05-08{'b' * 64}": -1,
        }
        self.expired = []
        self.deleted = []

    def scan_iter(self, match=None):
        if match == "giphy_pool_stale:*":
            return iter(["giphy_pool_stale:gm"])
        if match == "chat_history:*":
            return iter(["chat_history:123"])
        if match == "chat_message_ids:*":
            return iter(["chat_message_ids:123"])
        if match == "request_cache:*":
            return iter([])
        if match == "request_cache_history:*":
            return iter([])
        return iter([])

    def ttl(self, key):
        return self.ttls.get(key, -2)

    def expire(self, key, ttl):
        self.expired.append((key, ttl))
        self.ttls[key] = ttl
        return True

    def delete(self, *keys):
        self.deleted.extend(keys)
        return len(keys)

    def keys(self, pattern):
        if pattern == "[0-9a-f][0-9a-f]*":
            return ["a" * 64]
        return []


def test_prune_redis_growth_sets_ttls_and_deletes_legacy_cache_keys():
    redis_client = _FakeRedis()

    result = prune_redis_growth(
        redis_client,
        legacy_cache_keys=["a" * 64, f"2026-04-05-08{'b' * 64}"],
    )

    assert ("giphy_pool_stale:gm", GIPHY_STALE_TTL) in redis_client.expired
    assert ("chat_history:123", CHAT_STATE_TTL) in redis_client.expired
    assert ("chat_message_ids:123", CHAT_STATE_TTL) in redis_client.expired
    assert "a" * 64 in redis_client.deleted
    assert f"2026-04-05-08{'b' * 64}" in redis_client.deleted
    assert result["expired_keys"] == 3
    assert result["deleted_keys"] == 2
