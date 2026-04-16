import json

import api.services.redis_helpers as rh


class FakeRedis:
    def __init__(self, value):
        self._value = value

    def get(self, key):
        return self._value


def test_redis_get_json_decodes_bytes():
    payload = {"a": 1}
    fake = FakeRedis(json.dumps(payload).encode("utf-8"))
    got = rh.redis_get_json(fake, "k")
    assert got == payload


def test_redis_get_json_invalid_returns_none():
    fake = FakeRedis(b"not-json")
    got = rh.redis_get_json(fake, "k")
    assert got is None
