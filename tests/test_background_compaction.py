from tests.support import *


class _FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    def hexists(self, name, field):
        return field in self.hashes.get(name, {})

    def hsetnx(self, name, field, value):
        bucket = self.hashes.setdefault(name, {})
        if field in bucket:
            return 0
        bucket[field] = value
        return 1

    def hset(self, name, field, value):
        self.hashes.setdefault(name, {})[field] = value
        return 1

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def hdel(self, name, field):
        return int(self.hashes.get(name, {}).pop(field, None) is not None)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        return int(self.values.pop(key, None) is not None)


def _build_queue(redis_client, *, compact, save_result, credits):
    from api.memory.background import DurableCompactionQueue

    return DurableCompactionQueue(
        redis_factory=lambda: redis_client,
        compact=compact,
        get_summary=lambda _client, _chat_id: None,
        get_marker=lambda _client, _chat_id: None,
        save_result=save_result,
        estimate_reserve=lambda _plan: 3,
        credits=credits,
        logger=MagicMock(),
    )


def test_compaction_is_persisted_before_the_model_runs_and_survives_queue_restart():
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    compact = MagicMock(return_value=("[contexto anterior: resumen]", 100))
    save_result = MagicMock()
    credits = MagicMock()
    billing = MagicMock(
        user_id=42,
        message={"message_id": 99},
    )
    billing.reserve_ai_credits.return_value = (
        {
            "reserved_credit_units": 3,
            "chat_scope_id": None,
            "source": "user",
            "usage_tag": "memory_compaction:123:m2",
        },
        None,
    )
    plan = CompactionPlan(
        chat_id="123",
        messages=[{"id": "m1", "role": "user", "text": "hola"}],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
    )

    first_process = _build_queue(
        redis_client,
        compact=compact,
        save_result=save_result,
        credits=credits,
    )
    assert first_process.enqueue(plan, billing) is True
    compact.assert_not_called()

    restarted_process = _build_queue(
        redis_client,
        compact=compact,
        save_result=save_result,
        credits=credits,
    )
    assert restarted_process.run_pending_once() == 1

    compact.assert_called_once_with(plan.messages, None)
    save_result.assert_called_once_with(
        redis_client,
        "123",
        "[contexto anterior: resumen]",
        "m1",
    )
    assert redis_client.hgetall("memory:compaction:jobs") == {}
    credits.refund_ai_charge.assert_called_once()


def test_compaction_skips_reservation_when_a_chat_already_has_a_job():
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    redis_client.hset("memory:compaction:jobs", "123", "{}")
    billing = MagicMock()
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        credits=MagicMock(),
    )

    result = queue.enqueue(
        CompactionPlan("123", [], None, None, "m1"),
        billing,
    )

    assert result is False
    billing.reserve_ai_credits.assert_not_called()
