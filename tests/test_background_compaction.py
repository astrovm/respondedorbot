import json
from dataclasses import asdict

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


def _build_queue(
    redis_client,
    *,
    compact,
    save_result,
    settle_reservation,
    record_provider_usage=None,
    list_provider_usage=None,
):
    from api.memory.background import DurableCompactionQueue

    return DurableCompactionQueue(
        redis_factory=lambda: redis_client,
        compact=compact,
        get_summary=lambda _client, _chat_id: None,
        get_marker=lambda _client, _chat_id: None,
        save_result=save_result,
        estimate_reserve=lambda _plan: 3,
        settle_reservation=settle_reservation,
        record_provider_usage=record_provider_usage or MagicMock(return_value=True),
        list_provider_usage=list_provider_usage or MagicMock(return_value=[]),
        logger=MagicMock(),
    )


def test_background_compaction_preserves_known_overage_when_model_cost_is_missing():
    from api.memory.background import CompactionJob, DurableCompactionQueue

    settle_reservation = MagicMock(return_value={"applied": True})
    admin_report = MagicMock()
    queue = DurableCompactionQueue(
        redis_factory=lambda: _FakeRedis(),
        compact=MagicMock(),
        get_summary=MagicMock(),
        get_marker=MagicMock(),
        save_result=MagicMock(),
        estimate_reserve=MagicMock(),
        settle_reservation=settle_reservation,
        record_provider_usage=MagicMock(return_value=True),
        list_provider_usage=MagicMock(return_value=[]),
        logger=MagicMock(),
        admin_report=admin_report,
    )
    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
        },
        user_id=42,
        message_id="99",
        result_summary="summary",
        result_billing_segment={
            "kind": "summary",
            "model": "unknown/model",
            "usage": {},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "web_search_requests": 1,
                "firecrawl_credits_used": 1,
            },
        },
    )

    queue._settle(job, reason="memory_compaction_success")

    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 17
    admin_report.assert_called_once()


def test_failed_compaction_settles_persisted_provider_usage():
    from api.memory.background import CompactionJob

    redis_client = _FakeRedis()
    settle_reservation = MagicMock(return_value={"applied": True})
    record_provider_usage = MagicMock(return_value=True)
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=settle_reservation,
        record_provider_usage=record_provider_usage,
    )
    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
            "metadata": {"operation_id": "operation-1"},
        },
        user_id=42,
        message_id="99",
        attempts=2,
        result_summary="summary",
        result_billing_segment={
            "kind": "summary",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {"cost": 0.001},
            "source": "openrouter",
            "metadata": {
                "provider": "openrouter",
                "provider_generation_id": "generation-1",
            },
        },
    )
    redis_client.hset("memory:compaction:jobs", "123", "stored")

    queue._retry_or_refund(redis_client, job, RuntimeError("save failed"))

    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 20
    record_provider_usage.assert_called_once_with(
        user_id=42,
        chat_id=None,
        operation_id="operation-1",
        segment_id="openrouter:generation-1",
        segment=job.result_billing_segment,
    )
    assert settle_reservation.call_args.kwargs["metadata"]["operation_id"] == (
        "operation-1"
    )
    assert redis_client.hgetall("memory:compaction:jobs") == {}


def test_compaction_marks_its_operation_active_while_processing():
    from api.billing.reconciliation import is_ai_operation_active
    from api.memory.background import CompactionJob

    redis_client = _FakeRedis()
    operation_id = "operation-1"

    def compact(_messages, _summary):
        assert is_ai_operation_active(operation_id)
        return (
            "summary",
            100,
            {
                "kind": "summary",
                "model": "deepseek/deepseek-v4-flash-0731",
                "usage": {"cost": 0.0001},
                "source": "openrouter",
                "metadata": {"provider_generation_id": "generation-1"},
            },
        )

    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
            "metadata": {"operation_id": operation_id},
        },
        user_id=42,
        message_id="99",
    )
    redis_client.hset(
        "memory:compaction:jobs",
        "123",
        json.dumps(asdict(job)),
    )
    queue = _build_queue(
        redis_client,
        compact=compact,
        save_result=MagicMock(),
        settle_reservation=MagicMock(return_value={"applied": True}),
    )

    assert queue.run_pending_once() == 1
    assert is_ai_operation_active(operation_id) is False


def test_compaction_records_provider_usage_before_redis_result_write():
    from api.billing.reconciliation import is_ai_operation_active
    from api.memory.background import CompactionJob

    class FailingRedis(_FakeRedis):
        def hset(self, name, field, value):
            if name == "memory:compaction:jobs":
                raise RuntimeError("redis unavailable")
            return super().hset(name, field, value)

    operation_id = "operation-1"
    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
            "metadata": {"operation_id": operation_id},
        },
        user_id=42,
        message_id="99",
    )
    redis_client = FailingRedis()
    redis_client.hashes["memory:compaction:jobs"] = {
        "123": json.dumps(asdict(job))
    }
    record_provider_usage = MagicMock(return_value=True)
    queue = _build_queue(
        redis_client,
        compact=MagicMock(
            return_value=(
                "summary",
                100,
                {
                    "kind": "summary",
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "usage": {"cost": 0.0001},
                    "source": "openrouter",
                    "metadata": {"provider_generation_id": "generation-1"},
                },
            )
        ),
        save_result=MagicMock(),
        settle_reservation=MagicMock(return_value={"applied": True}),
        record_provider_usage=record_provider_usage,
    )

    with pytest.raises(RuntimeError, match="redis unavailable"):
        queue.run_pending_once()

    record_provider_usage.assert_called_once()
    assert record_provider_usage.call_args.kwargs["segment_id"] == (
        "openrouter:generation-1"
    )
    assert is_ai_operation_active(operation_id) is False


def test_provider_retries_have_distinct_durable_segment_ids():
    from api.billing.provider_usage import provider_segment_id

    segment = {
        "kind": "summary",
        "source": "openrouter",
        "metadata": {"provider_generation_id": "generation-1"},
    }
    retry = {
        **segment,
        "metadata": {"provider_generation_id": "generation-2"},
    }

    assert provider_segment_id(segment) == "openrouter:generation-1"
    assert provider_segment_id(retry) == "openrouter:generation-2"


def test_compaction_settles_every_durable_provider_retry():
    from api.ai.pricing import calculate_billing_for_segments
    from api.memory.background import CompactionJob

    def segment(generation_id):
        return {
            "kind": "summary",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {"cost": 0.001},
            "source": "openrouter",
            "metadata": {"provider_generation_id": generation_id},
        }

    segments = [segment("generation-1"), segment("generation-2")]
    settle_reservation = MagicMock(return_value={"applied": True})
    queue = _build_queue(
        _FakeRedis(),
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=settle_reservation,
        list_provider_usage=MagicMock(return_value=segments),
    )
    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary=None,
        expected_marker=None,
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
            "metadata": {"operation_id": "operation-1"},
        },
        user_id=42,
        message_id="99",
        result_summary="summary",
        result_billing_segment=segments[-1],
    )

    queue._settle(job, reason="memory_compaction_success")

    expected = calculate_billing_for_segments(segments)["charged_credit_units"]
    assert settle_reservation.call_args.kwargs["actual_credit_units"] == expected
    assert settle_reservation.call_args.kwargs["metadata"]["billing_segments"] == (
        segments
    )


def test_obsolete_compaction_settles_persisted_provider_usage():
    from api.memory.background import CompactionJob, DurableCompactionQueue

    redis_client = _FakeRedis()
    settle_reservation = MagicMock(return_value={"applied": True})
    queue = DurableCompactionQueue(
        redis_factory=lambda: redis_client,
        compact=MagicMock(),
        get_summary=MagicMock(return_value="newer summary"),
        get_marker=MagicMock(return_value="newer-marker"),
        save_result=MagicMock(),
        estimate_reserve=MagicMock(),
        settle_reservation=settle_reservation,
        record_provider_usage=MagicMock(return_value=True),
        list_provider_usage=MagicMock(return_value=[]),
        logger=MagicMock(),
    )
    job = CompactionJob(
        chat_id="123",
        messages=[],
        prior_summary="old summary",
        expected_marker="old-marker",
        target_marker="m1",
        reservation={
            "reserved_credit_units": 3,
            "credit_scale": 100,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
        },
        user_id=42,
        message_id="99",
        result_summary="generated summary",
        result_billing_segment={
            "kind": "summary",
            "model": "deepseek/deepseek-v4-flash-0731",
            "usage": {"cost": 0.001},
            "source": "openrouter",
            "metadata": {"provider": "openrouter"},
        },
    )

    queue._process(redis_client, job)

    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 20


def test_compaction_is_persisted_before_the_model_runs_and_survives_queue_restart():
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    compact = MagicMock(return_value=("[contexto anterior: resumen]", 100))
    save_result = MagicMock()
    settle_reservation = MagicMock(return_value={"applied": True})
    billing = MagicMock(
        user_id=42,
        message={"message_id": 99},
    )
    billing.reserve_background_ai_credits.return_value = (
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
        settle_reservation=settle_reservation,
    )
    assert first_process.enqueue(plan, billing) is True
    compact.assert_not_called()

    restarted_process = _build_queue(
        redis_client,
        compact=compact,
        save_result=save_result,
        settle_reservation=settle_reservation,
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
    settle_reservation.assert_called_once()
    assert settle_reservation.call_args.kwargs["reserved_credit_units"] == 3
    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 2


def test_background_compaction_restores_the_enqueued_locale():
    from api.i18n import current_locale, use_locale
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    compact = MagicMock(side_effect=lambda _messages, _summary: (current_locale(), 100))
    save_result = MagicMock()
    billing = MagicMock(user_id=42, message={"message_id": 99})
    billing.reserve_background_ai_credits.return_value = (
        {
            "reserved_credit_units": 3,
            "chat_scope_id": None,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
        },
        None,
    )
    queue = _build_queue(
        redis_client,
        compact=compact,
        save_result=save_result,
        settle_reservation=MagicMock(return_value={"applied": True}),
    )
    plan = CompactionPlan("123", [{"id": "m1"}], None, None, "m1")

    with use_locale("en"):
        assert queue.enqueue(plan, billing)
    stored_job = json.loads(redis_client.hgetall("memory:compaction:jobs")["123"])
    assert stored_job["locale"] == "en"

    assert queue.run_pending_once() == 1
    save_result.assert_called_once_with(redis_client, "123", "en", "m1")


def test_background_compaction_defaults_old_jobs_to_spanish():
    from api.i18n import current_locale, use_locale
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    compact = MagicMock(side_effect=lambda _messages, _summary: (current_locale(), 100))
    save_result = MagicMock()
    billing = MagicMock(user_id=42, message={"message_id": 99})
    billing.reserve_background_ai_credits.return_value = (
        {
            "reserved_credit_units": 3,
            "chat_scope_id": None,
            "source": "user",
            "usage_tag": "memory_compaction:123:m1",
        },
        None,
    )
    queue = _build_queue(
        redis_client,
        compact=compact,
        save_result=save_result,
        settle_reservation=MagicMock(return_value={"applied": True}),
    )
    plan = CompactionPlan("123", [{"id": "m1"}], None, None, "m1")
    with use_locale("en"):
        assert queue.enqueue(plan, billing)

    stored_job = json.loads(redis_client.hgetall("memory:compaction:jobs")["123"])
    stored_job.pop("locale")
    redis_client.hset("memory:compaction:jobs", "123", json.dumps(stored_job))

    assert queue.run_pending_once() == 1
    save_result.assert_called_once_with(redis_client, "123", "es", "m1")


def test_compaction_skips_reservation_when_a_chat_already_has_a_job():
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    redis_client.hset("memory:compaction:jobs", "123", "{}")
    billing = MagicMock()
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=MagicMock(),
    )

    result = queue.enqueue(
        CompactionPlan("123", [], None, None, "m1"),
        billing,
    )

    assert result is False
    billing.reserve_background_ai_credits.assert_not_called()


def test_enqueue_race_uses_distinct_settlement_ids_and_refunds_loser():
    from api.memory.compaction import CompactionPlan

    redis_client = _FakeRedis()
    redis_client.hexists = MagicMock(return_value=False)
    settle_reservation = MagicMock(return_value={"applied": True})

    def build_billing(message_id):
        billing = MagicMock(user_id=42, message={"message_id": message_id})

        def reserve(usage_tag, _amount, *, metadata):
            return (
                {
                    "reserved_credit_units": 3,
                    "chat_scope_id": 100,
                    "source": "chat",
                    "usage_tag": usage_tag,
                    "metadata": metadata,
                },
                None,
            )

        billing.reserve_background_ai_credits.side_effect = reserve
        return billing

    winner_billing = build_billing(99)
    loser_billing = build_billing(100)
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=settle_reservation,
    )

    plan = CompactionPlan("123", [{"id": "m1"}], None, None, "m1")
    assert queue.enqueue(plan, winner_billing) is True
    stored_job = json.loads(redis_client.hgetall("memory:compaction:jobs")["123"])

    assert queue.enqueue(plan, loser_billing) is False

    settle_reservation.assert_called_once()
    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 0
    loser_usage_tag = settle_reservation.call_args.kwargs["usage_tag"]
    assert loser_usage_tag != stored_job["reservation"]["usage_tag"]
    assert loser_usage_tag.startswith("memory_compaction:123:m1:")
    winner_billing.refund_reserved_ai_credits.assert_not_called()
    loser_billing.refund_reserved_ai_credits.assert_not_called()


def test_incompatible_job_refunds_reservation_before_deletion():
    redis_client = _FakeRedis()
    redis_client.hset(
        "memory:compaction:jobs",
        "123",
        json.dumps(
            {
                "chat_id": "123",
                "removed_schema_field": True,
                "user_id": 42,
                "reservation": {
                    "reserved_credit_units": 3,
                    "chat_scope_id": None,
                    "source": "user",
                    "usage_tag": "memory_compaction:123:m1",
                },
            }
        ),
    )
    settle_reservation = MagicMock(return_value={"applied": True})
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=settle_reservation,
    )

    queue.run_pending_once()

    settle_reservation.assert_called_once()
    assert settle_reservation.call_args.kwargs["reserved_credit_units"] == 30
    assert settle_reservation.call_args.kwargs["actual_credit_units"] == 0
    assert redis_client.hgetall("memory:compaction:jobs") == {}


def test_undecodable_job_is_quarantined_for_manual_recovery():
    redis_client = _FakeRedis()
    redis_client.hset("memory:compaction:jobs", "123", "not-json")
    queue = _build_queue(
        redis_client,
        compact=MagicMock(),
        save_result=MagicMock(),
        settle_reservation=MagicMock(),
    )

    queue.run_pending_once()

    assert redis_client.hgetall("memory:compaction:jobs") == {}
    dead_jobs = redis_client.hgetall("memory:compaction:dead_jobs")
    assert len(dead_jobs) == 1
    dead_job = json.loads(next(iter(dead_jobs.values())))
    assert dead_job["chat_id"] == "123"
    assert dead_job["payload"] == "not-json"
    assert dead_job["reason"] == "undecodable"
