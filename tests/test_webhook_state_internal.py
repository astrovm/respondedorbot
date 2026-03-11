from unittest.mock import patch

from api.webhook_state import (
    _acquire_webhook_processing_lock,
    _clear_persisted_webhook_reservation,
    _extract_webhook_operation_key,
    _load_persisted_webhook_reservation,
    _mark_webhook_completed,
    _persist_webhook_reservation,
    _restore_webhook_request_context,
    _set_webhook_request_context,
)


class FakeWebhookRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    def setex(self, key, ttl, value):
        self.data[key] = value
        return True

    def delete(self, key):
        self.data.pop(key, None)
        return 1

    def expire(self, key, ttl):
        return key in self.data


def test_extract_webhook_operation_key_for_supported_update_types():
    assert _extract_webhook_operation_key({"callback_query": {"id": "cbq_1"}}) == "callback:cbq_1"
    assert _extract_webhook_operation_key({"pre_checkout_query": {"id": "pcq_1"}}) == "pre_checkout:pcq_1"
    assert _extract_webhook_operation_key(
        {"message": {"message_id": 7, "chat": {"id": 12}}}
    ) == "message:12:7"


def test_webhook_processing_lock_returns_completed_when_completed_marker_exists():
    redis_client = FakeWebhookRedis()
    _mark_webhook_completed(redis_client, "message:1:2")

    assert (
        _acquire_webhook_processing_lock(
            redis_client,
            "message:1:2",
            "owner-1",
        )
        == "completed"
    )


def test_persisted_webhook_reservation_round_trip_uses_request_context():
    redis_client = FakeWebhookRedis()
    tokens = _set_webhook_request_context(
        request_started_at=1.0,
        operation_key="message:1:3",
        redis_client=redis_client,
        force_paid_retry=False,
    )

    try:
        reservation = {
            "reserved_credit_units": 10,
            "chat_scope_id": 1,
            "source": "user",
            "usage_tag": "ai_response_base",
            "metadata": {"cached": True},
        }
        with patch(
            "api.webhook_state._get_webhook_force_paid_retry_ttl_seconds",
            return_value=300,
        ):
            _persist_webhook_reservation("ai_response_base", reservation)

        assert _load_persisted_webhook_reservation("ai_response_base") == reservation

        _clear_persisted_webhook_reservation("ai_response_base")
        assert _load_persisted_webhook_reservation("ai_response_base") is None
    finally:
        _restore_webhook_request_context(tokens)
