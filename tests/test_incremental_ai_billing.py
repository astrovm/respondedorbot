from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.ai.pricing import AIUsageResult
from api.billing.ai import AICreditAuthorizer
from api.billing.authorization import (
    AI_COST_AUTHORIZER_KEY,
    AI_SEGMENT_RECORDER_KEY,
    AIAuthorizationDenied,
)
from api.billing.reconciliation import (
    AIBillingReconciler,
    mark_ai_operation_active,
    mark_ai_operation_inactive,
)
from api.services import credits_db
from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
from api.providers.openrouter import OpenRouterProvider
from api.tools.runtime import ToolRuntime
from tests.support import make_ai_message_billing
from tests.test_credits_db import _FakeConnection, _FakeCursor


def test_concurrent_duplicate_extension_is_reserved_once():
    billing = MagicMock()
    billing.reserve_ai_credits.return_value = (
        {"source": "user", "reserved_credit_units": 20},
        None,
    )
    authorizer = AICreditAuthorizer(
        billing=billing,
        operation_id="operation-1",
        reservations=[{"source": "user", "reserved_credit_units": 10}],
    )

    authorizer("model", 10, {"round": 1, "attempt": 1})
    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = list(
            pool.map(
                lambda _index: authorizer(
                    "web_search",
                    20,
                    {"tool_call_id": "search-1"},
                ),
                range(8),
            )
        )

    assert errors == [None] * 8
    billing.reserve_ai_credits.assert_called_once()
    assert len(authorizer.reservations) == 2
    assert billing.reserve_ai_credits.call_args.kwargs["metadata"]["operation_id"] == (
        "operation-1"
    )


def test_separate_provider_invocations_extend_the_same_operation():
    billing = MagicMock()
    billing.reserve_ai_credits.return_value = (
        {"source": "user", "reserved_credit_units": 10},
        None,
    )
    authorizer = AICreditAuthorizer(
        billing=billing,
        operation_id="operation-1",
        reservations=[{"source": "user", "reserved_credit_units": 10}],
    )

    authorizer(
        "model",
        10,
        {"invocation_id": "first", "round": 1, "attempt": 1},
    )
    authorizer(
        "model",
        10,
        {"invocation_id": "second", "round": 1, "attempt": 1},
    )
    authorizer(
        "model",
        10,
        {"invocation_id": "second", "round": 1, "attempt": 1},
    )

    billing.reserve_ai_credits.assert_called_once()
    assert len(authorizer.reservations) == 2


def test_first_model_request_reserves_only_the_admission_gap():
    billing = MagicMock()
    billing.reserve_ai_credits.return_value = (
        {"source": "user", "reserved_credit_units": 7},
        None,
    )
    authorizer = AICreditAuthorizer(
        billing=billing,
        operation_id="operation-1",
        reservations=[{"source": "user", "reserved_credit_units": 3}],
    )

    error = authorizer(
        "model",
        10,
        {"invocation_id": "first", "round": 1, "attempt": 1},
    )

    assert error is None
    billing.reserve_ai_credits.assert_called_once()
    assert billing.reserve_ai_credits.call_args.args[1] == 7


def test_billing_session_keeps_the_initial_payer_for_extensions():
    db = MagicMock()
    db.is_configured.return_value = True
    db.charge_ai_credits.side_effect = lambda **kwargs: {
        "ok": True,
        "applied": True,
        "source": kwargs.get("source") or "chat",
    }
    billing = make_ai_message_billing(
        chat_type="group",
        numeric_chat_id=557,
        credits_db_service=db,
        message={"message_id": 44, "from": {"first_name": "Ana"}},
    )
    operation_id = billing.operation_id("ai_response")
    base, error = billing.reserve_ai_credits(
        "ai_response_base",
        10,
        metadata={"operation_id": operation_id},
    )
    assert error is None
    session = billing.create_authorizer([base])

    session("model", 10, {"round": 1, "attempt": 1})
    session("model", 20, {"round": 2, "attempt": 1})

    assert db.charge_ai_credits.call_count == 2
    assert db.charge_ai_credits.call_args_list[1].kwargs["source"] == "chat"
    assert {item["source"] for item in session.reservations} == {"chat"}


def _web_search_call():
    return SimpleNamespace(
        id="search-1",
        function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
    )


def test_web_search_is_not_executed_when_authorization_is_denied():
    execute_tool = MagicMock()
    runtime = ToolRuntime(
        execute_tool_fn=execute_tool,
        tool_registry={"web_search": object()},
    )
    context = {AI_COST_AUTHORIZER_KEY: lambda *_args: "insufficient credits"}

    with pytest.raises(AIAuthorizationDenied, match="insufficient credits"):
        runtime.apply_tool_calls(
            SimpleNamespace(content=""),
            [_web_search_call()],
            [],
            context,
        )

    execute_tool.assert_not_called()


def test_completed_firecrawl_usage_is_persisted_before_another_round():
    recorder = MagicMock()
    execute_tool = MagicMock(
        return_value=SimpleNamespace(
            output='{"results": []}',
            metadata={"credits_used": 2},
        )
    )
    runtime = ToolRuntime(
        execute_tool_fn=execute_tool,
        tool_registry={"web_search": object()},
    )
    context = {
        AI_COST_AUTHORIZER_KEY: lambda *_args: None,
        AI_SEGMENT_RECORDER_KEY: recorder,
    }

    runtime.apply_tool_calls(
        SimpleNamespace(content=""),
        [_web_search_call()],
        [],
        context,
    )

    segment = recorder.call_args.args[0]
    assert segment["kind"] == "web_search"
    assert segment["metadata"]["firecrawl_credits_used"] == 2


def test_second_model_round_is_authorized_before_provider_execution():
    events = []
    tool_call = SimpleNamespace(
        id="calc-1",
        function=SimpleNamespace(name="calc", arguments="{}"),
    )
    responses = [
        SimpleNamespace(
            id="generation-1",
            model="deepseek/deepseek-v4-flash-0731",
            usage={"cost": 0.0001},
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(content="", tool_calls=[tool_call]),
                )
            ],
        ),
        SimpleNamespace(
            id="generation-2",
            model="deepseek/deepseek-v4-flash-0731",
            usage={"cost": 0.0001},
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="done", tool_calls=[]),
                )
            ],
        ),
    ]

    def create(**_kwargs):
        events.append("provider")
        return responses.pop(0)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {},
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage=dict(kwargs["response"].usage),
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda response: dict(response.usage),
            primary_model="deepseek/deepseek-v4-flash-0731",
        ),
        ToolRuntime(
            execute_tool_fn=lambda *_args: (
                events.append("tool")
                or SimpleNamespace(output="1", metadata={})
            ),
            tool_registry={"calc": object()},
        ),
    )

    def authorize(_kind, _units, metadata):
        events.append(f"authorize-{metadata['round']}")
        return None

    context = {
        AI_COST_AUTHORIZER_KEY: authorize,
        AI_SEGMENT_RECORDER_KEY: lambda _segment: events.append("persist"),
    }
    result = runtime.complete(
        {"role": "system", "content": "system"},
        [{"role": "user", "content": "calculate"}],
        enable_web_search=False,
        extra_tools=[{"name": "calc"}],
        tool_context=context,
    )

    assert result is not None
    assert events == [
        "authorize-1",
        "provider",
        "persist",
        "tool",
        "authorize-2",
        "provider",
    ]


def test_denied_second_model_round_does_not_reach_provider():
    calls = 0
    tool_call = SimpleNamespace(
        id="calc-1",
        function=SimpleNamespace(name="calc", arguments="{}"),
    )
    response = SimpleNamespace(
        id="generation-1",
        model="deepseek/deepseek-v4-flash-0731",
        usage={"cost": 0.0001},
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content="", tool_calls=[tool_call]),
            )
        ],
    )

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        return response

    runtime = ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            ),
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {},
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage=dict(kwargs["response"].usage),
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda item: dict(item.usage),
            primary_model="deepseek/deepseek-v4-flash-0731",
        ),
        ToolRuntime(
            execute_tool_fn=lambda *_args: SimpleNamespace(output="1", metadata={}),
            tool_registry={"calc": object()},
        ),
    )

    def authorize(_kind, _units, metadata):
        return "insufficient credits" if metadata["round"] == 2 else None

    with pytest.raises(AIAuthorizationDenied, match="insufficient credits"):
        runtime.complete(
            {"role": "system", "content": "system"},
            [{"role": "user", "content": "calculate"}],
            enable_web_search=False,
            extra_tools=[{"name": "calc"}],
            tool_context={AI_COST_AUTHORIZER_KEY: authorize},
        )

    assert calls == 1


def test_streaming_web_denial_reports_completed_model_usage():
    tool_delta = SimpleNamespace(
        index=0,
        id="search-1",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query":"test"}'),
    )
    chunks = iter(
        [
            SimpleNamespace(
                id="generation-1",
                model="deepseek/deepseek-v4-flash-0731",
                usage={"cost": 0.0001},
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[tool_delta],
                            annotations=[],
                        ),
                    )
                ],
            ),
            SimpleNamespace(
                id="generation-1",
                model="deepseek/deepseek-v4-flash-0731",
                usage={"cost": 0.0001},
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[],
                            annotations=[],
                        ),
                    )
                ],
            ),
        ]
    )
    create = MagicMock(return_value=chunks)
    execute_tool = MagicMock()
    recorded = []
    recorder = MagicMock()
    provider = OpenRouterProvider(
        get_client=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
        admin_report=MagicMock(),
        increment_request_count=MagicMock(),
        build_web_search_tool=lambda: {},
        build_usage_result=lambda **kwargs: AIUsageResult(
            kind=kwargs["kind"],
            text=kwargs["text"],
            model=kwargs["model"],
            usage=dict(kwargs["response"].usage),
            source="openrouter",
            metadata=kwargs.get("metadata") or {},
        ),
        extract_usage_map=lambda response: dict(response.usage),
        primary_model="deepseek/deepseek-v4-flash-0731",
        tool_runtime=ToolRuntime(
            execute_tool_fn=execute_tool,
            tool_registry={"web_search": object()},
        ),
    )

    def authorize(kind, _units, _metadata):
        return "insufficient credits" if kind == "web_search" else None

    with pytest.raises(AIAuthorizationDenied, match="insufficient credits"):
        list(
            provider.stream(
                {"role": "system", "content": "system"},
                [{"role": "user", "content": "search"}],
                extra_tools=[{"name": "web_search"}],
                tool_context={
                    AI_COST_AUTHORIZER_KEY: authorize,
                    AI_SEGMENT_RECORDER_KEY: recorder,
                },
                on_usage_result=recorded.append,
            )
        )

    assert len(recorded) == 1
    assert recorded[0].metadata["provider_generation_id"] == "generation-1"
    execute_tool.assert_not_called()
    create.assert_called_once()


def _operation(segment, *, created_at=None, last_activity_at=None, authorized=50):
    created_at = created_at or datetime.now(UTC) - timedelta(minutes=10)
    return {
        "operation_id": "operation-1",
        "user_id": 10,
        "chat_id": None,
        "authorized_credit_units": authorized,
        "created_at": created_at,
        "last_activity_at": last_activity_at or created_at,
        "reserve_metadata": {"usage_tag": "ai_response_base"},
        "segments": [{"segment_id": "openrouter:generation-1", "segment": segment}],
    }


def _chat_segment(*, interrupted=False, cost=0):
    return AIUsageResult(
        kind="chat",
        text="answer",
        model="deepseek/deepseek-v4-flash-0731",
        usage={"cost": cost},
        source="openrouter",
        metadata={
            "provider": "openrouter",
            "provider_generation_id": "generation-1",
            "stream_interrupted": interrupted,
        },
    ).billing_segment()


def _reconciler(credits, **kwargs):
    credits.is_configured.return_value = True
    return AIBillingReconciler(
        credits=credits,
        admin_report=MagicMock(),
        interval_seconds=5,
        retry_window_seconds=60,
        stale_seconds=30,
        **kwargs,
    )


def test_reconciler_settles_durable_usage_left_by_a_crash():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(_chat_segment(cost=0.0001))
    ]

    result = _reconciler(credits).run_once()

    assert result == {"settled": 1, "pending": 0, "unresolved": 0}
    credits.settle_ai_operation_once.assert_called_once()
    assert credits.settle_ai_operation_once.call_args.kwargs["actual_credit_units"] > 0


def test_reconciler_refunds_a_stale_reservation_without_provider_usage():
    credits = MagicMock()
    operation = _operation(_chat_segment(cost=0.0001))
    operation["segments"] = []
    credits.list_unsettled_ai_operations.return_value = [operation]

    result = _reconciler(credits).run_once()

    assert result == {"settled": 1, "pending": 0, "unresolved": 0}
    settled = credits.settle_ai_operation_once.call_args.kwargs
    assert settled["actual_credit_units"] == 0
    assert settled["metadata"]["reason"] == "unused_stale_reservation"


def test_reconciler_does_not_settle_an_active_provider_operation():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(_chat_segment(cost=0.0001))
    ]
    mark_ai_operation_active("operation-1")

    try:
        result = _reconciler(credits).run_once()
    finally:
        mark_ai_operation_inactive("operation-1")

    assert result == {"settled": 0, "pending": 1, "unresolved": 0}
    credits.settle_ai_operation_once.assert_not_called()


def test_reconciler_recovers_interrupted_openrouter_generation():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(_chat_segment(interrupted=True))
    ]
    get_generation = MagicMock(
        return_value={
            "total_cost": 0,
            "upstream_inference_cost": 0.0002,
            "tokens_prompt": 100,
            "tokens_completion": 20,
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider_name": "DeepSeek",
        }
    )

    result = _reconciler(credits, get_generation=get_generation).run_once()

    assert result == {"settled": 1, "pending": 0, "unresolved": 0}
    credits.update_ai_provider_usage.assert_called_once()
    reconciled = credits.update_ai_provider_usage.call_args.args[2]
    assert reconciled["usage"]["cost"] == 0.0002
    assert reconciled["usage"]["cost_details"] == {
        "upstream_inference_cost": 0.0002
    }
    settled = credits.settle_ai_operation_once.call_args.kwargs
    assert settled["actual_credit_units"] > 0
    assert settled["metadata"]["reconciliation_unresolved"] is False


def test_reconciler_does_not_use_gateway_total_as_upstream_cost():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(
            _chat_segment(interrupted=True),
            created_at=datetime.now(UTC) - timedelta(seconds=45),
        )
    ]
    get_generation = MagicMock(
        return_value={
            "total_cost": 0.0002,
            "tokens_prompt": 100,
            "tokens_completion": 20,
            "model": "deepseek/deepseek-v4-flash-0731",
            "provider_name": "DeepSeek",
        }
    )

    result = _reconciler(credits, get_generation=get_generation).run_once()

    assert result == {"settled": 0, "pending": 1, "unresolved": 0}
    credits.update_ai_provider_usage.assert_not_called()
    credits.settle_ai_operation_once.assert_not_called()


def test_reconciler_keeps_recent_missing_generation_pending():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(
            _chat_segment(interrupted=True),
            created_at=datetime.now(UTC),
            last_activity_at=datetime.now(UTC),
        )
    ]

    result = _reconciler(credits, get_generation=MagicMock(return_value=None)).run_once()

    assert result == {"settled": 0, "pending": 1, "unresolved": 0}
    credits.settle_ai_operation_once.assert_not_called()


def test_reconciler_retains_safety_amount_after_retry_window():
    credits = MagicMock()
    credits.list_unsettled_ai_operations.return_value = [
        _operation(
            _chat_segment(interrupted=True),
            created_at=datetime.now(UTC) - timedelta(minutes=2),
            authorized=50,
        )
    ]
    admin_report = MagicMock()
    reconciler = AIBillingReconciler(
        credits=credits,
        admin_report=admin_report,
        get_generation=MagicMock(return_value=None),
        retry_window_seconds=60,
        safety_credit_units=10,
        stale_seconds=30,
    )
    credits.is_configured.return_value = True

    result = reconciler.run_once()

    assert result == {"settled": 0, "pending": 0, "unresolved": 1}
    assert credits.settle_ai_operation_once.call_args.kwargs["actual_credit_units"] == 10
    admin_report.assert_called_once()


def test_reconciler_never_caps_finalized_usage_at_authorized_amount():
    credits = MagicMock()
    known_segment = _chat_segment(cost=0.005)
    known_segment["metadata"]["provider_generation_id"] = "generation-known"
    pending_segment = _chat_segment(interrupted=True)
    pending_segment["metadata"]["provider_generation_id"] = "generation-pending"
    operation = _operation(
        pending_segment,
        created_at=datetime.now(UTC) - timedelta(minutes=2),
        authorized=50,
    )
    operation["segments"].insert(
        0,
        {"segment_id": "openrouter:generation-known", "segment": known_segment},
    )
    credits.list_unsettled_ai_operations.return_value = [operation]
    credits.is_configured.return_value = True
    reconciler = AIBillingReconciler(
        credits=credits,
        admin_report=MagicMock(),
        get_generation=MagicMock(return_value=None),
        retry_window_seconds=60,
        safety_credit_units=10,
        stale_seconds=30,
    )

    result = reconciler.run_once()

    assert result == {"settled": 0, "pending": 0, "unresolved": 1}
    assert credits.settle_ai_operation_once.call_args.kwargs["actual_credit_units"] == 100


def test_reconciler_retains_authorization_when_pricing_is_incomplete():
    credits = MagicMock()
    segment = {
        "kind": "chat",
        "model": "openai/gpt-oss-120b",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "source": "openrouter",
        "metadata": {
            "provider": "openrouter",
            "upstream_provider": "DeepInfra",
        },
    }
    credits.list_unsettled_ai_operations.return_value = [_operation(segment, authorized=20)]
    admin_report = MagicMock()
    credits.is_configured.return_value = True
    reconciler = AIBillingReconciler(
        credits=credits,
        admin_report=admin_report,
        retry_window_seconds=60,
        stale_seconds=30,
    )

    result = reconciler.run_once()

    assert result == {"settled": 0, "pending": 0, "unresolved": 1}
    settlement = credits.settle_ai_operation_once.call_args.kwargs
    assert settlement["actual_credit_units"] == 20
    assert settlement["metadata"]["pricing_complete"] is False
    assert settlement["metadata"]["reconciliation_unresolved"] is True
    assert settlement["metadata"]["reason"] == "reconciliation_incomplete_pricing"
    admin_report.assert_called_once()


def test_reconciler_continues_after_one_operation_fails():
    credits = MagicMock()
    failed = _operation(_chat_segment(interrupted=True))
    recoverable = _operation(_chat_segment())
    recoverable["operation_id"] = "operation-2"
    recoverable["segments"] = []
    credits.list_unsettled_ai_operations.return_value = [failed, recoverable]
    credits.is_configured.return_value = True
    admin_report = MagicMock()
    reconciler = AIBillingReconciler(
        credits=credits,
        admin_report=admin_report,
        get_generation=MagicMock(side_effect=RuntimeError("provider unavailable")),
        retry_window_seconds=60,
        stale_seconds=30,
    )

    result = reconciler.run_once()

    assert result == {"settled": 1, "pending": 1, "unresolved": 0}
    credits.settle_ai_operation_once.assert_called_once()
    assert credits.settle_ai_operation_once.call_args.kwargs["operation_id"] == (
        "operation-2"
    )
    assert admin_report.call_args.args[2] == {"operation_id": "operation-1"}


def test_atomic_operation_settlement_is_idempotent():
    class SettlementCursor(_FakeCursor):
        def __init__(self):
            super().__init__(hourly_count=0, daily_count=0, insert_granted=False)
            self.balance = 60
            self.settled = False

        def execute(self, query, params=None):
            normalized = " ".join(str(query).split())
            if "event_type = 'ai_settlement_result'" in normalized and "SELECT 1" in normalized:
                self.fetchone_result = (1,) if self.settled else None
                return
            if "COALESCE(SUM(-amount), 0)" in normalized:
                self.fetchone_result = (40, 1, "user")
                return
            if "VALUES ('ai_settlement_result'" in normalized:
                self.settled = True
                self.fetchone_result = None
                self.executed.append((normalized, params))
                return
            super().execute(query, params)

    cursor = SettlementCursor()
    connection = _FakeConnection(cursor)
    with (
        patch("api.services.credits_db.ensure_schema"),
        patch("api.services.credits_db.connect", return_value=connection),
    ):
        first = credits_db.settle_ai_operation_once(
            user_id=10,
            chat_id=None,
            operation_id="operation-1",
            actual_credit_units=10,
            metadata={},
        )
        second = credits_db.settle_ai_operation_once(
            user_id=10,
            chat_id=None,
            operation_id="operation-1",
            actual_credit_units=10,
            metadata={},
        )

    assert first["applied"] is True
    assert first["refunded_credit_units"] == 30
    assert second["applied"] is False
    assert cursor.balance == 90
