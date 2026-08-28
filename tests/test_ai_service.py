from tests.support import *


class _Authorizer:
    def __init__(self, reservations, **_kwargs):
        self.reservations = [item for item in reservations if item]

    def __call__(self, *_args, **_kwargs):
        return None

    def record_provider_segment(self, _segment):
        return None

    def close(self):
        return None


def _billing_mock():
    billing = MagicMock()
    billing.create_authorizer.side_effect = _Authorizer
    return billing


def test_run_conversation_rejects_before_history_or_prompt_preparation():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    get_chat_history = MagicMock()
    prepare_chat_memory = MagicMock()
    build_ai_messages = MagicMock()
    check_provider_available = MagicMock()
    handle_ai_response = MagicMock()
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=get_chat_history,
        prepare_chat_memory=prepare_chat_memory,
        build_ai_messages=build_ai_messages,
        check_provider_available=check_provider_available,
        has_openrouter_fallback=MagicMock(),
        handle_rate_limit=MagicMock(),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )
    billing_helper = _billing_mock()
    billing_helper.reserve_ai_credits.return_value = (None, "sin créditos")

    response = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="900",
            message={"chat": {"id": 900, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage("hola", None, None),
            billing_helper=billing_helper,
            prompt_text="hola",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert response == ("sin créditos", False)
    get_chat_history.assert_not_called()
    prepare_chat_memory.assert_not_called()
    build_ai_messages.assert_not_called()
    check_provider_available.assert_not_called()
    handle_ai_response.assert_not_called()


def test_run_conversation_settles_transcription_when_base_reserve_fails():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    segment = {
        "kind": "transcribe",
        "model": "whisper-large-v3",
        "audio_seconds": 10,
        "source": "groq",
    }
    media_reservation = {
        "reserved_credit_units": 8,
        "source": "user",
        "usage_tag": "auto_audio_media",
    }
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(),
        has_openrouter_fallback=MagicMock(),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )
    billing_helper = _billing_mock()
    billing_helper.reserve_ai_credits.return_value = (None, "sin créditos")

    response = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="900",
            message={"chat": {"id": 900, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage(
                "transcripción",
                None,
                "audio-1",
                media_charge_meta=media_reservation,
                media_billing_segments=(segment,),
            ),
            billing_helper=billing_helper,
            prompt_text="transcripción",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert response == ("sin créditos", False)
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once_with(
        [media_reservation],
        [segment],
        reason="ai_response_base_reserve_failed",
    )
    billing_helper.refund_reserved_ai_credits.assert_not_called()


def test_run_summary_rejects_before_history_or_provider_checks():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    stream_summary_command = MagicMock()
    check_provider_available = MagicMock()
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=check_provider_available,
        has_openrouter_fallback=MagicMock(),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=stream_summary_command,
    )
    billing_helper = _billing_mock()
    billing_helper.reserve_ai_credits.return_value = (None, "sin créditos")

    response = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="resumí",
            redis_client=MagicMock(),
        ),
        stream_consumer=MagicMock(),
    )

    assert response == ("sin créditos", None, True)
    check_provider_available.assert_not_called()
    stream_summary_command.assert_not_called()


def test_run_summary_settles_from_streamed_provider_usage():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    segment = {
        "kind": "summary",
        "model": "deepseek/deepseek-v4-flash-0731",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001},
        "source": "openrouter",
        "metadata": {"provider": "openrouter"},
    }

    def stream_summary(_chat_id, _redis, _prompt, *, response_meta):
        def tokens():
            response_meta.setdefault("billing_segments", []).append(segment)
            yield "openrouter", "summary"

        return tokens(), None

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=stream_summary,
    )
    billing_helper = _billing_mock()
    reservation = {"reserved_credit_units": 3, "usage_tag": "ai_response_base"}
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="summarize",
            redis_client=MagicMock(),
        ),
        stream_consumer=lambda iterator: "".join(token for _provider, token in iterator),
    )

    assert result == ("summary", None, False)
    billing_helper.create_authorizer.assert_called_once_with(
        [reservation],
        model_reserve_credit_units=3,
    )
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once_with(
        [reservation],
        [segment],
        reason="summary_command_stream_success",
    )


def test_run_summary_refunds_first_round_authorization_denial():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    def denied_summary(_chat_id, _redis, _prompt, *, response_meta):
        response_meta["authorization_denied"] = True
        response_meta["ai_fallback"] = True
        return iter([("none", "insufficient credits")]), None

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=denied_summary,
    )
    billing_helper = _billing_mock()
    reservation = {"reserved_credit_units": 3, "usage_tag": "ai_response_base"}
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="summarize",
            redis_client=MagicMock(),
        ),
        stream_consumer=lambda iterator: "".join(token for _provider, token in iterator),
    )

    assert result == ("insufficient credits", None, True)
    billing_helper.refund_reserved_ai_credits.assert_called_once_with(
        reservation,
        reason="summary_stream_fallback",
    )
    billing_helper.settle_reserved_ai_credits_batch.assert_not_called()


def test_run_summary_settles_usage_before_authorization_denial():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    segment = {
        "kind": "summary",
        "model": "deepseek/deepseek-v4-flash-0731",
        "usage": {"cost": 0.001},
        "source": "openrouter",
        "metadata": {"provider": "openrouter"},
    }

    def denied_summary(_chat_id, _redis, _prompt, *, response_meta):
        response_meta["billing_segments"] = [segment]
        response_meta["authorization_denied"] = True
        response_meta["ai_fallback"] = True
        return iter([("none", "insufficient credits")]), None

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=denied_summary,
    )
    billing_helper = _billing_mock()
    reservation = {"reserved_credit_units": 3, "usage_tag": "ai_response_base"}
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="summarize",
            redis_client=MagicMock(),
        ),
        stream_consumer=lambda iterator: "".join(token for _provider, token in iterator),
    )

    assert result == ("insufficient credits", None, True)
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once_with(
        [reservation],
        [segment],
        reason="summary_stream_provider_usage_before_fallback",
    )
    billing_helper.refund_reserved_ai_credits.assert_not_called()


def test_run_summary_settles_usage_when_delivery_fails_after_generation():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    segment = {
        "kind": "summary",
        "model": "deepseek/deepseek-v4-flash-0731",
        "usage": {"cost": 0.001},
        "source": "openrouter",
        "metadata": {"provider": "openrouter"},
    }

    def stream_summary(_chat_id, _redis, _prompt, *, response_meta):
        def tokens():
            response_meta.setdefault("billing_segments", []).append(segment)
            yield "openrouter", "summary"

        return tokens(), None

    def consume_then_fail(iterator):
        list(iterator)
        raise RuntimeError("telegram edit failed")

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=stream_summary,
    )
    billing_helper = _billing_mock()
    reservation = {"reserved_credit_units": 3, "usage_tag": "ai_response_base"}
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="summarize",
            redis_client=MagicMock(),
        ),
        stream_consumer=consume_then_fail,
    )

    assert result[2] is True
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once_with(
        [reservation],
        [segment],
        reason="summary_stream_provider_usage_before_delivery_failure",
    )
    billing_helper.refund_reserved_ai_credits.assert_not_called()


def test_run_summary_refunds_when_stream_provider_is_unavailable():
    from api.ai.service import SummaryCommandRequest, build_ai_service

    def unavailable_summary(_chat_id, _redis, _prompt, *, response_meta):
        response_meta["provider_unavailable"] = True
        return iter([("none", "summary error")]), None

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(),
        prepare_chat_memory=MagicMock(),
        build_ai_messages=MagicMock(),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(3, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        stream_summary_command=unavailable_summary,
    )
    billing_helper = _billing_mock()
    reservation = {"reserved_credit_units": 3, "usage_tag": "ai_response_base"}
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_summary_command_stream(
        SummaryCommandRequest(
            chat_id="901",
            message={"chat": {"id": 901, "type": "private"}},
            billing_helper=billing_helper,
            prompt_text="summarize",
            redis_client=MagicMock(),
        ),
        stream_consumer=lambda iterator: "".join(token for _provider, token in iterator),
    )

    assert result == ("summary error", None, False)
    billing_helper.refund_reserved_ai_credits.assert_called_once_with(
        reservation,
        reason="summary_provider_unavailable",
    )
    billing_helper.settle_reserved_ai_credits_batch.assert_not_called()


def test_run_conversation_settles_provider_rounds_before_local_fallback():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    segment = {
        "kind": "chat",
        "model": "deepseek/deepseek-v4-flash-0731",
        "usage": {"cost": "0.0019"},
        "source": "openrouter",
        "metadata": {"provider": "openrouter"},
    }

    def fallback_after_provider_usage(*_args, **kwargs):
        response_meta = kwargs["response_meta"]
        response_meta["billing_segments"] = [segment]
        response_meta["ai_fallback"] = True
        return "local fallback"

    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hello"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(),
        handle_ai_response=fallback_after_provider_usage,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(16, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )
    billing_helper = _billing_mock()
    reservation = {
        "reserved_credit_units": 16,
        "source": "user",
        "usage_tag": "ai_response_base",
    }
    billing_helper.reserve_ai_credits.return_value = (reservation, None)

    result = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="902",
            message={"chat": {"id": 902, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage("hello", None, None),
            billing_helper=billing_helper,
            prompt_text="hello",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert result == ("local fallback", True)
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once_with(
        [reservation],
        [segment],
        reason="ai_response_provider_usage_before_fallback",
    )
    billing_helper.refund_reserved_ai_credits.assert_not_called()


def test_run_conversation_rechecks_full_context_before_model_call():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    estimator = MagicMock(side_effect=[(2, {}), (5, {})])
    handle_ai_response = MagicMock()
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[{"role": "user", "text": "historial largo"}]),
        prepare_chat_memory=MagicMock(
            return_value=([{"role": "user", "text": "historial largo"}], None, [], 0)
        ),
        build_ai_messages=MagicMock(
            return_value=[
                {"role": "user", "content": "historial largo"},
                {"role": "user", "content": "hola"},
            ]
        ),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=estimator,
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )
    billing_helper = _billing_mock()
    initial_reservation = {
        "reserved_credit_units": 2,
        "source": "user",
        "usage_tag": "ai_response_base",
    }
    billing_helper.reserve_ai_credits.side_effect = [
        (initial_reservation, None),
        (None, "sin créditos para el contexto"),
    ]

    response = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="902",
            message={"chat": {"id": 902, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage("hola", None, None),
            billing_helper=billing_helper,
            prompt_text="hola",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert response == ("sin créditos para el contexto", False)
    assert billing_helper.reserve_ai_credits.call_count == 2
    assert billing_helper.reserve_ai_credits.call_args_list[1].args[:2] == (
        "ai_response_context_extension",
        3,
    )
    billing_helper.refund_reserved_ai_credits.assert_called_once_with(
        initial_reservation, reason="ai_response_reserve_adjustment_failed"
    )
    handle_ai_response.assert_not_called()


def test_image_reserve_failure_refunds_base_and_context_reservations():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    estimator = MagicMock(side_effect=[(2, {}), (5, {})])
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(),
        handle_ai_response=MagicMock(),
        estimate_ai_base_reserve_credits=estimator,
        estimate_image_context_reserve_credits=MagicMock(return_value=4),
    )
    billing_helper = _billing_mock()
    base_reservation = {"usage_tag": "ai_response_base"}
    context_reservation = {"usage_tag": "ai_response_context_extension"}
    billing_helper.reserve_ai_credits.side_effect = [
        (base_reservation, None),
        (context_reservation, None),
        (None, "sin créditos para la imagen"),
    ]

    result = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="902",
            message={"chat": {"id": 902, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage(
                "hola",
                "photo-1",
                None,
                resized_image_data=b"image",
            ),
            billing_helper=billing_helper,
            prompt_text="hola",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert result == ("sin créditos para la imagen", False)
    refund_calls = billing_helper.refund_reserved_ai_credits.call_args_list
    assert [item.args[0] for item in refund_calls] == [
        base_reservation,
        context_reservation,
    ]
    assert {item.kwargs["reason"] for item in refund_calls} == {
        "image_context_reserve_failed"
    }


def test_local_fallback_refunds_every_incremental_reservation():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    def local_fallback(*_args, **kwargs):
        kwargs["response_meta"]["ai_fallback"] = True
        return "fallback"

    estimator = MagicMock(side_effect=[(2, {}), (5, {})])
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(),
        handle_ai_response=local_fallback,
        estimate_ai_base_reserve_credits=estimator,
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )
    billing_helper = _billing_mock()
    base_reservation = {"usage_tag": "ai_response_base"}
    context_reservation = {"usage_tag": "ai_response_context_extension"}
    billing_helper.reserve_ai_credits.side_effect = [
        (base_reservation, None),
        (context_reservation, None),
    ]

    result = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="902",
            message={"chat": {"id": 902, "type": "private"}},
            user_id=10,
            prepared_message=PreparedMessage("hola", None, None),
            billing_helper=billing_helper,
            prompt_text="hola",
            reply_context_text=None,
            user_identity="10",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert result == ("fallback", True)
    refund_calls = billing_helper.refund_reserved_ai_credits.call_args_list
    assert [item.args[0] for item in refund_calls] == [
        base_reservation,
        context_reservation,
    ]
    assert {item.kwargs["reason"] for item in refund_calls} == {
        "ai_response_fallback"
    }


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_vision():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    handle_ai_response = MagicMock(return_value="respuesta ok")
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=False),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(return_value="no boludo"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = make_ai_message_billing(
        command="/ask",
        chat_id="557",
        user_id=101,
        numeric_chat_id=557,
    )

    response_msg, handled = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="557",
            message={"chat": {"id": 557, "type": "private"}},
            user_id=None,
            prepared_message=PreparedMessage(
                message_text="/ask describe",
                photo_file_id="img_1",
                audio_file_id=None,
                resized_image_data=b"resized",
            ),
            billing_helper=billing_helper,
            prompt_text="Describe",
            reply_context_text=None,
            user_identity="101",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert handled is True
    assert response_msg == "respuesta ok"
    handle_ai_response.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_transcribe():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    handle_ai_response = MagicMock(return_value="🖼️ en la imagen veo: todo piola")
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=False),
        has_openrouter_fallback=MagicMock(return_value=True),
        handle_rate_limit=MagicMock(return_value="no boludo"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = make_ai_message_billing(
        command="/transcribe",
        chat_id="558",
        user_id=102,
        numeric_chat_id=558,
    )

    response_msg, handled = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="558",
            message={"chat": {"id": 558, "type": "private"}},
            user_id=None,
            prepared_message=PreparedMessage(
                message_text="/transcribe",
                photo_file_id="img_reply",
                audio_file_id=None,
                resized_image_data=b"resized",
            ),
            billing_helper=billing_helper,
            prompt_text="Describe what you see in this image in detail.",
            reply_context_text=None,
            user_identity="102",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert handled is True
    assert response_msg == "🖼️ en la imagen veo: todo piola"
    handle_ai_response.assert_called_once()


def test_run_conversation_passes_summary_and_retrieval_into_prompt_builder():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    build_ai_messages = MagicMock(return_value=[{"role": "user", "content": "hola"}])
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], "summary abc", [{"text": "old hit"}], 0)),
        build_ai_messages=build_ai_messages,
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="no"),
        handle_ai_response=MagicMock(return_value="ok"),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = make_ai_message_billing(
        command="/ask",
        chat_id="559",
        user_id=103,
        numeric_chat_id=559,
    )

    ai_service.run_conversation(
        AIConversationRequest(
            chat_id="559",
            message={"chat": {"id": 559, "type": "private"}},
            user_id=None,
            prepared_message=PreparedMessage(
                message_text="hola",
                photo_file_id=None,
                audio_file_id=None,
                resized_image_data=None,
            ),
            billing_helper=billing_helper,
            prompt_text="que paso hoy",
            reply_context_text=None,
            user_identity="103",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert build_ai_messages.call_args.kwargs["summary_text"] == "summary abc"
    assert build_ai_messages.call_args.kwargs["retrieved_messages"] == [{"text": "old hit"}]


def test_run_conversation_schedules_compaction_after_answer_settlement():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    handle_ai_response = MagicMock(return_value="ok")
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], "summary abc", [], 0, "compaction-plan")),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="no"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        schedule_compaction=(schedule_compaction := MagicMock()),
    )

    billing_helper = _billing_mock()
    billing_helper.reserve_ai_credits.return_value = ({"reserved_credit_units": 1}, None)

    ai_service.run_conversation(
        AIConversationRequest(
            chat_id="560",
            message={"chat": {"id": 560, "type": "private"}},
            user_id=None,
            prepared_message=PreparedMessage(
                message_text="hola",
                photo_file_id=None,
                audio_file_id=None,
                resized_image_data=None,
            ),
            billing_helper=billing_helper,
            prompt_text="que paso hoy",
            reply_context_text=None,
            user_identity="104",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    settle_args = billing_helper.settle_reserved_ai_credits_batch.call_args.args
    billing_segments = settle_args[1]
    assert not any(segment.get("kind") == "summary" for segment in billing_segments)
    schedule_compaction.assert_called_once_with("compaction-plan", billing_helper)


def test_run_conversation_uses_fallback_metadata_not_response_text():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    fallback_text = "me quedé reculando y no te pude responder, probá de nuevo"
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], None, [], 0)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="no"),
        handle_ai_response=MagicMock(return_value=fallback_text),
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
    )

    billing_helper = _billing_mock()
    billing_helper.reserve_ai_credits.return_value = ({"reserved_credit_units": 1}, None)

    response_msg, handled = ai_service.run_conversation(
        AIConversationRequest(
            chat_id="561",
            message={"chat": {"id": 561, "type": "private"}},
            user_id=None,
            prepared_message=PreparedMessage(
                message_text="hola",
                photo_file_id=None,
                audio_file_id=None,
                resized_image_data=None,
            ),
            billing_helper=billing_helper,
            prompt_text="hola",
            reply_context_text=None,
            user_identity="105",
            handler_func=lambda: None,
            redis_client=MagicMock(),
        )
    )

    assert handled is True
    assert response_msg == fallback_text
    billing_helper.refund_reserved_ai_credits.assert_not_called()
    billing_helper.settle_reserved_ai_credits_batch.assert_called_once()
