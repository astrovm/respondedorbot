from tests.support import *


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
    billing_helper = MagicMock()
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
    billing_helper = MagicMock()
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


def test_run_conversation_rechecks_full_context_before_model_call():
    from api.ai.service import AIConversationRequest, build_ai_service
    from api.bot.message_handler import PreparedMessage

    estimator = MagicMock(side_effect=[(2, {}), (5, {})])
    handle_ai_response = MagicMock()
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(
            return_value=[{"role": "user", "text": "historial largo"}]
        ),
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
    billing_helper = MagicMock()
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
        "ai_response_base",
        5,
    )
    billing_helper.refund_reserved_ai_credits.assert_called_once_with(
        initial_reservation, reason="ai_response_reserve_adjustment"
    )
    handle_ai_response.assert_not_called()


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
        prepare_chat_memory=MagicMock(
            return_value=([], "summary abc", [], 0, "compaction-plan")
        ),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="no"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
        schedule_compaction=(schedule_compaction := MagicMock()),
    )

    billing_helper = MagicMock()
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

    billing_helper = MagicMock()
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
