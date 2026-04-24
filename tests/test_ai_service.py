from tests.support import *


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_vision():
    from api.ai_service import AIConversationRequest, build_ai_service
    from api.message_handler import PreparedMessage

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
    from api.ai_service import AIConversationRequest, build_ai_service
    from api.message_handler import PreparedMessage

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
    from api.ai_service import AIConversationRequest, build_ai_service
    from api.message_handler import PreparedMessage

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


def test_run_conversation_bills_summary_compaction_as_billing_segment():
    from api.ai_service import AIConversationRequest, build_ai_service
    from api.message_handler import PreparedMessage

    handle_ai_response = MagicMock(return_value="ok")
    ai_service = build_ai_service(
        credits_db_service=MagicMock(is_configured=MagicMock(return_value=True)),
        get_chat_history=MagicMock(return_value=[]),
        prepare_chat_memory=MagicMock(return_value=([], "summary abc", [], 1234)),
        build_ai_messages=MagicMock(return_value=[{"role": "user", "content": "hola"}]),
        check_provider_available=MagicMock(return_value=True),
        has_openrouter_fallback=MagicMock(return_value=False),
        handle_rate_limit=MagicMock(return_value="no"),
        handle_ai_response=handle_ai_response,
        estimate_ai_base_reserve_credits=MagicMock(return_value=(1, {})),
        estimate_image_context_reserve_credits=MagicMock(return_value=1),
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
    assert any(segment.get("kind") == "summary" for segment in billing_segments)



