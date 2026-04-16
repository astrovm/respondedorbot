from tests.support import *  # noqa: F401,F403


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_vision():
    from api.ai_billing import AIMessageBilling
    from api.ai_service import run_ai_flow
    from api.message_handler import PreparedMessage

    deps = MagicMock()
    deps.get_chat_history.return_value = []
    deps.build_ai_messages.return_value = [{"role": "user", "content": "hola"}]
    deps.estimate_ai_base_reserve_credits.return_value = (
        1,
        {},
    )
    deps.check_provider_available.return_value = False
    deps.has_openrouter_fallback.return_value = True
    deps.estimate_image_context_reserve_credits.return_value = 1
    deps.handle_ai_response.return_value = "respuesta ok"

    billing_helper = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=lambda **_: "insufficient",
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/ask",
        chat_id="557",
        chat_type="private",
        user_id=101,
        numeric_chat_id=557,
        message={"from": {"first_name": "Ana"}},
    )

    response_msg, handled = run_ai_flow(
        deps,
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

    assert handled is True
    assert response_msg == "respuesta ok"
    deps.handle_ai_response.assert_called_once()


def test_run_ai_flow_keeps_going_when_openrouter_fallback_is_allowed_for_transcribe():
    from api.ai_billing import AIMessageBilling
    from api.ai_service import run_ai_flow
    from api.message_handler import PreparedMessage

    deps = MagicMock()
    deps.get_chat_history.return_value = []
    deps.build_ai_messages.return_value = [{"role": "user", "content": "hola"}]
    deps.estimate_ai_base_reserve_credits.return_value = (
        1,
        {},
    )
    deps.check_provider_available.return_value = False
    deps.has_openrouter_fallback.return_value = True
    deps.estimate_image_context_reserve_credits.return_value = 1
    deps.handle_ai_response.return_value = "🖼️ en la imagen veo: todo piola"

    billing_helper = AIMessageBilling(
        credits_db_service=MagicMock(),
        admin_reporter=MagicMock(),
        gen_random_fn=lambda _: "random",
        build_insufficient_credits_message_fn=lambda **_: "insufficient",
        maybe_grant_onboarding_credits_fn=lambda _user_id: None,
        command="/transcribe",
        chat_id="558",
        chat_type="private",
        user_id=102,
        numeric_chat_id=558,
        message={"from": {"first_name": "Ana"}},
    )

    response_msg, handled = run_ai_flow(
        deps,
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

    assert handled is True
    assert response_msg == "🖼️ en la imagen veo: todo piola"
    deps.handle_ai_response.assert_called_once()
