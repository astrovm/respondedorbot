from api.random_replies import build_random_reply, resolve_random_reply_name


def test_resolve_random_reply_name_prefers_first_name():
    assert (
        resolve_random_reply_name({"first_name": "Ana", "username": "ana_user"})
        == "Ana"
    )


def test_resolve_random_reply_name_falls_back_to_username():
    assert resolve_random_reply_name({"username": "ana_user"}) == "ana_user"


def test_resolve_random_reply_name_returns_empty_string_when_sender_has_no_name():
    assert resolve_random_reply_name({}) == ""


def test_build_random_reply_uses_shared_name_resolution():
    assert (
        build_random_reply(lambda name: f"random:{name}", {"username": "ana_user"})
        == "random:ana_user"
    )
