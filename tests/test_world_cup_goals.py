from datetime import UTC, datetime
from unittest.mock import MagicMock

from api.markets.world_cup_goals import (
    Goal,
    MatchScore,
    TEAM_NAMES_ES,
    WORLD_CUP_TEAM_RANKING,
    WorldCupGoalMonitor,
    detect_goals,
    parse_scoreboard,
    preferred_team,
    team_name_es,
)


def _payload(home_score: int, away_score: int, *, state: str = "in"):
    return {
        "events": [
            {
                "id": "match-1",
                "status": {"type": {"state": state}},
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": str(home_score),
                                "team": {"displayName": "Argentina"},
                            },
                            {
                                "homeAway": "away",
                                "score": str(away_score),
                                "team": {"displayName": "England"},
                            },
                        ]
                    }
                ],
            }
        ]
    }


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    return response


def test_parse_scoreboard_extracts_match_scores():
    scores = parse_scoreboard(_payload(2, 1))

    assert scores == {
        "match-1": MatchScore(
            event_id="match-1",
            home_team="Argentina",
            away_team="England",
            home_score=2,
            away_score=1,
            state="in",
        )
    }


def test_detect_goals_returns_each_team_that_increased():
    previous = parse_scoreboard(_payload(0, 0))
    current = parse_scoreboard(_payload(1, 1))

    assert detect_goals(previous, current) == [
        Goal("match-1", "Argentina", "England", 1, 1),
        Goal("match-1", "England", "Argentina", 1, 1),
    ]


def test_detect_goals_preserves_multiple_score_increments_between_polls():
    previous = parse_scoreboard(_payload(0, 0))
    current = parse_scoreboard(_payload(2, 0))

    assert detect_goals(previous, current) == [
        Goal("match-1", "Argentina", "England", 1, 0),
        Goal("match-1", "Argentina", "England", 2, 0),
    ]


def test_preferred_team_uses_global_ranking():
    assert len(WORLD_CUP_TEAM_RANKING) == 48
    assert len(set(WORLD_CUP_TEAM_RANKING)) == 48
    assert WORLD_CUP_TEAM_RANKING[:3] == ("Argentina", "Japan", "South Korea")
    assert WORLD_CUP_TEAM_RANKING[-1] == "Brazil"
    assert preferred_team("England", "Argentina") == "Argentina"
    assert preferred_team("Japan", "Argentina") == "Argentina"
    assert preferred_team("Japan", "Brazil") == "Japan"
    assert preferred_team("South Korea", "Uruguay") == "South Korea"
    assert preferred_team("Uruguay", "Brazil") == "Uruguay"
    assert preferred_team("Mexico", "Brazil") == "Mexico"
    assert preferred_team("England", "Brazil") == "England"
    assert preferred_team("Germany", "Morocco") == "Germany"
    assert preferred_team("United States", "Senegal") == "United States"
    assert preferred_team("Australia", "Qatar") == "Australia"
    assert preferred_team("Haiti", "Scotland") == "Scotland"
    assert preferred_team("Germany", "Curaçao") == "Germany"
    assert preferred_team("Switzerland", "Canada") == "Switzerland"
    assert preferred_team("Portugal", "Uzbekistan") == "Portugal"
    assert preferred_team("Spain", "Mexico") == "Spain"
    assert preferred_team("Norway", "France") == "Norway"
    assert preferred_team("Australia", "Scotland") == "Australia"
    assert preferred_team("Czechia", "Uzbekistan") == "Czechia"
    assert preferred_team("Czechia", "Mexico") == "Czechia"
    assert preferred_team("Netherlands", "Sweden") == "Sweden"
    assert preferred_team("Switzerland", "Panama") == "Switzerland"
    assert preferred_team("Germany", "France") == "Germany"
    assert preferred_team("Panama", "Haiti") == "Panama"
    assert preferred_team("Panama", "Ghana") == "Panama"
    assert preferred_team("Senegal", "Uzbekistan") == "Senegal"
    assert preferred_team("Morocco", "Uzbekistan") == "Morocco"


def test_goal_prompt_keeps_the_ranked_team_side():
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: [],
        ask_ai=MagicMock(),
        send_message=MagicMock(),
    )

    argentina_scores = monitor._build_prompt(
        Goal("match-1", "Argentina", "England", 1, 0)
    )
    england_scores = monitor._build_prompt(
        Goal("match-1", "England", "Argentina", 1, 0)
    )

    assert "hinchás por Argentina" in argentina_scores
    assert "Tu equipo acaba de hacer el gol" in argentina_scores
    assert "hinchás por Argentina" in england_scores
    assert "El rival acaba de hacerle un gol a tu equipo" in england_scores


def test_goal_messages_use_spanish_team_names():
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: [],
        ask_ai=MagicMock(),
        send_message=MagicMock(),
    )

    prompt = monitor._build_prompt(
        Goal("match-1", "Ivory Coast", "Ecuador", 1, 0)
    )
    fallback = monitor._fallback_message(
        Goal("match-1", "Ivory Coast", "Ecuador", 1, 0)
    )

    assert team_name_es("Ivory Coast") == "Costa de Marfil"
    assert team_name_es("Japan") == "Japón"
    assert team_name_es("Netherlands") == "Países Bajos"
    assert set(TEAM_NAMES_ES) == set(WORLD_CUP_TEAM_RANKING)
    assert team_name_es("Canada") == "Canadá"
    assert team_name_es("Belgium") == "Bélgica"
    assert team_name_es("Croatia") == "Croacia"
    assert team_name_es("Uzbekistan") == "Uzbekistán"
    assert team_name_es("Iran") == "Irán"
    assert team_name_es("IR Iran") == "Irán"
    assert team_name_es("Korea Republic") == "Corea del Sur"
    assert team_name_es("Côte d'Ivoire") == "Costa de Marfil"
    assert team_name_es("Congo DR") == "Congo"
    assert team_name_es("DR Congo") == "Congo"
    assert "Costa de Marfil acaba de meterle un gol a Ecuador" in prompt
    assert "Ivory Coast" not in prompt
    assert "COSTA DE MARFIL" in fallback
    assert "IVORY COAST" not in fallback


def test_monitor_warms_up_then_announces_new_goal_to_enabled_chats():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(1, 0)),
        ]
    )
    ask_ai = MagicMock(return_value="goooooool, ingleses muertos")
    send_message = MagicMock()
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["chat-1", "chat-2"],
        ask_ai=ask_ai,
        send_message=send_message,
        http_get=http_get,
        now=lambda: datetime(2026, 6, 14, tzinfo=UTC),
    )

    assert monitor.poll_once() == []
    goals = monitor.poll_once()

    assert goals == [Goal("match-1", "Argentina", "England", 1, 0)]
    assert ask_ai.call_count == 2
    assert ask_ai.call_args_list[0].kwargs["enable_web_search"] is False
    assert ask_ai.call_args_list[0].kwargs["chat_id"] == "chat-1"
    assert ask_ai.call_args_list[0].kwargs["response_meta"] == {}
    assert ask_ai.call_args_list[1].kwargs["chat_id"] == "chat-2"
    send_message.assert_any_call("chat-1", "goooooool, ingleses muertos")
    send_message.assert_any_call("chat-2", "goooooool, ingleses muertos")
    assert http_get.call_args.kwargs["params"]["dates"] == "20260613-20260615"


def test_monitor_does_not_generate_message_when_no_chat_is_enabled():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(1, 0)),
        ]
    )
    ask_ai = MagicMock()
    send_message = MagicMock()
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: [],
        ask_ai=ask_ai,
        send_message=send_message,
        http_get=http_get,
    )

    monitor.poll_once()
    monitor.poll_once()

    ask_ai.assert_not_called()
    send_message.assert_not_called()


def test_monitor_uses_fallback_when_ai_fails():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(0, 1)),
        ]
    )
    send_message = MagicMock()
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["chat-1"],
        ask_ai=MagicMock(side_effect=RuntimeError("provider down")),
        send_message=send_message,
        http_get=http_get,
    )

    monitor.poll_once()
    monitor.poll_once()

    message = send_message.call_args.args[1]
    assert "LA PUTA MADRE, INGLATERRA" in message
    assert "VAMOS ARGENTINA" in message
    assert "1-0" in message


def test_monitor_charges_chat_for_ai_goal_message_and_refunds_unused_reserve():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(1, 0)),
        ]
    )

    def ask_ai(_messages, *, response_meta, **_kwargs):
        response_meta["billing_segments"] = [
            {
                "kind": "chat",
                "model": "deepseek/deepseek-v4-flash",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        ]
        return "goooooool"

    credits = MagicMock()
    credits.is_configured.return_value = True
    credits.charge_chat_ai_credits.return_value = {"ok": True, "source": "chat"}
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["-100"],
        ask_ai=ask_ai,
        send_message=MagicMock(),
        http_get=http_get,
        credits_db_service=credits,
        estimate_ai_base_reserve_credits=lambda **_kwargs: (10, {"model": "x"}),
    )

    monitor.poll_once()
    monitor.poll_once()

    credits.charge_chat_ai_credits.assert_called_once()
    charge_args = credits.charge_chat_ai_credits.call_args
    assert charge_args.args[:2] == (-100, 10)
    assert charge_args.kwargs["event_type"] == "ai_reserve"
    assert charge_args.kwargs["metadata"]["usage_tag"] == "world_cup_goal_alert"
    credits.refund_chat_ai_credits.assert_called_once()
    assert credits.refund_chat_ai_credits.call_args.args[:2] == (-100, 9)
    credits.apply_chat_ai_debt.assert_not_called()


def test_monitor_uses_local_fallback_when_chat_has_no_credits():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(1, 0)),
        ]
    )
    ask_ai = MagicMock()
    send_message = MagicMock()
    credits = MagicMock()
    credits.is_configured.return_value = True
    credits.charge_chat_ai_credits.return_value = {"ok": False}
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["-100"],
        ask_ai=ask_ai,
        send_message=send_message,
        http_get=http_get,
        credits_db_service=credits,
        estimate_ai_base_reserve_credits=lambda **_kwargs: (10, {}),
    )

    monitor.poll_once()
    monitor.poll_once()

    ask_ai.assert_not_called()
    assert send_message.call_args.args[1].startswith("GOOOOOOL DE ARGENTINA")


def test_monitor_uses_local_message_for_provider_fallback():
    http_get = MagicMock(
        side_effect=[
            _response(_payload(0, 0)),
            _response(_payload(1, 0)),
        ]
    )

    def fallback_ai(_messages, *, response_meta, **_kwargs):
        response_meta["ai_fallback"] = True
        return "respuesta genérica"

    send_message = MagicMock()
    credits = MagicMock()
    credits.is_configured.return_value = True
    credits.charge_chat_ai_credits.return_value = {"ok": True, "source": "chat"}
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["-100"],
        ask_ai=fallback_ai,
        send_message=send_message,
        http_get=http_get,
        credits_db_service=credits,
        estimate_ai_base_reserve_credits=lambda **_kwargs: (10, {}),
    )

    monitor.poll_once()
    monitor.poll_once()

    assert send_message.call_args.args[1].startswith("GOOOOOOL DE ARGENTINA")
    credits.refund_chat_ai_credits.assert_called_once()
