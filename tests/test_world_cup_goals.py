from datetime import UTC, datetime
from unittest.mock import MagicMock

from api.markets.world_cup_goals import (
    Goal,
    MatchScore,
    WorldCupGoalMonitor,
    detect_goals,
    parse_scoreboard,
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
    ask_ai.assert_called_once()
    assert ask_ai.call_args.kwargs["enable_web_search"] is False
    assert ask_ai.call_args.kwargs["response_meta"] == {}
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
    assert "GOOOOOOL DE ENGLAND" in message
    assert "ARGENTINA, SON UNOS MUERTOS" in message
    assert "1-0" in message


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
    monitor = WorldCupGoalMonitor(
        list_chat_ids=lambda: ["chat-1"],
        ask_ai=fallback_ai,
        send_message=send_message,
        http_get=http_get,
    )

    monitor.poll_once()
    monitor.poll_once()

    assert send_message.call_args.args[1].startswith("GOOOOOOL DE ARGENTINA")
