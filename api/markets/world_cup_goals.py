"""Monitor live FIFA World Cup scores and announce new goals."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from api.ai.pricing import calculate_billing_for_segments
from api.core.logging import get_logger
from api.services import http_client

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "fifa.world/scoreboard"
)
POLL_INTERVAL_SECONDS = 15

logger = get_logger(__name__)


@dataclass(frozen=True)
class MatchScore:
    event_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    state: str


@dataclass(frozen=True)
class Goal:
    event_id: str
    scoring_team: str
    opponent: str
    scoring_score: int
    opponent_score: int

    @property
    def dedupe_key(self) -> str:
        return f"{self.event_id}:{self.scoring_team}:{self.scoring_score}"


@dataclass(frozen=True)
class GoalCharge:
    chat_id: int
    reserved_credit_units: int
    metadata: dict[str, Any]


def _score(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_competitor(competitor: Any) -> tuple[str, str, int] | None:
    if not isinstance(competitor, Mapping):
        return None
    side = competitor.get("homeAway")
    team = competitor.get("team")
    score = _score(competitor.get("score"))
    if (
        side not in {"home", "away"}
        or not isinstance(team, Mapping)
        or score is None
    ):
        return None
    name = team.get("displayName") or team.get("name")
    if not name:
        return None
    return str(side), str(name), score


def _parse_event(event: Any) -> MatchScore | None:
    if not isinstance(event, Mapping):
        return None
    event_id = event.get("id")
    competitions = event.get("competitions")
    if not event_id or not isinstance(competitions, list) or not competitions:
        return None
    competition = competitions[0]
    if not isinstance(competition, Mapping):
        return None
    competitors = competition.get("competitors")
    if not isinstance(competitors, list):
        return None

    teams = {
        parsed[0]: (parsed[1], parsed[2])
        for competitor in competitors
        if (parsed := _parse_competitor(competitor)) is not None
    }
    if "home" not in teams or "away" not in teams:
        return None
    status = event.get("status")
    status_type = status.get("type") if isinstance(status, Mapping) else None
    state = status_type.get("state") if isinstance(status_type, Mapping) else ""
    home_team, home_score = teams["home"]
    away_team, away_score = teams["away"]
    return MatchScore(
        event_id=str(event_id),
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        state=str(state),
    )


def parse_scoreboard(payload: Mapping[str, Any]) -> dict[str, MatchScore]:
    events = payload.get("events")
    if not isinstance(events, list):
        return {}
    matches = (
        match for event in events if (match := _parse_event(event)) is not None
    )
    return {match.event_id: match for match in matches}


def detect_goals(
    previous: Mapping[str, MatchScore],
    current: Mapping[str, MatchScore],
) -> list[Goal]:
    goals: list[Goal] = []
    for event_id, match in current.items():
        old = previous.get(event_id)
        if old is None or match.state not in {"in", "post"}:
            continue
        for scoring_score in range(old.home_score + 1, match.home_score + 1):
            goals.append(
                Goal(
                    event_id=event_id,
                    scoring_team=match.home_team,
                    opponent=match.away_team,
                    scoring_score=scoring_score,
                    opponent_score=match.away_score,
                )
            )
        for scoring_score in range(old.away_score + 1, match.away_score + 1):
            goals.append(
                Goal(
                    event_id=event_id,
                    scoring_team=match.away_team,
                    opponent=match.home_team,
                    scoring_score=scoring_score,
                    opponent_score=match.home_score,
                )
            )
    return goals


def _date_range(now: datetime) -> str:
    utc_now = now.astimezone(UTC)
    start = (utc_now - timedelta(days=1)).strftime("%Y%m%d")
    end = (utc_now + timedelta(days=1)).strftime("%Y%m%d")
    return f"{start}-{end}"


class WorldCupGoalMonitor:
    def __init__(
        self,
        *,
        list_chat_ids: Callable[[], list[str]],
        ask_ai: Callable[..., str],
        send_message: Callable[..., Any],
        http_get: Callable[..., Any] = http_client.get,
        now: Callable[[], datetime] | None = None,
        credits_db_service: Any | None = None,
        estimate_ai_base_reserve_credits: Callable[..., tuple[int, dict[str, Any]]]
        | None = None,
    ) -> None:
        self._list_chat_ids = list_chat_ids
        self._ask_ai = ask_ai
        self._send_message = send_message
        self._http_get = http_get
        self._now = now or (lambda: datetime.now(UTC))
        self._credits_db_service = credits_db_service
        self._estimate_ai_base_reserve_credits = estimate_ai_base_reserve_credits
        self._scores: dict[str, MatchScore] | None = None
        self._announced: set[str] = set()

    def fetch_scores(self) -> dict[str, MatchScore]:
        response = self._http_get(
            SCOREBOARD_URL,
            params={"dates": _date_range(self._now()), "limit": 20},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("World Cup scoreboard returned an invalid payload")
        return parse_scoreboard(payload)

    def poll_once(self) -> list[Goal]:
        current = self.fetch_scores()
        if self._scores is None:
            self._scores = current
            return []

        goals = detect_goals(self._scores, current)
        self._scores = current
        fresh_goals = [
            goal for goal in goals if goal.dedupe_key not in self._announced
        ]
        for goal in fresh_goals:
            self._announced.add(goal.dedupe_key)
            self._announce(goal)
        return fresh_goals

    def _announce(self, goal: Goal) -> None:
        chat_ids = self._list_chat_ids()
        if not chat_ids:
            return
        for chat_id in chat_ids:
            message = self._generate_message(goal, chat_id=chat_id)
            self._send_message(chat_id, message)

    def _build_prompt(self, goal: Goal) -> str:
        return (
            f"{goal.scoring_team} acaba de meterle un gol a {goal.opponent}. "
            f"El partido está {goal.scoring_score}-{goal.opponent_score}. "
            "Escribí un solo mensaje corto en español argentino: gritá el gol "
            "con mucha euforia y descansá o insultá futbolísticamente al equipo "
            "rival. Sin markdown ni explicación."
        )

    def _fallback_message(self, goal: Goal) -> str:
        return (
            f"GOOOOOOL DE {goal.scoring_team.upper()}! "
            f"{goal.opponent.upper()}, SON UNOS MUERTOS, MIREN COMO DEFIENDEN! "
            f"{goal.scoring_score}-{goal.opponent_score}"
        )

    def _reserve_chat_credits(
        self,
        chat_id: str,
        goal: Goal,
        messages: list[dict[str, Any]],
    ) -> GoalCharge | None:
        if self._credits_db_service is None:
            return GoalCharge(chat_id=0, reserved_credit_units=0, metadata={})
        if (
            self._estimate_ai_base_reserve_credits is None
            or not self._credits_db_service.is_configured()
        ):
            return None
        try:
            numeric_chat_id = int(chat_id)
        except (TypeError, ValueError):
            return None

        reserve_credits, reserve_meta = self._estimate_ai_base_reserve_credits(
            messages=messages,
        )
        metadata = {
            "usage_tag": "world_cup_goal_alert",
            "reserved_credit_units": reserve_credits,
            "event_id": goal.event_id,
            "scoring_team": goal.scoring_team,
            "opponent": goal.opponent,
            "scoring_score": goal.scoring_score,
            "opponent_score": goal.opponent_score,
            **reserve_meta,
        }
        try:
            result = self._credits_db_service.charge_chat_ai_credits(
                numeric_chat_id,
                reserve_credits,
                event_type="ai_reserve",
                metadata=metadata,
            )
        except Exception:
            logger.exception("failed to reserve World Cup goal alert credits")
            return None
        if not result.get("ok"):
            return None
        return GoalCharge(
            chat_id=numeric_chat_id,
            reserved_credit_units=reserve_credits,
            metadata=metadata,
        )

    def _settle_chat_credits(
        self,
        charge: GoalCharge,
        billing_segments: list[Mapping[str, Any]],
    ) -> None:
        if self._credits_db_service is None or charge.reserved_credit_units <= 0:
            return
        breakdown = calculate_billing_for_segments(billing_segments)
        settled_credit_units = int(breakdown.get("charged_credit_units", 0) or 0)
        metadata = {
            **charge.metadata,
            "reason": "world_cup_goal_success",
            "settled_credit_units": settled_credit_units,
            "pricing_version": breakdown.get("pricing_version"),
            "raw_usd_micros": breakdown.get("raw_usd_micros", 0),
            "model_breakdown": breakdown.get("model_breakdown", []),
            "tool_breakdown": breakdown.get("tool_breakdown", []),
        }
        try:
            if settled_credit_units < charge.reserved_credit_units:
                self._credits_db_service.refund_chat_ai_credits(
                    charge.chat_id,
                    charge.reserved_credit_units - settled_credit_units,
                    event_type="ai_refund",
                    metadata=metadata,
                )
            elif settled_credit_units > charge.reserved_credit_units:
                self._credits_db_service.apply_chat_ai_debt(
                    charge.chat_id,
                    settled_credit_units - charge.reserved_credit_units,
                    event_type="ai_settlement_debt",
                    metadata=metadata,
                )
        except Exception:
            logger.exception("failed to settle World Cup goal alert credits")

    def _refund_chat_credits(self, charge: GoalCharge, *, reason: str) -> None:
        if self._credits_db_service is None or charge.reserved_credit_units <= 0:
            return
        try:
            self._credits_db_service.refund_chat_ai_credits(
                charge.chat_id,
                charge.reserved_credit_units,
                event_type="ai_refund",
                metadata={**charge.metadata, "reason": reason},
            )
        except Exception:
            logger.exception("failed to refund World Cup goal alert credits")

    def _generate_message(self, goal: Goal, *, chat_id: str) -> str:
        prompt = self._build_prompt(goal)
        messages = [{"role": "user", "content": prompt}]
        charge = self._reserve_chat_credits(chat_id, goal, messages)
        if charge is None:
            return self._fallback_message(goal)

        response_meta: dict[str, Any] = {}
        try:
            message = self._ask_ai(
                messages,
                enable_web_search=False,
                chat_id=chat_id,
                response_meta=response_meta,
            ).strip()
        except Exception:
            logger.exception("failed to generate World Cup goal message")
            self._refund_chat_credits(charge, reason="world_cup_goal_error")
            message = ""
        if message and not response_meta.get("ai_fallback"):
            self._settle_chat_credits(
                charge,
                list(response_meta.get("billing_segments") or []),
            )
            return message
        self._refund_chat_credits(charge, reason="world_cup_goal_fallback")
        return self._fallback_message(goal)


def run_world_cup_goal_monitor(
    monitor: WorldCupGoalMonitor,
    *,
    interval_seconds: int = POLL_INTERVAL_SECONDS,
    stop_event: threading.Event | None = None,
) -> None:
    stopper = stop_event or threading.Event()
    while not stopper.is_set():
        try:
            monitor.poll_once()
        except Exception:
            logger.exception("World Cup goal monitor poll failed")
        stopper.wait(interval_seconds)


def start_world_cup_goal_monitor(
    monitor: WorldCupGoalMonitor,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_world_cup_goal_monitor,
        args=(monitor,),
        daemon=True,
        name="world-cup-goals",
    )
    thread.start()
    return thread
