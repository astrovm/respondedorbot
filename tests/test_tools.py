"""Tests for individual tool executors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from api.tools.registry import execute_tool, get_all_tool_schemas
from api.tools.runtime import ToolRuntime


class TestCalculateTool:
    def test_simple_addition(self):
        result = execute_tool("calculate", {"expression": "2 + 3"})
        assert result.output == "5"

    def test_multiplication(self):
        result = execute_tool("calculate", {"expression": "10 * 5"})
        assert result.output == "50"

    def test_division(self):
        result = execute_tool("calculate", {"expression": "10 / 3"})
        assert result.output == "3.33333333"

    def test_power(self):
        result = execute_tool("calculate", {"expression": "2 ** 10"})
        assert result.output == "1024"

    def test_integer_division(self):
        result = execute_tool("calculate", {"expression": "7 // 2"})
        assert result.output == "3"

    def test_modulo(self):
        result = execute_tool("calculate", {"expression": "10 % 3"})
        assert result.output == "1"

    def test_division_by_zero(self):
        result = execute_tool("calculate", {"expression": "10 / 0"})
        assert "cero" in result.output

    def test_invalid_expression(self):
        result = execute_tool("calculate", {"expression": "__import__('os')"})
        assert "no permitida" in result.output

    def test_empty_expression(self):
        result = execute_tool("calculate", {"expression": ""})
        assert "expresion" in result.output

    def test_no_expression_param(self):
        result = execute_tool("calculate", {})
        assert "expresion" in result.output

    def test_negative(self):
        result = execute_tool("calculate", {"expression": "-5 + 10"})
        assert result.output == "5"


class TestCryptoPricesTool:
    def test_crypto_prices_calls_get_prices(self):
        mock_gp = MagicMock(return_value="BTC: $50000")
        result = execute_tool(
            "crypto_prices",
            {"symbols": "BTC"},
            {"get_prices": mock_gp},
        )
        assert result.output == "BTC: $50000"
        mock_gp.assert_called_once_with("BTC")

    def test_crypto_prices_no_context(self):
        result = execute_tool("crypto_prices", {"symbols": "ETH"}, {})
        assert "not available" in result.output

    def test_crypto_prices_none_result(self):
        mock_gp = MagicMock(return_value=None)
        result = execute_tool(
            "crypto_prices",
            {"symbols": "BTC"},
            {"get_prices": mock_gp},
        )
        assert "no se pudieron" in result.output


class TestOnDemandContextTools:
    def test_dollar_rates_uses_injected_service(self):
        get_rates = MagicMock(return_value="synthetic dollar rates")

        result = execute_tool(
            "dollar_rates",
            {"timeframe": "6h"},
            {"get_dollar_rates": get_rates},
        )

        assert result.output == "synthetic dollar rates"
        get_rates.assert_called_once_with("6h")

    def test_weather_uses_injected_service(self):
        get_weather = MagicMock(
            return_value={
                "apparent_temperature": 21,
                "precipitation_probability": 15,
                "description": "synthetic clear sky",
                "cloud_cover": 10,
                "visibility": 12000,
            }
        )

        result = execute_tool(
            "weather",
            {},
            {"get_weather_context": get_weather},
        )

        assert "synthetic clear sky" in result.output
        assert "12.0km" in result.output
        get_weather.assert_called_once_with()

    def test_hacker_news_uses_injected_service(self):
        get_news = MagicMock(
            return_value=[
                {
                    "title": "Synthetic technology story",
                    "url": "https://example.test/story",
                    "points": 42,
                    "comments": 7,
                }
            ]
        )

        result = execute_tool(
            "hacker_news",
            {"limit": 3},
            {"get_hacker_news_context": get_news},
        )

        assert "Synthetic technology story" in result.output
        assert "https://example.test/story" in result.output
        get_news.assert_called_once_with(3)

    def test_bot_capabilities_uses_injected_renderer(self):
        get_capabilities = MagicMock(return_value="synthetic capability catalog")

        result = execute_tool(
            "bot_capabilities",
            {},
            {"get_bot_capabilities": get_capabilities},
        )

        assert result.output == "synthetic capability catalog"
        get_capabilities.assert_called_once_with()


def test_production_tool_registry_exposes_all_context_tools():
    context = {
        "chat_id": "synthetic-chat",
        "user_id": 123,
        "web_search_enabled": True,
        "get_prices": MagicMock(),
        "get_dollar_rates": MagicMock(),
        "get_weather_context": MagicMock(),
        "get_hacker_news_context": MagicMock(),
        "get_bot_capabilities": MagicMock(),
        "config_redis": MagicMock(),
    }

    names = {schema["function"]["name"] for schema in get_all_tool_schemas(context)}

    assert {
        "stock_prices",
        "task_list",
        "task_cancel",
        "dollar_rates",
        "weather",
        "hacker_news",
        "bot_capabilities",
    } <= names


def test_task_set_schema_contains_scheduling_argument_guidance():
    task_schema = next(
        schema["function"]
        for schema in get_all_tool_schemas()
        if schema["function"]["name"] == "task_set"
    )

    assert "only the future instruction" in task_schema["description"]
    assert "preserve its subject" in task_schema["description"]


def test_tool_runtime_executes_on_demand_weather_service():
    get_weather = MagicMock(
        return_value={
            "apparent_temperature": 18,
            "precipitation_probability": 5,
            "description": "synthetic mild weather",
            "cloud_cover": 20,
            "visibility": 9000,
        }
    )
    tool_call = SimpleNamespace(
        id="synthetic-call",
        function=SimpleNamespace(name="weather", arguments="{}"),
    )

    messages = ToolRuntime(print_fn=lambda _message: None).apply_tool_calls(
        SimpleNamespace(content=""),
        [tool_call],
        [],
        {"get_weather_context": get_weather},
    )

    assert messages[-1]["role"] == "tool"
    assert "synthetic mild weather" in messages[-1]["content"]
    get_weather.assert_called_once_with()


class TestWebFetchTool:
    @patch("api.links.agent_tools.fetch_url_content")
    def test_web_fetch(self, mock_fetch):
        mock_fetch.return_value = {
            "url": "https://example.com",
            "title": "Example",
            "content": "Hello world",
        }
        result = execute_tool("web_fetch", {"url": "https://example.com"}, {})
        assert "Hello world" in result.output

    @patch("api.links.agent_tools.fetch_url_content")
    def test_web_fetch_no_url(self, mock_fetch):
        result = execute_tool("web_fetch", {}, {})
        assert "url" in result.output.lower()

    @patch("api.links.agent_tools.fetch_url_content")
    def test_web_fetch_error(self, mock_fetch):
        mock_fetch.return_value = {"url": "https://example.com", "error": "timeout"}
        result = execute_tool("web_fetch", {"url": "https://example.com"}, {})
        assert "error" in result.output.lower()

    @patch("api.utils.links.fetch_tweet_via_oembed")
    def test_web_fetch_reads_tweet_with_oembed(self, mock_oembed):
        mock_oembed.return_value = {
            "author_name": "Example User",
            "html": (
                "<blockquote><p>This is an example status update.</p>"
                "<a href='https://x.com/test_user/status/1234567890123456789'>"
                "Jan 1, 2020</a></blockquote>"
            ),
        }

        result = execute_tool(
            "web_fetch",
            {"url": "https://x.com/test_user/status/1234567890123456789"},
            {},
        )

        assert "Example User" in result.output
        assert "example status update" in result.output
        assert result.metadata["url"] == "https://x.com/test_user/status/1234567890123456789"

    @patch("api.utils.links.request_with_ssl_fallback")
    @patch("api.utils.links.fetch_tweet_via_oembed")
    def test_web_fetch_resolves_id_only_fixupx_from_metadata(
        self, mock_oembed, mock_request
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://fixupx.com/status/1234567890123456789"
        mock_response.text = (
            "<meta property='og:url' content='https://x.com/test_user/status/1234567890123456789'>"
            "<meta property='og:title' content='Example User (@test_user)'>"
            "<meta property='og:description' content='Following reports'>"
            "<meta property='og:image' content='https://example.com/image.jpg'>"
        )
        mock_request.return_value = mock_response
        mock_oembed.return_value = {
            "author_name": "Example User",
            "html": "<blockquote><p>All clear.</p></blockquote>",
        }

        result = execute_tool(
            "web_fetch",
            {"url": "https://fixupx.com/status/1234567890123456789"},
            {},
        )

        assert "All clear" in result.output
        assert result.metadata["url"] == "https://x.com/test_user/status/1234567890123456789"

    @patch("api.links.agent_tools.fetch_url_content")
    def test_web_fetch_rejects_x_error_page(self, mock_fetch):
        mock_fetch.return_value = {
            "url": "https://x.com/status/1",
            "content": "Something went wrong Try again privacy related extensions",
        }

        result = execute_tool("web_fetch", {"url": "https://example.com/x-error"}, {})

        assert "pagina de error" in result.output

    @patch("api.utils.links.inspect_embed_url")
    @patch("api.links.agent_tools.fetch_url_content")
    def test_web_fetch_does_not_treat_embedded_twitter_url_as_tweet(
        self, mock_fetch, mock_inspect
    ):
        mock_fetch.return_value = {
            "url": "https://example.com/read/fixupx.com/status/123",
            "content": "regular page",
        }

        result = execute_tool(
            "web_fetch",
            {"url": "https://example.com/read/fixupx.com/status/123"},
            {},
        )

        assert "regular page" in result.output
        mock_inspect.assert_not_called()


class TestTaskSetTool:
    @patch("api.tools.task_set.schedule_task")
    def test_one_shot(self, mock_schedule):
        mock_schedule.return_value = "abc123"
        result = execute_tool(
            "task_set",
            {"text": "comprar pizza", "delay_seconds": 1800},
            {"chat_id": "123", "user_name": "u"},
        )
        assert "listo" in result.output
        assert result.metadata["task_id"] == "abc123"

    @patch("api.tools.task_set.schedule_task")
    def test_recurring(self, mock_schedule):
        mock_schedule.return_value = "def456"
        result = execute_tool(
            "task_set",
            {"text": "noticias de sonic", "interval_seconds": 86400},
            {"chat_id": "123"},
        )
        assert "listo" in result.output
        assert result.metadata["task_id"] == "def456"

    def test_no_text(self):
        result = execute_tool("task_set", {"delay_seconds": 1800}, {"chat_id": "123"})
        assert "texto" in result.output.lower()

    def test_no_time_params(self):
        result = execute_tool("task_set", {"text": "algo"}, {"chat_id": "123"})
        assert "delay_seconds" in result.output or "interval_seconds" in result.output

    def test_no_chat(self):
        result = execute_tool("task_set", {"text": "algo", "delay_seconds": 1800}, {})
        assert "chat" in result.output.lower()

    def test_delay_too_long(self):
        result = execute_tool(
            "task_set",
            {"text": "algo", "delay_seconds": 86400 * 3651},
            {"chat_id": "123"},
        )
        assert "maximo" in result.output.lower()

    def test_interval_too_short(self):
        result = execute_tool(
            "task_set",
            {"text": "algo", "interval_seconds": 60},
            {"chat_id": "123"},
        )
        assert "minimo" in result.output.lower()

    def test_interval_too_long(self):
        result = execute_tool(
            "task_set",
            {"text": "algo", "interval_seconds": 86400 * 8},
            {"chat_id": "123"},
        )
        assert "maximo" in result.output.lower()

    @patch("api.tools.task_set.schedule_task")
    def test_schedule_fails(self, mock_schedule):
        mock_schedule.return_value = None
        result = execute_tool(
            "task_set",
            {"text": "algo", "delay_seconds": 1800},
            {"chat_id": "123"},
        )
        assert "no se pudo" in result.output

    @patch("api.tools.task_set.credits_db")
    def test_no_credits(self, mock_credits):
        mock_credits.is_configured.return_value = True
        mock_credits.get_balance.return_value = 0
        result = execute_tool(
            "task_set",
            {"text": "algo", "delay_seconds": 1800},
            {"chat_id": "123", "user_id": 42},
        )
        assert "creditos" in result.output
        mock_credits.get_balance.assert_called_once_with("user", 42)

    @patch("api.tools.task_set.schedule_task")
    def test_trigger_config_interval(self, mock_schedule):
        mock_schedule.return_value = "interval123"
        result = execute_tool(
            "task_set",
            {
                "text": "cada 3 dias",
                "trigger_config": {"type": "interval", "days": 3},
            },
            {"chat_id": "123"},
        )
        assert "listo" in result.output
        assert "cada 3 dias" in result.output
        mock_schedule.assert_called_once()
        request = mock_schedule.call_args.args[0]
        assert request.trigger.kind == "interval_days"
        assert request.trigger.days == 3

    @patch("api.tools.task_set.schedule_task")
    def test_trigger_config_cron_daily(self, mock_schedule):
        mock_schedule.return_value = "cron123"
        result = execute_tool(
            "task_set",
            {
                "text": "a las 4:20",
                "trigger_config": {"type": "cron", "hour": 4, "minute": 20},
            },
            {"chat_id": "123"},
        )
        assert "listo" in result.output
        assert "04:20" in result.output
        mock_schedule.assert_called_once()
        request = mock_schedule.call_args.args[0]
        assert request.trigger.kind == "cron"
        assert request.trigger.hour == 4
        assert request.trigger.minute == 20

    @patch("api.tools.task_set.schedule_task")
    def test_trigger_config_cron_weekdays(self, mock_schedule):
        mock_schedule.return_value = "weekdays123"
        result = execute_tool(
            "task_set",
            {
                "text": "recordar los lunes",
                "trigger_config": {
                    "type": "cron",
                    "hour": 9,
                    "minute": 0,
                    "day_of_week": "mon",
                },
            },
            {"chat_id": "123"},
        )
        assert "listo" in result.output
        mock_schedule.assert_called_once()

    @patch("api.tools.task_set.schedule_task")
    def test_trigger_config_cron_weekdays_accepts_spanish_tokens(self, mock_schedule):
        mock_schedule.return_value = "weekdays123"
        result = execute_tool(
            "task_set",
            {
                "text": "recordar los lunes y miercoles",
                "trigger_config": {
                    "type": "cron",
                    "hour": 9,
                    "minute": 0,
                    "day_of_week": "lun,mie",
                },
            },
            {"chat_id": "123"},
        )
        assert "listo" in result.output
        mock_schedule.assert_called_once()
        request = mock_schedule.call_args.args[0]
        assert request.trigger.weekdays == ("mon", "wed")

    def test_trigger_config_invalid_type(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "invalid"},
            },
            {"chat_id": "123"},
        )
        assert "interval" in result.output.lower() or "cron" in result.output.lower()

    def test_trigger_config_invalid_hour(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "cron", "hour": 25},
            },
            {"chat_id": "123"},
        )
        assert "0-23" in result.output

    @patch("api.tools.task_set.schedule_task")
    def test_trigger_config_invalid_day_of_week(self, mock_schedule):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {
                    "type": "cron",
                    "hour": 9,
                    "minute": 0,
                    "day_of_week": "foo",
                },
            },
            {"chat_id": "123"},
        )
        assert "day_of_week" in result.output.lower()
        mock_schedule.assert_not_called()

    def test_trigger_config_cron_requires_hour(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "cron", "minute": 15},
            },
            {"chat_id": "123"},
        )
        assert "hour" in result.output.lower()
        assert "requerido" in result.output.lower()

    def test_trigger_config_cron_requires_minute(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "cron", "hour": 9},
            },
            {"chat_id": "123"},
        )
        assert "minute" in result.output.lower()
        assert "requerido" in result.output.lower()

    def test_trigger_config_invalid_days(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "interval", "days": -1},
            },
            {"chat_id": "123"},
        )
        assert "positivo" in result.output.lower()

    def test_trigger_config_missing_days(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "interval"},
            },
            {"chat_id": "123"},
        )
        assert "requerido" in result.output.lower()

    def test_trigger_config_days_too_large(self):
        result = execute_tool(
            "task_set",
            {
                "text": "algo",
                "trigger_config": {"type": "interval", "days": 91},
            },
            {"chat_id": "123"},
        )
        assert "maximo" in result.output.lower()


class TestTaskListTool:
    @patch("api.tools.task_list.list_tasks")
    def test_empty(self, mock_list):
        mock_list.return_value = []
        result = execute_tool("task_list", {}, {"chat_id": "123"})
        assert "no hay" in result.output.lower()

    @patch("api.tools.task_list.list_tasks")
    def test_with_one_shot(self, mock_list):
        mock_list.return_value = [
            {
                "id": "r1",
                "text": "comprar",
                "next_run": "2026-01-01",
                "interval_seconds": None,
            },
        ]
        result = execute_tool("task_list", {}, {"chat_id": "123"})
        assert "comprar" in result.output

    @patch("api.tools.task_list.list_tasks")
    def test_with_recurring(self, mock_list):
        mock_list.return_value = [
            {"text": "noticias sonic", "interval_seconds": 86400, "id": "t1"},
        ]
        result = execute_tool("task_list", {}, {"chat_id": "123"})
        assert "noticias sonic" in result.output

    @patch("api.tools.task_list.list_tasks")
    def test_with_cron_recurring(self, mock_list):
        mock_list.return_value = [
            {
                "id": "t1",
                "text": "cuanta aura farmeaste hoy",
                "interval_seconds": None,
                "trigger_config": {"type": "cron", "hour": 20, "minute": 30},
                "next_run": "16/04 20:30",
            }
        ]
        result = execute_tool("task_list", {}, {"chat_id": "123"})
        assert "todos los dias a las 20:30" in result.output

    @patch("api.tools.task_list.list_tasks")
    def test_with_cron_weekdays_recurring_shows_spanish_days(self, mock_list):
        mock_list.return_value = [
            {
                "id": "t1",
                "text": "cuanta aura farmeaste hoy",
                "interval_seconds": None,
                "trigger_config": {
                    "type": "cron",
                    "hour": 20,
                    "minute": 30,
                    "day_of_week": "mon,wed",
                },
                "next_run": "16/04 20:30",
            }
        ]
        result = execute_tool("task_list", {}, {"chat_id": "123"})
        assert "los lun, mie a las 20:30" in result.output

    def test_no_chat(self):
        result = execute_tool("task_list", {}, {})
        assert "chat" in result.output.lower()


class TestTaskCancelTool:
    @patch("api.tools.task_cancel.list_tasks")
    @patch("api.tools.task_cancel.cancel_task")
    def test_cancel_success(self, mock_cancel, mock_list):
        mock_list.return_value = [{"id": "synthetic-task"}]
        mock_cancel.return_value = True
        result = execute_tool(
            "task_cancel",
            {"task_id": "synthetic-task"},
            {"chat_id": "synthetic-chat"},
        )
        assert "cancelada" in result.output
        mock_list.assert_called_once_with("synthetic-chat")
        mock_cancel.assert_called_once_with("synthetic-task")

    def test_cancel_no_id(self):
        result = execute_tool("task_cancel", {}, {})
        assert "id" in result.output.lower()

    @patch("api.tools.task_cancel.cancel_task")
    @patch("api.tools.task_cancel.list_tasks", return_value=[])
    def test_rejects_task_from_another_chat(self, _mock_list, mock_cancel):
        result = execute_tool(
            "task_cancel",
            {"task_id": "synthetic-task"},
            {"chat_id": "synthetic-chat"},
        )

        assert "no existe en este chat" in result.output
        mock_cancel.assert_not_called()


class TestGetChatMembersTool:
    @patch("api.tools.get_chat_members.get_chat_members")
    def test_returns_formatted_members(self, mock_get):
        mock_get.return_value = [
            {"user_id": "42", "first_name": "Juan", "username": "juan123", "last_seen": 1000},
            {"user_id": "99", "first_name": "Maria", "username": "", "last_seen": 2000},
        ]
        mock_redis = MagicMock()
        result = execute_tool(
            "get_chat_members",
            {},
            {"chat_id": "-100123", "config_redis": lambda: mock_redis},
        )
        assert "Juan" in result.output
        assert "juan123" in result.output
        assert "Maria" in result.output

    @patch("api.tools.get_chat_members.get_chat_members")
    def test_returns_empty_message_when_no_members(self, mock_get):
        mock_get.return_value = []
        mock_redis = MagicMock()
        result = execute_tool(
            "get_chat_members",
            {},
            {"chat_id": "-100123", "config_redis": lambda: mock_redis},
        )
        assert "no conozco" in result.output.lower()

    def test_returns_not_available_without_chat_id(self):
        result = execute_tool("get_chat_members", {}, {"config_redis": lambda: MagicMock()})
        assert "no disponible" in result.output.lower()

    def test_returns_not_available_without_config_redis(self):
        result = execute_tool("get_chat_members", {}, {"chat_id": "-100123"})
        assert "no disponible" in result.output.lower()
