"""Tests for individual tool executors."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from api.tools.registry import execute_tool


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


class TestWebFetchTool:
    @patch("api.agent_tools.fetch_url_content")
    def test_web_fetch(self, mock_fetch):
        mock_fetch.return_value = {
            "url": "https://example.com",
            "title": "Example",
            "content": "Hello world",
        }
        result = execute_tool("web_fetch", {"url": "https://example.com"}, {})
        assert "Hello world" in result.output

    @patch("api.agent_tools.fetch_url_content")
    def test_web_fetch_no_url(self, mock_fetch):
        result = execute_tool("web_fetch", {}, {})
        assert "url" in result.output.lower()

    @patch("api.agent_tools.fetch_url_content")
    def test_web_fetch_error(self, mock_fetch):
        mock_fetch.return_value = {"url": "https://example.com", "error": "timeout"}
        result = execute_tool("web_fetch", {"url": "https://example.com"}, {})
        assert "error" in result.output.lower()

    @patch("api.utils.links.fetch_tweet_via_oembed")
    def test_web_fetch_reads_tweet_with_oembed(self, mock_oembed):
        mock_oembed.return_value = {
            "author_name": "OSINTdefender",
            "html": (
                "<blockquote><p>Reports of shots fired were unfounded.</p>"
                "<a href='https://x.com/sentdefender/status/2048202539770802483'>"
                "Apr 26, 2026</a></blockquote>"
            ),
        }

        result = execute_tool(
            "web_fetch",
            {"url": "https://x.com/sentdefender/status/2048202539770802483"},
            {},
        )

        assert "OSINTdefender" in result.output
        assert "Reports of shots fired" in result.output
        assert result.metadata["url"] == "https://x.com/sentdefender/status/2048202539770802483"

    @patch("api.utils.links.request_with_ssl_fallback")
    @patch("api.utils.links.fetch_tweet_via_oembed")
    def test_web_fetch_resolves_id_only_fixupx_from_metadata(
        self, mock_oembed, mock_request
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.url = "https://fixupx.com/status/2048202539770802483"
        mock_response.text = (
            "<meta property='og:url' content='https://x.com/sentdefender/status/2048202539770802483'>"
            "<meta property='og:title' content='OSINTdefender (@sentdefender)'>"
            "<meta property='og:description' content='Following reports'>"
            "<meta property='og:image' content='https://example.com/image.jpg'>"
        )
        mock_request.return_value = mock_response
        mock_oembed.return_value = {
            "author_name": "OSINTdefender",
            "html": "<blockquote><p>All clear.</p></blockquote>",
        }

        result = execute_tool(
            "web_fetch",
            {"url": "https://fixupx.com/status/2048202539770802483"},
            {},
        )

        assert "All clear" in result.output
        assert result.metadata["url"] == "https://x.com/sentdefender/status/2048202539770802483"

    @patch("api.agent_tools.fetch_url_content")
    def test_web_fetch_rejects_x_error_page(self, mock_fetch):
        mock_fetch.return_value = {
            "url": "https://x.com/status/1",
            "content": "Something went wrong Try again privacy related extensions",
        }

        result = execute_tool("web_fetch", {"url": "https://example.com/x-error"}, {})

        assert "pagina de error" in result.output

    @patch("api.utils.links.inspect_embed_url")
    @patch("api.agent_tools.fetch_url_content")
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
        call_kwargs = mock_schedule.call_args.kwargs
        assert call_kwargs["trigger_config"]["type"] == "interval"
        assert call_kwargs["trigger_config"]["days"] == 3

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
        call_kwargs = mock_schedule.call_args.kwargs
        assert call_kwargs["trigger_config"]["type"] == "cron"
        assert call_kwargs["trigger_config"]["hour"] == 4
        assert call_kwargs["trigger_config"]["minute"] == 20

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
        call_kwargs = mock_schedule.call_args.kwargs
        assert call_kwargs["trigger_config"]["day_of_week"] == "mon,wed"

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
    @patch("api.tools.task_cancel.cancel_task")
    def test_cancel_success(self, mock_cancel):
        mock_cancel.return_value = True
        result = execute_tool("task_cancel", {"task_id": "abc123"}, {})
        assert "cancelada" in result.output

    def test_cancel_no_id(self):
        result = execute_tool("task_cancel", {}, {})
        assert "id" in result.output.lower()
