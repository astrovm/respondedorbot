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


class TestPriceLookupTool:
    def test_price_lookup_calls_get_prices(self):
        mock_gp = MagicMock(return_value="BTC: $50000")
        result = execute_tool(
            "price_lookup",
            {"symbols": "BTC"},
            {"get_prices": mock_gp},
        )
        assert result.output == "BTC: $50000"
        mock_gp.assert_called_once_with("BTC")

    def test_price_lookup_no_context(self):
        result = execute_tool("price_lookup", {"symbols": "ETH"}, {})
        assert "not available" in result.output

    def test_price_lookup_none_result(self):
        mock_gp = MagicMock(return_value=None)
        result = execute_tool(
            "price_lookup",
            {"symbols": "BTC"},
            {"get_prices": mock_gp},
        )
        assert "no se pudieron" in result.output


class TestDollarLookupTool:
    def test_dollar_lookup(self):
        mock_gd = MagicMock(return_value="Blue: 1200")
        result = execute_tool(
            "dollar_lookup",
            {},
            {"get_dollar_rates": mock_gd},
        )
        assert result.output == "Blue: 1200"

    def test_dollar_lookup_no_context(self):
        result = execute_tool("dollar_lookup", {}, {})
        assert "not available" in result.output


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


class TestReminderSetTool:
    @patch("api.tools.reminder_set.schedule_reminder")
    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set(self, mock_parse, mock_schedule):
        mock_parse.return_value = 1800
        mock_schedule.return_value = "abc123"
        result = execute_tool(
            "reminder_set",
            {"text": "comprar pizza", "delay": "30 min"},
            {"chat_id": "123", "user_name": "testuser"},
        )
        assert "listo" in result.output
        assert result.metadata["reminder_id"] == "abc123"

    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set_no_text(self, mock_parse):
        result = execute_tool(
            "reminder_set",
            {"delay": "30 min"},
            {"chat_id": "123"},
        )
        assert "texto" in result.output.lower()

    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set_no_delay(self, mock_parse):
        result = execute_tool(
            "reminder_set",
            {"text": "algo"},
            {"chat_id": "123"},
        )
        assert "tiempo" in result.output.lower()

    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set_no_chat(self, mock_parse):
        result = execute_tool(
            "reminder_set",
            {"text": "algo", "delay": "30 min"},
            {},
        )
        assert "chat" in result.output.lower()

    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set_unparseable_delay(self, mock_parse):
        mock_parse.return_value = None
        result = execute_tool(
            "reminder_set",
            {"text": "algo", "delay": "xyz"},
            {"chat_id": "123"},
        )
        assert "no entendi" in result.output

    @patch("api.tools.reminder_set.schedule_reminder")
    @patch("api.tools.reminder_set.parse_delay")
    def test_reminder_set_schedule_fails(self, mock_parse, mock_schedule):
        mock_parse.return_value = 1800
        mock_schedule.return_value = None
        result = execute_tool(
            "reminder_set",
            {"text": "algo", "delay": "30 min"},
            {"chat_id": "123"},
        )
        assert "no se pudo" in result.output


class TestReminderListTool:
    @patch("api.tools.reminder_list.list_reminders")
    def test_reminder_list_empty(self, mock_list):
        mock_list.return_value = []
        result = execute_tool("reminder_list", {}, {"chat_id": "123"})
        assert "no hay" in result.output.lower()

    @patch("api.tools.reminder_list.list_reminders")
    def test_reminder_list_with_items(self, mock_list):
        mock_list.return_value = [
            {"text": "comprar", "next_run": "2026-01-01", "user_name": "u"},
        ]
        result = execute_tool("reminder_list", {}, {"chat_id": "123"})
        assert "comprar" in result.output

    def test_reminder_list_no_chat(self):
        result = execute_tool("reminder_list", {}, {})
        assert "chat" in result.output.lower()


class TestScheduledTaskSetTool:
    @patch("api.tools.scheduled_task_set.schedule_recurring_task")
    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set(self, mock_parse, mock_schedule):
        mock_parse.return_value = 86400
        mock_schedule.return_value = "t1"
        result = execute_tool(
            "scheduled_task_set",
            {"prompt": "noticias de sonic", "interval": "diario"},
            {"chat_id": "123", "user_name": "u"},
        )
        assert "listo" in result.output
        assert result.metadata["task_id"] == "t1"

    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set_no_prompt(self, mock_parse):
        result = execute_tool(
            "scheduled_task_set",
            {"interval": "diario"},
            {"chat_id": "123"},
        )
        assert "prompt" in result.output.lower()

    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set_no_interval(self, mock_parse):
        result = execute_tool(
            "scheduled_task_set",
            {"prompt": "algo"},
            {"chat_id": "123"},
        )
        assert "frecuencia" in result.output.lower()

    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set_unparseable(self, mock_parse):
        mock_parse.return_value = None
        result = execute_tool(
            "scheduled_task_set",
            {"prompt": "algo", "interval": "xyz"},
            {"chat_id": "123"},
        )
        assert "no entendi" in result.output

    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set_too_frequent(self, mock_parse):
        mock_parse.return_value = 60
        result = execute_tool(
            "scheduled_task_set",
            {"prompt": "algo", "interval": "1m"},
            {"chat_id": "123"},
        )
        assert "minimo" in result.output.lower() or "ansioso" in result.output.lower()

    @patch("api.tools.scheduled_task_set.schedule_recurring_task")
    @patch("api.tools.scheduled_task_set.parse_interval")
    def test_scheduled_task_set_schedule_fails(self, mock_parse, mock_schedule):
        mock_parse.return_value = 86400
        mock_schedule.return_value = None
        result = execute_tool(
            "scheduled_task_set",
            {"prompt": "algo", "interval": "diario"},
            {"chat_id": "123"},
        )
        assert "no se pudo" in result.output


class TestScheduledTaskListTool:
    @patch("api.tools.scheduled_task_list.list_scheduled_tasks")
    def test_scheduled_task_list_empty(self, mock_list):
        mock_list.return_value = []
        result = execute_tool("scheduled_task_list", {}, {"chat_id": "123"})
        assert "no hay" in result.output.lower()

    @patch("api.tools.scheduled_task_list.list_scheduled_tasks")
    def test_scheduled_task_list_with_items(self, mock_list):
        mock_list.return_value = [
            {"prompt": "noticias sonic", "interval_seconds": 86400, "id": "t1"},
        ]
        result = execute_tool("scheduled_task_list", {}, {"chat_id": "123"})
        assert "noticias sonic" in result.output

    def test_scheduled_task_list_no_chat(self):
        result = execute_tool("scheduled_task_list", {}, {})
        assert "chat" in result.output.lower()
