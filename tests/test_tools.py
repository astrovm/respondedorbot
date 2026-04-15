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
            {"text": "algo", "delay_seconds": 86400 * 31},
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
