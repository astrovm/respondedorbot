"""Tests for the /recordame and /remindme command handler."""

from __future__ import annotations

from unittest.mock import patch

from api.index import recordame_command


class TestRecordameCommand:
    def test_empty_text(self):
        result = recordame_command("")
        assert "decime" in result.lower()

    def test_whitespace_only(self):
        result = recordame_command("   ")
        assert "decime" in result.lower()

    @patch("api.tools.reminder_scheduler.schedule_reminder")
    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_simple_reminder(self, mock_parse, mock_schedule):
        mock_parse.return_value = 1800
        mock_schedule.return_value = "r1"
        result = recordame_command("comprar pizza en 30 min", "123", "user")
        assert "listo" in result.lower()
        assert "30 minuto" in result

    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_unparseable_delay(self, mock_parse):
        mock_parse.return_value = None
        result = recordame_command("comprar pizza en xyz", "123", "user")
        assert "no entendi" in result.lower()

    @patch("api.tools.reminder_scheduler.schedule_reminder")
    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_schedule_fails(self, mock_parse, mock_schedule):
        mock_parse.return_value = 1800
        mock_schedule.return_value = None
        result = recordame_command("algo en 30 min", "123", "user")
        assert "no se pudo" in result.lower()

    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_no_chat_id(self, mock_parse):
        mock_parse.return_value = 1800
        result = recordame_command("algo en 30 min", "", "user")
        assert "chat" in result.lower()

    @patch("api.tools.reminder_scheduler.schedule_reminder")
    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_hours(self, mock_parse, mock_schedule):
        mock_parse.return_value = 7200
        mock_schedule.return_value = "r2"
        result = recordame_command("algo en 2 horas", "123", "user")
        assert "2 hora" in result

    @patch("api.tools.reminder_scheduler.schedule_reminder")
    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_days(self, mock_parse, mock_schedule):
        mock_parse.return_value = 86400
        mock_schedule.return_value = "r3"
        result = recordame_command("algo en 1 dia", "123", "user")
        assert "1 dia" in result

    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_single_word_text_no_delay(self, mock_parse):
        mock_parse.return_value = None
        result = recordame_command("algo", "123", "user")
        assert "no entendi" in result.lower()

    @patch("api.tools.reminder_scheduler.schedule_reminder")
    @patch("api.tools.reminder_scheduler.parse_delay")
    def test_two_word_fallback(self, mock_parse, mock_schedule):
        mock_parse.side_effect = [None, 3600]
        mock_schedule.return_value = "r4"
        result = recordame_command("comprar pizza 1h", "123", "user")
        assert "listo" in result.lower()
