"""Tests for the reminder scheduler (parse_delay, parse_interval, etc.)."""

from __future__ import annotations

from api.tools.reminder_scheduler import parse_delay, parse_interval


class TestParseDelay:
    def test_minutes(self):
        assert parse_delay("30 min") == 1800

    def test_minutos(self):
        assert parse_delay("30 minutos") == 1800

    def test_hours(self):
        assert parse_delay("2 horas") == 7200

    def test_h(self):
        assert parse_delay("3h") == 10800

    def test_days(self):
        assert parse_delay("1 dia") == 86400

    def test_dias(self):
        assert parse_delay("2 dias") == 172800

    def test_seconds(self):
        assert parse_delay("60 segundos") == 60

    def test_bare_number_defaults_minutes(self):
        assert parse_delay("30") == 1800

    def test_bare_number_too_large(self):
        assert parse_delay("2000") is None

    def test_empty(self):
        assert parse_delay("") is None

    def test_garbage(self):
        assert parse_delay("asdf") is None

    def test_combined(self):
        assert parse_delay("1 hora 30 min") == 5400


class TestParseInterval:
    def test_diario(self):
        assert parse_interval("diario") == 86400

    def test_daily(self):
        assert parse_interval("daily") == 86400

    def test_cada_dia(self):
        assert parse_interval("cada dia") == 86400

    def test_semanal(self):
        assert parse_interval("semanal") == 86400 * 7

    def test_weekly(self):
        assert parse_interval("weekly") == 86400 * 7

    def test_hourly(self):
        assert parse_interval("cada hora") == 3600

    def test_cada_6_horas(self):
        assert parse_interval("cada 6 horas") == 21600

    def test_every_30_min(self):
        assert parse_interval("every 30 min") == 1800

    def test_cada_2_dias(self):
        assert parse_interval("cada 2 dias") == 172800

    def test_empty(self):
        assert parse_interval("") is None

    def test_garbage(self):
        assert parse_interval("xyz") is None
