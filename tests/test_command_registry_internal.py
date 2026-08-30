import json
from pathlib import Path

from api.bot import command_registry as command_registry_module
from api.bot.command_registry import build_command_registry, parse_command


def _command_contract():
    path = Path(__file__).parents[1] / "contracts" / "command_parsing.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_command_registry_reuses_alias_metadata():
    def prices(_: str) -> str:
        return "ok"

    registry = build_command_registry(
        {
            "ask_ai": prices,
            "config_command": lambda: "config",
            "language_command": lambda: "language",
            "convert_base": prices,
            "select_random": prices,
            "get_prices": prices,
            "get_crypto_prices": prices,
            "get_weather": prices,
            "get_dollar_rates": lambda: "usd",
            "get_oil_price": lambda: "oil",
            "get_stock_prices": prices,
            "summary_command": lambda: "resumen",
            "get_polymarket_global_elections": lambda: "election",
            "get_rulo": lambda: "rulo",
            "get_devo": prices,
            "powerlaw": lambda: "powerlaw",
            "rainbow": lambda: "rainbow",
            "satoshi": lambda: "satoshi",
            "get_timestamp": lambda: "time",
            "convert_to_command": prices,
            "get_instance_name": lambda: "instance",
            "get_help": lambda: "help",
            "handle_transcribe": lambda: "transcribe",
            "handle_bcra_variables": lambda: "bcra",
            "topup_command": lambda: "topup",
            "balance_command": lambda: "balance",
            "charges_command": prices,
            "printcredits_command": lambda x: "printcredits",
            "creditlog_command": lambda x: "creditlog",
            "transfer_command": prices,
            "get_good_morning": lambda: "gm",
            "get_good_night": lambda: "gn",
            "task_command": lambda _: "tareas",
        }
    )

    assert registry["/prices"] == registry["/price"]
    assert registry["/prices"] == registry["/precios"]
    assert registry["/prices"] == registry["/c"]
    assert registry["/crypto"][0] == prices
    assert registry["/transcribe"] == registry["/describe"]
    assert registry["/resumen"] == registry["/tldr"]
    assert registry["/eleccion"] == registry["/elecciones"]
    assert registry["/eleccion"] == registry["/election"]
    assert registry["/eleccion"] == registry["/elections"]
    assert registry["/config"] == registry["/configs"]
    assert registry["/config"] == registry["/settings"]
    assert registry["/language"] == registry["/idioma"]
    assert registry["/tarea"] == registry["/tareas"]
    assert registry["/tarea"] == registry["/task"]
    assert registry["/tarea"] == registry["/tasks"]
    assert registry["/charges"] == registry["/history"]
    assert registry["/charges"] == registry["/gastos"]
    assert "/purgeailog" not in registry
    assert "/updatecommands" not in registry


def test_parse_command_normalizes_hangul_filler_alias():
    command, text = parse_command("/ㅤ hola", "@gordo")
    assert command == "/ask"
    assert text == "hola"


def test_python_command_parser_matches_shared_contract(monkeypatch):
    monkeypatch.delenv("RUST_COMMAND_PARSING_ENABLED", raising=False)
    for case in _command_contract()["cases"]:
        assert parse_command(case["input"], case["bot_name"]) == (
            case["command"],
            case["message_text"],
        )


def test_enabled_rust_command_parser_is_authoritative(monkeypatch):
    class FakeRustCommandParser:
        def parse_command(self, message_text, bot_name):
            assert message_text == "/ask hola"
            assert bot_name == "@gordo"
            return "/rust", "parsed"

    monkeypatch.setenv("RUST_COMMAND_PARSING_ENABLED", "true")
    monkeypatch.setattr(
        command_registry_module,
        "_load_rust_command_parser",
        lambda: FakeRustCommandParser(),
    )

    assert parse_command("/ask hola", "@gordo") == ("/rust", "parsed")


def test_rust_command_parser_failure_falls_back(monkeypatch, caplog):
    class FailingRustCommandParser:
        def parse_command(self, _message_text, _bot_name):
            raise RuntimeError("synthetic failure")

    monkeypatch.setenv("RUST_COMMAND_PARSING_ENABLED", "1")
    monkeypatch.setattr(
        command_registry_module,
        "_load_rust_command_parser",
        lambda: FailingRustCommandParser(),
    )

    assert parse_command("/ASK hola", "@gordo") == ("/ask", "hola")
    assert len(caplog.records) == 1
