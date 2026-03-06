from api.command_registry import build_command_registry, parse_command


def test_build_command_registry_reuses_alias_metadata():
    def prices(_: str) -> str:
        return "ok"

    registry = build_command_registry(
        {
            "ask_ai": prices,
            "show_agent_thoughts": lambda: "agent",
            "config_command": lambda: "config",
            "convert_base": prices,
            "select_random": prices,
            "get_prices": prices,
            "get_dollar_rates": lambda: "usd",
            "get_polymarket_argentina_election": lambda: "election",
            "get_rulo": lambda: "rulo",
            "get_devo": prices,
            "powerlaw": lambda: "powerlaw",
            "rainbow": lambda: "rainbow",
            "satoshi": lambda: "satoshi",
            "get_timestamp": lambda: "time",
            "convert_to_command": prices,
            "search_command": prices,
            "get_instance_name": lambda: "instance",
            "get_help": lambda: "help",
            "handle_transcribe": lambda: "transcribe",
            "handle_bcra_variables": lambda: "bcra",
            "topup_command": lambda: "topup",
            "balance_command": lambda: "balance",
            "transfer_command": prices,
        }
    )

    assert registry["/prices"] == registry["/price"]
    assert registry["/prices"] == registry["/precios"]


def test_parse_command_normalizes_hangul_filler_alias():
    command, text = parse_command("/ㅤ hola", "@gordo")
    assert command == "/ask"
    assert text == "hola"
