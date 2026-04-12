from api.command_registry import build_command_registry, parse_command


def test_build_command_registry_reuses_alias_metadata():
    def prices(_: str) -> str:
        return "ok"

    registry = build_command_registry(
        {
            "ask_ai": prices,
            "config_command": lambda: "config",
            "convert_base": prices,
            "select_random": prices,
            "get_prices": prices,
            "get_dollar_rates": lambda: "usd",
            "get_oil_price": lambda: "oil",
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
            "printcredits_command": lambda x: "printcredits",
            "creditlog_command": lambda x: "creditlog",
            "transfer_command": prices,
            "get_good_morning": lambda: "gm",
            "get_good_night": lambda: "gn",
        }
    )

    assert registry["/prices"] == registry["/price"]
    assert registry["/prices"] == registry["/precios"]
    assert registry["/transcribe"] == registry["/describe"]
    assert "/purgeailog" not in registry
    assert "/updatecommands" not in registry


def test_parse_command_normalizes_hangul_filler_alias():
    command, text = parse_command("/ㅤ hola", "@gordo")
    assert command == "/ask"
    assert text == "hola"
