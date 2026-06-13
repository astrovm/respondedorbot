from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1] / "api"


def test_api_root_contains_only_composition_modules() -> None:
    root_modules = {path.name for path in API_ROOT.glob("*.py")}

    assert root_modules == {"__init__.py", "application.py", "index.py"}


def test_api_domains_are_explicit_packages() -> None:
    expected_domains = {
        "admin",
        "ai",
        "billing",
        "bot",
        "cache",
        "core",
        "links",
        "markets",
        "media",
        "memory",
        "providers",
        "services",
        "storage",
        "tasks",
        "tools",
        "utils",
    }

    package_domains = {
        path.name
        for path in API_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert package_domains == expected_domains
