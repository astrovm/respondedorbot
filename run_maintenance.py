#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path


def _load_dotenv() -> None:
    env_paths = [
        Path.home() / "respondedorbot" / ".env",
        Path(__file__).with_name(".env"),
    ]
    env_path = next((path for path in env_paths if path.is_file()), None)
    if env_path is None:
        return

    try:
        dotenv = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv, "load_dotenv")
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    with env_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> int:
    _load_dotenv()

    from api.services.maintenance import run_maintenance

    print(json.dumps(run_maintenance(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
