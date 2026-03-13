import json
from pathlib import Path


def test_vercel_config_sets_120_second_runtime_for_api_index():
    vercel_config_path = Path(__file__).resolve().parents[1] / "vercel.json"

    config = json.loads(vercel_config_path.read_text())

    assert config["functions"]["api/index.py"]["maxDuration"] == 120
    assert config["rewrites"] == [{"source": "/(.*)", "destination": "/api/index"}]
