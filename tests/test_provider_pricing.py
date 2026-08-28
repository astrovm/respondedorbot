from datetime import UTC, datetime
from unittest.mock import MagicMock

from api.providers.pricing import (
    clear_openrouter_pricing_cache,
    get_openrouter_provider_pricing,
    needs_published_provider_pricing,
)


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_openrouter_provider_pricing_uses_exact_routed_endpoint():
    clear_openrouter_pricing_cache()
    request_get = MagicMock(
        return_value=_response(
            {
                "data": {
                    "endpoints": [
                        {
                            "provider_name": "DeepInfra",
                            "pricing": {
                                "prompt": "0.00000008",
                                "completion": "0.00000018",
                                "input_cache_read": "0.000000016",
                            },
                        },
                        {
                            "provider_name": "Relace",
                            "pricing": {
                                "prompt": "0.00000006",
                                "completion": "0.00000012",
                            },
                        },
                    ]
                }
            }
        )
    )

    pricing = get_openrouter_provider_pricing(
        "deepseek/deepseek-v4-flash-0731",
        "deepinfra",
        base_url="https://openrouter.ai/api/v1",
        request_get=request_get,
    )

    assert pricing == {
        "input_per_million": 80_000,
        "output_per_million": 180_000,
        "cached_input_per_million": 16_000,
    }
    request_get.assert_called_once_with(
        "https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints",
        timeout=3.0,
    )


def test_openrouter_provider_pricing_applies_current_utc_override():
    clear_openrouter_pricing_cache()
    request_get = MagicMock(
        return_value=_response(
            {
                "data": {
                    "endpoints": [
                        {
                            "provider_name": "DeepSeek",
                            "pricing": {
                                "prompt": "0.00000022",
                                "completion": "0.00000066",
                                "overrides": [
                                    {
                                        "utc_days": ["friday"],
                                        "utc_start": 100,
                                        "utc_end": 400,
                                        "prompt": "0.00000044",
                                        "completion": "0.00000132",
                                    }
                                ],
                            },
                        }
                    ]
                }
            }
        )
    )

    pricing = get_openrouter_provider_pricing(
        "deepseek/deepseek-v4-flash-0731",
        "DeepSeek",
        base_url="https://openrouter.ai/api/v1",
        request_get=request_get,
        now=datetime(2026, 8, 28, 2, 0, tzinfo=UTC),
    )

    assert pricing == {
        "input_per_million": 440_000,
        "output_per_million": 1_320_000,
    }


def test_openrouter_provider_pricing_rejects_ambiguous_provider_variants():
    clear_openrouter_pricing_cache()
    request_get = MagicMock(
        return_value=_response(
            {
                "data": {
                    "endpoints": [
                        {
                            "provider_name": "Example",
                            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                        },
                        {
                            "provider_name": "Example",
                            "pricing": {"prompt": "0.0000002", "completion": "0.0000003"},
                        },
                    ]
                }
            }
        )
    )


def test_openrouter_provider_pricing_selects_returned_service_tier():
    clear_openrouter_pricing_cache()
    request_get = MagicMock(
        return_value=_response(
            {
                "data": {
                    "endpoints": [
                        {
                            "provider_name": "Google AI Studio",
                            "tag": "google-ai-studio",
                            "pricing": {"prompt": "0.00000025", "completion": "0.0000015"},
                        },
                        {
                            "provider_name": "Google AI Studio",
                            "tag": "google-ai-studio/flex",
                            "pricing": {"prompt": "0.000000125", "completion": "0.00000075"},
                        },
                        {
                            "provider_name": "Google AI Studio",
                            "tag": "google-ai-studio/priority",
                            "pricing": {"prompt": "0.00000045", "completion": "0.0000027"},
                        },
                    ]
                }
            }
        )
    )

    assert get_openrouter_provider_pricing(
        "google/gemini-3.1-flash-lite-preview",
        "Google AI Studio",
        "flex",
        base_url="https://openrouter.ai/api/v1",
        request_get=request_get,
    ) == {
        "input_per_million": 125_000,
        "output_per_million": 750_000,
    }
    assert get_openrouter_provider_pricing(
        "google/gemini-3.1-flash-lite-preview",
        "Google AI Studio",
        "priority",
        base_url="https://openrouter.ai/api/v1",
        request_get=request_get,
    ) == {
        "input_per_million": 450_000,
        "output_per_million": 2_700_000,
    }
    assert get_openrouter_provider_pricing(
        "google/gemini-3.1-flash-lite-preview",
        "Google AI Studio",
        "default",
        base_url="https://openrouter.ai/api/v1",
        request_get=request_get,
    ) == {
        "input_per_million": 250_000,
        "output_per_million": 1_500_000,
    }
    request_get.assert_called_once()

    assert (
        get_openrouter_provider_pricing(
            "author/model",
            "Example",
            base_url="https://openrouter.ai/api/v1",
            request_get=request_get,
        )
        is None
    )


def test_published_pricing_is_only_needed_without_positive_provider_cost():
    assert needs_published_provider_pricing(None) is True
    assert needs_published_provider_pricing({"cost": 0}) is True
    assert needs_published_provider_pricing({"cost": "0.001"}) is False
    assert (
        needs_published_provider_pricing(
            {
                "cost": "0.001",
                "cost_details": {"upstream_inference_cost": 0},
            }
        )
        is True
    )
    assert (
        needs_published_provider_pricing({"cost_details": {"upstream_inference_cost": "0.002"}})
        is False
    )
