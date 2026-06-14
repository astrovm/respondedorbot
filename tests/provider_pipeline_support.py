from types import SimpleNamespace

from tests.support import *
from api.ai.pipeline import strip_markdown_formatting as strip_markdown_formatting
from api.providers.base import ProviderResult as ProviderResult


def _build_provider_runtime(*, client, tool_runtime=None):
    from api.ai.pricing import AIUsageResult
    from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
    from api.tools.runtime import ToolRuntime

    runtime_tool_runtime = tool_runtime or ToolRuntime(print_fn=lambda *_args: None)
    return ProviderRuntime(
        ProviderRuntimeDeps(
            get_client=lambda: client,
            admin_report=MagicMock(),
            increment_request_count=MagicMock(),
            build_web_search_tool=lambda: {
                "type": "openrouter:web_search",
                "parameters": {
                    "engine": "firecrawl",
                    "max_results": 10,
                    "max_total_results": 30,
                },
            },
            build_usage_result=lambda **kwargs: AIUsageResult(
                kind=kwargs["kind"],
                text=kwargs["text"],
                model=kwargs["model"],
                usage={},
                metadata=kwargs.get("metadata") or {},
            ),
            extract_usage_map=lambda response: getattr(response, "usage", {}),
            primary_model="test-model",
            max_tool_rounds=5,
        ),
        runtime_tool_runtime,
    )


def _build_chat_response(*, text, finish_reason="stop", annotations=None, usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=text,
                    annotations=annotations or [],
                    tool_calls=[],
                ),
            )
        ],
        usage=usage or {"prompt_tokens": 0, "completion_tokens": 0},
    )
