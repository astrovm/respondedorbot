"""Tests for the tool registry system."""

from __future__ import annotations


from api.tools.registry import (
    ToolResult,
    parse_tool_call_arguments,
    register_tool,
    execute_tool,
    get_all_tool_schemas,
)


def test_register_and_execute_tool():
    def my_executor(params, context):
        return ToolResult(output=f"got {params.get('x', '')}")

    register_tool(
        name="test_tool_reg",
        description="A test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        executor=my_executor,
    )

    result = execute_tool("test_tool_reg", {"x": "hello"})
    assert result.output == "got hello"


def test_execute_unknown_tool():
    result = execute_tool("nonexistent_tool_xyz", {})
    assert "Unknown tool" in result.output


def test_execute_tool_handles_exception():
    def bad_executor(params, context):
        raise ValueError("boom")

    register_tool(
        name="test_bad_tool",
        description="A failing tool",
        parameters={"type": "object", "properties": {}},
        executor=bad_executor,
    )

    result = execute_tool("test_bad_tool", {})
    assert "error" in result.output


def test_parse_tool_call_arguments_dict():
    assert parse_tool_call_arguments({"a": 1}) == {"a": 1}


def test_parse_tool_call_arguments_json_string():
    assert parse_tool_call_arguments('{"a": 1}') == {"a": 1}


def test_parse_tool_call_arguments_invalid_string():
    assert parse_tool_call_arguments("not json") == {}


def test_parse_tool_call_arguments_other_type():
    assert parse_tool_call_arguments(42) == {}


def test_get_all_tool_schemas_returns_list():
    schemas = get_all_tool_schemas()
    assert isinstance(schemas, list)
    for s in schemas:
        assert s["type"] == "function"
        assert "function" in s
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


def test_tool_result_metadata_default():
    r = ToolResult(output="ok")
    assert r.metadata == {}


def test_tool_result_metadata_custom():
    r = ToolResult(output="ok", metadata={"id": "abc"})
    assert r.metadata["id"] == "abc"
