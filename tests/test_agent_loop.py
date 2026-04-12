from tests.support import *  # noqa: F401,F403


def test_run_agent_loop_executes_tool_then_returns_final_answer():
    from api.agent_loop import run_agent_loop

    billing_segment = {"provider": "test", "input_tokens": 3, "output_tokens": 1}
    model_outputs = [
        {
            "type": "tool_calls",
            "tool_calls": [
                {"name": "lookup_weather", "arguments": {"city": "Buenos Aires"}}
            ],
            "billing_segment": billing_segment,
        },
        {
            "type": "final",
            "text": "Hace calor.",
            "billing_segment": None,
        },
    ]
    model_calls = []
    tool_calls = []

    def model_call_fn(transcript):
        model_calls.append(transcript)
        return model_outputs[len(model_calls) - 1]

    def lookup_weather(arguments):
        tool_calls.append(arguments)
        return {"forecast": "soleado", "temperature_c": 27}

    result = run_agent_loop(
        system_message={"role": "system", "content": "sos util"},
        conversation=[{"role": "user", "content": "como esta el clima?"}],
        model_call_fn=model_call_fn,
        tools={"lookup_weather": lookup_weather},
        max_iterations=3,
        max_tool_calls=3,
    )

    assert result["text"] == "Hace calor."
    assert result["iterations"] == 2
    assert result["tool_calls"] == 1
    assert result["final_reason"] == "final"
    assert result["billing_segments"] == [billing_segment]
    assert tool_calls == [{"city": "Buenos Aires"}]
    assert model_calls[0] == [
        {"role": "system", "content": "sos util"},
        {"role": "user", "content": "como esta el clima?"},
    ]
    assert model_calls[1][-2:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"name": "lookup_weather", "arguments": {"city": "Buenos Aires"}}
            ],
        },
        {
            "role": "tool",
            "name": "lookup_weather",
            "content": {
                "forecast": "soleado",
                "temperature_c": 27,
            },
        },
    ]
    assert result["transcript"][-2:] == model_calls[1][-2:]


def test_run_agent_loop_stops_when_iteration_limit_is_hit():
    from api.agent_loop import run_agent_loop

    def model_call_fn(_transcript):
        return {
            "type": "tool_calls",
            "tool_calls": [{"name": "ping", "arguments": {}}],
            "billing_segment": {"provider": "test", "input_tokens": 1},
        }

    result = run_agent_loop(
        system_message={"role": "system", "content": "sos util"},
        conversation=[{"role": "user", "content": "segui"}],
        model_call_fn=model_call_fn,
        tools={"ping": lambda _arguments: {"ok": True}},
        max_iterations=1,
        max_tool_calls=5,
    )

    assert result["text"] == ""
    assert result["iterations"] == 1
    assert result["tool_calls"] == 1
    assert result["final_reason"] == "max_iterations"
    assert result["billing_segments"] == [{"provider": "test", "input_tokens": 1}]
    assert result["transcript"] == [
        {"role": "system", "content": "sos util"},
        {"role": "user", "content": "segui"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "ping", "arguments": {}}],
        },
        {"role": "tool", "name": "ping", "content": {"ok": True}},
    ]
