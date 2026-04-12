from typing import Any, Callable, Dict, List, Mapping


def run_agent_loop(
    *,
    system_message: Mapping[str, Any],
    conversation: List[Mapping[str, Any]],
    model_call_fn: Callable[[List[Dict[str, Any]]], Mapping[str, Any]],
    tools: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    max_iterations: int,
    max_tool_calls: int,
) -> Dict[str, Any]:
    transcript = [dict(system_message), *[dict(message) for message in conversation]]
    billing_segments = []
    iterations = 0
    total_tool_calls = 0

    while iterations < max_iterations:
        iterations += 1
        result = dict(model_call_fn(list(transcript)))
        billing_segment = result.get("billing_segment")
        if billing_segment is not None:
            billing_segments.append(billing_segment)

        if result.get("type") == "final":
            transcript.append({"role": "assistant", "content": result.get("text", "")})
            return {
                "text": result.get("text", ""),
                "iterations": iterations,
                "tool_calls": total_tool_calls,
                "final_reason": "final",
                "billing_segments": billing_segments,
                "transcript": transcript,
            }

        tool_calls = list(result.get("tool_calls", []))
        if total_tool_calls + len(tool_calls) > max_tool_calls:
            return {
                "text": "",
                "iterations": iterations,
                "tool_calls": total_tool_calls,
                "final_reason": "max_tool_calls",
                "billing_segments": billing_segments,
                "transcript": transcript,
            }

        transcript.append(
            {"role": "assistant", "content": None, "tool_calls": tool_calls}
        )

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_result = tools[tool_name](tool_call.get("arguments", {}))
            transcript.append(
                {"role": "tool", "name": tool_name, "content": tool_result}
            )
            total_tool_calls += 1

    final_reason = "max_iterations"
    if total_tool_calls >= max_tool_calls:
        final_reason = "max_tool_calls"

    return {
        "text": "",
        "iterations": iterations,
        "tool_calls": total_tool_calls,
        "final_reason": final_reason,
        "billing_segments": billing_segments,
        "transcript": transcript,
    }
