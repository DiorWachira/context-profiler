import json

from ctxprof.importers import load_jsonl, load_otel


def test_jsonl_openai_and_anthropic_shapes(tmp_path):
    lines = [
        {"id": "a", "model": "gpt-x", "temperature": 0,
         "messages": [{"role": "system", "content": "S"},
                      {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
         "tools": [{"type": "function",
                    "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}],
         "response": {"choices": [{"message": {"content": "ok", "tool_calls": [
             {"id": "1", "function": {"name": "f", "arguments": "{\"q\": 1}"}}]}}]}},
        {"model": "claude-x", "system": "Be nice",
         "messages": [{"role": "user", "content": "yo"}],
         "tools": [{"name": "g", "description": "", "input_schema": {"type": "object"}}],
         "response": {"content": [{"type": "text", "text": "hey"},
                                  {"type": "tool_use", "name": "g", "input": {"a": 2}}]},
         "intent": "status"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n", encoding="utf-8")
    a, b = load_jsonl(p)
    assert a.id == "a" and a.request.params == {"temperature": 0}
    assert a.request.messages[1].content == "hi"
    assert a.response.tool_calls[0].name == "f" and a.response.tool_calls[0].arguments == {"q": 1}
    assert b.request.messages[0].role == "system" and b.request.messages[0].content == "Be nice"
    assert b.request.tools[0].parameters == {"type": "object"}
    assert b.response.text == "hey" and b.response.tool_calls[0].arguments == {"a": 2}
    assert b.meta == {"intent": "status"} and len(b.id) == 12


def test_otel_genai_span(tmp_path):
    span = {
        "spanId": "abc123", "traceId": "t9",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "llama",
            "gen_ai.request.temperature": 0,
            "gen_ai.system_instructions": json.dumps([{"type": "text", "content": "## Role\nHelp."}]),
            "gen_ai.input.messages": json.dumps([{"role": "user", "parts": [{"type": "text", "content": "status?"}]}]),
            "gen_ai.output.messages": json.dumps([{"role": "assistant", "parts": [
                {"type": "text", "content": "Shortlisted."},
                {"type": "tool_call", "name": "get_status", "arguments": {"id": "JB-1"}}]}]),
            "gen_ai.tool.definitions": json.dumps([{"type": "function", "name": "get_status", "parameters": {}}]),
        },
    }
    other = {"spanId": "x", "attributes": {"gen_ai.operation.name": "embeddings"}}
    p = tmp_path / "spans.json"
    p.write_text(json.dumps([span, other]), encoding="utf-8")
    [t] = load_otel(p)
    assert t.id == "abc123" and t.request.model == "llama" and t.request.params == {"temperature": 0}
    assert [m.role for m in t.request.messages] == ["system", "user"]
    assert t.request.tools[0].name == "get_status"
    assert t.response.text == "Shortlisted." and t.response.tool_calls[0].arguments == {"id": "JB-1"}
