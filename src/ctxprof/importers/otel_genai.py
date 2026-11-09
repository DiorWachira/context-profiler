"""OpenTelemetry GenAI spans -> traces.

The GenAI semantic conventions are still in *development* status, so every attribute name the
importer depends on lives in ``MAPPING`` below. When the conventions change, edit this table
(and its test) only.

Input: a JSON file holding a list of spans, an OTLP-style ``{"resourceSpans": ...}`` export, or
JSONL with one span per line. Content capture must be enabled in the instrumentation
(``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``), otherwise messages are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import Message, Request, Response, Tool, ToolCall, Trace

# Attribute names follow the GenAI conventions' JSON-attribute form (gen_ai.input.messages etc.).
# Check them against the current spec before relying on a new instrumentation library.
MAPPING_VERSION = "otel-genai-semconv/development"
MAPPING = {
    "operation": "gen_ai.operation.name",
    "model": "gen_ai.request.model",
    "system_instructions": "gen_ai.system_instructions",
    "input_messages": "gen_ai.input.messages",
    "output_messages": "gen_ai.output.messages",
    "tool_definitions": "gen_ai.tool.definitions",
    "params": {
        "temperature": "gen_ai.request.temperature",
        "max_tokens": "gen_ai.request.max_tokens",
        "top_p": "gen_ai.request.top_p",
    },
    "chat_operations": ("chat", "generate_content", "text_completion"),
}


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    raw = span.get("attributes", {})
    if isinstance(raw, list):  # OTLP JSON: [{"key": k, "value": {"stringValue": v}}]
        out = {}
        for kv in raw:
            val = kv.get("value", {})
            out[kv["key"]] = next(iter(val.values())) if isinstance(val, dict) and val else val
        return out
    return dict(raw)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parts_text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    texts = []
    for p in parts or []:
        if isinstance(p, dict) and p.get("type") in ("text", None):
            texts.append(str(p.get("content", p.get("text", ""))))
        elif isinstance(p, dict) and p.get("type") == "tool_call_response":
            texts.append(json.dumps(p.get("response"), ensure_ascii=False))
    return "\n".join(t for t in texts if t)


def _span_to_trace(span: dict[str, Any]) -> Trace | None:
    a = _attrs(span)
    if a.get(MAPPING["operation"]) not in MAPPING["chat_operations"]:
        return None
    messages: list[Message] = []
    sys_inst = _json(a.get(MAPPING["system_instructions"]))
    if sys_inst:
        messages.append(Message(role="system", content=_parts_text(sys_inst)))
    for m in _json(a.get(MAPPING["input_messages"])) or []:
        role = m.get("role", "user")
        extra: dict[str, Any] = {}
        calls = [p for p in m.get("parts", []) if isinstance(p, dict) and p.get("type") == "tool_call"]
        if calls:
            extra["tool_calls"] = [{"id": p.get("id"), "function": {"name": p.get("name"),
                                                                   "arguments": p.get("arguments")}} for p in calls]
        responses = [p for p in m.get("parts", []) if isinstance(p, dict) and p.get("type") == "tool_call_response"]
        if responses and responses[0].get("id"):
            extra["tool_call_id"] = responses[0]["id"]
        messages.append(Message(role=role if role in ("system", "user", "assistant", "tool") else "user",
                                content=_parts_text(m.get("parts")), extra=extra))
    tools = []
    for t in _json(a.get(MAPPING["tool_definitions"])) or []:
        fn = t.get("function", t)
        tools.append(Tool(name=fn.get("name", ""), description=fn.get("description", "") or "",
                          parameters=fn.get("parameters") or {}))
    params = {k: a[v] for k, v in MAPPING["params"].items() if v in a}
    out_text, out_calls = "", []
    for m in _json(a.get(MAPPING["output_messages"])) or []:
        out_text += _parts_text(m.get("parts"))
        for p in m.get("parts", []):
            if isinstance(p, dict) and p.get("type") == "tool_call":
                args = p.get("arguments") or {}
                parsed = _json(args) if isinstance(args, str) else args
                out_calls.append(ToolCall(name=p.get("name", ""), arguments=parsed))
    if not messages:
        return None
    trace_id = str(span.get("spanId") or span.get("span_id") or span.get("context", {}).get("span_id"))
    return Trace(
        id=trace_id,
        request=Request(model=str(a.get(MAPPING["model"], "unknown")), params=params, messages=messages, tools=tools),
        response=Response(text=out_text, tool_calls=out_calls),
        meta={"otel_trace_id": span.get("traceId") or span.get("trace_id"), "mapping": MAPPING_VERSION},
    )


def _iter_spans(data: Any):
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and "resourceSpans" in data:
        for rs in data["resourceSpans"]:
            for ss in rs.get("scopeSpans", []):
                yield from ss.get("spans", [])
    elif isinstance(data, dict):
        yield data


def load_otel(path: str | Path) -> list[Trace]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        spans = list(_iter_spans(data))
    except json.JSONDecodeError:
        spans = [json.loads(line) for line in text.splitlines() if line.strip()]
    traces = [t for t in (_span_to_trace(s) for s in spans) if t is not None]
    return traces
