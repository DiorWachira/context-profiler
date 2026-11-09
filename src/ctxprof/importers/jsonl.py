"""JSONL traces: one request per line.

Accepted shapes per line (OpenAI-style and Anthropic-style are both normalised)::

    {"id": "...", "model": "...", "params": {...},
     "system": "optional top-level system prompt (Anthropic style)",
     "messages": [{"role": "user", "content": "..." | [{"type": "text", "text": "..."}]}],
     "tools": [{"type": "function", "function": {"name", "description", "parameters"}}
               | {"name", "description", "input_schema" | "parameters"}],
     "response": "text" | {"text"|"content": ..., "tool_calls": [...]}
               | {"choices": [{"message": {...}}]}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import Message, Request, Response, Tool, ToolCall, Trace

_PASS_THROUGH = ("tool_calls", "tool_call_id", "name")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text", "output_text"):
                    parts.append(str(part.get("text", part.get("content", ""))))
                elif part.get("type") == "tool_result":
                    parts.append(_content_text(part.get("content")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool(raw: dict[str, Any]) -> Tool:
    fn = raw.get("function", raw)
    return Tool(
        name=fn["name"],
        description=fn.get("description", "") or "",
        parameters=fn.get("parameters") or fn.get("input_schema") or {},
    )


def _args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"_": parsed}
        except json.JSONDecodeError:
            return {"_raw": value}
    return {}


def _tool_calls(raw: Any) -> list[ToolCall]:
    calls = []
    for c in raw or []:
        fn = c.get("function", c)
        calls.append(ToolCall(name=fn.get("name", ""), arguments=_args(fn.get("arguments", fn.get("input")))))
    return calls


def _response(raw: Any) -> Response:
    if raw is None:
        return Response()
    if isinstance(raw, str):
        return Response(text=raw)
    if "choices" in raw:
        raw = raw["choices"][0].get("message", {})
    text = _content_text(raw.get("text", raw.get("content")))
    calls = raw.get("tool_calls")
    if calls is None and isinstance(raw.get("content"), list):  # Anthropic tool_use blocks
        calls = [
            {"name": b.get("name"), "arguments": b.get("input", {})}
            for b in raw["content"] if isinstance(b, dict) and b.get("type") == "tool_use"
        ]
    return Response(text=text, tool_calls=_tool_calls(calls))


def parse_record(rec: dict[str, Any], fallback_id: str) -> Trace:
    messages: list[Message] = []
    if rec.get("system"):
        messages.append(Message(role="system", content=_content_text(rec["system"])))
    for m in rec.get("messages", []):
        extra = {k: m[k] for k in _PASS_THROUGH if k in m}
        messages.append(Message(role=m["role"], content=_content_text(m.get("content")), extra=extra))
    params = dict(rec.get("params", {}))
    for key in ("temperature", "max_tokens", "top_p"):
        if key in rec and key not in params:
            params[key] = rec[key]
    request = Request(
        model=rec.get("model", "unknown"),
        params=params,
        messages=messages,
        tools=[_tool(t) for t in rec.get("tools", []) or []],
    )
    trace_id = str(rec.get("id") or fallback_id)
    meta = {k: v for k, v in rec.items() if k not in {"id", "model", "params", "system", "messages", "tools",
                                                       "response", "temperature", "max_tokens", "top_p"}}
    return Trace(id=trace_id, request=request, response=_response(rec.get("response")), meta=meta)


def load_jsonl(path: str | Path) -> list[Trace]:
    path = Path(path)
    traces = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fallback = hashlib.sha1(f"{path.name}:{line_no}".encode()).hexdigest()[:12]
            traces.append(parse_record(rec, fallback))
    return traces
