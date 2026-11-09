"""Core data model: a trace (one recorded LLM request) and the segments it is split into."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

SegmentKind = Literal["system", "tool", "retrieved", "history", "user"]
Source = Literal["trusted", "untrusted"]
ToolPart = Literal["description", "parameters"]
KIND_ORDER: tuple[SegmentKind, ...] = ("system", "tool", "retrieved", "history", "user")


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    # Opaque provider fields (tool_calls, tool_call_id, name...) are kept so a request can be replayed.
    extra: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    def schema_text(self) -> str:
        """Canonical text of the schema, used for token counting and hashing."""
        return json.dumps(
            {"name": self.name, "description": self.description, "parameters": self.parameters},
            sort_keys=True,
            ensure_ascii=False,
        )


class Request(BaseModel):
    """The replayable part of a trace: everything sent to the model."""

    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Trace(BaseModel):
    id: str
    request: Request
    response: Response
    meta: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """A character range [start, end) inside one message's content."""

    message: int
    start: int
    end: int


class Locator(BaseModel):
    """Where a segment lives in the request, so it can be removed for ablation.

    Exactly one of: ``tool`` (index into request.tools; with ``tool_part`` only that part of the
    schema), ``whole_message`` (index into request.messages), or ``spans`` (character ranges
    inside message contents).
    """

    tool: int | None = None
    tool_part: ToolPart | None = None
    whole_message: int | None = None
    spans: list[Span] = Field(default_factory=list)


class Segment(BaseModel):
    trace_id: str
    index: int
    kind: SegmentKind
    source: Source
    fingerprint: str  # stable across traces, e.g. "system/## Tone rules", "tool/search_jobs"
    label: str  # short display name
    tokens: int
    content_hash: str
    locator: Locator
    parent: str | None = None  # optional finer grouping (reserved for sub-segments)

    @property
    def uid(self) -> str:
        return f"{self.trace_id}:{self.index}"


def content_hash(text: str) -> str:
    normalised = " ".join(text.split())
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]
