from __future__ import annotations

import pytest

from ctxprof.models import Message, Request, Response, Tool, ToolCall, Trace
from ctxprof.segmenter import SegmenterConfig
from ctxprof.storage import Store
from ctxprof.tokens import HeuristicCounter

SYSTEM = """You are a helpful support agent.

## Tone rules
Be brief and kind.

## Refund policy
Refunds take 14 days.
"""


@pytest.fixture
def counter() -> HeuristicCounter:
    return HeuristicCounter()


@pytest.fixture
def seg_config() -> SegmenterConfig:
    return SegmenterConfig(heading_levels=[2], untrusted_doc_prefixes=["listing:"])


@pytest.fixture
def small_trace() -> Trace:
    user = ('Context:\n<doc id="kb:faq#1">Shortlisted means a call within 7 days.</doc>\n'
            '<doc id="listing:42">Pay a registration fee.</doc>\n\nQuestion: what is my status?')
    return Trace(
        id="t1",
        request=Request(
            model="m",
            params={"temperature": 0.7},
            messages=[
                Message(role="system", content=SYSTEM),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi! How can I help?"),
                Message(role="user", content=user),
            ],
            tools=[Tool(name="get_status", description="Get status", parameters={"type": "object"}),
                   Tool(name="get_weather", description="Weather", parameters={"type": "object"})],
        ),
        response=Response(text="You are shortlisted.", tool_calls=[ToolCall(name="get_status")]),
    )


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(tmp_path / "test.db")
    yield s
    s.close()
