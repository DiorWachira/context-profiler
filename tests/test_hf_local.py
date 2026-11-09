"""Real-model checks (slow, optional): pytest -m slow. Needs pip install -e ".[local]" and a model download."""

import asyncio

import pytest

from ctxprof.models import Message, Request, Response, Tool, ToolCall
from ctxprof.providers.hf_local import parse_tool_calls

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_parse_tool_calls():
    text, calls = parse_tool_calls('Sure.\n<tool_call>\n{"name": "get_job", "arguments": {"listing_id": "JB-1"}}\n'
                                   '</tool_call>')
    assert text == "Sure." and calls == [ToolCall(name="get_job", arguments={"listing_id": "JB-1"})]
    assert parse_tool_calls("<tool_call>{bad json}</tool_call>")[1] == []


@pytest.fixture(scope="module")
def hf():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ctxprof.providers.hf_local import HFLocalProvider

    return HFLocalProvider(MODEL, max_new_tokens=64)


@pytest.mark.slow
def test_logprob_rises_when_the_needed_context_is_present(hf):
    with_fact = Request(model="m", messages=[
        Message(role="system", content="The JobBridge support line is open until 6pm. The access code is 4417."),
        Message(role="user", content="What is the access code?")])
    without = with_fact.model_copy(update={"messages": [with_fact.messages[1]]})
    target = Response(text="The access code is 4417.")
    lp_with = asyncio.run(hf.target_logprob(with_fact, target))
    lp_without = asyncio.run(hf.target_logprob(without, target))
    assert lp_with < 0 and lp_with > lp_without + 2.0, (lp_with, lp_without)


@pytest.mark.slow
def test_tool_call_target_and_generation(hf):
    req = Request(model="m", params={"max_tokens": 64}, messages=[
        Message(role="user", content="Check the status of my application JB-88213.")],
        tools=[Tool(name="get_application_status", description="Look up an application's status",
                    parameters={"type": "object", "properties": {"application_id": {"type": "string"}},
                                "required": ["application_id"]})])
    target = Response(tool_calls=[ToolCall(name="get_application_status", arguments={"application_id": "JB-88213"})])
    assert "<tool_call>" in hf.render_target(req, target)
    assert asyncio.run(hf.target_logprob(req, target)) > asyncio.run(
        hf.target_logprob(req.model_copy(update={"tools": []}), target))
    out = asyncio.run(hf.generate(req))
    assert out.tool_calls and out.tool_calls[0].name == "get_application_status"
