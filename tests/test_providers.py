"""OpenAI-compatible provider and embedder against httpx.MockTransport; embedding metric end to end."""

import asyncio
import json

import httpx
import pytest

from ctxprof.attribution import ProfileConfig, run_profile
from ctxprof.models import Message, Request, Tool
from ctxprof.providers import Effect, FakeProvider, OpenAICompatProvider, ProviderError
from ctxprof.providers.embeddings import CachedEmbedder, OpenAICompatEmbedder, cosine
from ctxprof.scoring import blackbox_score
from ctxprof.segmenter import segment_trace

REQ = Request(
    model="llama3.1:8b", params={"temperature": 0, "max_tokens": 50, "stream": True},
    messages=[Message(role="system", content="Be brief."),
              Message(role="assistant", content="", extra={"tool_calls": [
                  {"id": "c1", "type": "function", "function": {"name": "get_job", "arguments": "{}"}}]}),
              Message(role="tool", content="{\"title\": \"Driver\"}", extra={"tool_call_id": "c1"}),
              Message(role="user", content="Status?")],
    tools=[Tool(name="get_application_status", description="Status", parameters={})],
)


def chat_handler(seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "x", "type": "function",
                 "function": {"name": "get_application_status", "arguments": "{\"application_id\": \"JB-1\"}"}},
                {"id": "y", "type": "function", "function": {"name": "broken", "arguments": "{not json"}},
            ]}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9},
        })
    return handle


def test_openai_compat_payload_and_tool_call_parsing():
    seen: list = []
    prov = OpenAICompatProvider("http://llm.test/v1/", api_key="sk-test",
                                transport=httpx.MockTransport(chat_handler(seen)))
    out = asyncio.run(prov.generate(REQ))
    path, auth, body = seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["model"] == "llama3.1:8b" and body["temperature"] == 0 and body["max_tokens"] == 50
    assert "stream" not in body  # only replay-safe params are forwarded
    assert body["messages"][1]["tool_calls"][0]["id"] == "c1" and body["messages"][2]["tool_call_id"] == "c1"
    assert body["tools"][0] == {"type": "function", "function": {
        "name": "get_application_status", "description": "Status", "parameters": {"type": "object"}}}
    assert out.text == "" and out.input_tokens == 120 and out.output_tokens == 9
    assert out.tool_calls[0].arguments == {"application_id": "JB-1"}
    assert out.tool_calls[1].arguments == {"_raw": "{not json"}
    asyncio.run(prov.aclose())


def test_openai_compat_model_override_changes_cache_identity():
    t = httpx.MockTransport(chat_handler([]))
    assert OpenAICompatProvider("http://x/v1", transport=t).name == "openai-compat"
    assert OpenAICompatProvider("http://x/v1", model="qwen2.5:7b", transport=t).name == "openai-compat:qwen2.5:7b"


def test_openai_compat_errors_become_provider_errors(monkeypatch):
    monkeypatch.delenv("CTXPROF_API_KEY", raising=False)
    seen = []

    def fail(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(429, json={"error": "rate limited"})

    prov = OpenAICompatProvider("http://llm.test/v1", transport=httpx.MockTransport(fail))
    with pytest.raises(ProviderError, match="429"):
        asyncio.run(prov.generate(REQ))
    assert seen == [None]  # no key configured -> no Authorization header


def embed_handler(calls: list):
    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        vecs = [[1.0, 0.0] if "shortlisted" in t.lower() else [0.0, 1.0] for t in body["input"]]
        data = [{"index": i, "embedding": v} for i, v in enumerate(vecs)]
        return httpx.Response(200, json={"data": list(reversed(data))})  # order must come from "index"
    return handle


def test_embedder_batches_and_orders_by_index(store):
    calls: list = []
    emb = OpenAICompatEmbedder("http://e.test/v1", "nomic-embed-text", batch=2,
                               transport=httpx.MockTransport(embed_handler(calls)))
    vecs = asyncio.run(emb.embed(["Shortlisted!", "", "no idea"]))
    assert vecs == [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    assert calls == [["Shortlisted!", " "], ["no idea"]]  # empty input replaced, batched by 2

    cached = CachedEmbedder(emb, store)
    asyncio.run(cached.embed(["a", "b"]))
    asyncio.run(cached.embed(["b", "a", "a"]))
    assert cached.calls == 1  # second call fully served from the store
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0 and cosine([0, 0], [0, 0]) == 1.0


def test_embedding_metric_scores_and_profiles(store, counter, small_trace, seg_config):
    from ctxprof.models import Response
    a = Response(text="You are shortlisted.")
    assert blackbox_score(a, Response(text="x"), "embedding", 0.8) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        blackbox_score(a, a, "embedding")

    effects = [Effect(needs=["Shortlisted means"], phrase="You are shortlisted.")]
    prov = FakeProvider(effects)
    small_trace.response = asyncio.run(prov.generate(small_trace.request)).as_response()
    store.save_trace(small_trace, segment_trace(small_trace, seg_config, counter))
    emb = OpenAICompatEmbedder("http://e.test/v1", "m", transport=httpx.MockTransport(embed_handler([])))
    cfg = ProfileConfig(mode="blackbox", metric="embedding", n_masks=16, budget_usd=1e9, n_boot=5)
    run_id = asyncio.run(run_profile(store, prov, cfg, counter, embedder=emb))
    _, segs = store.load_trace(small_trace.id)
    attrs = store.attributions(run_id, small_trace.id)
    top = max((s for s in segs if attrs[s.index]["weight"] is not None), key=lambda s: attrs[s.index]["weight"])
    assert top.fingerprint == "rag/kb:faq#1"
