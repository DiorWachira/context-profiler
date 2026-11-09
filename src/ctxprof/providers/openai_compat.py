"""OpenAI-compatible chat endpoint (hosted APIs, Ollama's /v1, vLLM's OpenAI server).

Supports ``generate`` (black-box mode). Teacher-forced log-probabilities are not part of the chat
API, so ``target_logprob`` is left to a dedicated provider (see roadmap: vLLM prompt_logprobs).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..models import Request, ToolCall
from .base import Completion, Provider, ProviderError


class OpenAICompatProvider(Provider):
    name = "openai-compat"
    supports_logprob = False

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        if model:  # the cache key uses provider name + the trace's model; an override must change it
            self.name = f"openai-compat:{model}"
        key = api_key if api_key is not None else os.environ.get("CTXPROF_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout,
                                         transport=transport)

    def _payload(self, request: Request) -> dict[str, Any]:
        messages = []
        for m in request.messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            msg.update(m.extra)
            messages.append(msg)
        payload: dict[str, Any] = {"model": self.model or request.model, "messages": messages}
        payload.update({k: v for k, v in request.params.items() if k in ("temperature", "max_tokens", "top_p", "seed")})
        if request.tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters or {"type": "object"}}}
                for t in request.tools
            ]
        return payload

    async def generate(self, request: Request) -> Completion:
        try:
            r = await self._client.post("/chat/completions", json=self._payload(request))
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"chat completion failed: {exc}") from exc
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                parsed = {"_raw": args}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=parsed if isinstance(parsed, dict) else {}))
        usage = data.get("usage") or {}
        return Completion(text=msg.get("content") or "", tool_calls=calls,
                          input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))

    async def aclose(self) -> None:
        await self._client.aclose()
