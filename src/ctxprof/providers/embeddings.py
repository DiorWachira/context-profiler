"""Embedders for the ``embedding`` black-box metric (cosine similarity of answer embeddings).

Any OpenAI-compatible ``/embeddings`` endpoint works (hosted APIs, Ollama's ``/v1``, vLLM, TEI's
OpenAI route). Vectors are cached in the store like every other model call, so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Protocol

import httpx

from .base import ProviderError


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else (1.0 if na == nb else 0.0)


class OpenAICompatEmbedder:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60.0,
                 batch: int = 64, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model = model
        self.name = f"embed:{model}"
        self.batch = batch
        key = api_key if api_key is not None else os.environ.get("CTXPROF_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
                                         transport=transport)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            chunk = [t if t.strip() else " " for t in texts[i:i + self.batch]]  # APIs reject empty input
            try:
                r = await self._client.post("/embeddings", json={"model": self.model, "input": chunk})
                r.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(f"embedding request failed: {exc}") from exc
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            if len(data) != len(chunk):
                raise ProviderError(f"embedding endpoint returned {len(data)} vectors for {len(chunk)} inputs")
            out += [list(map(float, d["embedding"])) for d in data]
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


class CachedEmbedder:
    """Wraps an embedder with the store's call cache (key: embedder name + sha256 of the text)."""

    def __init__(self, inner: Embedder, store) -> None:
        self.inner, self.store, self.name = inner, store, inner.name
        self.calls = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(json.dumps({"kind": "embed", "embedder": self.name, "text": text}).encode()).hexdigest()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        found: dict[str, list[float]] = {}
        for t in set(texts):
            if (hit := self.store.cache_get(self._key(t))) is not None:
                found[t] = hit["v"]
        missing = sorted(set(texts) - set(found))
        if missing:
            vecs = await self.inner.embed(missing)
            self.calls += 1
            for t, v in zip(missing, vecs, strict=True):
                found[t] = v
                self.store.cache_put(self._key(t), "embed", self.name, {"v": v}, 0.0)
        return [found[t] for t in texts]

    async def aclose(self) -> None:
        await self.inner.aclose()
