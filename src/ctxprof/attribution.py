"""Attribution engine: ablate segments with random masks, score each answer, fit the surrogate.

Every model call goes through :class:`CallGate`, which caches by
(provider, model, params, masked-request hash, call kind, target hash) and enforces a hard
spend cap. A re-run of the same profile is therefore free and reproducible from cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .masking import apply_mask, sample_masks
from .models import Request, Response, Segment, Trace
from .providers.base import Completion, Provider
from .providers.embeddings import CachedEmbedder, cosine
from .providers.fake import logit_of_logprob
from .scoring import blackbox_score
from .storage import Store
from .surrogate import fit_surrogate, normalise
from .tokens import TokenCounter


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class ProfileConfig:
    n_masks: int = 32
    p_keep: float = 0.5
    mode: str = "logprob"  # "logprob" | "blackbox"
    metric: str = "auto"  # black-box metric: auto | text | tool | json
    seed: int = 0
    concurrency: int = 4
    budget_usd: float = 5.0
    price_in_per_m: float = 3.0
    price_out_per_m: float = 15.0
    holdout: float = 0.25
    n_boot: int = 50
    poor_fit_r2: float = 0.5
    force_temperature_zero: bool = True
    # Segment kinds that are always kept and never attributed. The current user turn is the
    # question, not context (as in ContextCite); ablating it creates strong interactions that a
    # linear surrogate cannot represent and that leak into every other weight.
    pin_kinds: tuple[str, ...] = ("user",)
    # More segments need more masks (ContextCite: N on the order of the number of sources).
    # N = max(n_masks, ceil(masks_per_segment * active segments)); 0 disables scaling.
    masks_per_segment: float = 0.0
    trace_concurrency: int = 1  # traces profiled at once
    interaction_top: int = 0  # pairwise interaction check among the top-k segments (0 = off)

    @property
    def min_effect(self) -> float:
        return 0.5 if self.mode == "logprob" else 0.05


class CallGate:
    """Cache + budget + concurrency limit around a provider."""

    def __init__(self, store: Store, provider: Provider, config: ProfileConfig, counter: TokenCounter,
                 embedder=None) -> None:
        self.store, self.provider, self.config, self.counter = store, provider, config, counter
        self.embedder = CachedEmbedder(embedder, store) if embedder is not None else None
        self.spent = 0.0
        self.calls = 0
        self.cache_hits = 0
        self._sem = asyncio.Semaphore(config.concurrency)
        self._lock = asyncio.Lock()

    def _key(self, kind: str, request: Request, target: Response | None) -> str:
        blob = json.dumps({
            "kind": kind, "provider": self.provider.name, "model": request.model, "params": request.params,
            "request": request.fingerprint(),
            "target": hashlib.sha256(target.model_dump_json().encode()).hexdigest() if target else None,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _estimate(self, request: Request, out_tokens: int) -> float:
        n_in = sum(self.counter.count(m.content) for m in request.messages)
        n_in += sum(self.counter.count(t.schema_text()) for t in request.tools)
        c = self.config
        return n_in * c.price_in_per_m / 1e6 + out_tokens * c.price_out_per_m / 1e6

    async def _reserve(self, cost: float) -> None:
        async with self._lock:
            if self.spent + cost > self.config.budget_usd:
                raise BudgetExceeded(f"budget cap ${self.config.budget_usd:.2f} reached (spent ${self.spent:.4f})")
            self.spent += cost

    async def generate(self, request: Request, out_tokens: int) -> Completion:
        key = self._key("generate", request, None)
        if (hit := self.store.cache_get(key)) is not None:
            self.cache_hits += 1
            return Completion.model_validate(hit)
        cost = self._estimate(request, out_tokens)
        await self._reserve(cost)
        async with self._sem:
            result = await self.provider.generate(request)
        self.calls += 1
        self.store.cache_put(key, "generate", request.model, result.model_dump(mode="json"), cost)
        return result

    async def similarities(self, reference: str, texts: list[str]) -> list[float] | None:
        """Embedding cosine of each text to ``reference`` (None without an embedder)."""
        if self.embedder is None:
            return None
        vecs = await self.embedder.embed([reference, *texts])
        return [cosine(vecs[0], v) for v in vecs[1:]]

    async def target_logprob(self, request: Request, target: Response, out_tokens: int) -> float:
        key = self._key("logprob", request, target)
        if (hit := self.store.cache_get(key)) is not None:
            self.cache_hits += 1
            return float(hit["logprob"])
        cost = self._estimate(request, out_tokens)
        await self._reserve(cost)
        async with self._sem:
            value = await self.provider.target_logprob(request, target)
        self.calls += 1
        self.store.cache_put(key, "logprob", request.model, {"logprob": value}, cost)
        return value


def _prepare(request: Request, config: ProfileConfig) -> Request:
    if config.force_temperature_zero:
        return request.model_copy(update={"params": {**request.params, "temperature": 0}})
    return request


async def profile_trace(gate: CallGate, trace: Trace, segments: list[Segment], config: ProfileConfig,
                        seed: int) -> dict[str, Any]:
    active = [j for j, s in enumerate(segments) if s.kind not in config.pin_kinds]
    n_masks = max(config.n_masks, math.ceil(config.masks_per_segment * len(active)))
    masks = sample_masks(n_masks, len(active), config.p_keep, seed=seed)
    full = np.ones((len(masks), len(segments)), dtype=bool)
    full[:, active] = masks
    target = trace.response
    out_tokens = max(1, gate.counter.count(target.text) + 8 * len(target.tool_calls))
    base = _prepare(trace.request, config)
    ablated = [apply_mask(base, segments, m) for m in full]

    texts: list[str | None] = [None] * len(masks)
    if config.mode == "logprob":
        original = logit_of_logprob(await gate.target_logprob(base, target, out_tokens))
        lps = await asyncio.gather(*(gate.target_logprob(r, target, out_tokens) for r in ablated))
        scores = np.array([logit_of_logprob(v) for v in lps])
    elif config.mode == "blackbox":
        original = 1.0
        outs = await asyncio.gather(*(gate.generate(r, out_tokens) for r in ablated))
        sims: list[float | None] = [None] * len(outs)
        if config.metric == "embedding":
            got = await gate.similarities(target.text, [o.text for o in outs])
            if got is None:
                raise ValueError("metric 'embedding' needs an embedder (--embed-url / --embed-model)")
            sims = list(got)
        scores = np.array([blackbox_score(target, o.as_response(), config.metric, sim)
                           for o, sim in zip(outs, sims, strict=True)])
        texts = [o.text for o in outs]
    else:
        raise ValueError(f"unknown mode {config.mode!r}")

    fit = fit_surrogate(masks, scores, holdout=config.holdout, n_boot=config.n_boot, seed=seed)
    influence = normalise(fit.weights, config.min_effect)
    return {
        "masks": full, "active": active, "scores": scores, "texts": texts, "fit": fit, "influence": influence,
        "original_score": float(original),
        "poor_fit": fit.r2_heldout is not None and fit.r2_heldout < config.poor_fit_r2,
    }


def _seed_for(trace_id: str, base_seed: int) -> int:
    return (int(hashlib.sha1(trace_id.encode()).hexdigest()[:8], 16) + base_seed) % (2**31)


async def interaction_check(gate: CallGate, trace: Trace, segments: list[Segment], config: ProfileConfig,
                            res: dict[str, Any]) -> list[dict[str, Any]]:
    """For the top-k segments, compare removing a pair with the sum of removing each alone.

    A linear surrogate assumes effect(a+b) = effect(a) + effect(b). ``interaction`` is the
    difference, so an AND (both needed) or a redundancy (either suffices) shows up even when the
    held-out R² looks fine. Costs 1 + k + k(k-1)/2 extra calls per trace (cached like all calls).
    """
    k = config.interaction_top
    w = res["fit"].weights
    order = [res["active"][c] for c in np.argsort(-w)[:k] if w[c] > 0]
    if len(order) < 2:
        return []
    target = trace.response
    out_tokens = max(1, gate.counter.count(target.text) + 8 * len(target.tool_calls))
    base = _prepare(trace.request, config)
    drops = [()] + [(a,) for a in order] + [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    reqs = [apply_mask(base, segments, [j not in d for j in range(len(segments))]) for d in drops]
    if config.mode == "logprob":
        vals = await asyncio.gather(*(gate.target_logprob(r, target, out_tokens) for r in reqs))
        score = {d: logit_of_logprob(v) for d, v in zip(drops, vals, strict=True)}
    else:
        outs = await asyncio.gather(*(gate.generate(r, out_tokens) for r in reqs))
        sims: list[float | None] = [None] * len(outs)
        if config.metric == "embedding":
            sims = list(await gate.similarities(target.text, [o.text for o in outs]) or sims)
        score = {d: blackbox_score(target, o.as_response(), config.metric, sim)
                 for d, o, sim in zip(drops, outs, sims, strict=True)}
    full = score[()]
    rows = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            ea, eb, eab = full - score[(a,)], full - score[(b,)], full - score[(a, b)]
            inter = eab - (ea + eb)
            rows.append({"a": a, "b": b, "effect_a": ea, "effect_b": eb, "effect_ab": eab, "interaction": inter,
                         "relative": inter / max(abs(ea) + abs(eb), config.min_effect)})
    return rows


async def run_profile(store: Store, provider: Provider, config: ProfileConfig, counter: TokenCounter,
                      trace_ids: list[str] | None = None, on_progress=None, embedder=None) -> int:
    """Profile the given traces (all stored traces by default). Returns the run id.

    Up to ``config.trace_concurrency`` traces run at once (model calls are further limited by
    ``config.concurrency``). Results do not depend on scheduling: masks are seeded per trace and
    every call is cached. A run that hits its budget stops starting new traces and keeps the
    finished ones.
    """
    if config.mode == "logprob" and not provider.supports_logprob:
        raise ValueError(f"provider {provider.name!r} has no log-probabilities; use mode='blackbox'")
    ids = trace_ids if trace_ids is not None else store.trace_ids()
    run_id = store.create_run({**asdict(config), "provider": provider.name, "counter": counter.name,
                               "traces": len(ids)})
    gate = CallGate(store, provider, config, counter, embedder)
    sem = asyncio.Semaphore(max(1, config.trace_concurrency))
    state: dict[str, Any] = {"done": 0, "stop": None}

    async def one(tid: str) -> None:
        async with sem:
            if state["stop"]:
                return
            trace, segments = store.load_trace(tid)
            if not segments:
                return
            try:
                res = await profile_trace(gate, trace, segments, config, _seed_for(tid, config.seed))
                inter = await interaction_check(gate, trace, segments, config, res) if config.interaction_top else []
            except BudgetExceeded as exc:
                state["stop"] = state["stop"] or str(exc)
                return
            _save(store, run_id, tid, segments, config, res, inter)
            state["done"] += 1
            if on_progress:
                on_progress(state["done"], len(ids), gate)

    await asyncio.gather(*(one(t) for t in ids))
    status = "budget" if state["stop"] else "done"
    note = f"stopped after {state['done']} traces: {state['stop']}" if state["stop"] else None
    store.finish_run(run_id, status, gate.spent, gate.calls, gate.cache_hits, note)
    return run_id


def _save(store: Store, run_id: int, tid: str, segments: list[Segment], config: ProfileConfig,
          res: dict[str, Any], inter: list[dict[str, Any]]) -> None:
    fit = res["fit"]
    col = {seg_j: f_j for f_j, seg_j in enumerate(res["active"])}
    rows: list[tuple[int, float | None, float | None, float | None, float | None]] = []
    for j, s in enumerate(segments):
        if j in col:
            c = col[j]
            rows.append((s.index, float(fit.weights[c]), float(res["influence"][c]),
                         float(fit.ci_lo[c]), float(fit.ci_hi[c])))
        else:  # pinned: kept in every mask, not attributed
            rows.append((s.index, None, None, None, None))
    abl = [(i, "".join("1" if x else "0" for x in m), float(res["scores"][i]), res["texts"][i])
           for i, m in enumerate(res["masks"])]
    store.save_attribution(run_id, tid, {
        "mode": config.mode, "n_masks": len(res["masks"]), "alpha": fit.alpha, "intercept": fit.intercept,
        "original_score": res["original_score"], "r2_heldout": fit.r2_heldout, "poor_fit": res["poor_fit"],
    }, rows, abl, [(segments[r["a"]].index, segments[r["b"]].index, r["effect_a"], r["effect_b"], r["effect_ab"],
                    r["interaction"], r["relative"]) for r in inter])
