"""Attribution engine: ablate segments with random masks, score each answer, fit the surrogate.

Every model call goes through :class:`CallGate`, which caches by
(provider, model, params, masked-request hash, call kind, target hash) and enforces a hard
spend cap. A re-run of the same profile is therefore free and reproducible from cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .masking import apply_mask, sample_masks
from .models import Request, Response, Segment, Trace
from .providers.base import Completion, Provider
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

    @property
    def min_effect(self) -> float:
        return 0.5 if self.mode == "logprob" else 0.05


class CallGate:
    """Cache + budget + concurrency limit around a provider."""

    def __init__(self, store: Store, provider: Provider, config: ProfileConfig, counter: TokenCounter) -> None:
        self.store, self.provider, self.config, self.counter = store, provider, config, counter
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
    masks = sample_masks(config.n_masks, len(active), config.p_keep, seed=seed)
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
        scores = np.array([blackbox_score(target, o.as_response(), config.metric) for o in outs])
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


async def run_profile(store: Store, provider: Provider, config: ProfileConfig, counter: TokenCounter,
                      trace_ids: list[str] | None = None, on_progress=None) -> int:
    """Profile the given traces (all stored traces by default). Returns the run id."""
    if config.mode == "logprob" and not provider.supports_logprob:
        raise ValueError(f"provider {provider.name!r} has no log-probabilities; use mode='blackbox'")
    ids = trace_ids if trace_ids is not None else store.trace_ids()
    run_id = store.create_run({**asdict(config), "provider": provider.name, "counter": counter.name,
                               "traces": len(ids)})
    gate = CallGate(store, provider, config, counter)
    status, note = "done", None
    for n, tid in enumerate(ids, start=1):
        trace, segments = store.load_trace(tid)
        if not segments:
            continue
        try:
            res = await profile_trace(gate, trace, segments, config, _seed_for(tid, config.seed))
        except BudgetExceeded as exc:
            status, note = "budget", f"stopped after {n - 1} traces: {exc}"
            break
        fit = res["fit"]
        col = {seg_j: f_j for f_j, seg_j in enumerate(res["active"])}
        rows = []
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
            "mode": config.mode, "n_masks": config.n_masks, "alpha": fit.alpha, "intercept": fit.intercept,
            "original_score": res["original_score"], "r2_heldout": fit.r2_heldout, "poor_fit": res["poor_fit"],
        }, rows, abl)
        if on_progress:
            on_progress(n, len(ids), gate)
    store.finish_run(run_id, status, gate.spent, gate.calls, gate.cache_hits, note)
    return run_id
