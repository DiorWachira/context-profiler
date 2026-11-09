"""Verification: replay held-out traces with and without the pruned segments and compare outcomes.

For each held-out trace the ORIGINAL request and the PRUNED request (the plan's cut fingerprints
removed with :func:`ctxprof.masking.apply_mask`) are replayed at temperature 0. The pruned
outcome is compared with the original *replay*, not the recorded answer, so drift between the
recorded model and the replay model is not blamed on pruning.

Agreement is the black-box metric (``score >= min_score``) or a user eval command that gets the
paths of two JSON files and exits 0 when the outcomes agree.

Per-hunk evidence, so a reviewer (and the UI) can see what each hunk costs:

* every regression is re-run once per cut hunk with that hunk restored; the hunks whose
  restoration fixes it are the *suspects* (if none fixes it alone, every cut hunk is suspect);
* every kept candidate (``--keep``) is probed on agreeing traces by also removing it, so the
  report says what cutting it would break.

``breaks`` per trace is the union of both; the estimate for any selection S of hunks is
"regression iff S ∩ breaks ≠ ∅". It is exact for the verified selection and an estimate otherwise
(it assumes removals do not interact beyond what the probes saw); re-run verify to confirm.
"""

from __future__ import annotations

import asyncio
import json
import random
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attribution import BudgetExceeded, CallGate, ProfileConfig, _prepare
from .masking import apply_mask
from .models import Response, Segment
from .providers.base import Provider
from .scoring import blackbox_score
from .storage import Store
from .tokens import TokenCounter


@dataclass
class VerifyConfig:
    held_out: int | None = None  # sample K traces from the held-out pool (None = all of it)
    seed: int = 0
    metric: str = "auto"
    min_score: float = 0.9  # black-box score at or above => same outcome
    threshold: float = 0.95  # agreement needed to recommend
    eval_cmd: str | None = None  # e.g. "python my_eval.py {original} {pruned}"; exit 0 = agree
    probe: bool = True  # attribute regressions to hunks and probe kept hunks
    price_in_per_m: float = 3.0
    price_out_per_m: float = 15.0
    ksh_per_usd: float = 129.0
    budget_usd: float = 5.0
    concurrency: int = 4


def choose_held_out(store: Store, run_id: int, k: int | None, seed: int = 0) -> tuple[list[str], bool]:
    """Traces the run did not profile; if there are none, a sample of profiled ones (flagged in-sample)."""
    profiled = set(store.run_trace_ids(run_id))
    pool = [t for t in store.trace_ids() if t not in profiled]
    in_sample = not pool
    if in_sample:
        pool = sorted(profiled)
    if k is not None and k < len(pool):
        pool = sorted(random.Random(seed).sample(pool, k))
    return pool, in_sample


def _resp(r: Response) -> dict[str, Any]:
    return r.model_dump(mode="json")


class _Judge:
    def __init__(self, cfg: VerifyConfig, gate: CallGate) -> None:
        self.cfg, self.gate = cfg, gate

    async def __call__(self, original: Response, pruned: Response) -> tuple[bool, float]:
        sim = None
        if self.cfg.metric == "embedding":
            sims = await self.gate.similarities(original.text, [pruned.text])
            if sims is None:
                raise ValueError("metric 'embedding' needs an embedder")
            sim = sims[0]
        score = blackbox_score(original, pruned, self.cfg.metric, sim)
        if not self.cfg.eval_cmd:
            return score >= self.cfg.min_score, score
        return self._run_eval(original, pruned), score

    def _run_eval(self, original: Response, pruned: Response) -> bool:
        assert self.cfg.eval_cmd is not None
        with tempfile.TemporaryDirectory(prefix="ctxprof-eval-") as d:
            a, b = Path(d) / "original.json", Path(d) / "pruned.json"
            a.write_text(json.dumps(_resp(original), ensure_ascii=False), encoding="utf-8")
            b.write_text(json.dumps(_resp(pruned), ensure_ascii=False), encoding="utf-8")
            cmd = self.cfg.eval_cmd.format(original=shlex.quote(str(a)), pruned=shlex.quote(str(b)))
            return subprocess.run(cmd, shell=True, capture_output=True).returncode == 0  # noqa: S602


def _hunk_of(s: Segment, hunks: set[str]) -> str | None:
    """The hunk fingerprint that removes this segment (a tool hunk covers its sub-segments)."""
    if s.fingerprint in hunks:
        return s.fingerprint
    return s.parent if s.parent in hunks else None


def _keep(segments: list[Segment], removed: set[str]) -> list[bool]:
    return [_hunk_of(s, removed) is None for s in segments]


async def verify_trace(gate: CallGate, judge: _Judge, store: Store, trace_id: str, cut: set[str],
                       kept: set[str], cfg: VerifyConfig, out_tokens_fn: Callable[[Response], int],
                       pcfg: ProfileConfig) -> dict[str, Any]:
    trace, segs = store.load_trace(trace_id)
    base = _prepare(trace.request, pcfg)
    out_tokens = out_tokens_fn(trace.response)
    present = {s.fingerprint for s in segs} | {s.parent for s in segs if s.parent}
    cut_here = sorted(cut & present)

    def run(removed: set[str]):
        return gate.generate(apply_mask(base, segs, _keep(segs, removed)), out_tokens)

    original, pruned = await asyncio.gather(run(set()), run(cut))
    o, p = original.as_response(), pruned.as_response()
    agree, score = await judge(o, p)

    tokens_before = sum(s.tokens for s in segs)
    removed_tokens: dict[str, int] = {}
    for s in segs:
        if (h := _hunk_of(s, cut | kept)) is not None:
            removed_tokens[h] = removed_tokens.get(h, 0) + s.tokens

    rec: dict[str, Any] = {
        "trace_id": trace_id, "intent": trace.meta.get("intent"), "agree": agree, "score": round(score, 4),
        "tokens_before": tokens_before, "removed_tokens": removed_tokens,
        "breaks": [], "explained": True, "probes": {},
    }
    if not agree:
        rec["question"] = next((m.content for m in reversed(trace.request.messages) if m.role == "user"), "")
        rec["original"], rec["pruned"] = _resp(o), _resp(p)
        if cfg.probe and cut_here:
            restored = await asyncio.gather(*(run(cut - {h}) for h in cut_here))
            verdicts = [await judge(o, r.as_response()) for r in restored]
            suspects = [h for h, (ok, _) in zip(cut_here, verdicts, strict=True) if ok]
            rec["explained"] = bool(suspects)
            rec["breaks"] = suspects or cut_here
        else:
            rec["explained"] = False
            rec["breaks"] = cut_here
    elif cfg.probe:
        probe_here = sorted(kept & present)
        results = await asyncio.gather(*(run(cut | {k}) for k in probe_here))
        for k, r in zip(probe_here, results, strict=True):
            ok, sc = await judge(o, r.as_response())
            if not ok:
                rec["breaks"].append(k)
                rec["probes"][k] = {"score": round(sc, 4), "pruned": _resp(r.as_response())}
                rec.setdefault("question", next(
                    (m.content for m in reversed(trace.request.messages) if m.role == "user"), ""))
                rec.setdefault("original", _resp(o))
    return rec


def summarise(records: list[dict[str, Any]], selection: set[str], cfg: VerifyConfig) -> dict[str, Any]:
    """Ledger + verdict for a selection of hunks (exact for the verified one, estimated otherwise)."""
    n = len(records)
    regress = [r for r in records if selection & set(r["breaks"]) or (not r["agree"] and not r["breaks"])]
    before = sum(r["tokens_before"] for r in records) / n if n else 0.0
    saved = sum(v for r in records for fp, v in r["removed_tokens"].items() if fp in selection) / n if n else 0.0
    after = before - saved
    price = cfg.price_in_per_m / 1e6
    agree_n = n - len(regress)
    agreement = agree_n / n if n else 0.0
    saved_100k = saved * price * 1e5
    return {
        "traces": n, "agree_n": agree_n, "agreement": agreement,
        "recommend": bool(n) and agreement >= cfg.threshold,
        "tokens_before": before, "tokens_after": after, "tokens_saved": saved,
        "reduction": saved / before if before else 0.0,
        "cost_before": before * price, "cost_after": after * price,
        "saved_per_100k_usd": saved_100k, "saved_per_100k_ksh": saved_100k * cfg.ksh_per_usd,
        "regressions": [r["trace_id"] for r in regress],
    }


async def run_verify(store: Store, provider: Provider, prune_id: int, cfg: VerifyConfig, counter: TokenCounter,
                     trace_ids: list[str] | None = None, on_progress=None, embedder=None) -> dict[str, Any]:
    prune = store.get_prune(prune_id)
    if prune is None:
        raise KeyError(f"prune {prune_id} not found")
    hunks = prune["hunks"]
    cut = {h["fingerprint"] for h in hunks if h["cut"]}
    kept = {h["fingerprint"] for h in hunks if not h["cut"]}
    in_sample = False
    if trace_ids is None:
        trace_ids, in_sample = choose_held_out(store, prune["run_id"], cfg.held_out, cfg.seed)

    pcfg = ProfileConfig(mode="blackbox", metric=cfg.metric, budget_usd=cfg.budget_usd, concurrency=cfg.concurrency,
                         price_in_per_m=cfg.price_in_per_m, price_out_per_m=cfg.price_out_per_m)
    gate = CallGate(store, provider, pcfg, counter, embedder)
    judge = _Judge(cfg, gate)

    def out_tokens(r: Response) -> int:
        return max(1, counter.count(r.text) + 8 * len(r.tool_calls))

    records: list[dict[str, Any]] = []
    status = "verified"
    for n, tid in enumerate(trace_ids, start=1):
        try:
            records.append(await verify_trace(gate, judge, store, tid, cut, kept, cfg, out_tokens, pcfg))
        except BudgetExceeded:
            status = "budget"
            break
        if on_progress:
            on_progress(n, len(trace_ids), gate)
    store.commit()

    summary = summarise(records, cut, cfg)
    if status == "budget":
        summary["recommend"] = False
    regressions = [
        {k: r.get(k) for k in ("trace_id", "intent", "score", "breaks", "explained", "question", "original", "pruned")}
        for r in records if not r["agree"]
    ]
    report = {
        "status": status, "in_sample": in_sample, "summary": summary, "regressions": regressions,
        "records": records, "config": asdict(cfg), "provider": provider.name,
        "spent_usd": gate.spent, "calls": gate.calls, "cache_hits": gate.cache_hits,
    }
    store.save_verification(prune_id, report)
    return report
