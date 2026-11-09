"""M5 case study: the JobBridge prompt against a REAL model (Qwen2.5-0.5B-Instruct on CPU).

    python examples/case_study.py            # resumable: every model call is cached in case.db

1. Draw demo requests (5 to profile, one per intent, plus held-out ones) and re-record each answer
   with the real model at temperature 0, so the profiled target is the model's own answer.
2. Profile the same traces in logprob mode (exact teacher-forced log-probs) and black-box mode.
3. Compare the two rankings: Spearman rank correlation and top-5 overlap per trace and aggregated.
4. Prune with the logprob labels and verify on the held-out traces with the real model.

Writes examples/case_study/results.json and results.md. CPU only: expect a few hours.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ctxprof.aggregate import LabelRules, aggregate, injection_signals
from ctxprof.attribution import ProfileConfig, run_profile
from ctxprof.demo import SEGMENTER_YAML, generate_traces, write_demo
from ctxprof.models import Trace
from ctxprof.prune import Bundle, build_plan
from ctxprof.providers.hf_local import HFLocalProvider
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.storage import Store
from ctxprof.tokens import HeuristicCounter
from ctxprof.verify import VerifyConfig, run_verify

OUT = Path(__file__).parent / "case_study"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROFILE_INTENTS = ["status", "search", "scam", "salary", "detail"]
HELD_OUT = 8
N_MASKS = 24
MAX_NEW = 64


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pick(traces: list[Trace]) -> tuple[list[Trace], list[Trace]]:
    """One trace per intent to profile; the next HELD_OUT traces (any intent, two scams) held out."""
    profiled, used = [], set()
    for intent in PROFILE_INTENTS:
        t = next(t for t in traces if t.meta["intent"] == intent and t.id not in used)
        profiled.append(t)
        used.add(t.id)
    rest = [t for t in traces if t.id not in used]
    scams = [t for t in rest if t.meta["intent"] == "scam"][:2]
    others = [t for t in rest if t.meta["intent"] != "scam"][: HELD_OUT - len(scams)]
    return profiled, sorted(scams + others, key=lambda t: t.id)


def ranking_stats(store: Store, run_a: int, run_b: int, ids: list[str]) -> dict:
    per_trace = []
    for tid in ids:
        a, b = store.attributions(run_a, tid), store.attributions(run_b, tid)
        _, segs = store.load_trace(tid)
        idx = [s.index for s in segs if a[s.index]["weight"] is not None]
        wa = np.array([a[i]["weight"] for i in idx])
        wb = np.array([b[i]["weight"] for i in idx])
        fps = {s.index: s.fingerprint for s in segs}
        top_a = {fps[i] for i in np.array(idx)[np.argsort(-wa)[:5]]}
        top_b = {fps[i] for i in np.array(idx)[np.argsort(-wb)[:5]]}
        rho = spearmanr(wa, wb).statistic if wa.std() and wb.std() else float("nan")
        per_trace.append({"trace_id": tid, "spearman": float(rho), "top5_overlap": len(top_a & top_b),
                          "top5_logprob": sorted(top_a), "top5_blackbox": sorted(top_b)})
    agg_a = {r["fingerprint"]: r["mean_influence"] for r in aggregate(store.run_rows(run_a), exclude_poor_fit=False)}
    agg_b = {r["fingerprint"]: r["mean_influence"] for r in aggregate(store.run_rows(run_b), exclude_poor_fit=False)}
    common = sorted(set(agg_a) & set(agg_b))
    va, vb = np.array([agg_a[f] for f in common]), np.array([agg_b[f] for f in common])
    ta = set(sorted(common, key=lambda f: -agg_a[f])[:5])
    tb = set(sorted(common, key=lambda f: -agg_b[f])[:5])
    return {
        "per_trace": per_trace,
        "aggregate": {"fingerprints": len(common), "spearman": float(spearmanr(va, vb).statistic),
                      "top5_overlap": len(ta & tb), "top5_logprob": sorted(ta), "top5_blackbox": sorted(tb)},
    }


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    demo_dir = OUT / "bundle"
    paths = write_demo(demo_dir, n=1)  # just the prompt bundle + segmenter
    store = Store(OUT / "case.db")
    counter = HeuristicCounter()
    sc = SegmenterConfig.model_validate(SEGMENTER_YAML)
    log(f"loading {MODEL}")
    hf = HFLocalProvider(MODEL, max_new_tokens=MAX_NEW)

    profiled, held = pick(generate_traces(60, seed=7))
    log(f"profile {[t.id for t in profiled]} | held out {[t.id for t in held]}")
    for t in profiled + held:  # re-record the answer with the real model (cached in the store)
        tid = "q" + t.id[1:]
        try:
            store.load_trace(tid)
            continue
        except KeyError:
            pass
        t.request.params = {**t.request.params, "max_tokens": MAX_NEW}
        comp = await hf.generate(t.request.model_copy(update={"params": {**t.request.params, "temperature": 0}}))
        rec = Trace(id=tid, request=t.request.model_copy(update={"model": MODEL}), response=comp.as_response(),
                    meta={**t.meta, "source_trace": t.id})
        store.save_trace(rec, segment_trace(rec, sc, counter))
        log(f"recorded {tid} ({t.meta['intent']}): {comp.text[:60]!r} calls={[c.name for c in comp.tool_calls]}")
    prof_ids = ["q" + t.id[1:] for t in profiled]
    held_ids = ["q" + t.id[1:] for t in held]

    def progress(n, total, gate):
        log(f"  {n}/{total} traces | calls {gate.calls} | cache hits {gate.cache_hits}")

    runs = {}
    for mode in ("logprob", "blackbox"):
        cfg = ProfileConfig(n_masks=N_MASKS, mode=mode, concurrency=1, budget_usd=100.0, n_boot=30)
        log(f"profiling {mode} ({N_MASKS} masks x {len(prof_ids)} traces)")
        t0 = time.time()
        runs[mode] = await run_profile(store, hf, cfg, counter, prof_ids, on_progress=progress)
        log(f"{mode} run {runs[mode]} done in {(time.time() - t0) / 60:.1f} min")

    stats = ranking_stats(store, runs["logprob"], runs["blackbox"], prof_ids)
    fits = {m: [store.trace_fit(r, t) for t in prof_ids] for m, r in runs.items()}

    rules = LabelRules(min_traces=len(prof_ids))
    rows = store.run_rows(runs["logprob"])
    agg = aggregate(rows, rules, exclude_poor_fit=False)
    labels = defaultdict(list)
    for a in agg:
        labels[a["label"] or "-"].append(a["fingerprint"])
    plan = build_plan(agg, Bundle.load(paths["prompt"]), injection=injection_signals(rows, rules),
                      heading_levels=sc.heading_levels, n_traces=len(prof_ids))
    pid = store.create_prune(runs["logprob"], paths["prompt"].as_posix(), [], plan.model_dump(mode="json"))
    log(f"prune {pid}: cutting {plan.removed}")
    vcfg = VerifyConfig(budget_usd=100.0, concurrency=1)
    report = await run_verify(store, hf, pid, vcfg, counter, trace_ids=held_ids)
    s = report["summary"]
    log(f"verify: {s['agree_n']}/{s['traces']} agree, tokens {s['tokens_before']:.0f} -> {s['tokens_after']:.0f}")

    results = {
        "model": MODEL, "device": "cpu", "n_masks": N_MASKS, "profiled": prof_ids, "held_out": held_ids,
        "runs": runs, "ranking": stats,
        "fits": {m: [{"trace_id": f["trace_id"], "r2_heldout": f["r2_heldout"], "poor_fit": bool(f["poor_fit"])}
                     for f in fs] for m, fs in fits.items()},
        "labels_logprob": dict(labels),
        "prune": {"id": pid, "removed": plan.removed, "skipped": [x.model_dump() for x in plan.skipped],
                  "summary": s, "regressions": report["regressions"], "calls": report["calls"]},
        "responses": {t: store.load_trace(t)[0].response.model_dump() for t in prof_ids + held_ids},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
