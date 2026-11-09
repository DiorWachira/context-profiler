"""Verification on held-out traces against the fake ground-truth model."""

import asyncio
import sys

from ctxprof.demo import SEGMENTER_YAML, demo_provider, generate_traces
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.verify import VerifyConfig, choose_held_out, run_verify, summarise

HUNKS = ["system/## Company history", "system/## Refund policy (2023, legacy)", "tool/get_weather",
         "tool/report_listing", "rag/kb:press-2019#2"]


def setup(store, counter, n=40, seed=4):
    sc = SegmenterConfig.model_validate(SEGMENTER_YAML)
    traces = generate_traces(n, seed=seed)
    for t in traces:
        store.save_trace(t, segment_trace(t, sc, counter))
    return traces, store.create_run({"test": True})


def make_prune(store, run_id, keep=()):
    hunks = [{"fingerprint": fp, "kind": fp.split("/")[0], "file": "x", "traces": 40, "tokens_per_request": 100.0,
              "mean_influence": 0.0, "p95_influence": 0.0, "reason": "", "cut": fp not in keep, "lines": []}
             for fp in HUNKS]
    return store.create_prune(run_id, "bundle", list(keep), {"hunks": hunks, "skipped": [], "diff": ""})


def test_verify_finds_report_listing_regression_and_keep_fixes_it(store, counter):
    traces, run_id = setup(store, counter)
    scam = {t.id for t in traces if t.meta["intent"] == "scam"}
    assert scam, "fixture needs scam traces"

    pid = make_prune(store, run_id)
    rep = asyncio.run(run_verify(store, demo_provider(), pid, VerifyConfig(), counter))
    s = rep["summary"]
    assert {r["trace_id"] for r in rep["regressions"]} == scam
    assert all(r["breaks"] == ["tool/report_listing"] and r["explained"] for r in rep["regressions"])
    assert s["agree_n"] == len(traces) - len(scam) and s["tokens_after"] < s["tokens_before"]
    assert not rep["in_sample"]  # nothing was profiled, so every trace is held out
    stored = store.get_prune(pid)
    assert stored["status"] == "verified" and stored["n_regressions"] == len(scam)
    assert stored["recommend"] is (s["agreement"] >= 0.95)

    # the per-hunk estimate for "keep report_listing" ...
    est = summarise(rep["records"], set(HUNKS) - {"tool/report_listing"}, VerifyConfig())
    assert est["agree_n"] == len(traces) and est["recommend"]

    # ... matches a real re-verification with --keep, which also probes the kept hunk
    pid2 = make_prune(store, run_id, keep=["tool/report_listing"])
    rep2 = asyncio.run(run_verify(store, demo_provider(), pid2, VerifyConfig(), counter))
    assert rep2["summary"]["agree_n"] == len(traces) and rep2["summary"]["recommend"]
    assert rep2["summary"]["tokens_saved"] == est["tokens_saved"]
    probed = {r["trace_id"] for r in rep2["records"] if "tool/report_listing" in r["breaks"]}
    assert probed == scam
    again = summarise(rep2["records"], set(HUNKS), VerifyConfig())  # cut it back in: estimate = first run
    assert again["agree_n"] == s["agree_n"]
    assert rep2["cache_hits"] > 0  # original replays are shared with the first verification


def test_held_out_pool_excludes_profiled_traces(store, counter):
    traces, run_id = setup(store, counter, n=10)
    fit = {"mode": "blackbox", "n_masks": 1, "alpha": 0, "intercept": 0, "original_score": 1, "r2_heldout": 1,
           "poor_fit": False}
    for t in traces[:6]:
        store.save_attribution(run_id, t.id, fit, [], [])
    ids, in_sample = choose_held_out(store, run_id, None)
    assert ids == [t.id for t in traces[6:]] and not in_sample
    assert choose_held_out(store, run_id, 2, seed=1)[0] == sorted(choose_held_out(store, run_id, 2, seed=1)[0])
    assert len(choose_held_out(store, run_id, 2)[0]) == 2


def test_eval_command_decides_agreement(store, counter):
    _, run_id = setup(store, counter, n=6)
    pid = make_prune(store, run_id)
    always_fail = f'"{sys.executable}" -c "import sys; sys.exit(1)" {{original}} {{pruned}}'
    rep = asyncio.run(run_verify(store, demo_provider(), pid, VerifyConfig(eval_cmd=always_fail, probe=False),
                                 counter))
    assert rep["summary"]["agree_n"] == 0 and not rep["summary"]["recommend"]


def test_budget_stops_verification(store, counter):
    _, run_id = setup(store, counter, n=6)
    pid = make_prune(store, run_id)
    rep = asyncio.run(run_verify(store, demo_provider(), pid, VerifyConfig(budget_usd=0.02), counter))
    assert rep["status"] == "budget" and not rep["summary"]["recommend"] and rep["spent_usd"] <= 0.02
