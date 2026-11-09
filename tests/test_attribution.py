"""Synthetic ground truth: a fake model whose output depends only on known segments."""

import asyncio

import numpy as np

from ctxprof.aggregate import aggregate, injection_signals
from ctxprof.attribution import ProfileConfig, run_profile
from ctxprof.demo import SEGMENTER_YAML, demo_provider, generate_traces
from ctxprof.providers import Effect, FakeProvider
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.surrogate import fit_surrogate


def test_surrogate_recovers_sparse_weights():
    rng = np.random.default_rng(0)
    X = rng.random((64, 12)) < 0.5
    true = np.zeros(12)
    true[[1, 4, 9]] = [2.0, 1.0, 3.0]
    y = X @ true + rng.normal(0, 0.05, 64)
    fit = fit_surrogate(X, y, seed=0)
    assert set(np.argsort(fit.weights)[-3:]) == {1, 4, 9}
    assert fit.r2_heldout is not None and fit.r2_heldout > 0.9
    assert np.all(fit.ci_lo <= fit.weights + 1e-9) and np.all(fit.weights <= fit.ci_hi + 1e-9)


def _profile(store, provider, traces, cfg, counter):
    sc = SegmenterConfig.model_validate(SEGMENTER_YAML)
    for t in traces:
        store.save_trace(t, segment_trace(t, sc, counter))
    return asyncio.run(run_profile(store, provider, cfg, counter))


def test_top_k_matches_true_causal_segments(store, counter):
    traces = [t for t in generate_traces(40, seed=3) if t.meta["intent"] == "status"][:6]
    assert traces, "need some status traces"
    cfg = ProfileConfig(n_masks=48, budget_usd=1e9, n_boot=10)
    run_id = _profile(store, demo_provider(), traces, cfg, counter)
    for t in traces:
        _, segs = store.load_trace(t.id)
        attrs = store.attributions(run_id, t.id)
        assert attrs[segs[-1].index]["weight"] is None, "the user turn is pinned, not attributed"
        scored = [s for s in segs if attrs[s.index]["weight"] is not None]
        ranked = sorted(scored, key=lambda s: attrs[s.index]["weight"], reverse=True)
        # true causal segments for a status question (weights 3.0, 1.5, 0.8 nats) ...
        expected = ["tool/get_application_status", "rag/kb:application-status#1", "system/## Role"]
        if t.meta["sheng"]:
            expected.append("system/## Swahili & Sheng replies")  # 1.2 nats
        assert {s.fingerprint for s in ranked[:len(expected)]} == set(expected), (t.id, ranked[:5])
        dead = {"system/## Refund policy (2023, legacy)", "system/## Company history", "tool/get_weather"}
        for s in segs:
            if s.fingerprint in dead:
                assert abs(attrs[s.index]["influence"]) < 0.1, (s.fingerprint, attrs[s.index])
        assert store.trace_fit(run_id, t.id)["r2_heldout"] > 0.5


def test_cache_makes_rerun_free_and_budget_caps_spend(store, counter):
    traces = generate_traces(3, seed=1)
    cfg = ProfileConfig(n_masks=16, budget_usd=1e9, n_boot=5)
    prov = demo_provider()
    r1 = _profile(store, prov, traces, cfg, counter)
    calls_first = prov.calls
    r2 = asyncio.run(run_profile(store, prov, cfg, counter))
    runs = {r["id"]: r for r in store.list_runs()}
    assert prov.calls == calls_first, "second run must be served entirely from cache"
    assert runs[r2]["cache_hits"] == runs[r1]["calls"] and runs[r2]["spent_usd"] == 0
    assert store.attributions(r1, traces[0].id) == store.attributions(r2, traces[0].id)

    tight = ProfileConfig(n_masks=16, budget_usd=0.05, n_boot=5, seed=99)
    r3 = asyncio.run(run_profile(store, demo_provider(), tight, counter))
    run3 = {r["id"]: r for r in store.list_runs()}[r3]
    assert run3["status"] == "budget" and run3["spent_usd"] <= 0.05


def test_blackbox_mode_with_generated_text(store, counter):
    effects = [Effect(needs=["## Role"], weight=1, phrase="I am the JobBridge assistant."),
               Effect(needs=["[tool:get_application_status]"], weight=1, phrase="Status: Shortlisted.",
                      tool_call="get_application_status")]
    traces = generate_traces(2, seed=5)
    prov = FakeProvider(effects)
    for t in traces:  # re-record the responses with this model
        t.response = asyncio.run(prov.generate(t.request)).as_response()
    run_id = _profile(store, prov, traces, ProfileConfig(mode="blackbox", n_masks=32, budget_usd=1e9, n_boot=5),
                      counter)
    _, segs = store.load_trace(traces[0].id)
    attrs = store.attributions(run_id, traces[0].id)
    scored = [s for s in segs if attrs[s.index]["weight"] is not None]
    top2 = {s.fingerprint for s in sorted(scored, key=lambda s: attrs[s.index]["weight"], reverse=True)[:2]}
    assert top2 == {"system/## Role", "tool/get_application_status"}


def test_aggregate_labels_and_injection_signal(store, counter):
    traces = generate_traces(60, seed=11)
    run_id = _profile(store, demo_provider(), traces, ProfileConfig(n_masks=32, budget_usd=1e9, n_boot=5), counter)
    rows = store.run_rows(run_id)
    agg = {a["fingerprint"]: a for a in aggregate(rows)}
    assert agg["system/## Company history"]["label"] == "DEAD"
    assert agg["tool/get_weather"]["label"] == "DEAD"
    assert agg["system/## Role"]["mean_influence"] > agg["system/## Holiday hours 2024"]["mean_influence"]
    assert "user/current" not in agg  # pinned
    assert agg["tool/search_jobs"]["label"] in ("CRITICAL", "VOLATILE")
    hits = injection_signals(rows)
    scam = {t.id for t in traces if t.meta["intent"] == "scam"}
    assert scam and any(h["trace_id"] in scam and h["fingerprint"].startswith("rag/listing:") for h in hits)
