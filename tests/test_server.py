import asyncio

from fastapi.testclient import TestClient

from ctxprof.attribution import ProfileConfig, run_profile
from ctxprof.demo import SEGMENTER_YAML, demo_provider, generate_traces
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.server import create_app
from ctxprof.storage import Store
from ctxprof.tokens import HeuristicCounter


def test_api_end_to_end(tmp_path):
    db = tmp_path / "api.db"
    store, counter = Store(db), HeuristicCounter()
    sc = SegmenterConfig.model_validate(SEGMENTER_YAML)
    traces = generate_traces(4, seed=2)
    for t in traces:
        store.save_trace(t, segment_trace(t, sc, counter))

    client = TestClient(create_app(db, web_dist=None))
    assert client.get("/api/health").json() == {"status": "ok"}
    listed = client.get("/api/traces").json()
    assert [t["id"] for t in listed] == [t.id for t in traces]
    assert listed[0]["intent"] == traces[0].meta["intent"] and listed[0]["profiled"] is False
    flame = client.get(f"/api/traces/{traces[0].id}/flame").json()
    assert flame["run"] is None and flame["tree"]["tokens"] == listed[0]["total_tokens"]
    assert [c["kind"] for c in flame["tree"]["children"]][:2] == ["system", "tool"]
    assert sum(c["tokens"] for c in flame["tree"]["children"]) == flame["tree"]["tokens"]

    run_id = asyncio.run(run_profile(store, demo_provider(), ProfileConfig(n_masks=16, budget_usd=1e9, n_boot=5),
                                     counter))
    flame = client.get(f"/api/traces/{traces[0].id}/flame").json()
    assert flame["run"] == run_id
    leaves = [leaf for kind in flame["tree"]["children"] for leaf in kind["children"]]
    assert all((leaf["influence"] is None) == (leaf["kind"] == "user") for leaf in leaves)
    detail = client.get(f"/api/traces/{traces[0].id}").json()
    assert len(detail["ablations"]) == 16 and detail["fit"]["n_masks"] == 16
    agg = client.get(f"/api/runs/{run_id}/aggregate").json()
    assert agg["summary"]["traces"] == 4 and agg["segments"]
    assert client.get("/api/traces/nope").status_code == 404
    store.close()


def test_prune_endpoints(tmp_path):
    from ctxprof.verify import VerifyConfig, run_verify

    db = tmp_path / "api.db"
    store, counter = Store(db), HeuristicCounter()
    sc = SegmenterConfig.model_validate(SEGMENTER_YAML)
    for t in generate_traces(6, seed=2):
        store.save_trace(t, segment_trace(t, sc, counter))
    run_id = store.create_run({})
    hunk = {"fingerprint": "tool/get_weather", "kind": "tool", "file": "tools.json", "traces": 6,
            "tokens_per_request": 113.0, "mean_influence": 0.0, "p95_influence": 0.0, "reason": "DEAD", "cut": True,
            "lines": [{"no": 1, "sign": "-", "text": "get_weather"}]}
    pid = store.create_prune(run_id, "examples/demo/prompt", [], {"hunks": [hunk], "skipped": [], "diff": "--- a"})
    client = TestClient(create_app(db, web_dist=None))
    listed = client.get("/api/prunes").json()
    assert listed[0]["id"] == pid and listed[0]["status"] == "unverified" and "hunks" not in listed[0]
    asyncio.run(run_verify(store, demo_provider(), pid, VerifyConfig(), counter))
    detail = client.get(f"/api/prunes/{pid}").json()
    assert detail["status"] == "verified" and detail["recommend"] is True and detail["run"]["id"] == run_id
    assert detail["verification"]["summary"]["traces"] == 6 and detail["hunks"][0]["lines"]
    assert client.get("/api/prunes/999").status_code == 404
    store.close()
