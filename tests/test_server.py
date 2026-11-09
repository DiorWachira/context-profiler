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
