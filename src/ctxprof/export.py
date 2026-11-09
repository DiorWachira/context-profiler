"""Static snapshot of the API, for hosting the UI without a server (e.g. GitHub Pages).

Every GET route the UI uses is rendered through the real FastAPI app (so the snapshot cannot
drift from the API) and written as ``<out>/api/<path>.json``. A UI built with ``VITE_STATIC=1``
reads ``./api/<path>.json`` instead of ``/api/<path>``.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from .server import create_app
from .storage import Store


def export_static(db_path: str | Path, out_dir: str | Path) -> int:
    """Write the snapshot; returns the number of files written."""
    out = Path(out_dir)
    store = Store(db_path)
    try:
        trace_ids = store.trace_ids()
        run_ids = [r["id"] for r in store.list_runs() if r["status"] in ("done", "budget")]
        prune_ids = [p["id"] for p in store.list_prunes()]
    finally:
        store.close()

    paths = ["/api/health", "/api/runs", "/api/traces", "/api/prunes"]
    paths += [f"/api/runs/{r}/aggregate" for r in run_ids]
    paths += [f"/api/prunes/{p}" for p in prune_ids]
    for t in trace_ids:
        paths += [f"/api/traces/{t}", f"/api/traces/{t}/flame"]

    client = TestClient(create_app(db_path, web_dist=None))
    n = 0
    for path in paths:
        r = client.get(path)
        if r.status_code == 404 and "/aggregate" in path:
            continue  # a run with no attributions
        r.raise_for_status()
        target = out / (path.lstrip("/") + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(r.json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        n += 1
    return n
