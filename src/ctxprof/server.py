"""FastAPI app: JSON API for the web UI, plus the built UI itself when ``web/dist`` exists."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .aggregate import aggregate, injection_signals
from .flame import flame_tree
from .storage import Store

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


def create_app(db_path: str | Path = "ctxprof.db", web_dist: Path | None = WEB_DIST) -> FastAPI:
    app = FastAPI(title="Context Profiler", version="0.1.0")

    def store() -> Iterator[Store]:
        s = Store(db_path)
        try:
            yield s
        finally:
            s.close()

    def resolve_run(s: Store, run: int | None) -> int | None:
        return run if run is not None else s.latest_run_id()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def runs(s: Store = Depends(store)) -> list[dict[str, Any]]:
        return s.list_runs()

    @app.get("/api/traces")
    def traces(s: Store = Depends(store)) -> list[dict[str, Any]]:
        return s.list_traces()

    @app.get("/api/traces/{trace_id}/flame")
    def trace_flame(trace_id: str, run: int | None = None, s: Store = Depends(store)) -> dict[str, Any]:
        try:
            _, segs = s.load_trace(trace_id)
        except KeyError as exc:
            raise HTTPException(404, f"trace {trace_id} not found") from exc
        rid = resolve_run(s, run)
        attrs = s.attributions(rid, trace_id) if rid is not None else {}
        return {"run": rid if attrs else None, "fit": s.trace_fit(rid, trace_id) if rid else None,
                "tree": flame_tree(trace_id, segs, attrs)}

    @app.get("/api/traces/{trace_id}")
    def trace_detail(trace_id: str, run: int | None = None, s: Store = Depends(store)) -> dict[str, Any]:
        try:
            trace, segs = s.load_trace(trace_id)
        except KeyError as exc:
            raise HTTPException(404, f"trace {trace_id} not found") from exc
        rid = resolve_run(s, run)
        attrs = s.attributions(rid, trace_id) if rid is not None else {}
        user = [m.content for m in trace.request.messages if m.role == "user"]
        return {
            "id": trace.id, "model": trace.request.model, "meta": trace.meta,
            "question": user[-1] if user else "", "response": trace.response.model_dump(),
            "segments": [{**seg.model_dump(exclude={"locator"}), **(attrs.get(seg.index) or {})} for seg in segs],
            "run": rid if attrs else None,
            "fit": s.trace_fit(rid, trace_id) if rid else None,
            "ablations": s.ablations(rid, trace_id) if rid else [],
        }

    @app.get("/api/runs/{run_id}/aggregate")
    def run_aggregate(run_id: int, s: Store = Depends(store)) -> dict[str, Any]:
        rows = s.run_rows(run_id)
        if not rows:
            raise HTTPException(404, f"run {run_id} has no attributions")
        agg = aggregate(rows)
        dead = [a for a in agg if a["label"] == "DEAD"]
        traces = {r["trace_id"] for r in rows}
        mean_tokens = sum(r["tokens"] for r in rows) / max(1, len(traces))
        return {
            "run": run_id, "segments": agg, "injection": injection_signals(rows),
            "summary": {
                "traces": len(traces),
                "mean_tokens": mean_tokens,
                "dead_count": len(dead),
                "dead_tokens_per_request": sum(a["tokens_paid"] for a in dead) / max(1, len(traces)),
                "poor_fit": len({r["trace_id"] for r in rows if r["poor_fit"]}),
            },
        }

    @app.get("/api/prunes")
    def prunes(s: Store = Depends(store)) -> list[dict[str, Any]]:
        return s.list_prunes()

    @app.get("/api/prunes/{prune_id}")
    def prune_detail(prune_id: int, s: Store = Depends(store)) -> dict[str, Any]:
        p = s.get_prune(prune_id)
        if p is None:
            raise HTTPException(404, f"prune {prune_id} not found")
        p["run"] = next((r for r in s.list_runs() if r["id"] == p["run_id"]), None)
        return p

    if web_dist and web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app
