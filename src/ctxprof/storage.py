"""SQLite storage for traces, segments, profiling runs, attributions and the model-call cache."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import Locator, Segment, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  request_json TEXT NOT NULL,
  response_json TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  total_tokens INTEGER NOT NULL DEFAULT 0,
  imported_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS segments (
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  label TEXT NOT NULL,
  tokens INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  locator_json TEXT NOT NULL,
  parent TEXT,
  PRIMARY KEY (trace_id, idx)
);
CREATE INDEX IF NOT EXISTS segments_fp ON segments(fingerprint);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at REAL NOT NULL,
  config_json TEXT NOT NULL,
  status TEXT NOT NULL,
  spent_usd REAL NOT NULL DEFAULT 0,
  calls INTEGER NOT NULL DEFAULT 0,
  cache_hits INTEGER NOT NULL DEFAULT 0,
  note TEXT
);
CREATE TABLE IF NOT EXISTS trace_fits (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  trace_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  n_masks INTEGER NOT NULL,
  alpha REAL NOT NULL,
  intercept REAL NOT NULL,
  original_score REAL NOT NULL,
  r2_heldout REAL,
  poor_fit INTEGER NOT NULL,
  PRIMARY KEY (run_id, trace_id)
);
CREATE TABLE IF NOT EXISTS attributions (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  trace_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  weight REAL,      -- NULL for pinned (never ablated) segments
  influence REAL,
  ci_lo REAL,
  ci_hi REAL,
  PRIMARY KEY (run_id, trace_id, idx)
);
CREATE TABLE IF NOT EXISTS ablations (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  trace_id TEXT NOT NULL,
  mask_idx INTEGER NOT NULL,
  mask TEXT NOT NULL,
  score REAL NOT NULL,
  response_text TEXT,
  PRIMARY KEY (run_id, trace_id, mask_idx)
);
CREATE TABLE IF NOT EXISTS interactions (
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  trace_id TEXT NOT NULL,
  a INTEGER NOT NULL,          -- segment idx
  b INTEGER NOT NULL,
  effect_a REAL NOT NULL,      -- score drop when a alone is removed
  effect_b REAL NOT NULL,
  effect_ab REAL NOT NULL,     -- score drop when both are removed
  interaction REAL NOT NULL,   -- effect_ab - (effect_a + effect_b); 0 if additive
  relative REAL NOT NULL,
  PRIMARY KEY (run_id, trace_id, a, b)
);
CREATE TABLE IF NOT EXISTS calls (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  model TEXT NOT NULL,
  value_json TEXT NOT NULL,
  cost_usd REAL NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prunes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,            -- unverified | verified | budget
  bundle TEXT NOT NULL,
  keep_json TEXT NOT NULL,
  removed_json TEXT NOT NULL,      -- fingerprints the diff removes
  hunks_json TEXT NOT NULL,
  skipped_json TEXT NOT NULL,
  diff TEXT NOT NULL,
  tokens_saved REAL NOT NULL,      -- per request: estimate from the run, then measured by verify
  agreement REAL,
  recommend INTEGER,
  regressions_json TEXT,
  verify_json TEXT,
  verified_at REAL
);
"""


class Store:
    def __init__(self, path: str | Path = "ctxprof.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(segments)")}
        if "parent" not in cols:  # databases created before sub-segments existed
            with self.conn:
                self.conn.execute("ALTER TABLE segments ADD COLUMN parent TEXT")

    def close(self) -> None:
        self.conn.close()

    # ---- traces & segments -------------------------------------------------
    def save_trace(self, trace: Trace, segments: list[Segment]) -> None:
        total = sum(s.tokens for s in segments)
        with self.conn:
            self.conn.execute("DELETE FROM segments WHERE trace_id = ?", (trace.id,))
            self.conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?)",
                (trace.id, trace.request.model, trace.request.model_dump_json(), trace.response.model_dump_json(),
                 json.dumps(trace.meta, ensure_ascii=False), total, time.time()),
            )
            self.conn.executemany(
                "INSERT INTO segments (trace_id, idx, kind, source, fingerprint, label, tokens, content_hash, "
                "locator_json, parent) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(s.trace_id, s.index, s.kind, s.source, s.fingerprint, s.label, s.tokens, s.content_hash,
                  s.locator.model_dump_json(), s.parent) for s in segments],
            )

    def trace_ids(self) -> list[str]:
        return [r["id"] for r in self.conn.execute("SELECT id FROM traces ORDER BY id")]

    def list_traces(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT t.id, t.model, t.total_tokens, t.meta_json, COUNT(s.idx) AS n_segments FROM traces t "
            "LEFT JOIN segments s ON s.trace_id = t.id GROUP BY t.id ORDER BY t.id"
        )
        out = []
        for r in rows:
            d = dict(r)
            meta = json.loads(d.pop("meta_json") or "{}")
            d["intent"] = meta.get("intent")
            out.append(d)
        return out

    def poor_fit_ids(self, run_id: int) -> dict[str, bool]:
        """trace id -> poor fit, for every trace the run profiled."""
        rows = self.conn.execute("SELECT trace_id, poor_fit FROM trace_fits WHERE run_id = ?", (run_id,))
        return {r["trace_id"]: bool(r["poor_fit"]) for r in rows}

    def load_trace(self, trace_id: str) -> tuple[Trace, list[Segment]]:
        row = self.conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(trace_id)
        trace = Trace.model_validate({
            "id": row["id"],
            "request": json.loads(row["request_json"]),
            "response": json.loads(row["response_json"]),
            "meta": json.loads(row["meta_json"]),
        })
        segs = [
            Segment(trace_id=r["trace_id"], index=r["idx"], kind=r["kind"], source=r["source"],
                    fingerprint=r["fingerprint"], label=r["label"], tokens=r["tokens"],
                    content_hash=r["content_hash"], locator=Locator.model_validate_json(r["locator_json"]),
                    parent=r["parent"])
            for r in self.conn.execute("SELECT * FROM segments WHERE trace_id = ? ORDER BY idx", (trace_id,))
        ]
        return trace, segs

    # ---- runs --------------------------------------------------------------
    def create_run(self, config: dict[str, Any]) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (created_at, config_json, status) VALUES (?,?,?)",
                (time.time(), json.dumps(config), "running"),
            )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def finish_run(self, run_id: int, status: str, spent: float, calls: int, cache_hits: int,
                   note: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status=?, spent_usd=?, calls=?, cache_hits=?, note=? WHERE id=?",
                (status, spent, calls, cache_hits, note, run_id),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for r in self.conn.execute("SELECT * FROM runs ORDER BY id DESC"):
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["n_traces"] = self.conn.execute(
                "SELECT COUNT(*) FROM trace_fits WHERE run_id = ?", (r["id"],)).fetchone()[0]
            out.append(d)
        return out

    def latest_run_id(self) -> int | None:
        row = self.conn.execute("SELECT id FROM runs WHERE status IN ('done','budget') ORDER BY id DESC").fetchone()
        return int(row["id"]) if row else None

    def save_attribution(self, run_id: int, trace_id: str, fit: dict[str, Any], rows: Iterable[tuple],
                         ablations: Iterable[tuple], interactions: Iterable[tuple] = ()) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO trace_fits VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, trace_id, fit["mode"], fit["n_masks"], fit["alpha"], fit["intercept"],
                 fit["original_score"], fit["r2_heldout"], int(fit["poor_fit"])),
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO attributions VALUES (?,?,?,?,?,?,?)",
                [(run_id, trace_id, *r) for r in rows],
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO ablations VALUES (?,?,?,?,?,?)",
                [(run_id, trace_id, *a) for a in ablations],
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO interactions VALUES (?,?,?,?,?,?,?,?,?)",
                [(run_id, trace_id, *x) for x in interactions],
            )

    def interactions(self, run_id: int, trace_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT i.*, sa.fingerprint AS fp_a, sb.fingerprint AS fp_b FROM interactions i "
            "JOIN segments sa ON sa.trace_id = i.trace_id AND sa.idx = i.a "
            "JOIN segments sb ON sb.trace_id = i.trace_id AND sb.idx = i.b "
            "WHERE i.run_id = ? AND i.trace_id = ? ORDER BY ABS(i.relative) DESC", (run_id, trace_id))
        return [{k: r[k] for k in r.keys() if k not in ("run_id", "trace_id")} for r in rows]

    def attributions(self, run_id: int, trace_id: str) -> dict[int, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT idx, weight, influence, ci_lo, ci_hi FROM attributions WHERE run_id=? AND trace_id=?",
            (run_id, trace_id),
        )
        return {r["idx"]: dict(r) for r in rows}

    def trace_fit(self, run_id: int, trace_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM trace_fits WHERE run_id=? AND trace_id=?", (run_id, trace_id)).fetchone()
        return dict(r) if r else None

    def ablations(self, run_id: int, trace_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT mask_idx, mask, score, response_text FROM ablations WHERE run_id=? AND trace_id=? "
            "ORDER BY mask_idx", (run_id, trace_id))
        return [dict(r) for r in rows]

    def run_rows(self, run_id: int) -> list[dict[str, Any]]:
        """Joined attribution + segment rows for aggregation."""
        rows = self.conn.execute(
            "SELECT a.trace_id, a.idx, a.weight, a.influence, s.kind, s.source, s.fingerprint, s.label, s.tokens, "
            "s.parent, f.poor_fit FROM attributions a "
            "JOIN segments s ON s.trace_id = a.trace_id AND s.idx = a.idx "
            "JOIN trace_fits f ON f.run_id = a.run_id AND f.trace_id = a.trace_id "
            "WHERE a.run_id = ? AND a.influence IS NOT NULL", (run_id,))
        return [dict(r) for r in rows]

    def run_trace_ids(self, run_id: int) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT trace_id FROM trace_fits WHERE run_id = ? ORDER BY trace_id", (run_id,))]

    def tool_call_counts(self, trace_ids: Iterable[str] | None = None) -> dict[str, int]:
        """How many recorded responses call each tool (evidence shown next to a tool hunk)."""
        wanted = set(trace_ids) if trace_ids is not None else None
        counts: dict[str, int] = {}
        for r in self.conn.execute("SELECT id, response_json FROM traces"):
            if wanted is not None and r["id"] not in wanted:
                continue
            names = {c.get("name") for c in json.loads(r["response_json"]).get("tool_calls", [])}
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        return counts

    # ---- prunes ------------------------------------------------------------
    def create_prune(self, run_id: int, bundle: str, keep: list[str], plan: dict[str, Any]) -> int:
        removed = [h["fingerprint"] for h in plan["hunks"] if h["cut"]]
        saved = sum(h["tokens_per_request"] for h in plan["hunks"] if h["cut"])
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO prunes (run_id, created_at, status, bundle, keep_json, removed_json, hunks_json, "
                "skipped_json, diff, tokens_saved) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, time.time(), "unverified", bundle, json.dumps(keep), json.dumps(removed),
                 json.dumps(plan["hunks"], ensure_ascii=False), json.dumps(plan["skipped"], ensure_ascii=False),
                 plan["diff"], saved),
            )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def save_verification(self, prune_id: int, report: dict[str, Any]) -> None:
        s = report["summary"]
        with self.conn:
            self.conn.execute(
                "UPDATE prunes SET status=?, tokens_saved=?, agreement=?, recommend=?, regressions_json=?, "
                "verify_json=?, verified_at=? WHERE id=?",
                (report["status"], s["tokens_saved"], s["agreement"], int(s["recommend"]),
                 json.dumps(report["regressions"], ensure_ascii=False), json.dumps(report, ensure_ascii=False),
                 time.time(), prune_id),
            )

    @staticmethod
    def _prune_row(r: sqlite3.Row, full: bool) -> dict[str, Any]:
        d = {k: r[k] for k in ("id", "run_id", "created_at", "status", "bundle", "tokens_saved", "agreement",
                               "verified_at")}
        d["recommend"] = None if r["recommend"] is None else bool(r["recommend"])
        d["keep"] = json.loads(r["keep_json"])
        d["removed"] = json.loads(r["removed_json"])
        regs = json.loads(r["regressions_json"]) if r["regressions_json"] else []
        d["n_regressions"] = len(regs)
        if full:
            d["hunks"] = json.loads(r["hunks_json"])
            d["skipped"] = json.loads(r["skipped_json"])
            d["diff"] = r["diff"]
            d["regressions"] = regs
            d["verification"] = json.loads(r["verify_json"]) if r["verify_json"] else None
        return d

    def list_prunes(self) -> list[dict[str, Any]]:
        return [self._prune_row(r, False) for r in self.conn.execute("SELECT * FROM prunes ORDER BY id DESC")]

    def get_prune(self, prune_id: int) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM prunes WHERE id = ?", (prune_id,)).fetchone()
        return self._prune_row(r, True) if r else None

    def latest_prune_id(self) -> int | None:
        r = self.conn.execute("SELECT id FROM prunes ORDER BY id DESC").fetchone()
        return int(r["id"]) if r else None

    # ---- model-call cache -------------------------------------------------
    def cache_get(self, key: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT value_json FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(r["value_json"]) if r else None

    def cache_put(self, key: str, kind: str, model: str, value: dict[str, Any], cost: float) -> None:
        # No commit per call: the surrounding trace's save_attribution / finish_run commits.
        self.conn.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?)",
            (key, kind, model, json.dumps(value, ensure_ascii=False), cost, time.time()),
        )

    def commit(self) -> None:
        self.conn.commit()
