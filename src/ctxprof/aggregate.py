"""Aggregate attributions across traces by segment fingerprint and label them."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LabelRules:
    dead_eps: float = 0.05  # |mean influence| below this ...
    dead_p95: float = 0.10  # ... and 95th percentile below this ...
    min_traces: int = 20  # ... across at least this many traces => DEAD
    critical: float = 0.5  # mean influence at or above => CRITICAL
    volatile_sd: float = 0.15  # standard deviation at or above => VOLATILE
    injection_top_fraction: float = 0.1  # untrusted segment within the top 10% of its trace ...
    injection_min: float = 0.3  # ... and with influence at least this => injection signal


def label_for(stats: dict[str, Any], rules: LabelRules) -> str:
    if stats["mean_influence"] >= rules.critical:
        return "CRITICAL"
    if stats["sd_influence"] >= rules.volatile_sd:
        return "VOLATILE"
    if (stats["traces"] >= rules.min_traces and abs(stats["mean_influence"]) < rules.dead_eps
            and stats["p95_influence"] < rules.dead_p95):
        return "DEAD"
    return ""


def aggregate(rows: list[dict[str, Any]], rules: LabelRules | None = None,
              exclude_poor_fit: bool = True) -> list[dict[str, Any]]:
    """``rows`` come from :meth:`Store.run_rows`. Returns one dict per fingerprint."""
    rules = rules or LabelRules()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if exclude_poor_fit and r["poor_fit"]:
            continue
        groups[r["fingerprint"]].append(r)
    out = []
    for fp, rs in groups.items():
        infl = np.array([r["influence"] for r in rs], dtype=float)
        tokens = np.array([r["tokens"] for r in rs], dtype=float)
        mean = float(infl.mean())
        stats = {
            "fingerprint": fp,
            "kind": rs[0]["kind"],
            "source": rs[0]["source"],
            "label_name": rs[0]["label"],
            "parent": rs[0].get("parent"),
            "traces": len(rs),
            "tokens_mean": float(tokens.mean()),
            "tokens_paid": int(tokens.sum()),
            "mean_influence": mean,
            "sd_influence": float(infl.std(ddof=1)) if len(rs) > 1 else 0.0,
            "p95_influence": float(np.percentile(np.abs(infl), 95)),
            "influence_per_1k": mean / tokens.mean() * 1000 if tokens.mean() else 0.0,
        }
        stats["label"] = label_for(stats, rules)
        out.append(stats)
    out.sort(key=lambda s: s["tokens_paid"], reverse=True)
    return out


def injection_signals(rows: list[dict[str, Any]], rules: LabelRules | None = None) -> list[dict[str, Any]]:
    """Untrusted segments (other than the current user turn) unusually influential in their trace."""
    rules = rules or LabelRules()
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_trace[r["trace_id"]].append(r)
    hits = []
    for tid, rs in by_trace.items():
        ranked = sorted(rs, key=lambda r: r["influence"], reverse=True)
        top_n = max(1, math.ceil(len(ranked) * rules.injection_top_fraction))
        for rank, r in enumerate(ranked[:top_n], start=1):
            if r["source"] == "untrusted" and r["kind"] != "user" and r["influence"] >= rules.injection_min:
                hits.append({"trace_id": tid, "fingerprint": r["fingerprint"], "influence": r["influence"],
                             "rank": rank, "of": len(ranked)})
    hits.sort(key=lambda h: h["influence"], reverse=True)
    return hits
