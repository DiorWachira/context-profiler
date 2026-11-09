"""Build the context flame graph: one bar per request, width = tokens, nested by segment kind."""

from __future__ import annotations

from typing import Any

from .models import KIND_ORDER, Segment

KIND_LABEL = {
    "system": "System prompt",
    "tool": "Tool schemas",
    "retrieved": "Retrieved chunks",
    "history": "History",
    "user": "User turn",
}


def flame_tree(trace_id: str, segments: list[Segment],
               attributions: dict[int, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a nested tree ``{name, tokens, offset, influence, children}``.

    ``offset`` is the token position in window order, so the UI can draw a ruler.
    Influence is absent (``None``) until a profiling run has attributed the trace;
    kind-level influence is the sum of its segments' positive influence, capped at 1.
    """
    attributions = attributions or {}
    root: dict[str, Any] = {"id": "root", "name": f"request {trace_id}", "kind": "request", "tokens": 0,
                            "offset": 0, "influence": None, "children": []}
    offset = 0
    for kind in KIND_ORDER:
        segs = [s for s in segments if s.kind == kind]
        if not segs:
            continue
        node: dict[str, Any] = {"id": f"kind/{kind}", "name": KIND_LABEL[kind], "kind": kind, "tokens": 0,
                                "offset": offset, "influence": None, "children": []}
        infl_sum, scored = 0.0, False
        groups: dict[str, dict[str, Any]] = {}
        for s in segs:
            a = attributions.get(s.index)
            infl = a["influence"] if a else None
            if infl is not None:
                scored = True
                infl_sum += max(0.0, infl)
            leaf = {
                "id": s.fingerprint, "name": s.label, "fingerprint": s.fingerprint, "kind": s.kind,
                "source": s.source, "index": s.index, "tokens": s.tokens, "offset": offset,
                "influence": infl, "weight": a["weight"] if a else None,
                "ci": [a["ci_lo"], a["ci_hi"]] if a and a.get("ci_lo") is not None else None,
                "children": [],
            }
            if s.parent:  # sub-segment: group under a synthesized parent node (depth 3 in the UI)
                g = groups.get(s.parent)
                if g is None:
                    g = groups[s.parent] = {
                        "id": s.parent, "name": s.parent.split("/", 1)[-1], "fingerprint": s.parent,
                        "kind": s.kind, "source": s.source, "tokens": 0, "offset": offset, "influence": None,
                        "children": [],
                    }
                    node["children"].append(g)
                g["children"].append(leaf)
                g["tokens"] += s.tokens
                if infl is not None:
                    g["influence"] = min(1.0, (g["influence"] or 0.0) + max(0.0, infl))
            else:
                node["children"].append(leaf)
            offset += s.tokens
            node["tokens"] += s.tokens
        node["influence"] = min(1.0, infl_sum) if scored else None
        root["children"].append(node)
        root["tokens"] += node["tokens"]
    return root
