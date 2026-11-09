"""Outcome metrics: how much of the original answer survives an ablation.

* ``logprob`` mode scores the logit-scaled probability of the original response (ContextCite).
* ``blackbox`` mode scores 1 - normalised change, from text similarity, tool-call agreement or
  JSON field agreement. Text similarity is a character 3-gram cosine by default; an embedding
  model can replace it (see roadmap).
"""

from __future__ import annotations

import json
import math
from collections import Counter

from .models import Response


def _grams(text: str, n: int = 3) -> Counter[str]:
    t = " ".join(text.lower().split())
    if len(t) < n:
        return Counter([t]) if t else Counter()
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def text_similarity(a: str, b: str) -> float:
    ga, gb = _grams(a), _grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    dot = sum(v * gb[k] for k, v in ga.items())
    return dot / (math.sqrt(sum(v * v for v in ga.values())) * math.sqrt(sum(v * v for v in gb.values())))


def tool_call_agreement(a: Response, b: Response) -> float:
    """1 if the same tools are called with the same arguments; partial credit for same names."""
    if not a.tool_calls and not b.tool_calls:
        return 1.0
    if not a.tool_calls or not b.tool_calls:
        return 0.0
    total = 0.0
    for call in a.tool_calls:
        best = 0.0
        for other in b.tool_calls:
            if call.name == other.name:
                best = max(best, 1.0 if call.arguments == other.arguments else 0.5)
        total += best
    return total / max(len(a.tool_calls), len(b.tool_calls))


def json_field_agreement(a: str, b: str) -> float:
    try:
        ja, jb = json.loads(a), json.loads(b)
    except (json.JSONDecodeError, TypeError):
        return text_similarity(a, b)
    if not isinstance(ja, dict) or not isinstance(jb, dict):
        return 1.0 if ja == jb else 0.0
    keys = set(ja) | set(jb)
    return sum(ja.get(k) == jb.get(k) for k in keys) / len(keys) if keys else 1.0


def blackbox_score(original: Response, ablated: Response, metric: str = "auto") -> float:
    if metric == "text":
        return text_similarity(original.text, ablated.text)
    if metric == "tool":
        return tool_call_agreement(original, ablated)
    if metric == "json":
        return json_field_agreement(original.text, ablated.text)
    if metric == "auto":
        if original.tool_calls:
            return 0.5 * tool_call_agreement(original, ablated) + 0.5 * text_similarity(original.text, ablated.text)
        return text_similarity(original.text, ablated.text)
    raise ValueError(f"unknown metric {metric!r}")
