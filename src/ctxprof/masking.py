"""Ablation masks: sample which segments to keep, and rebuild a request without the others."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .models import Request, Segment

_BLANKS = re.compile(r"\n{3,}")


def sample_masks(n_masks: int, n_segments: int, p_keep: float = 0.5, seed: int = 0) -> np.ndarray:
    """Boolean matrix (n_masks, n_segments); True = segment kept."""
    rng = np.random.default_rng(seed)
    masks = rng.random((n_masks, n_segments)) < p_keep
    # Guarantee every segment is both kept and dropped at least once when possible,
    # otherwise its weight is unidentifiable.
    if n_masks >= 2:
        for j in range(n_segments):
            if masks[:, j].all():
                masks[rng.integers(n_masks), j] = False
            elif not masks[:, j].any():
                masks[rng.integers(n_masks), j] = True
    return masks


def apply_mask(request: Request, segments: list[Segment], keep: np.ndarray | list[bool]) -> Request:
    """Return a copy of ``request`` with every segment where ``keep`` is False removed."""
    drop_tools: set[int] = set()
    drop_parts: dict[int, set[str]] = defaultdict(set)
    part_count: dict[int, int] = defaultdict(int)
    for seg in segments:
        if seg.locator.tool is not None and seg.locator.tool_part:
            part_count[seg.locator.tool] += 1
    drop_msgs: set[int] = set()
    cut: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seg, k in zip(segments, keep, strict=True):
        if k:
            continue
        loc = seg.locator
        if loc.tool is not None and loc.tool_part:
            drop_parts[loc.tool].add(loc.tool_part)
            if len(drop_parts[loc.tool]) == part_count[loc.tool]:
                drop_tools.add(loc.tool)
        elif loc.tool is not None:
            drop_tools.add(loc.tool)
        elif loc.whole_message is not None:
            drop_msgs.add(loc.whole_message)
        else:
            for sp in loc.spans:
                cut[sp.message].append((sp.start, sp.end))

    messages = []
    for i, msg in enumerate(request.messages):
        if i in drop_msgs:
            continue
        m = msg.model_copy(deep=True)
        if i in cut:
            text = m.content
            for s, e in sorted(cut[i], reverse=True):
                text = text[:s] + text[e:]
            m.content = _BLANKS.sub("\n\n", text).strip("\n")
        messages.append(m)

    messages = _repair_tool_pairs(messages)
    tools = []
    for i, t in enumerate(request.tools):
        if i in drop_tools:
            continue
        parts = drop_parts.get(i)
        if parts:
            t = t.model_copy(update={
                "description": "" if "description" in parts else t.description,
                "parameters": {"type": "object", "properties": {}} if "parameters" in parts else t.parameters,
            })
        tools.append(t)
    return request.model_copy(update={"messages": messages, "tools": tools})


def _repair_tool_pairs(messages):
    """Keep tool calls and tool results paired so providers accept the ablated request."""
    call_ids = set()
    for m in messages:
        for call in m.extra.get("tool_calls", []) or []:
            if isinstance(call, dict) and call.get("id"):
                call_ids.add(call["id"])
    kept = [
        m for m in messages
        if not (m.role == "tool" and m.extra.get("tool_call_id") and m.extra["tool_call_id"] not in call_ids)
    ]
    result_ids = {m.extra.get("tool_call_id") for m in kept if m.role == "tool"}
    for m in kept:
        calls = m.extra.get("tool_calls")
        if calls:
            m.extra["tool_calls"] = [c for c in calls if not isinstance(c, dict) or c.get("id") in result_ids]
            if not m.extra["tool_calls"]:
                m.extra.pop("tool_calls")
    return kept
