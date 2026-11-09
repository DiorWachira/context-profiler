"""Pruning: turn DEAD fingerprints into a reviewable diff against the prompt bundle.

A *prompt bundle* is the directory the application builds its requests from:

* ``system.md``      the system prompt; sections are removed by heading;
* ``tools.json``     a JSON list of tool schemas (``{"name": ...}`` or OpenAI ``{"function": {"name": ...}}``);
* ``retriever.yaml`` optional; DEAD chunk ids are appended to its ``exclude:`` list (created if missing).

Only DEAD segments are candidates. VOLATILE and CRITICAL segments are never DEAD by definition; on
top of that, untrusted segments, pinned kinds, injection-flagged fingerprints and conversation
history (not part of the bundle) are never removed. ``keep`` overrides a candidate: its hunk is
still listed (so the UI can show what keeping it costs) but it is not applied.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SYSTEM_FILE, TOOLS_FILE, RETRIEVER_FILE = "system.md", "tools.json", "retriever.yaml"
_HEADING = re.compile(r"^(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*$")
_RETRIEVER_HEADER = "# Chunk ids the retriever must never return. Appended by `ctxprof prune`.\n"


class DiffLine(BaseModel):
    no: int  # 1-based line number (old file for "-", new file for "+")
    sign: str  # "-" | "+"
    text: str


class Hunk(BaseModel):
    fingerprint: str
    kind: str
    file: str
    traces: int  # traces in the run that carried this segment
    tokens_per_request: float  # tokens saved per request, averaged over all traces in the run
    mean_influence: float
    p95_influence: float
    reason: str
    cut: bool = True
    lines: list[DiffLine] = Field(default_factory=list)


class Skipped(BaseModel):
    fingerprint: str
    kind: str
    label: str
    why: str


@dataclass
class Bundle:
    root: Path
    files: dict[str, str] = field(default_factory=dict)  # name -> original text (missing files absent)

    @classmethod
    def load(cls, root: str | Path) -> Bundle:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"prompt bundle {root} is not a directory")
        files = {n: (root / n).read_text(encoding="utf-8") for n in (SYSTEM_FILE, TOOLS_FILE, RETRIEVER_FILE)
                 if (root / n).is_file()}
        if SYSTEM_FILE not in files and TOOLS_FILE not in files:
            raise FileNotFoundError(f"{root} has neither {SYSTEM_FILE} nor {TOOLS_FILE}")
        return cls(root, files)


# ---- per-file editors ------------------------------------------------------------------------

def _sections(text: str, heading_levels: Iterable[int]) -> dict[str, tuple[int, int]]:
    """Map ``system/<title>`` fingerprints to line ranges [start, end) of ``text``.

    Mirrors :func:`ctxprof.segmenter.segment_trace`: a section runs from its heading to the next
    heading at a configured level; text before the first heading is ``system/preamble``.
    """
    levels = set(heading_levels)
    lines = text.splitlines()
    heads = [(i, f"{m.group('hashes')} {m.group('title')}") for i, ln in enumerate(lines)
             if (m := _HEADING.match(ln)) and len(m.group("hashes")) in levels]
    out: dict[str, tuple[int, int]] = {}
    first = heads[0][0] if heads else len(lines)
    if any(ln.strip() for ln in lines[:first]):
        out["system/preamble"] = (0, first)
    seen: dict[str, int] = {}
    for n, (start, title) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        fp = f"system/{title}"
        if fp in seen:  # duplicate heading: same disambiguation as the segmenter
            seen[fp] += 1
            fp = f"{fp} #{seen[fp]}"
        else:
            seen[fp] = 1
        out[fp] = (start, end)
    return out


def _tool_name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    if isinstance(entry.get("function"), dict):
        return entry["function"].get("name")
    return entry.get("name")


def _chunk_id(fingerprint: str) -> str:
    return fingerprint.removeprefix("rag/")


class _Editor:
    """Knows which fingerprints each bundle file holds and how to remove a set of them."""

    def __init__(self, bundle: Bundle, heading_levels: Iterable[int]) -> None:
        self.bundle = bundle
        sys_text = bundle.files.get(SYSTEM_FILE)
        self.sections = _sections(sys_text, heading_levels) if sys_text is not None else {}
        self.tools: list[Any] = []
        if TOOLS_FILE in bundle.files:
            data = json.loads(bundle.files[TOOLS_FILE])
            if not isinstance(data, list):
                raise ValueError(f"{TOOLS_FILE} must be a JSON list of tool schemas")
            self.tools = data
        self.tool_names = {n for t in self.tools if (n := _tool_name(t))}

    def file_for(self, fingerprint: str, kind: str) -> str | None:
        if kind == "system" and fingerprint in self.sections:
            return SYSTEM_FILE
        if kind == "tool" and fingerprint.removeprefix("tool/") in self.tool_names:
            return TOOLS_FILE
        if kind == "retrieved":
            return RETRIEVER_FILE
        return None

    def _tool_ranges(self) -> dict[str, tuple[int, int]] | None:
        """Line ranges [start, end) of each tool object, when tools.json is in canonical indent-2 form."""
        text = self.bundle.files.get(TOOLS_FILE)
        if text is None or text != json.dumps(self.tools, indent=2, ensure_ascii=False) + "\n":
            return None
        lines, start = text.splitlines(), None
        out: dict[str, tuple[int, int]] = {}
        for i, ln in enumerate(lines):
            if ln == "  {":
                start = i
            elif start is not None and ln in ("  }", "  },"):
                name = _tool_name(self.tools[len(out)]) if len(out) < len(self.tools) else None
                if name:
                    out[name] = (start, i + 1)
                start = None
        return out if len(out) == len(self.tools) else None

    def hunk_lines(self, fingerprint: str, name: str) -> list[DiffLine]:
        """The lines one hunk removes or adds, numbered against the original file."""
        original = self.bundle.files.get(name)
        if name == SYSTEM_FILE:
            s, e = self.sections[fingerprint]
            lines = (original or "").splitlines()
            while e > s and not lines[e - 1].strip():  # trailing blank separator: not interesting
                e -= 1
            return [DiffLine(no=i + 1, sign="-", text=lines[i]) for i in range(s, e)]
        if name == TOOLS_FILE and (ranges := self._tool_ranges()) is not None:
            s, e = ranges[fingerprint.removeprefix("tool/")]
            lines = (original or "").splitlines()
            return [DiffLine(no=i + 1, sign="-", text=lines[i]) for i in range(s, e)]
        diff = _diff_lines(original, self.render(name, {fingerprint}))
        if name == RETRIEVER_FILE:  # only the new exclude entry, not the header of a new file
            cid = _chunk_id(fingerprint)
            diff = [d for d in diff if cid in d.text] or diff
        return diff

    def render(self, name: str, removed: set[str]) -> str | None:
        """New text of bundle file ``name`` with ``removed`` fingerprints applied (None = file absent)."""
        original = self.bundle.files.get(name)
        if name == SYSTEM_FILE:
            if original is None:
                return None
            drop = {i for fp, (s, e) in self.sections.items() if fp in removed for i in range(s, e)}
            if not drop:
                return original
            kept = [ln for i, ln in enumerate(original.splitlines()) if i not in drop]
            return "\n".join(kept).strip("\n") + "\n"
        if name == TOOLS_FILE:
            if original is None:
                return None
            gone = {fp.removeprefix("tool/") for fp in removed if fp.startswith("tool/")}
            if not gone & self.tool_names:
                return original
            kept = [t for t in self.tools if _tool_name(t) not in gone]
            return json.dumps(kept, indent=2, ensure_ascii=False) + "\n"
        if name == RETRIEVER_FILE:
            ids = sorted(_chunk_id(fp) for fp in removed if fp.startswith("rag/"))
            return _add_excludes(original, ids)
        raise ValueError(name)


def _add_excludes(original: str | None, ids: list[str]) -> str | None:
    """Append ids to the ``exclude:`` block list of retriever.yaml, editing text to keep its layout."""
    if original is None:
        if not ids:
            return None
        return _RETRIEVER_HEADER + "exclude:\n" + "".join(f"  - {_q(i)}\n" for i in ids)
    lines = original.splitlines()
    present = set(re.findall(r"^\s+-\s+['\"]?(.+?)['\"]?\s*$", original, re.MULTILINE))
    new = [i for i in ids if i not in present]
    if not new:
        return original
    key = next((n for n, ln in enumerate(lines) if re.match(r"^exclude:\s*(\[\s*\])?\s*$", ln)), None)
    if key is None:
        body = original.rstrip("\n") + "\n" if original.strip() else ""
        return body + "exclude:\n" + "".join(f"  - {_q(i)}\n" for i in new)
    lines[key] = "exclude:"
    end = key + 1
    while end < len(lines) and re.match(r"^\s+-\s", lines[end]):
        end += 1
    lines[end:end] = [f"  - {_q(i)}" for i in new]
    return "\n".join(lines) + "\n"


def _q(chunk_id: str) -> str:
    """YAML-quote ids that would otherwise parse as something else (``#`` starts a comment)."""
    return json.dumps(chunk_id) if re.search(r"[#:*&!|>'\"%@`{}\[\],]|^\s|\s$", chunk_id) else chunk_id


def _diff_lines(old: str | None, new: str | None) -> list[DiffLine]:
    a = (old or "").splitlines()
    b = (new or "").splitlines()
    out: list[DiffLine] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            out += [DiffLine(no=i + 1, sign="-", text=a[i]) for i in range(i1, i2)]
        if tag in ("insert", "replace"):
            out += [DiffLine(no=j + 1, sign="+", text=b[j]) for j in range(j1, j2)]
    return out


def unified_diff(bundle: Bundle, rendered: dict[str, str | None]) -> str:
    chunks: list[str] = []
    for name in (SYSTEM_FILE, TOOLS_FILE, RETRIEVER_FILE):
        old = bundle.files.get(name)
        new = rendered.get(name, old)
        if new == old:
            continue
        chunks.extend(difflib.unified_diff(
            (old or "").splitlines(keepends=True), (new or "").splitlines(keepends=True),
            fromfile=f"a/{name}" if old is not None else "/dev/null", tofile=f"b/{name}", n=1,
        ))
    return "".join(c if c.endswith("\n") else c + "\n\\ No newline at end of file\n" for c in chunks)


# ---- the plan --------------------------------------------------------------------------------

class PrunePlan(BaseModel):
    hunks: list[Hunk]
    skipped: list[Skipped]
    unknown_keep: list[str] = Field(default_factory=list)
    diff: str = ""
    files: dict[str, str] = Field(default_factory=dict)  # pruned bundle: name -> new text

    @property
    def removed(self) -> list[str]:
        return [h.fingerprint for h in self.hunks if h.cut]

    @property
    def tokens_saved(self) -> float:
        """Expected tokens saved per request, from the profiling run (verification measures it)."""
        return sum(h.tokens_per_request for h in self.hunks if h.cut)


def _reason(a: dict[str, Any], tool_calls: dict[str, int] | None) -> str:
    bits = [f"DEAD · {a['traces']} traces", f"mean {a['mean_influence']:+.2f}", f"p95 {a['p95_influence']:.2f}"]
    if a["kind"] == "tool" and tool_calls is not None:
        n = tool_calls.get(a["fingerprint"].removeprefix("tool/"), 0)
        bits.append(f"{n} call{'s' if n != 1 else ''} in recorded answers")
    return " · ".join(bits)


def _merge_tool_parts(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse tool sub-segments into one row per tool: DEAD only if every part is DEAD."""
    parts: dict[str, list[dict[str, Any]]] = {}
    out = []
    for a in agg:
        if a.get("parent") and a["kind"] == "tool":
            parts.setdefault(a["parent"], []).append(a)
        else:
            out.append(a)
    for parent, ps in parts.items():
        dead = [p for p in ps if p["label"] == "DEAD"]
        out.append({
            "fingerprint": parent, "kind": "tool", "source": ps[0]["source"], "label_name": parent,
            "traces": max(p["traces"] for p in ps), "tokens_mean": sum(p["tokens_mean"] for p in ps),
            "mean_influence": sum(p["mean_influence"] for p in ps),
            "p95_influence": max(p["p95_influence"] for p in ps),
            "label": "DEAD" if len(dead) == len(ps) else "",
            "dead_parts": [p["fingerprint"].rsplit(":", 1)[-1] for p in dead] if 0 < len(dead) < len(ps) else [],
        })
    return out


def build_plan(agg: list[dict[str, Any]], bundle: Bundle, *, keep: Iterable[str] = (),
               injection: Iterable[dict[str, Any]] = (), heading_levels: Iterable[int] = (1, 2, 3),
               pinned_kinds: Iterable[str] = ("user",), tool_calls: dict[str, int] | None = None,
               n_traces: int | None = None) -> PrunePlan:
    """Plan removals from an aggregate (see :func:`ctxprof.aggregate.aggregate`).

    ``n_traces`` (well-fit traces in the run) turns the aggregate's mean tokens over carrying traces
    into a saving per request. ``tool_calls`` counts how often each
    tool was called in the recorded responses (shown as evidence in the hunk reason).
    """
    keep = set(keep)
    flagged = {h["fingerprint"] for h in injection}
    pinned = set(pinned_kinds)
    editor = _Editor(bundle, heading_levels)
    total = n_traces or max((a["traces"] for a in agg), default=1)

    hunks: list[Hunk] = []
    skipped: list[Skipped] = []
    agg = _merge_tool_parts(agg)
    for a in agg:
        if a["label"] != "DEAD":
            if a.get("dead_parts"):
                skipped.append(Skipped(
                    fingerprint=a["fingerprint"], kind=a["kind"], label=a.get("label_name", a["fingerprint"]),
                    why=f"only part of this tool is DEAD ({', '.join(a['dead_parts'])}); not pruned automatically"))
            continue
        fp, kind = a["fingerprint"], a["kind"]

        def skip(why: str, fp=fp, kind=kind, a=a) -> None:
            skipped.append(Skipped(fingerprint=fp, kind=kind, label=a.get("label_name", fp), why=why))

        if kind in pinned:
            skip("pinned: never ablated, never pruned")
        elif a["source"] == "untrusted":
            skip("untrusted source: never pruned automatically")
        elif fp in flagged:
            skip("injection signal on this fingerprint")
        elif kind == "history":
            skip("conversation history is not part of the prompt bundle")
        elif "*" in fp and kind != "retrieved":
            skip("wildcard fingerprint")
        elif (file := editor.file_for(fp, kind)) is None:
            skip("not found in the prompt bundle")
        else:
            hunks.append(Hunk(
                fingerprint=fp, kind=kind, file=file, traces=a["traces"],
                tokens_per_request=round(a["tokens_mean"] * a["traces"] / total, 2), mean_influence=a["mean_influence"],
                p95_influence=a["p95_influence"], reason=_reason(a, tool_calls), cut=fp not in keep,
            ))

    order = {SYSTEM_FILE: 0, TOOLS_FILE: 1, RETRIEVER_FILE: 2}
    for h in hunks:
        h.lines = editor.hunk_lines(h.fingerprint, h.file)
    hunks.sort(key=lambda h: (order[h.file], min((ln.no for ln in h.lines), default=0)))

    removed = {h.fingerprint for h in hunks if h.cut}
    rendered = {name: editor.render(name, removed) for name in (SYSTEM_FILE, TOOLS_FILE, RETRIEVER_FILE)}
    files = {n: t for n, t in rendered.items() if t is not None and t != bundle.files.get(n)}
    candidates = {h.fingerprint for h in hunks}
    return PrunePlan(hunks=hunks, skipped=skipped, unknown_keep=sorted(keep - candidates),
                     diff=unified_diff(bundle, rendered), files=files)


def write_bundle(plan: PrunePlan, bundle: Bundle, out_dir: str | Path) -> list[Path]:
    """Write the full pruned bundle (unchanged files copied) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in (SYSTEM_FILE, TOOLS_FILE, RETRIEVER_FILE):
        text = plan.files.get(name, bundle.files.get(name))
        if text is None:
            continue
        (out / name).write_text(text, encoding="utf-8")
        written.append(out / name)
    return written
