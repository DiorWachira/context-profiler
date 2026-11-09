"""Split a request into named, ablatable segments with stable fingerprints.

Rules (configurable via ``segmenter.yaml``):

* each tool schema is one ``tool`` segment (``tool/<name>``);
* retrieved chunks are carved out of any message by ``doc_pattern`` (``rag/<doc id>``);
* system messages are split by markdown headings (``system/## Tone rules``);
* the last user message is the current turn (``user/current``);
* every earlier user/assistant/tool message is a ``history`` segment, fingerprinted by its
  role and distance from the current turn (``history/user -1``).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .models import KIND_ORDER, Locator, Segment, Source, Span, Trace, content_hash
from .tokens import TokenCounter


class SegmenterConfig(BaseModel):
    heading_levels: list[int] = Field(default_factory=lambda: [1, 2, 3])
    preamble_label: str = "preamble"
    doc_pattern: str = r'<doc id="(?P<id>[^"]+)"[^>]*>(?P<body>.*?)</doc>'
    # Retrieved chunks whose id starts with one of these prefixes are untrusted (third-party text).
    untrusted_doc_prefixes: list[str] = Field(default_factory=list)
    # Message roles whose content is untrusted input.
    untrusted_roles: list[str] = Field(default_factory=lambda: ["user", "tool"])
    # Regex rewrites applied to fingerprints so per-request ids aggregate together,
    # e.g. [["^rag/listing:.*$", "rag/listing:*"]]. The display label keeps the specific id.
    fingerprint_rewrites: list[tuple[str, str]] = Field(default_factory=list)
    min_tokens: int = 1

    @classmethod
    def load(cls, path: str | Path | None) -> SegmenterConfig:
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


_Range = tuple[int, int]


def _subtract(ranges: list[_Range], holes: list[_Range]) -> list[_Range]:
    """Remove ``holes`` from ``ranges``; both are half-open [start, end)."""
    out: list[_Range] = []
    for start, end in ranges:
        pieces = [(start, end)]
        for hs, he in holes:
            nxt: list[_Range] = []
            for ps, pe in pieces:
                if he <= ps or hs >= pe:
                    nxt.append((ps, pe))
                    continue
                if hs > ps:
                    nxt.append((ps, hs))
                if he < pe:
                    nxt.append((he, pe))
            pieces = nxt
        out.extend(pieces)
    return out


def _text_of(content: str, ranges: list[_Range]) -> str:
    return "".join(content[s:e] for s, e in ranges)


class _Draft(BaseModel):
    kind: str
    source: Source
    fingerprint: str
    label: str
    text: str
    locator: Locator
    order: tuple[int, int, int]


def segment_trace(trace: Trace, config: SegmenterConfig, counter: TokenCounter) -> list[Segment]:
    req = trace.request
    doc_re = re.compile(config.doc_pattern, re.DOTALL)
    heading_re = re.compile(
        r"^(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE
    )
    drafts: list[_Draft] = []

    user_idx = [i for i, m in enumerate(req.messages) if m.role == "user"]
    current = user_idx[-1] if user_idx else None
    convo = [i for i, m in enumerate(req.messages) if m.role != "system" and i != current]
    distance = {i: len(convo) - pos for pos, i in enumerate(convo)}  # nearest earlier message = 1

    def doc_source(doc_id: str) -> Source:
        return "untrusted" if any(doc_id.startswith(p) for p in config.untrusted_doc_prefixes) else "trusted"

    for t_i, tool in enumerate(req.tools):
        drafts.append(
            _Draft(
                kind="tool", source="trusted", fingerprint=f"tool/{tool.name}", label=tool.name,
                text=tool.schema_text(), locator=Locator(tool=t_i), order=(KIND_ORDER.index("tool"), t_i, 0),
            )
        )

    for m_i, msg in enumerate(req.messages):
        content = msg.content
        docs = [(m.start(), m.end(), m.group("id")) for m in doc_re.finditer(content)]
        holes = [(s, e) for s, e, _ in docs]
        for d_i, (s, e, doc_id) in enumerate(docs):
            drafts.append(
                _Draft(
                    kind="retrieved", source=doc_source(doc_id), fingerprint=f"rag/{doc_id}", label=doc_id,
                    text=content[s:e], locator=Locator(spans=[Span(message=m_i, start=s, end=e)]),
                    order=(KIND_ORDER.index("retrieved"), m_i, d_i),
                )
            )

        if msg.role == "system":
            heads = [
                h for h in heading_re.finditer(content)
                if len(h.group("hashes")) in config.heading_levels and not any(s <= h.start() < e for s, e in holes)
            ]
            bounds = [0] + [h.start() for h in heads] + [len(content)]
            titles = [config.preamble_label] + [f"{h.group('hashes')} {h.group('title')}" for h in heads]
            for s_i in range(len(bounds) - 1):
                ranges = _subtract([(bounds[s_i], bounds[s_i + 1])], holes)
                text = _text_of(content, ranges)
                if not text.strip():
                    continue
                title = titles[s_i]
                drafts.append(
                    _Draft(
                        kind="system", source="trusted", fingerprint=f"system/{title}",
                        label=title.lstrip("# ").strip() or title,
                        text=text, locator=Locator(spans=[Span(message=m_i, start=s, end=e) for s, e in ranges]),
                        order=(KIND_ORDER.index("system"), m_i, s_i),
                    )
                )
            continue

        source: Source = "untrusted" if msg.role in config.untrusted_roles else "trusted"
        if m_i == current:
            kind, fp, label = "user", "user/current", "user (current)"
        else:
            kind = "history"
            fp = f"history/{msg.role} -{distance[m_i]}"
            label = f"{msg.role} -{distance[m_i]}"
        ranges = _subtract([(0, len(content))], holes)
        text = _text_of(content, ranges)
        if holes:
            locator = Locator(spans=[Span(message=m_i, start=s, end=e) for s, e in ranges])
        else:
            locator = Locator(whole_message=m_i)
        if not text.strip() and kind == "history" and not msg.extra.get("tool_calls"):
            continue
        if msg.extra.get("tool_calls"):
            text = text + " " + str(msg.extra["tool_calls"])
        drafts.append(
            _Draft(kind=kind, source=source, fingerprint=fp, label=label, text=text, locator=locator,
                   order=(KIND_ORDER.index(kind), m_i, 0))
        )

    drafts.sort(key=lambda d: d.order)
    rewrites = [(re.compile(p), r) for p, r in config.fingerprint_rewrites]
    segments: list[Segment] = []
    seen: dict[str, int] = {}
    for d in drafts:
        tokens = counter.count(d.text)
        if tokens < config.min_tokens:
            continue
        fp = d.fingerprint
        for pattern, repl in rewrites:
            fp = pattern.sub(repl, fp)
        if fp in seen:  # duplicate heading or doc id inside one request: disambiguate
            seen[fp] += 1
            fp = f"{fp} #{seen[fp]}"
        else:
            seen[fp] = 1
        segments.append(
            Segment(
                trace_id=trace.id, index=len(segments), kind=d.kind, source=d.source, fingerprint=fp,
                label=d.label, tokens=tokens, content_hash=content_hash(d.text), locator=d.locator,
            )
        )
    return segments
