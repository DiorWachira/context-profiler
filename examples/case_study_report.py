"""Render examples/case_study/results.json as results.md (run after examples/case_study.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "case_study"


def fmt_rho(x: float) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:+.2f}"


def main() -> None:
    r = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    rank, prune = r["ranking"], r["prune"]
    s = prune["summary"]
    lines = [
        f"# Case study: the JobBridge prompt on {r['model']} ({r['device'].upper()})",
        "",
        f"Profiled {len(r['profiled'])} traces (one per intent) with {r['n_masks']} masks each, in both modes, "
        f"after re-recording every answer with the real model at temperature 0. "
        f"{len(r['held_out'])} further traces were held out for verification.",
        "",
        "## Logprob vs black-box rankings",
        "",
        "| Trace | Spearman ρ (weights) | Top-5 overlap | Logprob R² | Black-box R² |",
        "|---|---|---|---|---|",
    ]
    fits = {m: {f["trace_id"]: f for f in fs} for m, fs in r["fits"].items()}
    for t in rank["per_trace"]:
        lp, bb = fits["logprob"][t["trace_id"]], fits["blackbox"][t["trace_id"]]
        r2 = lambda f: "n/a" if f["r2_heldout"] is None else f"{f['r2_heldout']:.2f}"  # noqa: E731
        lines.append(f"| {t['trace_id']} | {fmt_rho(t['spearman'])} | {t['top5_overlap']}/5 | {r2(lp)} | {r2(bb)} |")
    a = rank["aggregate"]
    lines += [
        f"| **aggregate** ({a['fingerprints']} fingerprints, mean influence) | **{fmt_rho(a['spearman'])}** | "
        f"**{a['top5_overlap']}/5** | | |",
        "",
        f"Top 5 by logprob: {', '.join(f'`{x}`' for x in a['top5_logprob'])}",
        "",
        f"Top 5 by black-box: {', '.join(f'`{x}`' for x in a['top5_blackbox'])}",
        "",
        "## Labels (logprob run, DEAD needs all 5 traces)",
        "",
    ]
    for label, fps in sorted(r["labels_logprob"].items()):
        lines.append(f"- **{label}** ({len(fps)}): {', '.join(f'`{x}`' for x in fps)}")
    lines += [
        "",
        "## Pruning, verified on the held-out traces with the real model",
        "",
        f"- Removed: {', '.join(f'`{x}`' for x in prune['removed']) or 'nothing'}",
        f"- Tokens per request: {s['tokens_before']:,.0f} → {s['tokens_after']:,.0f} (−{s['reduction'] * 100:.1f}%)",
        f"- Agreement: {s['agree_n']}/{s['traces']} = {s['agreement'] * 100:.1f}% "
        f"→ {'recommend' if s['recommend'] else 'do not ship'} at the 95% threshold",
        f"- Model calls for verification: {prune['calls']}",
        "",
    ]
    if prune["regressions"]:
        lines += ["### Regressions", ""]
        for g in prune["regressions"]:
            o, p = g.get("original") or {}, g.get("pruned") or {}
            calls = lambda x: ", ".join(c["name"] for c in x.get("tool_calls", [])) or "none"  # noqa: E731
            lines.append(f"- `{g['trace_id']}` ({g.get('intent')}), score {g['score']:.2f}, suspects "
                         f"{', '.join(f'`{b}`' for b in g['breaks']) or 'unknown'}"
                         f"{'' if g['explained'] else ' (no single hunk explains it)'}: "
                         f"tool calls {calls(o)} → {calls(p)}")
        lines.append("")
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
