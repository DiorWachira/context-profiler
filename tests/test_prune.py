"""The diff writer: heading removal, tool removal, retriever exclude list, --keep and safety rules."""

import json

import yaml

from ctxprof.prune import Bundle, _add_excludes, _sections, build_plan, write_bundle

SYSTEM = """You are JobBridge Support.

## Role
Help job seekers.

## Company history
Founded in 2019 in Thika.
Expanded in 2021.

## Formatting
Use short lists.
"""

TOOLS = [
    {"name": "search_jobs", "description": "Search", "parameters": {"type": "object"}},
    {"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {}}},
    {"name": "report_listing", "description": "Report", "parameters": {"type": "object"}},
]


def agg_row(fp, kind, label="DEAD", source="trusted", traces=40, tokens=100.0, mean=0.0):
    return {"fingerprint": fp, "kind": kind, "source": source, "label_name": fp, "traces": traces,
            "tokens_mean": tokens, "tokens_paid": int(tokens * traces), "mean_influence": mean,
            "sd_influence": 0.01, "p95_influence": 0.02, "influence_per_1k": 0.0, "label": label}


def make_bundle(tmp_path, retriever: str | None = None) -> Bundle:
    root = tmp_path / "prompt"
    root.mkdir(parents=True)
    (root / "system.md").write_text(SYSTEM, encoding="utf-8")
    (root / "tools.json").write_text(json.dumps(TOOLS, indent=2) + "\n", encoding="utf-8")
    if retriever is not None:
        (root / "retriever.yaml").write_text(retriever, encoding="utf-8")
    return Bundle.load(root)


AGG = [
    agg_row("system/## Company history", "system"),
    agg_row("system/## Role", "system", label="CRITICAL", mean=0.8),
    agg_row("tool/get_weather", "tool"),
    agg_row("tool/report_listing", "tool"),
    agg_row("tool/search_jobs", "tool", label="VOLATILE", mean=0.3),
    agg_row("rag/kb:press-2019#2", "retrieved", traces=20),
    agg_row("rag/listing:*", "retrieved", source="untrusted"),
    agg_row("history/assistant -1", "history"),
    agg_row("system/## Not in this prompt", "system"),
]


def test_sections_match_segmenter_levels():
    secs = _sections(SYSTEM, [2])
    assert list(secs) == ["system/preamble", "system/## Role", "system/## Company history", "system/## Formatting"]
    s, e = secs["system/## Company history"]
    assert SYSTEM.splitlines()[s] == "## Company history" and SYSTEM.splitlines()[e] == "## Formatting"


def test_plan_removes_heading_tool_and_excludes_chunk(tmp_path):
    bundle = make_bundle(tmp_path)
    plan = build_plan(AGG, bundle, heading_levels=[2], tool_calls={"report_listing": 3}, n_traces=40)
    cut = set(plan.removed)
    assert cut == {"system/## Company history", "tool/get_weather", "tool/report_listing", "rag/kb:press-2019#2"}

    system = plan.files["system.md"]
    assert "Company history" not in system and "Founded in 2019" not in system
    assert "## Role\nHelp job seekers.\n\n## Formatting" in system  # neighbours untouched, one blank line
    tools = json.loads(plan.files["tools.json"])
    assert [t.get("name") or t["function"]["name"] for t in tools] == ["search_jobs"]
    excl = yaml.safe_load(plan.files["retriever.yaml"])
    assert excl == {"exclude": ["kb:press-2019#2"]}  # '#' survives YAML (quoted)

    assert "--- a/system.md" in plan.diff and "-## Company history" in plan.diff
    assert "--- /dev/null" in plan.diff and "+++ b/retriever.yaml" in plan.diff
    assert '-    "name": "report_listing",' in plan.diff

    hunk = {h.fingerprint: h for h in plan.hunks}
    assert [ln.text for ln in hunk["system/## Company history"].lines if ln.sign == "-"][0] == "## Company history"
    assert all(ln.sign == "+" for ln in hunk["rag/kb:press-2019#2"].lines)
    assert hunk["tool/report_listing"].reason.endswith("3 calls in recorded answers")
    assert hunk["rag/kb:press-2019#2"].tokens_per_request == 50.0  # 100 tokens in 20 of 40 traces


def test_safety_rules_never_prune_untrusted_volatile_history(tmp_path):
    plan = build_plan(AGG, make_bundle(tmp_path), heading_levels=[2])
    names = {h.fingerprint for h in plan.hunks}
    assert "tool/search_jobs" not in names and "system/## Role" not in names  # VOLATILE / CRITICAL
    why = {s.fingerprint: s.why for s in plan.skipped}
    assert why["rag/listing:*"].startswith("untrusted")
    assert why["history/assistant -1"].startswith("conversation history")
    assert why["system/## Not in this prompt"] == "not found in the prompt bundle"

    flagged = build_plan(AGG, make_bundle(tmp_path / "x"), heading_levels=[2],
                         injection=[{"fingerprint": "tool/get_weather"}])
    assert "tool/get_weather" not in flagged.removed
    assert {s.fingerprint: s.why for s in flagged.skipped}["tool/get_weather"].startswith("injection")


def test_keep_lists_the_hunk_but_does_not_apply_it(tmp_path):
    bundle = make_bundle(tmp_path)
    plan = build_plan(AGG, bundle, heading_levels=[2], keep=["tool/report_listing", "tool/search_jobs"])
    hunk = {h.fingerprint: h for h in plan.hunks}
    assert hunk["tool/report_listing"].cut is False and hunk["tool/report_listing"].lines  # still shown
    assert "report_listing" in plan.files["tools.json"] and "report_listing" not in plan.diff
    assert plan.unknown_keep == ["tool/search_jobs"]  # VOLATILE: never a candidate anyway

    out = tmp_path / "pruned"
    written = write_bundle(plan, bundle, out)
    assert {p.name for p in written} == {"system.md", "tools.json", "retriever.yaml"}


def test_exclude_list_editing_keeps_existing_layout():
    existing = "top_k: 6\nexclude:\n  - kb:old#1\nrerank: true\n"
    out = _add_excludes(existing, ["kb:new#2", "kb:old#1"])
    assert out == 'top_k: 6\nexclude:\n  - kb:old#1\n  - "kb:new#2"\nrerank: true\n'
    assert yaml.safe_load(out)["exclude"] == ["kb:old#1", "kb:new#2"]
    assert _add_excludes("exclude: []\n", ["a"]) == "exclude:\n  - a\n"
    assert yaml.safe_load(_add_excludes("top_k: 6\n", ["x:y"])) == {"top_k": 6, "exclude": ["x:y"]}
    assert _add_excludes(None, []) is None
