"""Tool sub-segments: description / parameters as separate segments under their tool."""

import json

from ctxprof.flame import flame_tree
from ctxprof.masking import apply_mask
from ctxprof.prune import Bundle, build_plan
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.verify import _keep


def parts_config() -> SegmenterConfig:
    return SegmenterConfig(heading_levels=[2], tool_parts=True)


def test_tools_split_into_description_and_parameters(small_trace, counter):
    segs = segment_trace(small_trace, parts_config(), counter)
    tools = [s for s in segs if s.kind == "tool"]
    assert [s.fingerprint for s in tools] == ["tool/get_status:description", "tool/get_status:parameters",
                                              "tool/get_weather:description", "tool/get_weather:parameters"]
    assert {s.parent for s in tools} == {"tool/get_status", "tool/get_weather"}


def test_masking_blanks_a_part_and_drops_a_fully_removed_tool(small_trace, counter):
    segs = segment_trace(small_trace, parts_config(), counter)
    drop = {"tool/get_status:description", "tool/get_weather:description", "tool/get_weather:parameters"}
    out = apply_mask(small_trace.request, segs, [s.fingerprint not in drop for s in segs])
    assert [t.name for t in out.tools] == ["get_status"]
    assert out.tools[0].description == "" and out.tools[0].parameters == small_trace.request.tools[0].parameters
    # verify's keep-vector: a whole-tool hunk removes all of its parts
    keep = _keep(segs, {"tool/get_weather"})
    assert [s.fingerprint for s, k in zip(segs, keep, strict=True) if not k] == [
        "tool/get_weather:description", "tool/get_weather:parameters"]


def test_flame_nests_parts_under_their_tool(small_trace, counter):
    segs = segment_trace(small_trace, parts_config(), counter)
    attrs = {s.index: {"influence": 0.3, "weight": 1.0, "ci_lo": 0.9, "ci_hi": 1.1} for s in segs}
    tree = flame_tree(small_trace.id, segs, attrs)
    tool_kind = next(c for c in tree["children"] if c["kind"] == "tool")
    assert [c["id"] for c in tool_kind["children"]] == ["tool/get_status", "tool/get_weather"]
    status = tool_kind["children"][0]
    assert [c["id"] for c in status["children"]] == ["tool/get_status:description", "tool/get_status:parameters"]
    assert status["tokens"] == sum(c["tokens"] for c in status["children"])
    first, second = status["children"]
    assert status["influence"] == 0.6 and second["offset"] == status["offset"] + first["tokens"]


def _row(fp, label, parent=None):
    return {"fingerprint": fp, "kind": "tool", "source": "trusted", "label_name": fp, "parent": parent, "traces": 30,
            "tokens_mean": 50.0, "tokens_paid": 1500, "mean_influence": 0.0, "sd_influence": 0.01,
            "p95_influence": 0.02, "influence_per_1k": 0.0, "label": label}


def test_prune_removes_a_tool_only_when_every_part_is_dead(tmp_path):
    root = tmp_path / "prompt"
    root.mkdir()
    tools = [{"name": "get_weather", "description": "w", "parameters": {}},
             {"name": "get_job", "description": "j", "parameters": {}}]
    (root / "tools.json").write_text(json.dumps(tools, indent=2) + "\n", encoding="utf-8")
    agg = [_row("tool/get_weather:description", "DEAD", "tool/get_weather"),
           _row("tool/get_weather:parameters", "DEAD", "tool/get_weather"),
           _row("tool/get_job:description", "DEAD", "tool/get_job"),
           _row("tool/get_job:parameters", "VOLATILE", "tool/get_job")]
    plan = build_plan(agg, Bundle.load(root), n_traces=30)
    assert plan.removed == ["tool/get_weather"]
    assert plan.hunks[0].tokens_per_request == 100.0  # both parts
    why = {s.fingerprint: s.why for s in plan.skipped}
    assert why["tool/get_job"].startswith("only part of this tool is DEAD (description)")
    assert [t["name"] for t in json.loads(plan.files["tools.json"])] == ["get_job"]
