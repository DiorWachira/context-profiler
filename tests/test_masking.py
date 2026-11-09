import numpy as np

from ctxprof.masking import apply_mask, sample_masks
from ctxprof.models import Message, Request
from ctxprof.segmenter import SegmenterConfig, segment_trace


def test_sample_masks_shape_and_coverage():
    m = sample_masks(32, 10, 0.5, seed=1)
    assert m.shape == (32, 10) and m.dtype == bool
    assert m.any(axis=0).all() and (~m).any(axis=0).all()
    assert np.array_equal(m, sample_masks(32, 10, 0.5, seed=1))


def test_apply_mask_removes_each_kind(small_trace, seg_config, counter):
    segs = segment_trace(small_trace, seg_config, counter)
    fp = [s.fingerprint for s in segs]
    keep = [f not in {"system/## Refund policy", "tool/get_weather", "rag/listing:42", "history/user -2"} for f in fp]
    out = apply_mask(small_trace.request, segs, keep)
    assert "Refunds take" not in out.messages[0].content and "Be brief" in out.messages[0].content
    assert [t.name for t in out.tools] == ["get_status"]
    assert "registration fee" not in out.messages[-1].content and "Shortlisted" in out.messages[-1].content
    assert [m.content for m in out.messages if m.role == "user"][0] != "Hello"
    assert len(out.messages) == 3
    # the original is untouched
    assert len(small_trace.request.messages) == 4 and len(small_trace.request.tools) == 2


def test_keep_all_is_identity(small_trace, seg_config, counter):
    segs = segment_trace(small_trace, seg_config, counter)
    out = apply_mask(small_trace.request, segs, [True] * len(segs))
    assert out.messages[0].content.strip() == small_trace.request.messages[0].content.strip()
    assert out.tools == small_trace.request.tools


def test_tool_call_pairs_are_repaired(counter):
    req = Request(model="m", messages=[
        Message(role="user", content="find jobs"),
        Message(role="assistant", content="", extra={"tool_calls": [{"id": "c1", "function": {"name": "s"}}]}),
        Message(role="tool", content='{"jobs": []}', extra={"tool_call_id": "c1"}),
        Message(role="user", content="thanks"),
    ])
    from ctxprof.models import Response, Trace
    t = Trace(id="p", request=req, response=Response(text="ok"))
    segs = segment_trace(t, SegmenterConfig(), counter)
    keep = [s.fingerprint != "history/assistant -2" for s in segs]
    out = apply_mask(req, segs, keep)
    assert all(m.role != "tool" for m in out.messages)  # orphaned tool result dropped
