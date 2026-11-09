from ctxprof.models import Message, Request, Response, Trace
from ctxprof.segmenter import SegmenterConfig, segment_trace
from ctxprof.tokens import HeuristicCounter


def test_fingerprints_and_kinds(small_trace, seg_config, counter):
    segs = segment_trace(small_trace, seg_config, counter)
    fps = [s.fingerprint for s in segs]
    assert fps == [
        "system/preamble", "system/## Tone rules", "system/## Refund policy",
        "tool/get_status", "tool/get_weather",
        "rag/kb:faq#1", "rag/listing:42",
        "history/user -2", "history/assistant -1",
        "user/current",
    ]
    assert [s.index for s in segs] == list(range(len(segs)))
    by_fp = {s.fingerprint: s for s in segs}
    assert by_fp["rag/listing:42"].source == "untrusted"
    assert by_fp["rag/kb:faq#1"].source == "trusted"
    assert by_fp["user/current"].source == "untrusted"
    assert by_fp["history/assistant -1"].source == "trusted"
    assert all(s.tokens > 0 for s in segs)


def test_docs_are_carved_out_of_user_turn(small_trace, seg_config, counter):
    segs = {s.fingerprint: s for s in segment_trace(small_trace, seg_config, counter)}
    user = small_trace.request.messages[3].content
    text = "".join(user[sp.start:sp.end] for sp in segs["user/current"].locator.spans)
    assert "registration fee" not in text and "Question: what is my status?" in text


def test_heading_inside_doc_is_not_a_section(counter):
    system = '## A\nalpha\n<doc id="d1">## Not a heading\nbody</doc>\n## B\nbeta'
    t = Trace(id="x", request=Request(model="m", messages=[Message(role="system", content=system),
                                                            Message(role="user", content="q")]),
              response=Response(text="a"))
    fps = [s.fingerprint for s in segment_trace(t, SegmenterConfig(heading_levels=[2]), counter)]
    assert fps == ["system/## A", "system/## B", "rag/d1", "user/current"]


def test_fingerprints_are_stable_across_traces(small_trace, seg_config, counter):
    other = small_trace.model_copy(deep=True)
    other.id = "t2"
    other.request.messages[3].content = other.request.messages[3].content.replace("status", "application")
    a = [s.fingerprint for s in segment_trace(small_trace, seg_config, counter)]
    b = [s.fingerprint for s in segment_trace(other, seg_config, counter)]
    assert a == b


def test_heuristic_counter_is_deterministic():
    c = HeuristicCounter()
    assert c.count("") == 0
    assert c.count("hello, world") == c.count("hello, world") == 5  # hell|o , worl|d
    assert c.count("internationalisation") == 5
