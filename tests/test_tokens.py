"""Token counters: the heuristic is deterministic; tiktoken (optional extra) matches known counts."""

import pytest

from ctxprof.tokens import HeuristicCounter, get_counter


def test_heuristic_counter_is_deterministic_and_bpe_like():
    c = HeuristicCounter()
    assert c.count("") == 0
    assert c.count("Hi, KSh 2,000!") == 7  # Hi , KSh 2 , 000 !
    assert c.count("internationalisation") == 5  # ceil(20 / 4)
    assert c.count("Habari yako?") == c.count("Habari yako?")


def test_unknown_counter_is_an_error():
    with pytest.raises(ValueError):
        get_counter("sentencepiece")


@pytest.mark.slow
def test_tiktoken_counter():
    pytest.importorskip("tiktoken")
    c = get_counter("tiktoken")
    assert c.name == "tiktoken"
    assert c.count("hello world") == 2
    assert 0.5 < c.count("Sasa, nimeapply kazi ya driver Thika juzi.") / HeuristicCounter().count(
        "Sasa, nimeapply kazi ya driver Thika juzi.") < 2.0
