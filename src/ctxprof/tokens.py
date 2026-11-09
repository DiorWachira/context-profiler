"""Token counting. Pluggable: tiktoken when installed, otherwise a deterministic heuristic.

The heuristic approximates BPE tokenisers (roughly one token per 4 characters of a word, one per
punctuation mark). It is deterministic and dependency-free, which keeps tests and the demo
reproducible. Use ``get_counter("tiktoken")`` for provider-accurate numbers.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicCounter:
    name = "heuristic"

    def count(self, text: str) -> int:
        total = 0
        for piece in _PIECE.findall(text):
            total += max(1, math.ceil(len(piece) / 4)) if piece[0].isalnum() or piece[0] == "_" else 1
        return total


class TiktokenCounter:
    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken  # optional dependency

        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def get_counter(name: str = "heuristic") -> TokenCounter:
    if name == "heuristic":
        return HeuristicCounter()
    if name == "tiktoken":
        return TiktokenCounter()
    raise ValueError(f"unknown token counter: {name!r} (use 'heuristic' or 'tiktoken')")
