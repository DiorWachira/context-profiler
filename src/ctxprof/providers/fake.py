"""A deterministic fake model whose output depends only on known segments.

Each :class:`Effect` fires when all of its ``needs`` substrings are present in the request. A
firing effect raises the log-probability of the original answer by ``weight`` nats and adds its
``phrase`` (and optional tool call) to generated responses. Everything else in the context has
no effect at all, so a correct attribution method must recover exactly the effects' segments.
"""

from __future__ import annotations

import hashlib
import math
import re

from pydantic import BaseModel, Field

from ..models import Request, Response, ToolCall
from .base import Completion, Provider


class Effect(BaseModel):
    needs: list[str]
    weight: float = 1.0
    phrase: str = ""
    tool_call: str | None = None
    # Optional gate: substrings looked up in the current user question (docs stripped). The effect
    # fires only when the gate is present too, so different traces need different segments, and
    # the user turn holding the gate is itself causal.
    when: list[str] = Field(default_factory=list)


def request_text(request: Request) -> str:
    parts = [m.content for m in request.messages]
    parts += [f"[tool:{t.name}]" for t in request.tools]
    return "\n".join(parts)


_DOC = re.compile(r"<doc\b[^>]*>.*?</doc>", re.DOTALL)


def user_question(request: Request) -> str:
    """The last user message with retrieved documents stripped: where effect gates are looked up."""
    users = [m.content for m in request.messages if m.role == "user"]
    return _DOC.sub("", users[-1]) if users else ""


class FakeProvider(Provider):
    name = "fake"
    supports_logprob = True

    def __init__(self, effects: list[Effect], base_nll: float = 6.0, noise: float = 0.05,
                 model: str = "fake-model") -> None:
        self.effects = effects
        self.base_nll = base_nll
        self.noise = noise
        self.model = model
        self.calls = 0

    def _fired(self, request: Request) -> list[Effect]:
        text = request_text(request)
        question = user_question(request)
        return [
            e for e in self.effects
            if all(w in question for w in e.when) and all(n in text for n in e.needs)
        ]

    @staticmethod
    def _in_target(effect: Effect, target: Response) -> bool:
        """An effect matters for a target answer if the answer contains its phrase or tool call."""
        if effect.phrase and effect.phrase in target.text:
            return True
        return bool(effect.tool_call) and any(c.name == effect.tool_call for c in target.tool_calls)

    def _jitter(self, request: Request) -> float:
        h = hashlib.sha256(request.canonical_json().encode()).digest()
        return (int.from_bytes(h[:4], "big") / 2**32 - 0.5) * 2 * self.noise

    async def generate(self, request: Request) -> Completion:
        self.calls += 1
        fired = self._fired(request)
        text = " ".join(e.phrase for e in fired if e.phrase).strip() or "Sorry, I can't help with that."
        calls = [ToolCall(name=e.tool_call) for e in fired if e.tool_call]
        n_in = len(request_text(request)) // 4
        return Completion(text=text, tool_calls=calls, input_tokens=n_in, output_tokens=max(1, len(text) // 4))

    async def target_logprob(self, request: Request, target: Response) -> float:
        self.calls += 1
        fired = {id(e) for e in self._fired(request)}
        missing = sum(e.weight for e in self.effects if self._in_target(e, target) and id(e) not in fired)
        return -(self.base_nll + missing) + self._jitter(request)


def logit_of_logprob(total_logprob: float) -> float:
    """Logit-scaled probability log(p / (1 - p)) for p = exp(total_logprob), numerically safe."""
    if total_logprob < -30:
        return total_logprob
    p = math.exp(total_logprob)
    p = min(p, 1 - 1e-12)
    return math.log(p) - math.log1p(-p)
