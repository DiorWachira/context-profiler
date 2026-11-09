from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from ..models import Request, Response, ToolCall


class ProviderError(RuntimeError):
    pass


class Completion(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_response(self) -> Response:
        return Response(text=self.text, tool_calls=self.tool_calls)


class Provider(ABC):
    """A model endpoint the profiler can replay ablated requests against."""

    name: str = "provider"
    #: True if :meth:`target_logprob` is implemented (enables ``logprob`` scoring mode).
    supports_logprob: bool = False

    @abstractmethod
    async def generate(self, request: Request) -> Completion:
        """Run the (ablated) request and return the model's new response."""

    async def target_logprob(self, request: Request, target: Response) -> float:
        """Total log-probability of ``target`` given ``request`` (teacher-forced)."""
        raise ProviderError(f"{self.name} does not expose log-probabilities; use --mode blackbox")

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None
