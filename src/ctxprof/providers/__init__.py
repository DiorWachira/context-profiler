"""Model providers. All are async; ``FakeProvider`` gives synthetic ground truth for tests and the demo."""

from .base import Completion, Provider, ProviderError
from .fake import Effect, FakeProvider
from .openai_compat import OpenAICompatProvider

__all__ = ["Completion", "Effect", "FakeProvider", "OpenAICompatProvider", "Provider", "ProviderError"]
