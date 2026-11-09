"""Trace importers. Each returns a list of :class:`ctxprof.models.Trace`."""

from .jsonl import load_jsonl, parse_record
from .otel_genai import load_otel

__all__ = ["load_jsonl", "load_otel", "parse_record"]
