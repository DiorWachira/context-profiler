"""A local Hugging Face ``transformers`` model: exact teacher-forced log-probabilities + generation.

This is the simplest provider to trust for ``logprob`` mode: the request is rendered with the
model's own chat template (tools included), the target answer is rendered the same way as the
assistant turn, and the log-probabilities of exactly those target tokens are summed from one
forward pass. Only the logits of the target positions are materialised (``logits_to_keep``), so a
4k-token prompt does not need a 2 GB logits tensor.

Install with ``pip install -e ".[local]"`` (torch + transformers). CPU works for small models
(e.g. ``Qwen/Qwen2.5-0.5B-Instruct``); it is slow (seconds per call) but free and exact.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any

from ..models import Request, Response, ToolCall
from .base import Completion, Provider, ProviderError

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split Hermes/Qwen-style ``<tool_call>{json}</tool_call>`` blocks out of generated text."""
    calls: list[ToolCall] = []
    for m in _TOOL_CALL.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append(ToolCall(name=str(obj.get("name", "")), arguments=args if isinstance(args, dict) else {}))
    return _TOOL_CALL.sub("", text).strip(), calls


def chat_messages(request: Request) -> list[dict[str, Any]]:
    out = []
    for m in request.messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        calls = m.extra.get("tool_calls")
        if calls:
            d["tool_calls"] = [
                {"type": "function", "function": {
                    "name": c.get("function", c).get("name"),
                    "arguments": _as_dict(c.get("function", c).get("arguments", {}))}}
                for c in calls if isinstance(c, dict)
            ]
        out.append(d)
    return out


def _as_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw": args}
    return args if isinstance(args, dict) else {}


def chat_tools(request: Request) -> list[dict[str, Any]] | None:
    if not request.tools:
        return None
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters or {"type": "object", "properties": {}}}}
            for t in request.tools]


def assistant_turn(target: Response) -> dict[str, Any]:
    turn: dict[str, Any] = {"role": "assistant", "content": target.text}
    if target.tool_calls:
        turn["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                              for c in target.tool_calls]
    return turn


class HFLocalProvider(Provider):
    name = "hf"
    supports_logprob = True

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 96,
                 dtype: str = "float32", device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ProviderError('the hf provider needs torch + transformers: pip install -e ".[local]"') from exc
        self._torch = torch
        self.model_id = model
        self.name = f"hf:{model}"  # part of the cache key: different models never share cached calls
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self._lock = threading.Lock()  # one model, one forward pass at a time
        self.calls = 0

    # ---- rendering ------------------------------------------------------------------------
    def render_prompt(self, request: Request) -> str:
        return self.tok.apply_chat_template(chat_messages(request), tools=chat_tools(request),
                                            add_generation_prompt=True, tokenize=False)

    def render_target(self, request: Request, target: Response, prompt: str | None = None) -> str:
        """The exact text the template puts after the prompt for this assistant answer."""
        prompt = prompt if prompt is not None else self.render_prompt(request)
        full = self.tok.apply_chat_template(chat_messages(request) + [assistant_turn(target)],
                                            tools=chat_tools(request), tokenize=False)
        if not full.startswith(prompt):
            raise ProviderError("chat template does not render the prompt as a prefix of the full conversation")
        return full[len(prompt):].rstrip("\n")

    # ---- calls ----------------------------------------------------------------------------
    def _logprob_sync(self, request: Request, target: Response) -> float:
        torch = self._torch
        prompt = self.render_prompt(request)
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        t_ids = self.tok(self.render_target(request, target, prompt), add_special_tokens=False).input_ids
        if not t_ids:
            return 0.0
        ids = torch.tensor([p_ids + t_ids], device=self.device)
        with self._lock, torch.inference_mode():
            # logits at positions len(p)-1 .. end predict the target tokens
            logits = self.model(ids, logits_to_keep=len(t_ids) + 1).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            total = lp.gather(1, torch.tensor(t_ids, device=self.device)[:, None]).sum().item()
        self.calls += 1
        return float(total)

    def _generate_sync(self, request: Request) -> Completion:
        torch = self._torch
        ids = self.tok(self.render_prompt(request), add_special_tokens=False, return_tensors="pt").input_ids
        ids = ids.to(self.device)
        limit = min(int(request.params.get("max_tokens") or self.max_new_tokens), self.max_new_tokens)
        with self._lock, torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=limit,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        text, calls = parse_tool_calls(self.tok.decode(new, skip_special_tokens=True))
        self.calls += 1
        return Completion(text=text, tool_calls=calls, input_tokens=int(ids.shape[1]), output_tokens=int(len(new)))

    async def target_logprob(self, request: Request, target: Response) -> float:
        return await asyncio.to_thread(self._logprob_sync, request, target)

    async def generate(self, request: Request) -> Completion:
        return await asyncio.to_thread(self._generate_sync, request)
