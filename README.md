# Context Profiler

A flame graph for your prompts. Context Profiler samples real LLM requests, removes parts of the
context (system-prompt sections, tool schemas, retrieved chunks, prior turns), measures how the
answer changes, fits a sparse linear surrogate (Lasso, as in ContextCite), and ranks every segment
by **tokens paid × influence** across hundreds of traces.

> "I built a profiler that tells you which parts of a prompt you're paying for but not using,
> and proves removing them is safe."

Status: milestones M1–M3 are in (import, segmenting, flame graph, attribution engine, aggregation,
labels, injection signal, API and UI). Pruning diff and verification (M4) are next. See
`roadmap.txt`.

## Quick start (Windows, Python 3.12, Node 20+)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/ctxprof demo            # writes examples/demo, imports 200 traces, profiles them
.venv/Scripts/ctxprof aggregate       # ranked fingerprint table + DEAD / injection summary
cd web && npm install && npm run build && cd ..
.venv/Scripts/ctxprof serve           # http://127.0.0.1:8000
```

On macOS/Linux use `python3.12 -m venv .venv` and `.venv/bin/...`.

## Commands

| Command | What it does |
|---|---|
| `ctxprof import traces.jsonl [--format otel] [--segmenter seg.yaml] [--counter tiktoken]` | Load traces, split into fingerprinted segments, count tokens |
| `ctxprof profile [--provider fake-demo\|openai] [--mode logprob\|blackbox] [--n 32] [--budget 5]` | Ablate, score, fit the surrogate per trace (cached, budget-capped) |
| `ctxprof aggregate [--run N] [--json]` | Aggregate by fingerprint; DEAD / CRITICAL / VOLATILE; injection signals |
| `ctxprof flame TRACE_ID` | Flame tree JSON for one trace |
| `ctxprof serve` | FastAPI on :8000, serves `web/dist` |
| `ctxprof prune`, `ctxprof verify` | Milestone M4 (not implemented yet) |

Black-box mode works against any OpenAI-compatible endpoint, e.g. Ollama:

```bash
ctxprof profile --provider openai --base-url http://localhost:11434/v1 --model llama3.1:8b --mode blackbox --sample 20
```

## Method, in one paragraph

For each trace, `N` random masks keep each segment with p = 0.5 (the current user question is pinned,
as in ContextCite, because the question is not context). Each masked request is replayed at
temperature 0 and scored: in **logprob** mode by the logit-scaled probability of the original
answer, in **black-box** mode by 1 − normalised change (text similarity, tool-call agreement, JSON
field agreement). A Lasso with α chosen by cross-validation maps mask → score; its weights are
attributions, reported with bootstrap 95% intervals and R² on held-out masks (traces with R² < 0.5
are flagged and excluded from aggregation). Influence is the weight relative to the strongest
segment in the trace (floored, so traces where nothing matters do not inflate noise).

## Limitations

- **Linearity.** Segments interact (a tool schema matters only if the question needs it). Pinning the
  question removes the biggest interaction; others remain and show up as lower held-out R².
- **Cost.** Attribution costs about N model calls per trace (N = 32 by default). The cache makes
  re-runs free; the budget cap stops a run before it overspends.
- **Rare-but-needed segments.** A segment used in 2% of traces can look DEAD. That is why pruning
  must be verified on held-out traces before it is recommended (M4).
- **Demo numbers come from a fake model** whose answers depend only on known segments, which makes
  the ground truth checkable. Real-model numbers belong in the M5 case study.

## Layout

```
src/ctxprof/      Python package (models, segmenter, masking, importers, providers, scoring,
                  surrogate, attribution, aggregate, flame, storage, server, cli, demo)
tests/            pytest suite, including the synthetic ground-truth test
web/              React + Vite + TypeScript UI (Graticule design)
roadmap.txt       build plan and hand-off notes
```
