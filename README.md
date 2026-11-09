# Context Profiler

A flame graph for your prompts. Context Profiler samples real LLM requests, removes parts of the
context (system-prompt sections, tool schemas, retrieved chunks, prior turns), measures how the
answer changes, fits a sparse linear surrogate (Lasso, as in ContextCite), and ranks every segment
by **tokens paid × influence** across hundreds of traces.

> "I built a profiler that tells you which parts of a prompt you're paying for but not using,
> and proves removing them is safe."

Status: M1–M4 are in: import, segmenting, flame graph, attribution engine, aggregation, labels,
injection signal, pruning diff, held-out verification with a cost report, API and UI. See
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
| `ctxprof prune [--bundle DIR] [--keep FP]... [--out prompt.diff] [--write DIR]` | Unified diff removing DEAD sections / tools / chunks from the prompt bundle |
| `ctxprof verify [--prune N] [--held-out K] [--threshold 0.95] [--eval CMD]` | Replay held-out traces original vs pruned; exit 0 = recommend |
| `ctxprof export [--out web/dist]` | Static JSON snapshot of the API (for a serverless UI build) |

Black-box mode works against any OpenAI-compatible endpoint, e.g. Ollama:

```bash
ctxprof profile --provider openai --base-url http://localhost:11434/v1 --model llama3.1:8b --mode blackbox --sample 20
```

## Pruning and verification (M4)

`ctxprof prune` turns DEAD fingerprints into a diff against the *prompt bundle* (`system.md`
sections removed by heading, `tools.json` schemas removed by name, chunk ids appended to
`retriever.yaml`'s `exclude:` list). VOLATILE and CRITICAL segments are never candidates;
untrusted, pinned, injection-flagged and history segments are listed as "not pruned" with the
reason. `--keep FP` keeps a candidate (its hunk is still shown, unticked).

`ctxprof verify` replays traces the profile never saw: the original request and the pruned one,
both at temperature 0, compared with the black-box metric (or your `--eval` command). Each
regression is re-run once per cut hunk with that hunk restored, so it names its suspect; kept
hunks are probed by cutting them. The UI's per-hunk "Cut" toggles recompute the ledger and
verdict from that evidence (exact for the verified selection, labelled an estimate otherwise).

Demo result (`ctxprof demo`: 200 traces profiled, 60 held out, fake model, heuristic tokens):

| Diff | Hunks | Tokens / request | Cost / request at $3/M | Agreement | Verdict |
|---|---|---|---|---|---|
| default (all DEAD) | 14 | 4,380 → 1,987 (−54.6%) | $0.01314 → $0.00596 | 54/60 = 90.0% | below 95%: 6 regressions, all scam questions, suspect `tool/report_listing` |
| `--keep tool/report_listing` | 13 | 4,380 → 2,150 (−50.9%) | $0.01314 → $0.00645 | 60/60 = 100% | recommend: saves $669.03 (KSh 86,305) per 100k requests |

`report_listing` is used only when a listing asks for money (6% of traffic), so its average
influence is ~0 and it is labelled DEAD; verification on held-out traces is what catches it. These
are fake-model numbers that prove the pipeline, not real-world savings.

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
- **Rare-but-needed segments.** A segment used in a few % of traces can look DEAD (the demo's
  `report_listing`). That is why a diff is only recommended after held-out verification.
- **Verification is a sample.** 60 held-out traces at a 95% threshold cannot see a regression
  rarer than a few %. The per-hunk estimate assumes removals do not interact beyond what the
  probes saw; re-run `verify` on the selection you ship.
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
