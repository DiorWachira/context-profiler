"""``ctxprof`` command line: import | profile | aggregate | flame | prune | verify | serve | demo."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Annotated

import typer

from .aggregate import LabelRules, aggregate, injection_signals
from .attribution import ProfileConfig, run_profile
from .flame import flame_tree
from .importers import load_jsonl, load_otel
from .providers import FakeProvider, OpenAICompatProvider, Provider
from .segmenter import SegmenterConfig, segment_trace
from .storage import Store
from .tokens import get_counter

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Context Profiler: which parts of your prompt you pay for but don't use.")

DbOpt = Annotated[Path, typer.Option("--db", help="SQLite database file")]
DEFAULT_DB = Path("ctxprof.db")


def _import(store: Store, path: Path, fmt: str, segmenter: Path | None, counter_name: str) -> tuple[int, int]:
    traces = load_otel(path) if fmt == "otel" else load_jsonl(path)
    cfg = SegmenterConfig.load(segmenter)
    counter = get_counter(counter_name)
    n_segments = 0
    for t in traces:
        segs = segment_trace(t, cfg, counter)
        store.save_trace(t, segs)
        n_segments += len(segs)
    return len(traces), n_segments


@app.command("import")
def import_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Traces file")],
    fmt: Annotated[str, typer.Option("--format", help="jsonl | otel")] = "jsonl",
    segmenter: Annotated[Path | None, typer.Option(help="Segmenter config (YAML)")] = None,
    counter: Annotated[str, typer.Option(help="Token counter: heuristic | tiktoken")] = "heuristic",
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Import traces and split each request into fingerprinted segments."""
    store = Store(db)
    n, k = _import(store, path, fmt, segmenter, counter)
    typer.echo(f"imported {n} traces, {k} segments -> {db}")


def _provider(name: str, base_url: str | None, model: str | None) -> Provider:
    if name == "fake-demo":
        from .demo import demo_provider

        return demo_provider()
    if name == "fake":
        return FakeProvider([])
    if name == "openai":
        if not base_url:
            raise typer.BadParameter("--base-url is required for --provider openai (e.g. http://localhost:11434/v1)")
        return OpenAICompatProvider(base_url, model=model)
    raise typer.BadParameter(f"unknown provider {name!r}: fake-demo | openai")


@app.command()
def profile(
    provider: Annotated[str, typer.Option(help="fake-demo | openai (any OpenAI-compatible endpoint)")] = "fake-demo",
    base_url: Annotated[str | None, typer.Option(help="Endpoint for --provider openai")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model name from the traces")] = None,
    mode: Annotated[str, typer.Option(help="logprob | blackbox")] = "logprob",
    metric: Annotated[str, typer.Option(help="Black-box metric: auto | text | tool | json")] = "auto",
    n_masks: Annotated[int, typer.Option("--n", help="Ablation masks per trace")] = 32,
    sample: Annotated[int | None, typer.Option(help="Profile a random sample of K traces")] = None,
    budget: Annotated[float, typer.Option(help="Hard spend cap in USD for this run")] = 5.0,
    price_in: Annotated[float, typer.Option(help="USD per 1M input tokens")] = 3.0,
    price_out: Annotated[float, typer.Option(help="USD per 1M output tokens")] = 15.0,
    concurrency: Annotated[int, typer.Option(help="Concurrent model calls")] = 4,
    seed: Annotated[int, typer.Option(help="Random seed for sampling and masks")] = 0,
    counter: Annotated[str, typer.Option(help="Token counter for cost estimates")] = "heuristic",
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Ablate segments, score answers, fit the Lasso surrogate for each trace."""
    store = Store(db)
    ids = store.trace_ids()
    if not ids:
        raise typer.BadParameter("no traces in the database; run `ctxprof import` or `ctxprof demo` first")
    if sample and sample < len(ids):
        ids = sorted(random.Random(seed).sample(ids, sample))
    prov = _provider(provider, base_url, model)
    cfg = ProfileConfig(n_masks=n_masks, mode=mode, metric=metric, seed=seed, concurrency=concurrency,
                        budget_usd=budget, price_in_per_m=price_in, price_out_per_m=price_out)

    def progress(n: int, total: int, gate) -> None:
        if n % 10 == 0 or n == total:
            typer.echo(f"  {n}/{total} traces | calls {gate.calls} | cache hits {gate.cache_hits} | "
                       f"spent ${gate.spent:.2f} / ${cfg.budget_usd:.2f}")

    async def go() -> int:
        try:
            return await run_profile(store, prov, cfg, get_counter(counter), ids, on_progress=progress)
        finally:
            await prov.aclose()

    run_id = asyncio.run(go())
    run = next(r for r in store.list_runs() if r["id"] == run_id)
    typer.echo(f"run {run_id}: {run['status']} | {run['n_traces']} traces | spent ${run['spent_usd']:.2f}"
               + (f" | {run['note']}" if run["note"] else ""))


def _run_or_latest(store: Store, run: int | None) -> int:
    rid = run if run is not None else store.latest_run_id()
    if rid is None:
        raise typer.BadParameter("no finished profiling run; run `ctxprof profile` first")
    return rid


@app.command("aggregate")
def aggregate_cmd(
    run: Annotated[int | None, typer.Option(help="Run id (default: latest)")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
    min_traces: Annotated[int, typer.Option(help="DEAD needs at least this many traces")] = 20,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Rank segment fingerprints across traces; label DEAD, CRITICAL and VOLATILE."""
    store = Store(db)
    rid = _run_or_latest(store, run)
    rows = store.run_rows(rid)
    rules = LabelRules(min_traces=min_traces)
    agg = aggregate(rows, rules)
    inj = injection_signals(rows, rules)
    if as_json:
        typer.echo(json.dumps({"run": rid, "segments": agg, "injection": inj}, indent=2))
        return
    typer.echo(f"run {rid} | {len(agg)} fingerprints\n")
    typer.echo(f"{'fingerprint':44} {'traces':>6} {'tok/req':>8} {'paid':>9} {'mean':>6} {'sd':>5} {'/1k':>6}  label")
    for s in agg:
        typer.echo(f"{s['fingerprint'][:44]:44} {s['traces']:>6} {s['tokens_mean']:>8.0f} {s['tokens_paid']:>9,} "
                   f"{s['mean_influence']:>6.2f} {s['sd_influence']:>5.2f} {s['influence_per_1k']:>6.2f}  {s['label']}")
    dead = [s for s in agg if s["label"] == "DEAD"]
    n_traces = len({r["trace_id"] for r in rows if not r["poor_fit"]}) or 1
    per_req = sum(s["tokens_paid"] for s in dead) / n_traces
    typer.echo(f"\nDEAD: {len(dead)} fingerprints, {per_req:,.0f} tokens per request on average "
               f"(over {n_traces} well-fit traces)")
    if inj:
        typer.echo(f"injection signals: {len(inj)} (untrusted segments in the top of their trace)")
        for h in inj[:10]:
            typer.echo(f"  {h['trace_id']} {h['fingerprint']} influence {h['influence']:.2f} "
                       f"rank {h['rank']}/{h['of']}")


@app.command()
def flame(
    trace_id: Annotated[str, typer.Argument(help="Trace id")],
    run: Annotated[int | None, typer.Option(help="Run id for influence (default: latest, if any)")] = None,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Print the flame tree of one trace as JSON."""
    store = Store(db)
    _, segs = store.load_trace(trace_id)
    rid = run if run is not None else store.latest_run_id()
    attrs = store.attributions(rid, trace_id) if rid is not None else {}
    typer.echo(json.dumps(flame_tree(trace_id, segs, attrs), indent=2))


@app.command()
def prune() -> None:
    """(Milestone M4) Write a unified diff removing DEAD segments."""
    typer.echo("prune is not implemented yet: milestone M4 in roadmap.txt")
    raise typer.Exit(2)


@app.command()
def verify() -> None:
    """(Milestone M4) Re-run held-out traces with the pruned context and report agreement."""
    typer.echo("verify is not implemented yet: milestone M4 in roadmap.txt")
    raise typer.Exit(2)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Serve the API (and the built web UI from web/dist, if present)."""
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(db), host=host, port=port)


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Where to write the demo files")] = Path("examples/demo"),
    n: Annotated[int, typer.Option(help="Number of synthetic traces")] = 200,
    run_profile_too: Annotated[bool, typer.Option("--profile/--no-profile", help="Also profile them")] = True,
    budget: Annotated[float, typer.Option(help="Spend cap for the demo profile (estimated USD)")] = 250.0,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Generate the bloated JobBridge support-agent demo, import it and (optionally) profile it."""
    from .demo import write_demo

    paths = write_demo(out, n=n)
    store = Store(db)
    k, segs = _import(store, paths["traces"], "jsonl", paths["segmenter"], "heuristic")
    typer.echo(f"demo: {k} traces, {segs} segments -> {db} (files in {out})")
    if run_profile_too:
        profile(provider="fake-demo", base_url=None, model=None, mode="logprob", metric="auto", n_masks=32,
                sample=None, budget=budget, price_in=3.0, price_out=15.0, concurrency=8, seed=0,
                counter="heuristic", db=db)


if __name__ == "__main__":  # pragma: no cover
    app()
