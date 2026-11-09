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
from .prune import Bundle, PrunePlan, build_plan, write_bundle
from .segmenter import SegmenterConfig, segment_trace
from .storage import Store
from .tokens import get_counter
from .verify import VerifyConfig, run_verify

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
    if name == "hf":
        from .providers.hf_local import HFLocalProvider

        return HFLocalProvider(model or "Qwen/Qwen2.5-0.5B-Instruct")
    raise typer.BadParameter(f"unknown provider {name!r}: fake-demo | openai | hf")


def _embedder(metric: str, url: str | None, model: str | None):
    if metric != "embedding":
        return None
    if not url or not model:
        raise typer.BadParameter("--metric embedding needs --embed-url and --embed-model")
    from .providers.embeddings import OpenAICompatEmbedder

    return OpenAICompatEmbedder(url, model)


@app.command()
def profile(
    provider: Annotated[str, typer.Option(
        help="fake-demo | openai (any OpenAI-compatible endpoint) | hf (local transformers model)")] = "fake-demo",
    base_url: Annotated[str | None, typer.Option(help="Endpoint for --provider openai")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model name from the traces")] = None,
    mode: Annotated[str, typer.Option(help="logprob | blackbox")] = "logprob",
    metric: Annotated[str, typer.Option(help="Black-box metric: auto | text | tool | json | embedding")] = "auto",
    embed_url: Annotated[str | None, typer.Option(help="OpenAI-compatible base URL for /embeddings")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Embedding model name (metric embedding)")] = None,
    n_masks: Annotated[int, typer.Option("--n", help="Ablation masks per trace")] = 32,
    sample: Annotated[int | None, typer.Option(help="Profile a random sample of K traces")] = None,
    budget: Annotated[float, typer.Option(help="Hard spend cap in USD for this run")] = 5.0,
    price_in: Annotated[float, typer.Option(help="USD per 1M input tokens")] = 3.0,
    price_out: Annotated[float, typer.Option(help="USD per 1M output tokens")] = 15.0,
    concurrency: Annotated[int, typer.Option(help="Concurrent model calls")] = 4,
    trace_concurrency: Annotated[int, typer.Option(help="Traces profiled at once")] = 4,
    masks_per_segment: Annotated[float, typer.Option(help="Scale masks with segments: N >= this x segments")] = 0.0,
    interactions: Annotated[int, typer.Option(help="Pairwise interaction check among the top K segments")] = 0,
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
    emb = _embedder(metric, embed_url, embed_model)
    cfg = ProfileConfig(n_masks=n_masks, mode=mode, metric=metric, seed=seed, concurrency=concurrency,
                        budget_usd=budget, price_in_per_m=price_in, price_out_per_m=price_out,
                        trace_concurrency=trace_concurrency, masks_per_segment=masks_per_segment,
                        interaction_top=interactions)

    def progress(n: int, total: int, gate) -> None:
        if n % 10 == 0 or n == total:
            typer.echo(f"  {n}/{total} traces | calls {gate.calls} | cache hits {gate.cache_hits} | "
                       f"spent ${gate.spent:.2f} / ${cfg.budget_usd:.2f}")

    async def go() -> int:
        try:
            return await run_profile(store, prov, cfg, get_counter(counter), ids, on_progress=progress,
                                     embedder=emb)
        finally:
            await prov.aclose()
            if emb is not None:
                await emb.aclose()

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


def _segmenter_for(bundle: Path, segmenter: Path | None) -> SegmenterConfig:
    """Heading levels must match the segmenter that produced the fingerprints."""
    if segmenter is not None:
        return SegmenterConfig.load(segmenter)
    for cand in (bundle / "segmenter.yaml", bundle.parent / "segmenter.yaml"):
        if cand.is_file():
            return SegmenterConfig.load(cand)
    return SegmenterConfig()


def make_prune(store: Store, run_id: int, bundle_dir: Path, keep: list[str], rules: LabelRules,
               segmenter: Path | None = None) -> tuple[int, PrunePlan, Bundle]:
    rows = store.run_rows(run_id)
    if not rows:
        raise typer.BadParameter(f"run {run_id} has no attributions")
    bundle = Bundle.load(bundle_dir)
    cfg = _segmenter_for(bundle_dir, segmenter)
    traces = {r["trace_id"] for r in rows if not r["poor_fit"]}
    plan = build_plan(aggregate(rows, rules), bundle, keep=keep, injection=injection_signals(rows, rules),
                      heading_levels=cfg.heading_levels, tool_calls=store.tool_call_counts(traces),
                      n_traces=len(traces))
    pid = store.create_prune(run_id, bundle_dir.as_posix(), sorted(set(keep)), plan.model_dump(mode="json"))
    return pid, plan, bundle


@app.command()
def prune(
    bundle: Annotated[Path, typer.Option(help="Prompt bundle: system.md, tools.json, optional retriever.yaml")] = Path(
        "examples/demo/prompt"),
    run: Annotated[int | None, typer.Option(help="Run id (default: latest)")] = None,
    keep: Annotated[list[str] | None, typer.Option(help="Never remove this fingerprint (repeatable)")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the unified diff here instead of stdout")] = None,
    write: Annotated[Path | None, typer.Option(help="Write the pruned bundle to this directory")] = None,
    segmenter: Annotated[Path | None, typer.Option(help="Segmenter config used at import (heading levels)")] = None,
    min_traces: Annotated[int, typer.Option(help="DEAD needs at least this many traces")] = 20,
    quiet: Annotated[bool, typer.Option(help="Do not print the diff")] = False,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Write a unified diff that removes DEAD segments from the prompt bundle."""
    store = Store(db)
    rid = _run_or_latest(store, run)
    try:
        pid, plan, b = make_prune(store, rid, bundle, keep or [], LabelRules(min_traces=min_traces), segmenter)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if out:
        out.write_text(plan.diff, encoding="utf-8")
    elif not quiet:
        typer.echo(plan.diff, nl=False)
    if write:
        write_bundle(plan, b, write)

    def err(m: str) -> None:
        typer.echo(m, err=True)

    cut = [h for h in plan.hunks if h.cut]
    err(f"prune {pid} (run {rid}): {len(cut)} of {len(plan.hunks)} DEAD candidates cut, "
        f"~{plan.tokens_saved:,.0f} tokens per request (estimate; run `ctxprof verify` to measure)")
    for h in plan.hunks:
        if not h.cut:
            err(f"  kept by --keep: {h.fingerprint}")
    for s in plan.skipped:
        err(f"  not pruned: {s.fingerprint} ({s.why})")
    for k in plan.unknown_keep:
        err(f"  note: --keep {k} is not a DEAD candidate in this run (never pruned anyway)")
    if out:
        err(f"diff -> {out}")
    if write:
        err(f"pruned bundle -> {write}")


@app.command()
def verify(
    prune_id: Annotated[int | None, typer.Option("--prune", help="Prune id (default: latest)")] = None,
    provider: Annotated[str, typer.Option(help="fake-demo | openai | hf")] = "fake-demo",
    base_url: Annotated[str | None, typer.Option(help="Endpoint for --provider openai")] = None,
    model: Annotated[str | None, typer.Option(help="Model name (overrides the traces' model)")] = None,
    held_out: Annotated[int | None, typer.Option(help="Random sample of K held-out traces")] = None,
    metric: Annotated[str, typer.Option(help="Black-box metric: auto | text | tool | json | embedding")] = "auto",
    embed_url: Annotated[str | None, typer.Option(help="OpenAI-compatible base URL for /embeddings")] = None,
    embed_model: Annotated[str | None, typer.Option(help="Embedding model name (metric embedding)")] = None,
    min_score: Annotated[float, typer.Option(help="Score at or above = same outcome")] = 0.9,
    threshold: Annotated[float, typer.Option(help="Agreement needed to recommend")] = 0.95,
    eval_cmd: Annotated[str | None, typer.Option(
        "--eval", help="Eval command with {original} {pruned} JSON paths; exit 0 = agree")] = None,
    probe: Annotated[bool, typer.Option("--probe/--no-probe", help="Attribute regressions to hunks")] = True,
    price_in: Annotated[float, typer.Option(help="USD per 1M input tokens")] = 3.0,
    price_out: Annotated[float, typer.Option(help="USD per 1M output tokens")] = 15.0,
    ksh_per_usd: Annotated[float, typer.Option(help="KSh per USD for the cost report")] = 129.0,
    budget: Annotated[float, typer.Option(help="Hard spend cap in USD for verification")] = 5.0,
    seed: Annotated[int, typer.Option(help="Seed for the held-out sample")] = 0,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Replay held-out traces with the pruned context; exit 0 if agreement meets the threshold."""
    store = Store(db)
    pid = prune_id if prune_id is not None else store.latest_prune_id()
    if pid is None:
        raise typer.BadParameter("no prune yet; run `ctxprof prune` first")
    prov = _provider(provider, base_url, model)
    emb = _embedder(metric, embed_url, embed_model)
    cfg = VerifyConfig(held_out=held_out, seed=seed, metric=metric, min_score=min_score, threshold=threshold,
                       eval_cmd=eval_cmd, probe=probe, price_in_per_m=price_in, price_out_per_m=price_out,
                       ksh_per_usd=ksh_per_usd, budget_usd=budget)

    async def go():
        try:
            return await run_verify(store, prov, pid, cfg, get_counter("heuristic"), embedder=emb)
        finally:
            await prov.aclose()
            if emb is not None:
                await emb.aclose()

    report = asyncio.run(go())
    raise typer.Exit(_print_verify(pid, report, cfg))


def _print_verify(pid: int, report: dict, cfg: VerifyConfig) -> int:
    s = report["summary"]
    typer.echo(f"prune {pid}: {s['traces']} held-out traces"
               + (" (IN-SAMPLE: no unprofiled traces were available)" if report["in_sample"] else ""))
    typer.echo(f"  tokens / request   {s['tokens_before']:,.0f} -> {s['tokens_after']:,.0f} "
               f"(-{s['reduction'] * 100:.1f}%)")
    typer.echo(f"  cost / request     ${s['cost_before']:.5f} -> ${s['cost_after']:.5f} "
               f"at ${cfg.price_in_per_m:.2f}/1M input")
    typer.echo(f"  saved / 100k req   ${s['saved_per_100k_usd']:,.2f} (KSh {s['saved_per_100k_ksh']:,.0f})")
    typer.echo(f"  agreement          {s['agree_n']}/{s['traces']} = {s['agreement'] * 100:.1f}% "
               f"(threshold {cfg.threshold * 100:.1f}%)")
    typer.echo(f"  calls {report['calls']} | cache hits {report['cache_hits']} | spent ${report['spent_usd']:.2f}"
               + (" | STOPPED AT BUDGET" if report["status"] == "budget" else ""))
    for r in report["regressions"]:
        typer.echo(f"  regression {r['trace_id']} ({r.get('intent') or '?'}): score {r['score']:.2f}, "
                   f"suspect {', '.join(r['breaks']) or 'unknown'}" + ("" if r["explained"] else " (interaction)"))
    probed = sorted({k for rec in report["records"] for k in rec["probes"]})
    if probed:
        n = sum(len(rec["probes"]) for rec in report["records"])
        typer.echo(f"  kept hunks that would break traces if cut: {', '.join(probed)} ({n} traces)")
    if s["recommend"]:
        typer.echo("VERIFIED: recommend this diff")
        return 0
    typer.echo("BELOW THRESHOLD: do not ship this diff (try --keep for the suspects)")
    return 1


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


@app.command("export")
def export_cmd(
    out: Annotated[Path, typer.Option(help="Directory of the built UI (web/dist); writes <out>/api/*.json")] = Path(
        "web/dist"),
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Write a static JSON snapshot of the API for serverless hosting (build the UI with VITE_STATIC=1)."""
    from .export import export_static

    n = export_static(db, out)
    typer.echo(f"exported {n} JSON files -> {out / 'api'}")


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Where to write the demo files")] = Path("examples/demo"),
    n: Annotated[int, typer.Option(help="Synthetic traces to profile")] = 200,
    held_out: Annotated[int, typer.Option(help="Extra traces kept out of profiling, for verify")] = 60,
    run_profile_too: Annotated[bool, typer.Option("--profile/--no-profile", help="Also profile them")] = True,
    run_prune_too: Annotated[bool, typer.Option("--prune/--no-prune", help="Also prune and verify")] = True,
    budget: Annotated[float, typer.Option(help="Spend cap for the demo profile (estimated USD)")] = 250.0,
    fresh: Annotated[bool, typer.Option("--fresh/--no-fresh", help="Start from an empty database")] = True,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Generate the bloated JobBridge support-agent demo, then profile, prune and verify it."""
    from .demo import demo_provider, write_demo

    if fresh and db.exists():
        db.unlink()
    paths = write_demo(out, n=n + held_out)
    store = Store(db)
    k, segs = _import(store, paths["traces"], "jsonl", paths["segmenter"], "heuristic")
    typer.echo(f"demo: {k} traces ({n} to profile, {held_out} held out), {segs} segments -> {db} (files in {out})")
    if not run_profile_too:
        return
    ids = store.trace_ids()[:n]
    cfg = ProfileConfig(n_masks=32, mode="logprob", seed=0, concurrency=8, budget_usd=budget, interaction_top=3)

    def progress(i: int, total: int, gate) -> None:
        if i % 25 == 0 or i == total:
            typer.echo(f"  profile {i}/{total} traces | calls {gate.calls} | spent ${gate.spent:.2f}")

    rid = asyncio.run(run_profile(store, demo_provider(), cfg, get_counter("heuristic"), ids, on_progress=progress))
    typer.echo(f"run {rid} done")
    if not run_prune_too or not held_out:
        return
    rules = LabelRules()
    vcfg = VerifyConfig()
    for keep in ([], ["tool/report_listing", "tool/schedule_interview"]):
        pid, plan, _ = make_prune(store, rid, paths["prompt"], keep, rules, paths["segmenter"])
        label = "default" if not keep else "--keep " + " --keep ".join(keep)
        typer.echo(f"\nprune {pid} ({label}): {len(plan.removed)} hunks cut")
        report = asyncio.run(run_verify(store, demo_provider(), pid, vcfg, get_counter("heuristic")))
        _print_verify(pid, report, vcfg)


if __name__ == "__main__":  # pragma: no cover
    app()
