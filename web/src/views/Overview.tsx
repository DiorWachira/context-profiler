import { api, type Aggregate, type PruneRow, type Run, type Spectrum, type TraceRow } from "../api";
import { fmt, heatClass, k } from "../heat";
import { useLoad } from "../hooks";
import { TraceList } from "../components/TraceList";
import { Waterfall } from "../components/Waterfall";
import { Seg7 } from "../components/Seg7";
import { Strip } from "../components/Strip";
import { Fault, Warmup } from "../components/Warmup";
import { HeatKey } from "../components/HeatKey";
import { openGuide } from "../guide/Guide";
import { stripsFrom } from "../spectrum";

interface Data { traces: TraceRow[]; run: Run | null; agg: Aggregate | null; prune: PruneRow | null; spec: Spectrum | null }

async function load(): Promise<Data> {
  const [traces, runs, prunes] = await Promise.all([api.traces(), api.runs(), api.prunes().catch(() => [])]);
  const run = runs.find((r) => r.status === "done" || r.status === "budget") ?? null;
  const [agg, spec] = run
    ? await Promise.all([api.aggregate(run.id).catch(() => null), api.spectrum(run.id).catch(() => null)])
    : [null, null];
  const prune = prunes.find((p) => p.run_id === run?.id && p.recommend) ?? null;
  return { traces, run, agg, prune, spec };
}

const open = (id: string) => { location.hash = `#/trace/${encodeURIComponent(id)}`; };

export function Overview() {
  const s = useLoad(load);
  if (s.state === "loading") return <Warmup label="Acquiring the run…" />;
  if (s.state === "error") return <Fault>Could not reach the API ({s.error}). Start it with <code>ctxprof serve</code>.</Fault>;
  const { traces, run, agg, prune, spec } = s.data;
  if (!traces.length) {
    return <Fault>No traces yet. Run <code>ctxprof demo</code> for the JobBridge demo, or <code>ctxprof import traces.jsonl</code>.</Fault>;
  }
  const mean = traces.reduce((a, t) => a + t.total_tokens, 0) / traces.length;
  const dead = agg?.segments.filter((x) => x.label === "DEAD") ?? [];
  const deadTok = dead.reduce((a, x) => a + x.tokens_paid, 0) / Math.max(1, agg?.summary.traces ?? 1);
  const top = [...(agg?.segments ?? [])].sort((a, b) => b.influence_per_1k - a.influence_per_1k).slice(0, 6);
  const price = Number(run?.config.price_in_per_m ?? 3);
  const strips = spec ? stripsFrom(spec) : null;
  const maxPaid = Math.max(1, ...dead.map((x) => x.tokens_paid));
  const firstTrace = traces.find((t) => t.profiled) ?? traces[0];

  return (
    <main className="wrap">
      <section className="hero" aria-labelledby="hero-h">
        <div className="hero-copy">
          <span className="lbl">
            {run ? `Run ${run.id} · ${run.n_traces} requests · ${String(run.config.n_masks)} ablations each · ${String(run.config.mode)} mode`
              : `${traces.length} requests imported · not profiled yet`}
          </span>
          <h1 className="hd" id="hero-h">
            {fmt(mean)} tokens in.<br />
            {agg ? <>{fmt(deadTok)} never move <em>the answer.</em></> : <em>Profile them to find out which matter.</em>}
          </h1>
          <p className="lede">
            Context Profiler removes pieces of real prompts, replays them, and measures what the answer actually
            depended on. Then it writes the diff that cuts the rest, and proves it on requests it has never seen.
          </p>
          <div className="cta">
            <a className="btn pri" href={`#/trace/${encodeURIComponent(firstTrace.id)}`}>Open a flame graph</a>
            {agg && <a className="btn" href="#/pruning">Pruning verdict</a>}
            <button type="button" className="btn ghost" onClick={() => openGuide({ action: "tour", scope: "all" })}>
              <span className="balloon sm" aria-hidden="true">?</span>Take the guide</button>
          </div>
          {!run && <p className="note dim">Next: <code>ctxprof profile --provider fake-demo</code> (or an OpenAI-compatible endpoint).</p>}
        </div>

        {spec && (
          <figure className="panel fig screen-panel" data-tour="waterfall">
            <div className="ph">
              <span className="n">FIG. 1</span><h2>Waterfall · {spec.rows.length} requests</h2>
              <span className="acq-live mono"><i className="lamp" aria-hidden="true" />ACQ</span>
            </div>
            <Waterfall spec={spec} height={Math.min(420, Math.max(220, spec.rows.length * 2))} onOpen={open} />
            <figcaption>
              <HeatKey />
              <span className="dim">Row = one request, in trace order. x = token position in its context window.
                Colour = influence of that segment on the answer. Hatched gutter tick = poor surrogate fit.</span>
            </figcaption>
          </figure>
        )}
      </section>

      {run && agg && (
        <section className="panel leds" aria-label="Run readouts" data-tour="readouts">
          <div className="led-cell">
            <span className="lbl">Requests profiled</span>
            <Seg7 value={agg.summary.traces} format={(v) => fmt(v)} label="Requests profiled" />
            <span className="mono dim sm">{agg.summary.poor_fit} poor fit (R² &lt; 0.5)</span>
          </div>
          <div className="led-cell">
            <span className="lbl">Model calls</span>
            <Seg7 value={run.calls} format={(v) => fmt(v)} label="Model calls" />
            <span className="mono dim sm">{fmt(run.cache_hits)} served from cache</span>
          </div>
          <div className="led-cell">
            <span className="lbl">Spend (est.) $</span>
            <Seg7 value={run.spent_usd} format={(v) => v.toFixed(2)} label="Estimated spend in dollars" />
            <div className="meter sm" aria-hidden="true"><b style={{ width: `${Math.min(100, (run.spent_usd / Number(run.config.budget_usd)) * 100)}%` }} /></div>
            <span className="mono dim sm">cap ${Number(run.config.budget_usd).toFixed(2)} · {run.status}</span>
          </div>
          <div className="led-cell">
            <span className="lbl">DEAD fingerprints</span>
            <Seg7 value={dead.length} format={(v) => String(Math.round(v))} label="DEAD fingerprints" />
            <span className="mono dim sm">{((deadTok / mean) * 100).toFixed(1)}% of the window</span>
          </div>
          {prune ? (
            <a className="led-cell rlink" href={`#/pruning/${prune.id}`}>
              <span className="lbl">Verified saving $ / 100k req</span>
              <Seg7 value={(prune.tokens_saved * price) / 1e6 * 1e5} format={(v) => v.toFixed(2)} label="Verified saving per 100k requests in dollars" />
              <span className="mono dim sm">prune {prune.id} · {((prune.agreement ?? 0) * 100).toFixed(1)}% agreement →</span>
            </a>
          ) : (
            <div className="led-cell">
              <span className="lbl">If all DEAD cut $ / 100k</span>
              <Seg7 value={(deadTok * price) / 1e6 * 1e5} format={(v) => v.toFixed(2)} label="Unverified saving" />
              <span className="mono dim sm">unverified: run prune + verify</span>
            </div>
          )}
        </section>
      )}

      {agg && (
        <div className="cols">
          <section className="panel main" aria-labelledby="dead-h" data-tour="dead">
            <div className="ph"><span className="n">02</span><h2 id="dead-h">Paid for, never used</h2>
              <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>tokens paid · influence in each request</span></div>
            <ul className="list deadlist">
              {dead.slice(0, 9).map((x) => (
                <li key={x.fingerprint}>
                  <span className="fp" title={x.fingerprint}>{x.fingerprint}</span>
                  <span className="paid"><i style={{ width: `${(x.tokens_paid / maxPaid) * 100}%` }} aria-hidden="true" />
                    <span className="mono">{k(x.tokens_paid)}</span></span>
                  {strips && <Strip values={strips.strips.get(x.fingerprint) ?? []} ids={strips.ids} label={x.fingerprint} width={150} height={12} />}
                </li>
              ))}
            </ul>
            {dead.length > 9 && <a className="more-link mono" href="#/segments">+{dead.length - 9} more in the segment table →</a>}
          </section>
          <div className="side">
            <section className="panel" aria-labelledby="crit-h">
              <div className="ph"><span className="n">03</span><h2 id="crit-h">Carries the answer</h2>
                <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>influence / 1k tok</span></div>
              <ul className="list">
                {top.map((x) => (
                  <li key={x.fingerprint}><i className={`sw ${heatClass(x.mean_influence)}`} aria-hidden="true" />
                    <span className="fp">{x.fingerprint}</span><span className="mono">{x.influence_per_1k.toFixed(2)}</span></li>
                ))}
              </ul>
            </section>
            <section className="panel" aria-labelledby="inj-h" data-tour="inject">
              <div className="ph"><span className="n">04</span><h2 id="inj-h">Injection signal</h2>
                <span className={`annun${agg.injection.length ? " on" : ""}`}>{agg.injection.length ? `⚑ ${agg.injection.length}` : "CLEAR"}</span></div>
              {agg.injection.length === 0 ? <p className="empty">No untrusted segment stood out.</p> : (
                <ul className="list">
                  {agg.injection.slice(0, 6).map((h) => (
                    <li key={h.trace_id + h.fingerprint}><i className={`sw ${heatClass(h.influence)}`} aria-hidden="true" />
                      <a className="fp" href={`#/trace/${encodeURIComponent(h.trace_id)}`}>{h.trace_id} · {h.fingerprint}</a>
                      <span className="mono">{h.influence.toFixed(2)}</span></li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </div>
      )}

      <TraceList traces={traces} />
    </main>
  );
}
