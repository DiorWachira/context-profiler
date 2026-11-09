import { api, type Aggregate, type PruneRow, type Run, type TraceRow } from "../api";
import { fmt, heatClass, k } from "../heat";
import { useLoad } from "../hooks";
import { TraceList } from "../components/TraceList";

async function load(): Promise<{ traces: TraceRow[]; run: Run | null; agg: Aggregate | null; prune: PruneRow | null }> {
  const [traces, runs, prunes] = await Promise.all([api.traces(), api.runs(), api.prunes().catch(() => [])]);
  const run = runs.find((r) => r.status === "done" || r.status === "budget") ?? null;
  const agg = run ? await api.aggregate(run.id).catch(() => null) : null;
  const prune = prunes.find((p) => p.run_id === run?.id && p.recommend) ?? null;
  return { traces, run, agg, prune };
}

export function Overview() {
  const s = useLoad(load);
  if (s.state === "loading") return <main className="wrap"><p className="empty">Loading…</p></main>;
  if (s.state === "error") {
    return (
      <main className="wrap"><div className="panel empty">
        Could not reach the API ({s.error}). Start it with <code>ctxprof serve</code>.
      </div></main>
    );
  }
  const { traces, run, agg, prune } = s.data;
  if (!traces.length) {
    return (
      <main className="wrap"><div className="panel empty">
        No traces yet. Run <code>ctxprof demo</code> for the JobBridge demo, or <code>ctxprof import traces.jsonl</code>.
      </div></main>
    );
  }
  const mean = traces.reduce((a, t) => a + t.total_tokens, 0) / traces.length;
  const dead = agg?.segments.filter((x) => x.label === "DEAD") ?? [];
  const deadTok = dead.reduce((a, x) => a + x.tokens_paid, 0) / Math.max(1, agg?.summary.traces ?? 1);
  const top = [...(agg?.segments ?? [])].sort((a, b) => b.influence_per_1k - a.influence_per_1k).slice(0, 6);
  const price = Number(run?.config.price_in_per_m ?? 3);

  return (
    <main className="wrap">
      <section aria-labelledby="hero-h">
        <span className="lbl">
          {run ? `Run ${run.id} | ${run.n_traces} traces | ${String(run.config.n_masks)} ablations each | ${String(run.config.mode)}`
            : `${traces.length} traces imported | not profiled yet`}
        </span>
        <h1 className="hd" id="hero-h">
          {fmt(mean)} tokens in.<br />
          {agg ? <>{fmt(deadTok)} never move <em>the answer.</em></> : <em>Profile them to find out which matter.</em>}
        </h1>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          <a className="btn pri" href={`#/trace/${encodeURIComponent(traces[0].id)}`}>Open a flame graph</a>
          {agg && <a className="btn" href="#/segments">Segment table</a>}
        </div>
        {!run && <p className="note dim">Next: <code>ctxprof profile --provider fake-demo</code> (or an OpenAI-compatible endpoint).</p>}
      </section>

      {run && agg && (
        <section className="panel readouts" aria-label="Run readouts">
          <div><span className="lbl">Traces profiled</span><span className="v">{agg.summary.traces}</span>
            <span className="mono dim" style={{ fontSize: 12 }}>{agg.summary.poor_fit} poor fit (R² &lt; 0.5)</span></div>
          <div><span className="lbl">Model calls</span><span className="v">{fmt(run.calls)}</span>
            <span className="mono dim" style={{ fontSize: 12 }}>{fmt(run.cache_hits)} cache hits</span></div>
          <div><span className="lbl">Spend (est.)</span><span className="v">${run.spent_usd.toFixed(2)}</span>
            <span className="mono dim" style={{ fontSize: 12 }}>cap ${Number(run.config.budget_usd).toFixed(2)} | {run.status}</span></div>
          <div><span className="lbl">DEAD fingerprints</span><span className="v">{dead.length}</span>
            <span className="mono dim" style={{ fontSize: 12 }}>{((deadTok / mean) * 100).toFixed(1)}% of tokens</span></div>
          {prune ? (
            <a className="rlink" href={`#/pruning/${prune.id}`}><span className="lbl">Verified saving / 100k req</span>
              <span className="v">${((prune.tokens_saved * price) / 1e6 * 1e5).toFixed(2)}</span>
              <span className="mono dim" style={{ fontSize: 12 }}>prune {prune.id} · {((prune.agreement ?? 0) * 100).toFixed(1)}% agreement →</span></a>
          ) : (
            <div><span className="lbl">If all DEAD removed / 100k req</span><span className="v">${((deadTok * price) / 1e6 * 1e5).toFixed(2)}</span>
              <span className="mono dim" style={{ fontSize: 12 }}>unverified: run ctxprof prune + verify</span></div>
          )}
        </section>
      )}

      {agg && (
        <div className="cols">
          <section className="panel main" aria-labelledby="dead-h">
            <div className="ph"><span className="n">02</span><h2 id="dead-h">Paid, not used</h2>
              <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>tokens paid across traces</span></div>
            <ul className="list">
              {dead.slice(0, 8).map((x) => (
                <li key={x.fingerprint}><i className="sw h0" aria-hidden="true" /><span className="fp">{x.fingerprint}</span>
                  <span className="mono">{k(x.tokens_paid)}</span></li>
              ))}
            </ul>
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
            <section className="panel" aria-labelledby="inj-h">
              <div className="ph"><span className="n">04</span><h2 id="inj-h">Injection signal</h2>
                <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>untrusted, top of its trace</span></div>
              {agg.injection.length === 0 ? <p className="empty">No untrusted segment stood out.</p> : (
                <ul className="list">
                  {agg.injection.slice(0, 6).map((h) => (
                    <li key={h.trace_id + h.fingerprint}><i className={`sw ${heatClass(h.influence)}`} aria-hidden="true" />
                      <a className="fp" href={`#/trace/${encodeURIComponent(h.trace_id)}`}>{h.trace_id} | {h.fingerprint}</a>
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
