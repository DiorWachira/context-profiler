import { useState } from "react";
import { api, type Aggregate, type SegmentAgg } from "../api";
import { fmt, heatBin, heatClass, k } from "../heat";
import { useLoad } from "../hooks";

type SortKey = "paid" | "per1k" | "mean";
const KINDS = ["all", "system", "tool", "retrieved", "history"] as const;
const LABELS = ["all", "DEAD", "CRITICAL", "VOLATILE"] as const;
const GLYPH: Record<string, string> = { DEAD: "□ DEAD", CRITICAL: "▲ CRITICAL", VOLATILE: "≈ VOLATILE" };

async function load(): Promise<Aggregate | null> {
  const runs = await api.runs();
  const run = runs.find((r) => r.status === "done" || r.status === "budget");
  return run ? api.aggregate(run.id) : null;
}

export function Segments() {
  const s = useLoad(load);
  const [sort, setSort] = useState<SortKey>("paid");
  const [kind, setKind] = useState<(typeof KINDS)[number]>("all");
  const [label, setLabel] = useState<(typeof LABELS)[number]>("all");

  if (s.state === "loading") return <main className="wrap"><p className="empty">Loading…</p></main>;
  if (s.state === "error") return <main className="wrap"><div className="panel empty">{s.error}</div></main>;
  if (!s.data) {
    return <main className="wrap"><div className="panel empty">No profiling run yet. Run <code>ctxprof profile</code>.</div></main>;
  }
  const agg = s.data;
  const key = (x: SegmentAgg) => (sort === "paid" ? x.tokens_paid : sort === "per1k" ? x.influence_per_1k : x.mean_influence);
  const rows = agg.segments
    .filter((x) => (kind === "all" || x.kind === kind) && (label === "all" || x.label === label))
    .sort((a, b) => key(b) - key(a));
  const maxPaid = Math.max(1, ...agg.segments.map((x) => x.tokens_paid));
  const aria = (kk: SortKey) => (sort === kk ? "descending" : "none");

  return (
    <main className="wrap">
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: "12px 20px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginRight: "auto" }}>
          <span className="lbl">Run {agg.run} | aggregated by stable fingerprint across {agg.summary.traces} traces</span>
          <h1 className="title">{agg.segments.length} segments, ranked by what they cost</h1>
        </div>
        <span className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>{rows.length} shown</span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 18px" }}>
        <div className="group" role="group" aria-label="Filter by kind">
          {KINDS.map((x) => <button key={x} className="btn" type="button" aria-pressed={kind === x} onClick={() => setKind(x)}>{x}</button>)}
        </div>
        <div className="group" role="group" aria-label="Filter by label">
          {LABELS.map((x) => <button key={x} className="btn" type="button" aria-pressed={label === x} onClick={() => setLabel(x)}>
            {x === "all" ? "any label" : GLYPH[x]}</button>)}
        </div>
      </div>
      <section className="panel" aria-labelledby="t-h">
        <div className="ph"><span className="n">03.1</span><h2 id="t-h">Fingerprints</h2>
          <span className="dim" style={{ fontSize: 12.5 }}>DEAD = influence ≈ 0 in enough traces; VOLATILE segments are never auto-pruned</span></div>
        <div className="tbox">
          <table>
            <caption className="sr">Segment fingerprints aggregated across traces</caption>
            <thead>
              <tr>
                <th scope="col">Fingerprint</th><th scope="col">Kind</th><th scope="col">Source</th>
                <th scope="col" className="r">Traces</th><th scope="col" className="r">Tok / req</th>
                <th scope="col" className="r" aria-sort={aria("paid")}><button type="button" onClick={() => setSort("paid")}>Tokens paid ↓</button></th>
                <th scope="col" aria-sort={aria("mean")}><button type="button" onClick={() => setSort("mean")}>Mean influence ↓</button></th>
                <th scope="col" className="r">σ</th>
                <th scope="col" className="r" aria-sort={aria("per1k")}><button type="button" onClick={() => setSort("per1k")}>Infl / 1k tok ↓</button></th>
                <th scope="col">Label</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => (
                <tr key={x.fingerprint}>
                  <td>{x.fingerprint}</td>
                  <td style={{ color: "var(--ink-2)" }}>{x.kind}</td>
                  <td style={{ color: "var(--ink-2)" }}>{x.source}</td>
                  <td className="r">{x.traces}</td>
                  <td className="r">{fmt(x.tokens_mean)}</td>
                  <td className="r"><span className="paid"><i style={{ width: Math.max(2, (x.tokens_paid / maxPaid) * 90) }} aria-hidden="true" />{k(x.tokens_paid)}</span></td>
                  <td><i className={`sw ${heatClass(x.mean_influence)}`} style={{ marginRight: 8, verticalAlign: -2 }} aria-hidden="true" />
                    {x.mean_influence.toFixed(2)} <span className="dim">H{heatBin(x.mean_influence)}</span></td>
                  <td className="r" style={{ color: "var(--ink-2)" }}>{x.sd_influence.toFixed(2)}</td>
                  <td className="r">{x.influence_per_1k.toFixed(2)}</td>
                  <td>{x.label && <span className={`chip ${x.label === "DEAD" ? "" : "solid"}`}>{GLYPH[x.label]}</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
