import { useMemo, useState } from "react";
import type { TraceRow } from "../api";
import { fmt } from "../heat";

type Flag = "all" | "profiled" | "held" | "poor" | "injection";
const FLAGS: { id: Flag; label: string }[] = [
  { id: "all", label: "all" }, { id: "profiled", label: "profiled" }, { id: "held", label: "held out" },
  { id: "poor", label: "poor fit" }, { id: "injection", label: "⚑ injection" },
];
const PAGE = 15;

export function TraceList({ traces }: { traces: TraceRow[] }) {
  const [q, setQ] = useState("");
  const [intent, setIntent] = useState("all");
  const [flag, setFlag] = useState<Flag>("all");
  const [limit, setLimit] = useState(PAGE);
  const intents = useMemo(() => ["all", ...new Set(traces.map((t) => t.intent).filter((x): x is string => !!x))], [traces]);
  const rows = traces.filter((t) =>
    (!q || t.id.toLowerCase().includes(q.trim().toLowerCase()))
    && (intent === "all" || t.intent === intent)
    && (flag === "all" || (flag === "profiled" && t.profiled) || (flag === "held" && !t.profiled)
      || (flag === "poor" && t.poor_fit) || (flag === "injection" && t.injection)));

  return (
    <section className="panel" aria-labelledby="tr-h">
      <div className="ph"><span className="n">05</span><h2 id="tr-h">Traces</h2>
        <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }} aria-live="polite">
          {rows.length} of {traces.length}</span></div>
      <div className="filters">
        <label className="search"><span className="lbl">Trace id</span>
          <input type="search" value={q} placeholder="t0014" onChange={(e) => { setQ(e.target.value); setLimit(PAGE); }} /></label>
        {intents.length > 2 && (
          <div className="group" role="group" aria-label="Filter by intent">
            {intents.map((x) => <button key={x} type="button" className="btn" aria-pressed={intent === x}
              onClick={() => { setIntent(x); setLimit(PAGE); }}>{x}</button>)}
          </div>
        )}
        <div className="group" role="group" aria-label="Filter by status">
          {FLAGS.map((f) => <button key={f.id} type="button" className="btn" aria-pressed={flag === f.id}
            onClick={() => { setFlag(f.id); setLimit(PAGE); }}>{f.label}</button>)}
        </div>
      </div>
      <div className="tbox">
        <table>
          <caption className="sr">Imported traces</caption>
          <thead><tr><th scope="col">Trace</th><th scope="col">Intent</th><th scope="col">Status</th><th scope="col">Model</th>
            <th scope="col" className="r">Segments</th><th scope="col" className="r">Tokens</th></tr></thead>
          <tbody>
            {rows.slice(0, limit).map((t) => (
              <tr key={t.id}>
                <td><a href={`#/trace/${encodeURIComponent(t.id)}`}>{t.id}</a></td>
                <td style={{ color: "var(--ink-2)" }}>{t.intent ?? "—"}</td>
                <td>
                  {t.profiled ? <span className="dim">profiled</span> : <span className="chip">held out</span>}
                  {t.poor_fit && <span className="chip" style={{ marginLeft: 6 }}>R² &lt; .5</span>}
                  {t.injection && <span className="chip solid" style={{ marginLeft: 6 }}>⚑ injection</span>}
                </td>
                <td style={{ color: "var(--ink-2)" }}>{t.model}</td>
                <td className="r">{t.n_segments}</td><td className="r">{fmt(t.total_tokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <p className="empty">No trace matches these filters.</p>}
      </div>
      {rows.length > limit && (
        <div style={{ padding: "10px 14px", borderTop: "1px solid var(--rule)" }}>
          <button type="button" className="btn" onClick={() => setLimit(limit + PAGE * 4)}>
            Show {Math.min(PAGE * 4, rows.length - limit)} more</button>
        </div>
      )}
    </section>
  );
}
