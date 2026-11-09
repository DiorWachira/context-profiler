import { useState } from "react";
import { api, type FlameNode, type TraceDetail } from "../api";
import { Flame } from "../components/Flame";
import { fmt, heatBin, heatClass } from "../heat";
import { useLoad } from "../hooks";

function findNode(n: FlameNode, id: string): FlameNode | null {
  if (n.id === id) return n;
  for (const c of n.children) { const f = findNode(c, id); if (f) return f; }
  return null;
}

function defaultSelection(tree: FlameNode): string {
  let best: FlameNode | null = null;
  for (const kind of tree.children) for (const leaf of kind.children) {
    if (leaf.influence != null && (!best || (leaf.influence ?? 0) > (best.influence ?? 0))) best = leaf;
  }
  return best?.id ?? tree.id;
}

export function TraceView({ id }: { id: string }) {
  const s = useLoad(() => Promise.all([api.flame(id), api.trace(id)]), [id]);
  const [sel, setSel] = useState<string | null>(null);
  const [mask, setMask] = useState(0);

  if (s.state === "loading") return <main className="wrap"><p className="empty">Loading trace {id}…</p></main>;
  if (s.state === "error") return <main className="wrap"><div className="panel empty">{s.error}</div></main>;
  const [{ tree, fit }, detail] = s.data;
  const selected = sel ?? defaultSelection(tree);
  const node = findNode(tree, selected) ?? tree;

  return (
    <main className="wrap">
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span className="lbl">
          Trace {detail.id} | {detail.model} | {fmt(tree.tokens)} tokens
          {fit ? ` | ${fit.mode} | held-out R² ${fit.r2_heldout?.toFixed(2) ?? "n/a"}${fit.poor_fit ? " (poor fit)" : ""}` : " | not profiled"}
        </span>
        <h1 className="title">Context window, trace {detail.id}</h1>
      </div>
      <div className="cols">
        <div className="main"><Flame tree={tree} selected={selected} onSelect={setSel} /></div>
        <aside className="side" aria-label="Selected segment">
          <SelectedPanel node={node} total={tree.tokens} />
        </aside>
      </div>
      <DrillDown detail={detail} mask={mask} setMask={setMask} />
    </main>
  );
}

function SelectedPanel({ node, total }: { node: FlameNode; total: number }) {
  const label = node.fingerprint ?? node.name;
  const untrustedHot = node.source === "untrusted" && (node.influence ?? 0) >= 0.55 && node.kind !== "user";
  return (
    <section className="panel">
      <div className="ph"><span className="n">02.2</span><h2>Selected</h2>
        <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>tok {fmt(node.offset)} – {fmt(node.offset + node.tokens)}</span></div>
      <div className="mono" style={{ fontSize: 16, fontWeight: 600, padding: "12px 14px 8px", wordBreak: "break-word" }}>{label}</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, padding: "0 14px 12px" }}>
        <span className="chip">{node.kind}</span>
        {node.source && <span className="chip">{node.source}</span>}
        {node.influence == null && node.kind === "user" && <span className="chip">pinned: not ablated</span>}
        {untrustedHot && <span className="chip solid">⚑ injection signal</span>}
      </div>
      <div className="readouts" style={{ borderTop: "1px solid var(--rule)" }}>
        <div><span className="lbl">Tokens</span><span className="v">{fmt(node.tokens)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>{((node.tokens / total) * 100).toFixed(1)}% of window</span></div>
        <div><span className="lbl">Influence</span><span className="v">{node.influence == null ? "—" : node.influence.toFixed(2)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>{node.influence == null ? "not scored" : `heat H${heatBin(node.influence)}`}</span></div>
        <div><span className="lbl">Surrogate weight</span><span className="v">{node.weight == null ? "—" : node.weight.toFixed(3)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>{node.ci ? `95% CI ${node.ci[0].toFixed(2)} – ${node.ci[1].toFixed(2)}` : ""}</span></div>
        <div><span className="lbl">Influence / 1k tok</span><span className="v">
          {node.influence == null ? "—" : ((node.influence / Math.max(1, node.tokens)) * 1000).toFixed(2)}</span></div>
      </div>
    </section>
  );
}

function DrillDown({ detail, mask, setMask }: { detail: TraceDetail; mask: number; setMask: (n: number) => void }) {
  const scored = detail.segments.filter((x) => x.weight != null).sort((a, b) => (b.weight ?? 0) - (a.weight ?? 0));
  const maxW = Math.max(1e-9, ...scored.map((x) => Math.abs(x.weight ?? 0)));
  const ab = detail.ablations;
  const sMin = Math.min(...ab.map((a) => a.score)), sMax = Math.max(...ab.map((a) => a.score));
  const cur = ab[mask];
  return (
    <div className="cols">
      <section className="panel main" aria-labelledby="ab-h">
        <div className="ph"><span className="n">04.1</span><h2 id="ab-h">Ablation raster</h2>
          <span className="dim" style={{ fontSize: 12.5 }}>{ab.length} masks × {detail.segments.length} segments | filled = kept, outlined = removed, grey = pinned</span></div>
        {ab.length === 0 ? <p className="empty">Not profiled yet. Run <code>ctxprof profile</code>.</p> : (
          <>
            <div className="tbox"><div className="raster">
              {ab.map((a) => (
                <button key={a.mask_idx} type="button" className="mrow" aria-pressed={a.mask_idx === mask}
                  aria-label={`Mask ${a.mask_idx + 1}, score ${a.score.toFixed(2)}`} onClick={() => setMask(a.mask_idx)}>
                  <span className="mono dim" style={{ fontSize: 10.5 }}>#{String(a.mask_idx + 1).padStart(2, "0")}</span>
                  <span className="cells" style={{ gridTemplateColumns: `repeat(${a.mask.length}, minmax(0, 1fr))` }}>
                    {a.mask.split("").map((c, i) => {
                      const pinned = detail.segments[i]?.weight == null;
                      return <i key={i} className={pinned ? "p" : c === "1" ? "k" : "x"} />;
                    })}
                  </span>
                  <span className="mono" style={{ fontSize: 10.5, color: "var(--ink-2)", display: "flex", gap: 6, alignItems: "center" }}>
                    <b style={{ display: "block", height: 6, background: "var(--ink-3)",
                      width: Math.max(2, ((a.score - sMin) / Math.max(1e-9, sMax - sMin)) * 44) }} />{a.score.toFixed(2)}
                  </span>
                </button>
              ))}
            </div></div>
            {cur && (
              <div style={{ borderTop: "1px solid var(--rule)", padding: "12px 14px" }} aria-live="polite">
                <span className="lbl">Mask #{String(cur.mask_idx + 1).padStart(2, "0")} | score {cur.score.toFixed(3)}
                  {detail.fit ? ` | original ${detail.fit.original_score.toFixed(3)}` : ""}</span>
                <p className="mono" style={{ margin: "6px 0 0", fontSize: 12.5, color: "var(--ink-2)" }}>
                  removed: {cur.mask.split("").map((c, i) => (c === "0" ? detail.segments[i]?.fingerprint : null)).filter(Boolean).join(", ") || "nothing"}
                </p>
                {cur.response_text && <p className="msg mono" style={{ padding: "8px 0 0" }}>{cur.response_text}</p>}
              </div>
            )}
          </>
        )}
      </section>
      <div className="side">
        <section className="panel" aria-labelledby="rs-h">
          <div className="ph"><span className="n">04.2</span><h2 id="rs-h">Original response</h2></div>
          <p className="msg">{detail.response.text}</p>
          {detail.response.tool_calls.length > 0 && (
            <p className="mono dim" style={{ margin: 0, padding: "0 14px 12px", fontSize: 12 }}>
              tool calls: {detail.response.tool_calls.map((c) => c.name).join(", ")}</p>
          )}
        </section>
        <section className="panel" aria-labelledby="wt-h">
          <div className="ph"><span className="n">04.3</span><h2 id="wt-h">Surrogate weights</h2>
            <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>Lasso | 95% bootstrap CI</span></div>
          {scored.slice(0, 12).map((x) => {
            const w = x.weight ?? 0;
            const lo = Math.max(0, x.ci_lo ?? w), hi = Math.max(lo, x.ci_hi ?? w);
            return (
              <div className="w" key={x.index}>
                <span className="nm" title={x.fingerprint}>{x.fingerprint}</span>
                <span className="track" role="img" aria-label={`${x.fingerprint} weight ${w.toFixed(2)}`}>
                  <b className={heatClass(x.influence)} style={{ width: `${(Math.max(0, w) / maxW) * 100}%` }} />
                  <s style={{ left: `${(lo / maxW) * 100}%`, width: `${((hi - lo) / maxW) * 100}%` }} />
                </span>
                <span style={{ textAlign: "right" }}>{w.toFixed(2)}</span>
              </div>
            );
          })}
          {scored.length > 12 && <div className="w" style={{ borderBottom: 0 }}><span className="nm dim">{scored.length - 12} more</span><span /><span /></div>}
        </section>
      </div>
    </div>
  );
}
