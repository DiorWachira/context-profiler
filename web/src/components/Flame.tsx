// The signature: an icicle flame graph of one context window.
// Width = tokens (window order), heat = influence. Scroll zooms at the cursor, Shift+scroll pans,
// W/A/S/D on the focused stack, the ruler drops M1/M2 markers that sum tokens and influence.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FlameNode } from "../api";
import { fmt, heatClass } from "../heat";

interface Flat { node: FlameNode; depth: number; offset: number; parent: string | null }

function flatten(root: FlameNode, mode: "window" | "heavy"): Flat[] {
  const out: Flat[] = [];
  const walk = (n: FlameNode, depth: number, offset: number, parent: string | null) => {
    out.push({ node: n, depth, offset, parent });
    const kids = mode === "heavy"
      ? [...n.children].sort((a, b) => (b.influence ?? -1) - (a.influence ?? -1))
      : n.children;
    let o = offset;
    for (const c of kids) { walk(c, depth + 1, o, n.id); o += c.tokens; }
  };
  walk(root, 0, 0, null);
  return out;
}

const STEPS = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
const MIN_SPAN = 24;

export interface FlameProps {
  tree: FlameNode;
  selected: string;
  onSelect: (id: string) => void;
  pricePerM?: number;
}

export function Flame({ tree, selected, onSelect, pricePerM = 3 }: FlameProps) {
  const T = Math.max(1, tree.tokens);
  const [mode, setMode] = useState<"window" | "heavy">("window");
  const [view, setView] = useState({ v0: 0, span: T });
  const [hx, setHx] = useState<number | null>(null);
  const [W, setW] = useState(1000);
  const [markers, setMarkers] = useState<{ m1: number; m2: number; next: 1 | 2 }>(() => {
    const tools = tree.children.find((c) => c.kind === "tool");
    return tools ? { m1: tools.offset, m2: tools.offset + tools.tokens, next: 1 } : { m1: 0, m2: T, next: 1 };
  });
  const [live, setLive] = useState(false);
  const vpRef = useRef<HTMLDivElement>(null);
  const flat = useMemo(() => flatten(tree, mode), [tree, mode]);
  const viewRef = useRef(view);
  viewRef.current = view;

  const set = useCallback((v0: number, span: number) => {
    const s = Math.max(MIN_SPAN, Math.min(T, span));
    setView({ v0: Math.max(0, Math.min(T - s, v0)), span: s });
  }, [T]);
  const zoomAt = useCallback((frac: number, f: number) => {
    const { v0, span } = viewRef.current;
    const c = v0 + frac * span;
    const s = Math.max(MIN_SPAN, Math.min(T, span * f));
    set(c - frac * s, s);
  }, [T, set]);

  useEffect(() => {
    const el = vpRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const k = e.deltaMode === 1 ? 16 : 1;
      const { v0, span } = viewRef.current;
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        set(v0 + ((e.deltaX || e.deltaY) * k) / r.width * span, span);
      } else {
        zoomAt(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), Math.exp(e.deltaY * k * 0.0016));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    const ro = new ResizeObserver(([en]) => setW(en.contentRect.width || 1000));
    ro.observe(el);
    return () => { el.removeEventListener("wheel", onWheel); ro.disconnect(); };
  }, [set, zoomAt]);

  const { v0, span } = view;
  const pct = (t: number) => ((t - v0) / span) * 100;
  const sel = flat.find((f) => f.node.id === selected) ?? flat[0];

  const fitNode = (id: string) => {
    const f = flat.find((x) => x.node.id === id);
    if (f) set(f.offset - f.node.tokens * 0.04, f.node.tokens * 1.08);
  };

  const onKey = (e: React.KeyboardEvent) => {
    const frac = hx != null ? Math.max(0, Math.min(1, hx / W)) : 0.5;
    const key = e.key.toLowerCase();
    if (key === "w" || key === "+" || key === "=") zoomAt(frac, 1 / 1.4);
    else if (key === "s" || key === "-") zoomAt(frac, 1.4);
    else if (key === "a" || key === "arrowleft") set(v0 - span * 0.12, span);
    else if (key === "d" || key === "arrowright") set(v0 + span * 0.12, span);
    else if (key === "f") fitNode(selected);
    else if (key === "0" || key === "escape") set(0, T);
    else return;
    e.preventDefault();
  };

  // ruler ticks re-step with zoom so the scale stays readable
  const raw = span / Math.max(4, W / 95);
  const step = STEPS.find((s) => s >= raw) ?? 5000;
  const minor = step / 5;
  const ticks: { t: number; major: boolean }[] = [];
  for (let t = Math.ceil(v0 / minor) * minor; t <= v0 + span && ticks.length < 300; t += minor) {
    ticks.push({ t, major: Math.abs(t / step - Math.round(t / step)) < 1e-6 });
  }

  const a = Math.min(markers.m1, markers.m2), b = Math.max(markers.m1, markers.m2);
  let sum = 0, cnt = 0;
  for (const f of flat) {
    if (f.depth === 2 && f.node.influence != null && f.offset >= a && f.offset + f.node.tokens <= b) {
      sum += f.node.influence; cnt++;
    }
  }

  const placeMarker = (e: React.MouseEvent<HTMLButtonElement>) => {
    let t: number;
    if (e.detail === 0) t = v0 + span / 2;
    else { const r = e.currentTarget.getBoundingClientRect(); t = v0 + ((e.clientX - r.left) / r.width) * span; }
    t = Math.round(Math.max(0, Math.min(T, t)));
    setMarkers((m) => (m.next === 1 ? { ...m, m1: t, next: 2 } : { ...m, m2: t, next: 1 }));
  };

  const miniAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    set(((e.clientX - r.left) / r.width) * T - span / 2, span);
  };

  const depth = Math.min(4, Math.max(...flat.map((f) => f.depth)) + 1);
  const rows: Flat[][] = Array.from({ length: depth }, () => []);
  for (const f of flat) if (f.depth < rows.length) rows[f.depth].push(f);
  const logMax = Math.log(T / MIN_SPAN);

  return (
    <section className="panel screen-panel" aria-labelledby="fg-h" data-tour="flame">
      <div className="ph" style={{ gap: "8px 14px" }}>
        <span className="n">02.1</span><h2 id="fg-h">Flame stack</h2>
        <span className="dim" style={{ fontSize: 12.5, marginRight: "auto" }}>width = tokens, heat = influence</span>
        <div className="group" role="group" aria-label="Ordering">
          <button className="btn" type="button" aria-pressed={mode === "window"} onClick={() => setMode("window")}>Window order</button>
          <button className="btn" type="button" aria-pressed={mode === "heavy"} onClick={() => setMode("heavy")}>Heaviest first</button>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <button className="btn sq" type="button" aria-label="Zoom out" onClick={() => zoomAt(0.5, 1.6)}>−</button>
          <label className="sr" htmlFor="zoom">Zoom level</label>
          <input id="zoom" type="range" min={0} max={100} value={Math.round((Math.log(T / span) / logMax) * 100)}
            style={{ width: 120, accentColor: "var(--ink)" }}
            onChange={(e) => { const s = T / Math.exp((+e.target.value / 100) * logMax); set(v0 + span / 2 - s / 2, s); }} />
          <button className="btn sq" type="button" aria-label="Zoom in" onClick={() => zoomAt(0.5, 1 / 1.6)}>+</button>
          <button className="btn" type="button" onClick={() => set(0, T)}>Fit</button>
        </div>
        <span className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>
          tok {fmt(v0)} – {fmt(v0 + span)} | {(T / span).toFixed(1)}×
        </span>
      </div>

      <div className="mini" aria-hidden="true"
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setLive(true); miniAt(e); }}
        onPointerMove={(e) => { if (e.buttons & 1) miniAt(e); }}
        onPointerUp={() => setLive(false)}>
        {flat.filter((f) => f.depth === 2).map((f) => (
          <i key={f.node.id + f.offset} className={heatClass(f.node.influence)}
            style={{ left: `${(f.offset / T) * 100}%`, width: `${(f.node.tokens / T) * 100}%` }} />
        ))}
        <i className="brush" style={{ left: `${(v0 / T) * 100}%`, width: `${(span / T) * 100}%` }} />
      </div>

      <button className="ruler" type="button" onClick={placeMarker} data-tour="ruler"
        aria-label={`Token ruler from ${fmt(v0)} to ${fmt(v0 + span)}. Activate to drop marker M${markers.next}.`}>
        {ticks.map(({ t, major }) => (
          <span key={t} className={`tk ${major ? "mj" : ""}`} style={{ left: `${pct(t)}%`, height: major ? 16 : 6 }}>
            {major && <span>{fmt(t)}</span>}
          </span>
        ))}
        {([["M1", markers.m1], ["M2", markers.m2]] as const).map(([n, t]) =>
          pct(t) >= 0 && pct(t) <= 100 ? <span key={n} className="flag" style={{ left: `${pct(t)}%` }}><span>{n}</span></span> : null)}
        {hx != null && <span className="ctag" style={{ left: hx }}>tok {fmt(v0 + (hx / W) * span)}</span>}
      </button>

      <div ref={vpRef} className={`vp grat ${live ? "live" : ""}`} tabIndex={0} role="group"
        aria-label="Flame stack. W and S zoom, A and D pan, F zooms to the selection, 0 fits the whole window."
        onKeyDown={onKey}
        onPointerMove={(e) => setHx(e.clientX - e.currentTarget.getBoundingClientRect().left)}
        onPointerLeave={() => setHx(null)}>
        <div className="sweep" aria-hidden="true" />
        {b > a && <div className="mband" aria-hidden="true"
          style={{ left: `${Math.max(0, pct(a))}%`, width: `${Math.max(0, Math.min(100, pct(b)) - Math.max(0, pct(a)))}%` }} />}
        {rows.map((row, depth) => (
          <div className="frow" key={depth}>
            {row.map((f) => {
              const l = pct(f.offset), r = pct(f.offset + f.node.tokens);
              const vis = r > 0 && l < 100;
              const L = Math.max(l, -1), w = Math.max(0, Math.min(r, 101) - L);
              const px = (w / 100) * W;
              const txt = px >= 160 ? `${f.node.name} | ${fmt(f.node.tokens)}` : px >= 46 ? f.node.name : "";
              const cls = f.depth === 0 ? "hn" : heatClass(f.node.influence);
              return (
                <button key={f.node.id + f.offset} type="button"
                  className={`seg ${cls}${vis ? "" : " gone"}${f.node.id === selected ? " sel" : ""}`}
                  style={{ left: `${L}%`, width: `${w}%`, animationDelay: `${Math.round((f.offset / T) * 950) + f.depth * 50}ms` }}
                  aria-pressed={f.node.id === selected}
                  aria-label={`${f.node.fingerprint ?? f.node.name}, ${fmt(f.node.tokens)} tokens${f.node.influence == null ? ", not scored" : `, influence ${f.node.influence.toFixed(2)}`}`}
                  onClick={() => onSelect(f.node.id)}
                  onDoubleClick={() => { onSelect(f.node.id); fitNode(f.node.id); }}>
                  <span>{txt}</span>
                </button>
              );
            })}
          </div>
        ))}
        {[markers.m1, markers.m2].map((t, i) =>
          pct(t) >= 0 && pct(t) <= 100 ? <div key={i} className="mline" style={{ left: `${pct(t)}%` }} aria-hidden="true" /> : null)}
        {hx != null && <div className="hair" style={{ left: hx }} aria-hidden="true" />}
      </div>

      <div className="readouts" aria-live="polite" style={{ borderTop: "1px solid var(--rule)", marginTop: 12 }}>
        <div><span className="lbl">M1</span><span className="v">tok {fmt(markers.m1)}</span></div>
        <div><span className="lbl">M2</span><span className="v">tok {fmt(markers.m2)}</span></div>
        <div><span className="lbl">Δ tokens</span><span className="v">{fmt(b - a)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>{(((b - a) / T) * 100).toFixed(1)}% of window</span></div>
        <div><span className="lbl">Σ influence inside</span><span className="v">{sum.toFixed(2)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>{cnt} whole segments</span></div>
        <div><span className="lbl">Δ cost / 100k req</span><span className="v">${(((b - a) * pricePerM) / 1e6 * 1e5).toFixed(2)}</span>
          <span className="mono dim" style={{ fontSize: 11.5 }}>at ${pricePerM.toFixed(2)} / 1M input</span></div>
      </div>
      <div className="legend" aria-label="Heat key">
        <span className="lbl" style={{ fontFamily: "inherit" }}>Heat = influence</span>
        {["<.05", ".05", ".15", ".25", ".40", ".55", "≥.70"].map((t, i) => (
          <span key={t}><i className={`h${i}`} />H{i} {t}</span>
        ))}
        <span><i className="hn" />not scored</span>
      </div>
      <div className="keys">
        <span><kbd>Wheel</kbd> zoom at cursor</span><span><kbd>Shift</kbd>+<kbd>Wheel</kbd> pan</span>
        <span><kbd>W</kbd><kbd>S</kbd> zoom</span><span><kbd>A</kbd><kbd>D</kbd> pan</span>
        <span><kbd>F</kbd> fit selection</span><span><kbd>0</kbd> fit window</span><span>Double-click a bar to dive</span>
      </div>
      <span className="sr" aria-live="polite">Selected {sel?.node.name}</span>
    </section>
  );
}
