// Fig. 1, the waterfall: every profiled request is one row (like an SDR waterfall's scan lines),
// x is the token position in its context window, colour is the segment's influence (heat bins).
// Hover or arrow keys move a crosshair readout; click or Enter opens the trace.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Spectrum } from "../api";
import { fmt, heatBin, readPalette } from "../heat";
import { useReducedMotion, useThemeVersion } from "../hooks";

type Order = "trace" | "length";
interface Cursor { row: number; seg: number }

function niceStep(max: number, target: number) {
  const raw = max / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw;
}

export function Waterfall({ spec, height = 380, onOpen }: { spec: Spectrum; height?: number; onOpen: (id: string) => void }) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(900);
  const [order, setOrder] = useState<Order>("trace");
  const [cur, setCur] = useState<Cursor | null>(null);
  const [progress, setProgress] = useState(0);
  const reduced = useReducedMotion();
  const theme = useThemeVersion();

  const rows = useMemo(() => {
    const r = [...spec.rows];
    if (order === "length") r.sort((a, b) => b[1] - a[1]);
    return r;
  }, [spec, order]);
  const max = Math.max(1, spec.max_tokens);
  const rowH = height / Math.max(1, rows.length);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // acquisition sweep: rows are written top to bottom once, like a waterfall filling
  useEffect(() => {
    if (reduced) { setProgress(1); return; }
    setProgress(0);
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / 1400);
      setProgress(1 - (1 - p) ** 3);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduced, order]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(height * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pal = readPalette();
    g.fillStyle = pal.well;
    g.fillRect(0, 0, w, height);
    const shown = Math.ceil(rows.length * progress);
    for (let i = 0; i < shown; i++) {
      const y = i * rowH;
      for (const [off, tok, infl] of rows[i][3]) {
        const x = (off / max) * w;
        const sw = (tok / max) * w;
        const b = heatBin(infl);
        g.fillStyle = b == null ? pal.rule : pal.heat[b];
        g.fillRect(x, y, Math.max(0.6, sw - (sw > 5 ? 1 : 0)), Math.max(0.6, rowH - (rowH > 4 ? 1 : 0)));
      }
      if (rows[i][2]) { // poor fit: a hollow tick in the gutter
        g.strokeStyle = pal.ink;
        g.strokeRect(w - 5.5, y + 0.5, 4, Math.max(1, rowH - 1));
      }
    }
    if (progress < 1) { // the write head
      g.fillStyle = pal.ink;
      g.fillRect(0, shown * rowH, w, 1.5);
    }
  }, [rows, w, height, rowH, max, progress, theme]);

  const at = useCallback((clientX: number, clientY: number): Cursor | null => {
    const r = canvas.current?.getBoundingClientRect();
    if (!r) return null;
    const row = Math.floor((clientY - r.top) / rowH);
    if (row < 0 || row >= rows.length) return null;
    const tok = ((clientX - r.left) / r.width) * max;
    const segs = rows[row][3];
    const seg = segs.findIndex(([o, t]) => tok >= o && tok < o + t);
    return seg < 0 ? null : { row, seg };
  }, [rows, rowH, max]);

  const onKey = (e: React.KeyboardEvent) => {
    const c = cur ?? { row: 0, seg: 0 };
    const segs = (r: number) => rows[r][3].length;
    let next: Cursor | null = null;
    if (e.key === "ArrowDown") next = { row: Math.min(rows.length - 1, c.row + (e.shiftKey ? 10 : 1)), seg: c.seg };
    else if (e.key === "ArrowUp") next = { row: Math.max(0, c.row - (e.shiftKey ? 10 : 1)), seg: c.seg };
    else if (e.key === "ArrowRight") next = { row: c.row, seg: Math.min(segs(c.row) - 1, c.seg + 1) };
    else if (e.key === "ArrowLeft") next = { row: c.row, seg: Math.max(0, c.seg - 1) };
    else if (e.key === "Enter" && cur) { onOpen(rows[cur.row][0]); e.preventDefault(); return; }
    else if (e.key === "Escape") { setCur(null); return; }
    if (next) {
      next.seg = Math.min(next.seg, segs(next.row) - 1);
      setCur(next);
      e.preventDefault();
    }
  };

  const step = niceStep(max, w < 500 ? 4 : 8);
  const ticks: number[] = [];
  for (let t = 0; t <= max; t += step) ticks.push(t);
  const hit = cur ? rows[cur.row] : null;
  const seg = hit ? hit[3][cur!.seg] : null;
  const sx = seg ? ((seg[0] + seg[1] / 2) / max) * w : 0;
  const sy = cur ? cur.row * rowH + rowH / 2 : 0;

  return (
    <div className="wf">
      <div className="wf-top">
        <div className="group" role="group" aria-label="Row order">
          <button type="button" className="btn" aria-pressed={order === "trace"} onClick={() => setOrder("trace")}>Trace order</button>
          <button type="button" className="btn" aria-pressed={order === "length"} onClick={() => setOrder("length")}>Longest first</button>
        </div>
        <span className="mono dim wf-hint">{rows.length} requests × {fmt(max)} tokens · hover or focus + arrows · click opens</span>
      </div>
      <div className="wf-ruler" aria-hidden="true">
        {ticks.map((t) => (
          <i key={t} style={{ left: `${(t / max) * 100}%` }}><span>{t >= 1000 ? `${t / 1000}k` : t}</span></i>
        ))}
      </div>
      <div ref={wrap} className="wf-screen grat" tabIndex={0} role="img" onKeyDown={onKey}
        aria-label={`Waterfall of ${rows.length} profiled requests. Each row is one request, x is token position, colour is influence. Use arrow keys to inspect segments and Enter to open a trace.`}
        onPointerMove={(e) => setCur(at(e.clientX, e.clientY))} onPointerLeave={() => setCur(null)}
        onClick={(e) => { const c = at(e.clientX, e.clientY); if (c) onOpen(rows[c.row][0]); }}
        onBlur={() => setCur(null)}>
        <canvas ref={canvas} style={{ width: w, height }} />
        {hit && seg && (
          <>
            <div className="wf-h" style={{ top: cur!.row * rowH, height: Math.max(2, rowH) }} aria-hidden="true" />
            <div className="wf-v" style={{ left: sx }} aria-hidden="true" />
            <div className={`wf-tip ${sx > w * 0.6 ? "l" : ""} ${sy > height * 0.6 ? "u" : ""}`}
              style={{ left: sx, top: sy }} role="status">
              <b className="mono">{hit[0]}</b>
              <span className="mono">{spec.fingerprints[seg[3]]}</span>
              <span className="mono dim">tok {fmt(seg[0])}–{fmt(seg[0] + seg[1])} · {fmt(seg[1])} tok</span>
              <span className="mono">{seg[2] == null ? "pinned · not scored" : `influence ${seg[2].toFixed(2)}`}
                {hit[2] ? " · poor fit" : ""}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
