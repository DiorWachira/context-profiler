// A one-row waterfall for a fingerprint: one cell per profiled trace (in trace order),
// coloured by its influence there; blank where the trace did not carry the segment.
import { useEffect, useRef, useState } from "react";
import { heatBin, readPalette } from "../heat";
import { useThemeVersion } from "../hooks";

export function Strip({ values, ids, label, width = 180, height = 16 }: {
  values: (number | null | undefined)[]; ids: string[]; label: string; width?: number; height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const [hover, setHover] = useState<number | null>(null);
  const n = Math.max(1, values.length);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    const g = c.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pal = readPalette();
    g.clearRect(0, 0, width, height);
    g.fillStyle = pal.well;
    g.fillRect(0, 0, width, height);
    const cw = width / n;
    values.forEach((v, i) => {
      if (v === undefined) return;
      const b = heatBin(v);
      g.fillStyle = b == null ? pal.rule : pal.heat[b];
      g.fillRect(i * cw, 0, Math.max(0.7, cw), height);
    });
  }, [values, width, height, n, theme]);

  const carried = values.filter((v) => v !== undefined).length;
  const peak = Math.max(0, ...values.map((v) => v ?? 0));
  const v = hover != null ? values[hover] : undefined;
  return (
    <span className="strip" role="img" aria-label={`${label}: carried by ${carried} of ${values.length} traces, peak influence ${peak.toFixed(2)}`}
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        setHover(Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))));
      }}
      onPointerLeave={() => setHover(null)}>
      <canvas ref={ref} style={{ width, height }} />
      {hover != null && (
        <span className="strip-tip mono" style={{ left: `${((hover + 0.5) / n) * 100}%` }} aria-hidden="true">
          {ids[hover]} · {v === undefined ? "not carried" : v === null ? "pinned" : v.toFixed(2)}
        </span>
      )}
    </span>
  );
}
