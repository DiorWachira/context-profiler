// Seven-segment LED readout, as on bench counters: unlit "ghost" segments stay faintly visible.
// Screen readers get the plain value via aria-label; the SVG is decorative.
import { useEffect, useState } from "react";
import { useReducedMotion } from "../hooks";

//            a  b  c  d  e  f  g
const DIGITS: Record<string, number[]> = {
  "0": [1, 1, 1, 1, 1, 1, 0], "1": [0, 1, 1, 0, 0, 0, 0], "2": [1, 1, 0, 1, 1, 0, 1], "3": [1, 1, 1, 1, 0, 0, 1],
  "4": [0, 1, 1, 0, 0, 1, 1], "5": [1, 0, 1, 1, 0, 1, 1], "6": [1, 0, 1, 1, 1, 1, 1], "7": [1, 1, 1, 0, 0, 0, 0],
  "8": [1, 1, 1, 1, 1, 1, 1], "9": [1, 1, 1, 1, 0, 1, 1], "-": [0, 0, 0, 0, 0, 0, 1], " ": [0, 0, 0, 0, 0, 0, 0],
};
// segment polygons in a 12 x 20 cell, slanted 6 degrees like real LED displays
const SEG = [
  "2.6,1 9.4,1 8.4,2.6 3.6,2.6", "9.8,1.4 10.8,2.4 10.3,9.3 9.1,9.9 8.6,3", "10.2,10.7 9.9,17.6 8.8,18.6 8.3,11.3 9.0,10.2",
  "2.6,19 9.4,19 8.4,17.4 3.6,17.4", "1.9,10.7 2.8,10.2 3.4,11.3 3.0,17.0 1.7,18.4", "2.2,2.4 3.2,1.4 3.6,3 3.2,9.9 2.0,9.3",
  "2.9,10 3.9,9.2 8.1,9.2 9.1,10 8.1,10.8 3.9,10.8",
];

function Glyph({ ch }: { ch: string }) {
  if (ch === "." || ch === ",") {
    return <svg viewBox="0 0 4 20" className="s7-g s7-dot" aria-hidden="true"><rect x="0.6" y="17.2" width="1.8" height="1.8" className="on" />
      {ch === "," && <rect x="0.2" y="19" width="1.2" height="1" className="on" />}</svg>;
  }
  if (!(ch in DIGITS)) return <span className="s7-ch" aria-hidden="true">{ch}</span>;
  const on = DIGITS[ch];
  return (
    <svg viewBox="0 0 12 20" className="s7-g" aria-hidden="true">
      <g transform="skewX(-6) translate(1 0)">
        {SEG.map((pts, i) => <polygon key={i} points={pts} className={on[i] ? "on" : "off"} />)}
      </g>
    </svg>
  );
}

/** Renders `format(value)`; counts up from 0 on mount unless motion is reduced. */
export function Seg7({ value, format, label, size = 30 }: {
  value: number; format: (v: number) => string; label: string; size?: number;
}) {
  const reduced = useReducedMotion();
  const [v, setV] = useState(reduced ? value : 0);
  useEffect(() => {
    if (reduced) { setV(value); return; }
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / 900);
      setV(value * (1 - (1 - p) ** 3));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, reduced]);
  const text = format(v);
  return (
    <span className="s7" style={{ fontSize: size }} role="img" aria-label={`${label}: ${format(value)}`}>
      {[...text].map((ch, i) => <Glyph key={i} ch={ch} />)}
    </span>
  );
}
