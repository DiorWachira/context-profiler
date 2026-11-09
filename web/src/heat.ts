// Influence -> heat bin. Influence is relative to the strongest segment of the trace (0..1).
// The bins are the only place colour gets meaning in this app.
export const BINS = [0.05, 0.15, 0.25, 0.4, 0.55, 0.7];

export function heatBin(v: number | null | undefined): number | null {
  if (v == null) return null;
  const x = Math.max(0, v);
  let b = 0;
  while (b < BINS.length && x >= BINS[b]) b++;
  return b;
}

export const heatClass = (v: number | null | undefined) => {
  const b = heatBin(v);
  return b == null ? "hn" : `h${b}`;
};

export const fmt = (n: number) => Math.round(n).toLocaleString("en-US");
export const k = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n)));
