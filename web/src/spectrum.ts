import type { Spectrum } from "./api";

/** Per-fingerprint influence in every profiled trace (trace order); undefined = not carried. */
export function stripsFrom(spec: Spectrum): { ids: string[]; strips: Map<string, (number | null | undefined)[]> } {
  const ids = spec.rows.map((r) => r[0]);
  const strips = new Map<string, (number | null | undefined)[]>();
  spec.fingerprints.forEach((fp) => strips.set(fp, new Array(ids.length).fill(undefined)));
  spec.rows.forEach((row, i) => {
    for (const [, , infl, fpi] of row[3]) strips.get(spec.fingerprints[fpi])![i] = infl;
  });
  return { ids, strips };
}
