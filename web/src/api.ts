// Typed client for the FastAPI backend (src/ctxprof/server.py).

export type Kind = "request" | "system" | "tool" | "retrieved" | "history" | "user";

export interface FlameNode {
  id: string;
  name: string;
  kind: Kind;
  tokens: number;
  offset: number;
  influence: number | null;
  weight?: number | null;
  ci?: [number, number] | null;
  fingerprint?: string;
  source?: "trusted" | "untrusted";
  index?: number;
  children: FlameNode[];
}

export interface Fit {
  mode: string;
  n_masks: number;
  alpha: number;
  intercept: number;
  original_score: number;
  r2_heldout: number | null;
  poor_fit: number;
}

export interface TraceRow { id: string; model: string; total_tokens: number; n_segments: number; }

export interface Run {
  id: number; created_at: number; status: string; spent_usd: number; calls: number; cache_hits: number;
  note: string | null; n_traces: number; config: Record<string, unknown>;
}

export interface SegmentAgg {
  fingerprint: string; kind: Kind; source: string; label_name: string; traces: number; tokens_mean: number;
  tokens_paid: number; mean_influence: number; sd_influence: number; p95_influence: number;
  influence_per_1k: number; label: "" | "DEAD" | "CRITICAL" | "VOLATILE";
}

export interface Aggregate {
  run: number;
  segments: SegmentAgg[];
  injection: { trace_id: string; fingerprint: string; influence: number; rank: number; of: number }[];
  summary: { traces: number; mean_tokens: number; dead_count: number; dead_tokens_per_request: number; poor_fit: number };
}

export interface TraceDetail {
  id: string; model: string; meta: Record<string, unknown>; question: string;
  response: { text: string; tool_calls: { name: string; arguments: Record<string, unknown> }[] };
  segments: {
    index: number; kind: Kind; source: string; fingerprint: string; label: string; tokens: number;
    weight?: number | null; influence?: number | null; ci_lo?: number | null; ci_hi?: number | null;
  }[];
  run: number | null; fit: Fit | null;
  ablations: { mask_idx: number; mask: string; score: number; response_text: string | null }[];
}

export interface DiffLine { no: number; sign: "-" | "+"; text: string; }
export interface Hunk {
  fingerprint: string; kind: Kind; file: string; traces: number; tokens_per_request: number;
  mean_influence: number; p95_influence: number; reason: string; cut: boolean; lines: DiffLine[];
}
export interface Resp { text: string; tool_calls: { name: string; arguments: Record<string, unknown> }[] }
export interface VerifyRecord {
  trace_id: string; intent?: string | null; agree: boolean; score: number; tokens_before: number;
  removed_tokens: Record<string, number>; breaks: string[]; explained: boolean;
  probes: Record<string, { score: number; pruned: Resp }>;
  question?: string; original?: Resp; pruned?: Resp;
}
export interface VerifySummary {
  traces: number; agree_n: number; agreement: number; recommend: boolean; tokens_before: number; tokens_after: number;
  tokens_saved: number; reduction: number; cost_before: number; cost_after: number;
  saved_per_100k_usd: number; saved_per_100k_ksh: number; regressions: string[];
}
export interface Verification {
  status: string; in_sample: boolean; summary: VerifySummary; records: VerifyRecord[]; provider: string;
  spent_usd: number; calls: number; cache_hits: number;
  config: { threshold: number; min_score: number; metric: string; price_in_per_m: number; ksh_per_usd: number;
    budget_usd: number; held_out: number | null; eval_cmd: string | null };
}
export interface PruneRow {
  id: number; run_id: number; created_at: number; status: "unverified" | "verified" | "budget"; bundle: string;
  tokens_saved: number; agreement: number | null; recommend: boolean | null; verified_at: number | null;
  keep: string[]; removed: string[]; n_regressions: number;
}
export interface PruneDetail extends PruneRow {
  hunks: Hunk[]; skipped: { fingerprint: string; kind: string; label: string; why: string }[]; diff: string;
  verification: Verification | null; run: Run | null;
}

/** Static build (GitHub Pages): the API is a snapshot of JSON files written by `ctxprof export`. */
export const STATIC = import.meta.env.VITE_STATIC === "1";

export function resolve(path: string): string {
  if (!STATIC) return path;
  const [p, q] = path.split("?");
  if (q) throw new Error(`query strings are not available in the static snapshot: ${path}`);
  return `.${p}.json`;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(resolve(path));
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${path}`);
  return (await r.json()) as T;
}

export const api = {
  traces: () => get<TraceRow[]>("/api/traces"),
  runs: () => get<Run[]>("/api/runs"),
  flame: (id: string) => get<{ run: number | null; fit: Fit | null; tree: FlameNode }>(`/api/traces/${encodeURIComponent(id)}/flame`),
  trace: (id: string) => get<TraceDetail>(`/api/traces/${encodeURIComponent(id)}`),
  aggregate: (run: number) => get<Aggregate>(`/api/runs/${run}/aggregate`),
  prunes: () => get<PruneRow[]>("/api/prunes"),
  prune: (id: number) => get<PruneDetail>(`/api/prunes/${id}`),
};
