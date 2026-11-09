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

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} for ${path}`);
  return (await r.json()) as T;
}

export const api = {
  traces: () => get<TraceRow[]>("/api/traces"),
  runs: () => get<Run[]>("/api/runs"),
  flame: (id: string) => get<{ run: number | null; fit: Fit | null; tree: FlameNode }>(`/api/traces/${encodeURIComponent(id)}/flame`),
  trace: (id: string) => get<TraceDetail>(`/api/traces/${encodeURIComponent(id)}`),
  aggregate: (run: number) => get<Aggregate>(`/api/runs/${run}/aggregate`),
};
