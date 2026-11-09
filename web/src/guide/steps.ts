// The guided tour, written like a service manual: one callout per feature, in reading order.
// Routes may contain {trace} (a profiled demo trace) and {prune} (the most instructive pruning
// report), resolved at runtime from the API so the tour works on any data set.
import { api } from "../api";

export interface Step {
  route: string;
  target: string; // value of a data-tour attribute
  screen: string; // shown in the card header and used for "tour this screen"
  title: string;
  body: string;
}

export const STEPS: Step[] = [
  { route: "#/", target: "waterfall", screen: "Overview", title: "Every row is one request",
    body: "Fig. 1 is a waterfall, like a radio's spectrum display: each row is a recorded request laid out by token position, and colour is influence, how much removing that piece changed the answer. Cool blue barely mattered; the warmest colours carried the answer. Hover to read a segment, click to open it." },
  { route: "#/", target: "readouts", screen: "Overview", title: "The front-panel readouts",
    body: "Traces profiled, model calls, spend against the hard budget cap, how many fingerprints are DEAD, and the saving that survived verification. Re-runs are served from cache and cost nothing." },
  { route: "#/", target: "dead", screen: "Overview", title: "Paid for, never used",
    body: "Segments with near-zero influence across at least 20 requests. They are candidates, not verdicts: a pruning diff is only recommended after it is verified on requests the profiler never saw." },
  { route: "#/", target: "inject", screen: "Overview", title: "Injection signal",
    body: "Untrusted text, such as an employer listing, that steers the answer more than it should. In the demo it is a fake \"registration fee\" planted inside a listing." },
  { route: "#/trace/{trace}", target: "flame", screen: "Trace", title: "One request under the probe",
    body: "Width is tokens, heat is influence. Scroll to zoom at the cursor, Shift+scroll to pan, or focus the stack and use W/S to zoom, A/D to pan, F to fit the selection and 0 to see the whole window." },
  { route: "#/trace/{trace}", target: "ruler", screen: "Trace", title: "Drop markers M1 and M2",
    body: "Click the token ruler to drop a marker. The readout sums tokens, influence and cost between M1 and M2, so you can price any slice of the window." },
  { route: "#/trace/{trace}", target: "raster", screen: "Trace", title: "The experiment itself",
    body: "Each row is one ablation: filled cells were kept, outlined cells removed. The model's score for each row is what the sparse surrogate learns influence from. Click a row to see the answer it produced." },
  { route: "#/segments", target: "seg-table", screen: "Segments", title: "Ranked by what they cost",
    body: "Every fingerprint aggregated across requests and sorted by tokens paid. The strip is a one-row waterfall: its influence in each request. A DEAD segment draws a long, cold line." },
  { route: "#/pruning/{prune}", target: "verdict", screen: "Pruning", title: "The verdict",
    body: "The diff is recommended only if requests it never saw still get the same answer. This report removed every DEAD segment and failed: something rarely used was still needed." },
  { route: "#/pruning/{prune}", target: "hunk", screen: "Pruning", title: "Cut, or keep",
    body: "Each hunk is one removal; removed lines are hatched, never coloured. Untick Cut on the suspect and the ledger and verdict recompute from the held-out evidence." },
  { route: "#/pruning/{prune}", target: "grid", screen: "Pruning", title: "Held-out evidence",
    body: "One cell per request the profiler never saw. A crossed-out cell is a regression; both answers are listed below it, with the hunk that broke it." },
  { route: "#/", target: "guide-key", screen: "Overview", title: "The service manual",
    body: "Open this guide again, tour a single screen, or look up shortcuts and terms. Press ? anywhere." },
];

let cache: Promise<{ trace: string; prune: string }> | null = null;

/** Picks a profiled trace with an injection signal (else any profiled one) and the most instructive prune. */
export function resolveVars(): Promise<{ trace: string; prune: string }> {
  cache ??= Promise.all([api.traces().catch(() => []), api.prunes().catch(() => [])]).then(([traces, prunes]) => {
    const t = traces.find((x) => x.injection) ?? traces.find((x) => x.profiled) ?? traces[0];
    const p = prunes.find((x) => x.n_regressions > 0) ?? prunes[0];
    return { trace: t?.id ?? "", prune: p ? String(p.id) : "" };
  });
  return cache;
}

export async function resolveRoute(route: string): Promise<string> {
  const v = await resolveVars();
  return route.replace("{trace}", encodeURIComponent(v.trace)).replace("{prune}", v.prune);
}

export function screenOf(hash: string): string {
  const h = hash.replace(/^#\/?/, "");
  if (h.startsWith("trace/")) return "Trace";
  if (h.startsWith("segments")) return "Segments";
  if (h.startsWith("pruning")) return "Pruning";
  return "Overview";
}
