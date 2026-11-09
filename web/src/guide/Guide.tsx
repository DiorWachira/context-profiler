// The guide: an opt-in, skippable tour drawn as service-manual callouts (numbered balloon + dog-leg
// leader line to the feature, per MIL-STD-38784 practice), a first-visit invitation that never
// forces itself (NN/g: tutorials pushed on users tend to lower success), and a "service manual"
// drawer with the tour, keyboard shortcuts and a glossary. Opened with the Guide key or "?".
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useReducedMotion } from "../hooks";
import { STEPS, type Step, resolveRoute, screenOf } from "./steps";

const SEEN_KEY = "ctxprof.guide.v1";
export const GUIDE_EVENT = "ctxprof:guide";
export type GuideRequest = { action: "drawer" } | { action: "tour"; scope: "all" | "screen" };

export function openGuide(req: GuideRequest) {
  window.dispatchEvent(new CustomEvent<GuideRequest>(GUIDE_EVENT, { detail: req }));
}

function seen(): boolean {
  try { return localStorage.getItem(SEEN_KEY) === "1"; } catch { return true; }
}
function markSeen() {
  try { localStorage.setItem(SEEN_KEY, "1"); } catch { /* storage unavailable: invite shows once per load */ }
}

interface Tour { steps: (Step & { href: string | null })[]; i: number }

export function Guide() {
  const [tour, setTour] = useState<Tour | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [invite, setInvite] = useState(false);
  const returnFocus = useRef<HTMLElement | null>(null);

  const start = useCallback((scope: "all" | "screen") => {
    markSeen();
    setInvite(false);
    setDrawer(false);
    returnFocus.current = document.activeElement as HTMLElement | null;
    const here = screenOf(location.hash);
    const steps = scope === "all"
      ? STEPS.map((s) => ({ ...s, href: null }))
      : STEPS.filter((s) => s.screen === here && s.target !== "guide-key").map((s) => ({ ...s, href: location.hash || "#/" }));
    if (steps.length) setTour({ steps, i: 0 });
  }, []);

  const end = useCallback(() => {
    setTour(null);
    markSeen();
    requestAnimationFrame(() => (returnFocus.current ?? document.querySelector<HTMLElement>("[data-tour=guide-key]"))?.focus());
  }, []);

  useEffect(() => {
    const on = (e: Event) => {
      const d = (e as CustomEvent<GuideRequest>).detail;
      if (d.action === "drawer") { returnFocus.current = document.activeElement as HTMLElement | null; setDrawer(true); }
      else start(d.scope);
    };
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === "?" && !/INPUT|TEXTAREA|SELECT/.test(t.tagName) && !t.isContentEditable) {
        e.preventDefault();
        returnFocus.current = t;
        setDrawer(true);
      }
    };
    window.addEventListener(GUIDE_EVENT, on);
    window.addEventListener("keydown", key);
    const timer = window.setTimeout(() => { if (!seen()) setInvite(true); }, 1400);
    return () => { window.removeEventListener(GUIDE_EVENT, on); window.removeEventListener("keydown", key); clearTimeout(timer); };
  }, [start]);

  return (
    <>
      {invite && !tour && (
        <aside className="invite" aria-labelledby="inv-h">
          <span className="balloon" aria-hidden="true">?</span>
          <div>
            <h2 id="inv-h">First time on the bench?</h2>
            <p>A 12-step guide shows how to read the waterfall, the flame graph and a pruning verdict. About a minute.</p>
            <div className="inv-act">
              <button type="button" className="btn pri" onClick={() => start("all")}>Start the guide</button>
              <button type="button" className="btn" onClick={() => { setInvite(false); markSeen(); }}>Not now</button>
            </div>
            <span className="mono dim sm">Reopen any time with <kbd>?</kbd></span>
          </div>
        </aside>
      )}
      {drawer && <Manual onClose={() => { setDrawer(false); requestAnimationFrame(() => returnFocus.current?.focus()); }} onTour={start} />}
      {tour && <Callout tour={tour} setI={(i) => setTour({ ...tour, i })} onEnd={end} />}
    </>
  );
}

// ---- the callout layer ---------------------------------------------------------------------

interface Box { top: number; left: number; width: number; height: number }
const PAD = 8;
const GAP = 64;

function findTarget(name: string): HTMLElement | null {
  const all = [...document.querySelectorAll<HTMLElement>(`[data-tour="${name}"]`)];
  return all.find((el) => el.getClientRects().length > 0) ?? null;
}

function Callout({ tour, setI, onEnd }: { tour: Tour; setI: (i: number) => void; onEnd: () => void }) {
  const step = tour.steps[tour.i];
  const n = tour.steps.length;
  const reduced = useReducedMotion();
  const [hole, setHole] = useState<Box | null>(null);
  const [missing, setMissing] = useState(false);
  const [cardH, setCardH] = useState(220);
  const [vp, setVp] = useState({ w: window.innerWidth, h: window.innerHeight });
  const card = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const target = useRef<HTMLElement | null>(null);

  const measure = useCallback(() => {
    setVp({ w: window.innerWidth, h: window.innerHeight });
    const el = target.current;
    if (!el || !el.isConnected) return;
    const r = el.getBoundingClientRect();
    const top = Math.max(4, r.top - PAD), left = Math.max(4, r.left - PAD);
    setHole({ top, left, width: Math.min(window.innerWidth - 8, r.right + PAD) - left,
      height: Math.min(window.innerHeight - 8, r.bottom + PAD) - top });
  }, []);

  // go to the step's screen, wait for its target, bring it into view
  useEffect(() => {
    let alive = true;
    let raf = 0;
    setHole(null);
    setMissing(false);
    target.current = null;
    (async () => {
      const href = step.href ?? await resolveRoute(step.route);
      if (!alive) return;
      const norm = (h: string) => h.replace(/^#\/?/, "").replace(/\/$/, "");
      if (norm(location.hash) !== norm(href)) location.hash = href;
      const t0 = performance.now();
      const wait = () => {
        if (!alive) return;
        const el = findTarget(step.target);
        if (el) {
          target.current = el;
          el.scrollIntoView({ block: el.offsetHeight > window.innerHeight * 0.6 ? "start" : "center",
            behavior: reduced ? "auto" : "smooth" });
          window.setTimeout(() => alive && measure(), reduced ? 30 : 420);
        } else if (performance.now() - t0 < 6000) {
          raf = requestAnimationFrame(wait);
        } else {
          setMissing(true);
        }
      };
      wait();
    })();
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [step, reduced, measure]);

  useEffect(() => {
    let raf = 0;
    const on = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(measure); };
    window.addEventListener("resize", on);
    window.addEventListener("scroll", on, true);
    return () => { window.removeEventListener("resize", on); window.removeEventListener("scroll", on, true); cancelAnimationFrame(raf); };
  }, [measure]);

  useLayoutEffect(() => {
    if (card.current) setCardH(card.current.offsetHeight);
    heading.current?.focus({ preventScroll: true });
  }, [tour.i, hole, missing]);

  const next = () => (tour.i < n - 1 ? setI(tour.i + 1) : onEnd());
  const back = () => tour.i > 0 && setI(tour.i - 1);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onEnd(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); next(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); back(); }
    else if (e.key === "Tab" && card.current) { // keep focus inside the card while the veil is up
      const f = [...card.current.querySelectorAll<HTMLElement>("button")];
      const i = f.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
    }
  };

  // ---- geometry: pick the side with room, else dock (phones) or sit inside a huge target ----
  const { w: vw, h: vh } = vp;
  const mobile = vw < 640;
  const cw = Math.min(360, vw - 32);
  let pos: { x: number; y: number; side: "r" | "l" | "b" | "t" | "in" | "dock" } = { x: (vw - cw) / 2, y: (vh - cardH) / 2, side: "in" };
  let dot: [number, number] | null = null;
  let ball: [number, number] | null = null;
  const clampY = (y: number) => Math.max(16, Math.min(vh - cardH - 16, y));
  const clampX = (x: number) => Math.max(16, Math.min(vw - cw - 16, x));
  if (hole) {
    const h = hole;
    const midY = (lo: number) => Math.max(h.top + 14, Math.min(h.top + h.height - 14, lo));
    const midX = (lo: number) => Math.max(h.left + 14, Math.min(h.left + h.width - 14, lo));
    if (mobile) {
      const below = h.top + h.height / 2 < vh / 2;
      pos = { x: 16, y: below ? vh - cardH - 16 : 16, side: "dock" };
      ball = [Math.min(vw - 36, Math.max(36, h.left + 24)), below ? pos.y - 26 : pos.y + cardH + 26];
      dot = [ball[0], below ? h.top + h.height : h.top];
      if ((below && dot[1] > ball[1]) || (!below && dot[1] < ball[1])) dot = null; // target hidden behind the card
    } else if (h.left + h.width + GAP + cw <= vw - 16) {
      pos = { x: h.left + h.width + GAP, y: clampY(h.top), side: "r" };
    } else if (h.left - GAP - cw >= 16) {
      pos = { x: h.left - GAP - cw, y: clampY(h.top), side: "l" };
    } else if (h.top + h.height + GAP + cardH <= vh - 16) {
      pos = { x: clampX(h.left + 24), y: h.top + h.height + GAP, side: "b" };
    } else if (h.top - GAP - cardH >= 16) {
      pos = { x: clampX(h.left + 24), y: h.top - GAP - cardH, side: "t" };
    } else {
      pos = { x: clampX(h.left + h.width - cw - 24), y: clampY(h.top + h.height - cardH - 24), side: "in" };
    }
    if (pos.side === "r") { ball = [pos.x, pos.y + 30]; dot = [h.left + h.width, midY(pos.y + 30)]; }
    if (pos.side === "l") { ball = [pos.x + cw, pos.y + 30]; dot = [h.left, midY(pos.y + 30)]; }
    if (pos.side === "b") { ball = [pos.x + 30, pos.y]; dot = [midX(pos.x + 30), h.top + h.height]; }
    if (pos.side === "t") { ball = [pos.x + 30, pos.y + cardH]; dot = [midX(pos.x + 30), h.top]; }
  }
  let leader = "";
  if (dot && ball) { // dog-leg: a short run straight out of the feature, then to the balloon
    const [dx, dy] = dot, [bx, by] = ball;
    const horiz = pos.side === "r" || pos.side === "l";
    leader = horiz ? `M${dx},${dy} H${dx + Math.sign(bx - dx) * 18} L${bx},${by}`
      : `M${dx},${dy} V${dy + Math.sign(by - dy) * 18} L${bx},${by}`;
  }
  const num = String(tour.i + 1).padStart(2, "0");

  return (
    <div className="tour" onKeyDown={onKey}>
      {hole ? (
        <>
          <div className="veil" style={{ top: 0, left: 0, right: 0, height: hole.top }} />
          <div className="veil" style={{ top: hole.top + hole.height, left: 0, right: 0, bottom: 0 }} />
          <div className="veil" style={{ top: hole.top, left: 0, width: hole.left, height: hole.height }} />
          <div className="veil" style={{ top: hole.top, left: hole.left + hole.width, right: 0, height: hole.height }} />
          <div className={`hole${reduced ? "" : " anim"}`} style={hole} aria-hidden="true"><i /><i /><i /><i /></div>
        </>
      ) : <div className="veil" style={{ inset: 0 }} />}
      <svg className="leader" width={vw} height={vh} aria-hidden="true">
        {leader && <path d={leader} className={reduced ? "" : "draw"} />}
        {dot && <circle cx={dot[0]} cy={dot[1]} r={3.5} />}
      </svg>
      {ball && <span className="balloon big" style={{ left: ball[0], top: ball[1] }} aria-hidden="true">{tour.i + 1}</span>}
      <div ref={card} className={`callout${pos.side === "dock" ? " dock" : ""}`} role="dialog" aria-modal="true"
        aria-labelledby="co-h" aria-describedby="co-b" style={{ left: pos.x, top: pos.y, width: cw }}>
        <div className="co-head">
          <span className="mono">{num} / {String(n).padStart(2, "0")}</span>
          <span className="lbl">Service manual · {step.screen}</span>
          <button type="button" className="co-x" aria-label="End the guide" onClick={onEnd}>×</button>
        </div>
        <h2 id="co-h" ref={heading} tabIndex={-1}>{step.title}</h2>
        <p id="co-b">{missing ? "This part is not on screen with the current data, so this step has nothing to point at. Skip ahead." : step.body}</p>
        <div className="co-ticks" aria-hidden="true">{tour.steps.map((_, i) => <i key={i} className={i <= tour.i ? "on" : ""} />)}</div>
        <div className="co-act">
          <button type="button" className="btn" onClick={back} disabled={tour.i === 0}>Back</button>
          <button type="button" className="btn pri" onClick={next}>{tour.i === n - 1 ? "Finish" : "Next →"}</button>
        </div>
        <span className="mono dim sm co-keys"><kbd>←</kbd> <kbd>→</kbd> step · <kbd>Esc</kbd> close</span>
        <span className="sr" aria-live="polite">Step {tour.i + 1} of {n}: {step.title}</span>
      </div>
    </div>
  );
}

// ---- the service manual drawer --------------------------------------------------------------

const KEYS: [string, string][] = [
  ["?", "Open this service manual"],
  ["W / S", "Flame graph: zoom in / out at the cursor (stack focused)"],
  ["A / D", "Flame graph: pan left / right"],
  ["F · 0", "Flame graph: fit the selection · fit the whole window"],
  ["Shift + wheel", "Flame graph: pan"],
  ["← → ↑ ↓", "Waterfall: move the readout cursor (Shift = 10 rows)"],
  ["Enter", "Waterfall: open the request under the cursor"],
  ["Space", "Pruning: toggle the focused Cut checkbox"],
];

const TERMS: [string, string][] = [
  ["Segment", "A removable piece of a request: a system-prompt section, a tool schema, a retrieved chunk, a past turn."],
  ["Fingerprint", "A segment's stable name across requests, e.g. system/## Tone rules or tool/search_jobs."],
  ["Ablation", "Replaying a request with some segments removed, at temperature 0, and scoring how the answer changed."],
  ["Influence", "A segment's surrogate weight relative to the strongest segment in that request, from 0 to 1. Colour encodes only this."],
  ["DEAD", "Near-zero influence (mean below 0.05, p95 below 0.10) across at least 20 requests. A pruning candidate."],
  ["VOLATILE", "Influence varies a lot between requests (sd ≥ 0.15): needed sometimes. Never pruned automatically."],
  ["CRITICAL", "Mean influence ≥ 0.5: carries the answer."],
  ["Injection signal", "Untrusted text in the top 10% of its request with influence ≥ 0.3."],
  ["Held-out", "Requests the profiler never saw, kept back to verify a pruning diff."],
  ["Agreement", "Share of held-out requests whose pruned answer matches the original. 95% is needed to recommend."],
];

function Manual({ onClose, onTour }: { onClose: () => void; onTour: (s: "all" | "screen") => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const here = screenOf(location.hash);
  const count = STEPS.filter((s) => s.screen === here && s.target !== "guide-key").length;
  useEffect(() => { ref.current?.querySelector<HTMLElement>("h2")?.focus(); }, []);
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") { e.preventDefault(); onClose(); }
    if (e.key === "Tab" && ref.current) {
      const f = [...ref.current.querySelectorAll<HTMLElement>("button, a[href]")];
      const i = f.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
    }
  };
  return (
    <div className="manual-wrap" onKeyDown={onKey}>
      <div className="manual-veil" onClick={onClose} aria-hidden="true" />
      <div ref={ref} className="manual" role="dialog" aria-modal="true" aria-labelledby="mn-h">
        <div className="manual-plate">
          <span className="lbl">Model CP-8200A · Context Profiler</span>
          <button type="button" className="co-x" aria-label="Close the service manual" onClick={onClose}>×</button>
        </div>
        <h2 id="mn-h" tabIndex={-1}>Service manual</h2>
        <p className="manual-lede">How to read the bench: what each screen measures, the keys that drive it, and the words it uses.</p>

        <section aria-labelledby="mn-tour">
          <h3 id="mn-tour"><span className="balloon sm">1</span>Guided tour</h3>
          <div className="manual-tours">
            <button type="button" className="btn pri" onClick={() => onTour("all")}>Full tour · {STEPS.length} steps</button>
            <button type="button" className="btn" onClick={() => onTour("screen")} disabled={!count}>This screen · {count} steps</button>
          </div>
        </section>

        <section aria-labelledby="mn-how">
          <h3 id="mn-how"><span className="balloon sm">2</span>How a verdict is made</h3>
          <ol className="pipeline">
            <li><b>Segment</b> each recorded request into sections, tools, chunks and turns.</li>
            <li><b>Ablate</b>: replay it N times with random pieces removed, temperature 0, cached and budget-capped.</li>
            <li><b>Fit</b> a sparse linear surrogate (Lasso): its weights are each piece's influence.</li>
            <li><b>Aggregate</b> by fingerprint across hundreds of requests: DEAD, VOLATILE, CRITICAL.</li>
            <li><b>Prune</b>: write a diff that removes what never mattered.</li>
            <li><b>Verify</b> on held-out requests; recommend only above the agreement threshold.</li>
          </ol>
        </section>

        <section aria-labelledby="mn-keys">
          <h3 id="mn-keys"><span className="balloon sm">3</span>Keys</h3>
          <dl className="keys-list">
            {KEYS.map(([k, d]) => <div key={k}><dt><kbd>{k}</kbd></dt><dd>{d}</dd></div>)}
          </dl>
        </section>

        <section aria-labelledby="mn-terms">
          <h3 id="mn-terms"><span className="balloon sm">4</span>Terms</h3>
          <dl className="terms">
            {TERMS.map(([t, d]) => <div key={t}><dt>{t}</dt><dd>{d}</dd></div>)}
          </dl>
        </section>
        <p className="mono dim sm" style={{ marginTop: 18 }}>
          Method and limitations: <a href="https://github.com/DiorWachira/context-profiler#method-in-one-paragraph">README</a>
        </p>
      </div>
    </div>
  );
}
