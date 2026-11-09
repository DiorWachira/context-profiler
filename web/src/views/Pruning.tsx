import { useEffect, useMemo, useState } from "react";
import { api, type Hunk, type PruneDetail, type PruneRow, type Resp, type VerifyRecord } from "../api";
import { fmt } from "../heat";
import { useLoad } from "../hooks";

const RATE_KEY = "ctxprof.kshPerUsd";
const PREVIEW_LINES = 6;

function readRate(fallback: number): number {
  try {
    const v = Number(localStorage.getItem(RATE_KEY));
    return Number.isFinite(v) && v > 0 ? v : fallback;
  } catch { return fallback; }
}

function writeRate(v: number) {
  try { localStorage.setItem(RATE_KEY, String(v)); } catch { /* storage unavailable: keep in memory only */ }
}

/** Mirrors ctxprof.verify.summarise: a trace regresses under a selection if it cuts any hunk in `breaks`. */
export function estimate(records: VerifyRecord[], sel: Set<string>) {
  const n = records.length;
  const regress = records.filter((r) => r.breaks.some((b) => sel.has(b)) || (!r.agree && r.breaks.length === 0));
  const before = n ? records.reduce((a, r) => a + r.tokens_before, 0) / n : 0;
  let saved = 0;
  for (const r of records) for (const [fp, v] of Object.entries(r.removed_tokens)) if (sel.has(fp)) saved += v;
  saved = n ? saved / n : 0;
  const agreeN = n - regress.length;
  return { n, regress, before, saved, after: before - saved, agreeN, agreement: n ? agreeN / n : 0 };
}

async function load(id: number | null): Promise<{ list: PruneRow[]; prune: PruneDetail | null }> {
  const list = await api.prunes();
  const pick = id ?? list.find((p) => p.recommend)?.id ?? list[0]?.id ?? null;
  return { list, prune: pick == null ? null : await api.prune(pick) };
}

export function Pruning({ id }: { id: number | null }) {
  const s = useLoad(() => load(id), [id]);
  if (s.state === "loading") return <main className="wrap"><p className="empty">Loading pruning report…</p></main>;
  if (s.state === "error") return <main className="wrap"><div className="panel empty">{s.error}</div></main>;
  const { list, prune } = s.data;
  if (!prune) {
    return (
      <main className="wrap"><div className="panel empty">
        No pruning report yet. Run <code>ctxprof prune --bundle path/to/prompt</code>, then <code>ctxprof verify</code>.
      </div></main>
    );
  }
  return <Report key={prune.id} prune={prune} list={list} />;
}

function Report({ prune, list }: { prune: PruneDetail; list: PruneRow[] }) {
  const v = prune.verification;
  const cfg = v?.config;
  const price = cfg?.price_in_per_m ?? Number(prune.run?.config.price_in_per_m ?? 3);
  const threshold = cfg?.threshold ?? 0.95;
  const [cut, setCut] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(prune.hunks.map((h) => [h.fingerprint, h.cut])));
  const [rate, setRate] = useState(() => readRate(cfg?.ksh_per_usd ?? 129));
  useEffect(() => writeRate(rate), [rate]);

  const sel = useMemo(() => new Set(Object.keys(cut).filter((k) => cut[k])), [cut]);
  const verifiedSel = useMemo(() => new Set(prune.hunks.filter((h) => h.cut).map((h) => h.fingerprint)), [prune]);
  const exact = sel.size === verifiedSel.size && [...sel].every((x) => verifiedSel.has(x));
  const est = v ? estimate(v.records, sel) : null;
  const savedPlan = prune.hunks.reduce((a, h) => a + (sel.has(h.fingerprint) ? h.tokens_per_request : 0), 0);
  const saved = est ? est.saved : savedPlan;
  const before = est?.before ?? 0;
  const pass = !!est && v!.status !== "budget" && est.agreement >= threshold;
  const usd100k = saved * price / 1e6 * 1e5;
  const keepArgs = prune.hunks.filter((h) => !sel.has(h.fingerprint)).map((h) => ` --keep "${h.fingerprint}"`).join("");

  let stamp: string;
  if (!v) stamp = "○ UNVERIFIED · RUN ctxprof verify";
  else if (exact) stamp = pass ? "✓ VERIFIED · RECOMMEND" : "× BELOW THRESHOLD · DO NOT SHIP";
  else stamp = pass ? "✓ ESTIMATE PASSES · RE-VERIFY" : "× ESTIMATE FAILS · DO NOT SHIP";

  return (
    <main className="wrap">
      <section className="verdict" aria-labelledby="v-h">
        <div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
            <span className={`stamp${v && pass ? "" : " fail"}`} aria-live="polite">{stamp}</span>
            <span className="lbl">
              {est ? `Agreement ${(est.agreement * 100).toFixed(1)}% · threshold ${(threshold * 100).toFixed(1)}%` : "Not verified"}
              {` · ${sel.size} of ${prune.hunks.length} hunks cut`}{!exact && v ? " · estimated" : ""}
            </span>
          </div>
          <h1 className="hd vh" id="v-h">−{fmt(saved)} tok
            <small>
              per request{before ? `, ${((saved / before) * 100).toFixed(1)}% of the context window` : " (estimate from the profiling run)"}.
              {" "}Untick “Cut” to keep a segment: the ledger and verdict recompute from the held-out runs
              {v ? "" : " once verified"}.
            </small>
          </h1>
          {list.length > 1 && (
            <nav className="group" aria-label="Pruning reports" style={{ marginTop: 14 }}>
              {[...list].sort((a, b) => a.id - b.id).map((p) => (
                <a key={p.id} className="btn" href={`#/pruning/${p.id}`} aria-current={p.id === prune.id ? "page" : undefined}
                  aria-pressed={p.id === prune.id}>
                  #{p.id} {p.recommend == null ? "○" : p.recommend ? "✓" : "×"} {p.keep.length ? `keep ${p.keep.length}` : "default"}
                </a>
              ))}
            </nav>
          )}
        </div>
        <section className="panel" aria-labelledby="l-h">
          <div className="ph"><span className="n">05.1</span><h2 id="l-h">Ledger</h2>
            <label className="mono dim rate">${price.toFixed(2)} / 1M in · KSh
              <input type="number" min={1} step={0.5} value={rate} aria-label="KSh per US dollar"
                onChange={(e) => { const x = Number(e.target.value); if (x > 0) setRate(x); }} /> / USD</label>
          </div>
          <div className="ledger" aria-live="polite">
            <div><span className="lbl">Tokens / request</span>
              <span className="v">{est ? `${fmt(before)} → ${fmt(est.after)}` : `−${fmt(saved)}`}</span></div>
            <div><span className="lbl">Cost / request</span>
              <span className="v">{est ? `$${(before * price / 1e6).toFixed(5)} → $${(est.after * price / 1e6).toFixed(5)}` : "—"}</span></div>
            <div><span className="lbl">Saved / 100k requests</span><span className="v">${usd100k.toFixed(2)}</span>
              <span className="mono dim sm">≈ KSh {fmt(Math.round(usd100k * rate / 100) * 100)}</span></div>
            <div><span className="lbl">Held-out agreement</span>
              <span className="v">{est ? `${(est.agreement * 100).toFixed(1)}%` : "—"}</span>
              <span className="mono dim sm">{est ? `${est.agreeN} of ${est.n} traces${v!.in_sample ? " · in-sample" : ""}` : "run ctxprof verify"}</span></div>
          </div>
        </section>
      </section>

      <div className="split">
        <section className="panel" aria-labelledby="d-h">
          <div className="ph"><span className="n">05.2</span><h2 id="d-h">prompt.diff</h2>
            <span className="dim" style={{ fontSize: 12.5 }}>removed lines are hatched and struck, never coloured: colour belongs to influence</span></div>
          <div className="diff">
            {prune.hunks.map((h) => (
              <HunkView key={h.fingerprint} h={h} cut={sel.has(h.fingerprint)}
                toggle={() => setCut((c) => ({ ...c, [h.fingerprint]: !c[h.fingerprint] }))} />
            ))}
          </div>
          <pre className="cmd">{`$ ctxprof prune --run ${prune.run_id} --bundle ${prune.bundle}${keepArgs} --out prompt.diff
$ ctxprof verify --threshold ${threshold}`}</pre>
          {exact && prune.diff && <DiffDownload diff={prune.diff} id={prune.id} />}
        </section>

        <div className="stack">
          {v && est && <Verification prune={prune} est={est} sel={sel} />}
          <Budget prune={prune} />
        </div>
      </div>
    </main>
  );
}

function HunkView({ h, cut, toggle }: { h: Hunk; cut: boolean; toggle: () => void }) {
  const [open, setOpen] = useState(false);
  const lines = open ? h.lines : h.lines.slice(0, PREVIEW_LINES);
  const long = h.lines.length > PREVIEW_LINES || h.lines.some((l) => l.text.length > 240);
  const id = `hunk-${h.fingerprint.replace(/[^a-z0-9]+/gi, "-")}`;
  return (
    <div className={`hunk${cut ? "" : " kept"}${open ? " open" : ""}`}>
      <div className="hh">
        <code>@@ {h.file} · {h.fingerprint} @@</code>
        <span className="htk mono dim">−{fmt(h.tokens_per_request)} tok/req · {h.reason}</span>
        <label className="keep"><input type="checkbox" checked={cut} onChange={toggle}
          aria-describedby={id} aria-label={`Cut ${h.fingerprint}`} />Cut</label>
      </div>
      <div id={id}>
        {lines.map((l, i) => (
          <div key={i} className={`ln ${l.sign === "-" ? "del" : "add"}`}>
            <span className="no">{l.no}</span><span className="sg" aria-hidden="true">{l.sign === "-" ? "−" : "+"}</span>
            <span className="tx"><span className="sr">{l.sign === "-" ? (cut ? "removed: " : "kept: ") : "added: "}</span>{l.text}</span>
          </div>
        ))}
      </div>
      {long && (
        <button type="button" className="more" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)}>
          {open ? "Show less" : h.lines.length > PREVIEW_LINES ? `Show ${h.lines.length - PREVIEW_LINES} more lines` : "Show full text"}
        </button>
      )}
    </div>
  );
}

function DiffDownload({ diff, id }: { diff: string; id: number }) {
  const [href, setHref] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(new Blob([diff], { type: "text/x-diff" }));
    setHref(url);
    return () => URL.revokeObjectURL(url);
  }, [diff]);
  return href ? (
    <div style={{ padding: "10px 14px", borderTop: "1px solid var(--rule)" }}>
      <a className="btn" href={href} download={`prompt-${id}.diff`}>Download verified diff</a>
    </div>
  ) : null;
}

function question(q?: string) {
  if (!q) return "";
  const i = q.lastIndexOf("Question:");
  return (i >= 0 ? q.slice(i + 9) : q).trim();
}

function RespBox({ title, r }: { title: string; r?: Resp }) {
  return (
    <div className="resp">
      <span className="lbl">{title}</span>
      <p>{r?.text || <span className="dim">(no text)</span>}</p>
      {!!r?.tool_calls.length && <span className="mono sm">calls {r.tool_calls.map((c) => c.name).join(", ")}</span>}
      {r && !r.tool_calls.length && <span className="mono sm dim">no tool call</span>}
    </div>
  );
}

function Verification({ prune, est, sel }: { prune: PruneDetail; est: ReturnType<typeof estimate>; sel: Set<string> }) {
  const v = prune.verification!;
  const bad = new Set(est.regress.map((r) => r.trace_id));
  const resolved = v.records.filter((r) => !r.agree && !bad.has(r.trace_id));
  const tok = (fps: string[]) => fmt(prune.hunks.filter((h) => fps.includes(h.fingerprint)).reduce((a, h) => a + h.tokens_per_request, 0));
  return (
    <section className="panel" aria-labelledby="g-h">
      <div className="ph"><span className="n">05.3</span><h2 id="g-h">Verification · {v.records.length} held-out traces</h2>
        <span className="mono dim" style={{ marginLeft: "auto", fontSize: 11.5 }}>{v.provider} · score ≥ {v.config.min_score}</span></div>
      <div className="vgrid" role="img" aria-label={`${est.agreeN} of ${est.n} held-out traces kept the same outcome`}>
        {v.records.map((r) => <i key={r.trace_id} className={bad.has(r.trace_id) ? "bad" : ""} title={r.trace_id} />)}
      </div>
      <div className="mono dim keyrow">
        <span><i className="kc" />same outcome</span><span><i className="kc bad" />regression</span>
      </div>
      <ul className="reg">
        {est.regress.map((r) => {
          const suspects = r.breaks.filter((b) => sel.has(b));
          const probe = !r.pruned ? r.probes[suspects[0]] : undefined;
          return (
            <li key={r.trace_id}>
              <span className="mono" style={{ fontWeight: 600 }}>
                <a href={`#/trace/${encodeURIComponent(r.trace_id)}`}>trace {r.trace_id}</a> · {suspects.join(", ") || "interaction"}
                {r.intent ? <span className="dim"> · {r.intent}</span> : null}
              </span>
              <span className="q">“{question(r.question)}”</span>
              <div className="resps">
                <RespBox title="Original replay" r={r.original} />
                <RespBox title={probe ? "If cut (probe)" : "Pruned replay"} r={probe ? probe.pruned : r.pruned} />
              </div>
              <span className="mono dim sm">
                REGRESSION · score {(probe?.score ?? r.score).toFixed(2)}
                {r.explained ? ` · untick “Cut” on ${suspects[0] ?? "the suspect"} to restore` : " · no single hunk explains it"}
              </span>
            </li>
          );
        })}
        {resolved.map((r) => (
          <li key={r.trace_id} className="ok">
            <span className="mono" style={{ fontWeight: 600 }}>trace {r.trace_id} · {r.breaks.join(", ")}</span>
            <span className="mono dim sm">RESOLVED · kept (+{tok(r.breaks)} tok/req)</span>
          </li>
        ))}
        {!est.regress.length && !resolved.length && <li className="ok"><span>No regressions on the held-out traces.</span></li>}
      </ul>
    </section>
  );
}

function Meter({ label, spent, cap }: { label: string; spent: number; cap: number }) {
  const pct = cap > 0 ? Math.min(100, (spent / cap) * 100) : 0;
  return (
    <div style={{ padding: "12px 14px" }}>
      <span className="lbl">{label}</span>
      <div className="meter" role="img" aria-label={`${spent.toFixed(2)} dollars spent of a ${cap.toFixed(2)} dollar cap`}>
        <b style={{ width: `${pct}%` }} /><s style={{ left: "calc(100% - 2px)" }} />
      </div>
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
        <span>${spent.toFixed(2)} spent</span><span className="dim">cap ${cap.toFixed(2)}</span>
      </div>
    </div>
  );
}

function Budget({ prune }: { prune: PruneDetail }) {
  const v = prune.verification;
  const run = prune.run;
  return (
    <section className="panel" aria-labelledby="b-h">
      <div className="ph"><span className="n">05.4</span><h2 id="b-h">Budget and reproducibility</h2></div>
      {run && <Meter label={`Profiling, run ${run.id}`} spent={run.spent_usd} cap={Number(run.config.budget_usd)} />}
      {v && <Meter label="Verification" spent={v.spent_usd} cap={v.config.budget_usd} />}
      <div className="ledger" style={{ borderTop: "1px solid var(--rule)" }}>
        <div><span className="lbl">Model calls</span><span className="v">{fmt((run?.calls ?? 0) + (v?.calls ?? 0))}</span>
          <span className="mono dim sm">profile {fmt(run?.calls ?? 0)} · verify {fmt(v?.calls ?? 0)}</span></div>
        <div><span className="lbl">Cache hits</span><span className="v">{fmt((run?.cache_hits ?? 0) + (v?.cache_hits ?? 0))}</span>
          <span className="mono dim sm">replay = $0.00</span></div>
        <div><span className="lbl">Never pruned</span><span className="v">{prune.skipped.length}</span>
          <span className="mono dim sm">DEAD but unsafe to cut</span></div>
        <div><span className="lbl">Kept by --keep</span><span className="v">{prune.keep.length}</span>
          <span className="mono dim sm">{prune.keep.join(", ") || "none"}</span></div>
      </div>
      {prune.skipped.length > 0 && (
        <details className="skipped">
          <summary>Why {prune.skipped.length} DEAD segments stay</summary>
          <ul className="list">
            {prune.skipped.map((x) => (
              <li key={x.fingerprint}><i className="sw hn" aria-hidden="true" /><span className="fp">{x.fingerprint}</span>
                <span className="mono dim sm">{x.why}</span></li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
