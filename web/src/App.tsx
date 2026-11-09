import { useEffect, useState } from "react";
import { Overview } from "./views/Overview";
import { TraceView } from "./views/TraceView";
import { Segments } from "./views/Segments";
import { Pruning } from "./views/Pruning";
import { STATIC } from "./api";
import { THEME_EVENT } from "./heat";
import { Guide, openGuide } from "./guide/Guide";

type Route =
  | { name: "overview" }
  | { name: "trace"; id: string }
  | { name: "segments" }
  | { name: "pruning"; id: number | null }
  | { name: "missing"; path: string };

function parse(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  if (h === "" ) return { name: "overview" };
  if (h.startsWith("trace/")) return { name: "trace", id: decodeURIComponent(h.slice(6)) };
  if (h === "segments") return { name: "segments" };
  if (h === "pruning") return { name: "pruning", id: null };
  if (h.startsWith("pruning/")) {
    const id = Number(h.slice(8));
    return { name: "pruning", id: Number.isInteger(id) && id > 0 ? id : null };
  }
  return { name: "missing", path: h };
}

type Theme = "light" | "dark";
const THEME_KEY = "ctxprof.theme";

function initialTheme(): Theme | null {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : null;
  } catch { return null; }
}

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(initialTheme);
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    window.dispatchEvent(new Event(THEME_EVENT));
  }, [theme]);
  const shown = theme ?? systemTheme();
  const next: Theme = shown === "dark" ? "light" : "dark";
  return (
    <button type="button" className="toggle" role="switch" aria-checked={shown === "light"}
      aria-label={`Light panel (currently ${shown})`} title={`Switch to ${next} panel`}
      onClick={() => {
        setTheme(next);
        try { localStorage.setItem(THEME_KEY, next); } catch { /* storage unavailable */ }
      }}>
      <span className="toggle-lbl" aria-hidden="true">DARK</span>
      <span className="toggle-track" aria-hidden="true"><span className="toggle-lever" /></span>
      <span className="toggle-lbl" aria-hidden="true">LIGHT</span>
    </button>
  );
}

const KEYS: { name: Route["name"]; href: string; label: string; code: string }[] = [
  { name: "overview", href: "#/", label: "Overview", code: "F1" },
  { name: "segments", href: "#/segments", label: "Segments", code: "F2" },
  { name: "pruning", href: "#/pruning", label: "Pruning", code: "F3" },
];

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => { setRoute(parse(location.hash)); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const routeKey = route.name === "trace" ? `t:${route.id}` : route.name === "pruning" ? `p:${route.id}` : route.name;
  return (
    <>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>
        Skip to content</a>
      <header className="bar">
        <a className="brand" href="#/" aria-label="Context Profiler overview">
          <svg width="30" height="30" viewBox="0 0 30 30" fill="none" aria-hidden="true">
            <rect x="1" y="1" width="28" height="28" stroke="currentColor" strokeWidth="1.4" />
            <path d="M4 22h4V12h3v10h3V7h4v15h3v-6h3v6h2" stroke="currentColor" strokeWidth="1.4" />
            <path d="M4 26h22" stroke="currentColor" strokeWidth="1" strokeDasharray="1 2" />
          </svg>
          <span className="brand-txt">
            <b>Context Profiler</b>
            <span className="mono">CP-8200A · context spectrum analyser</span>
          </span>
        </a>
        <nav className="softkeys" aria-label="Screens">
          {KEYS.map((k) => (
            <a key={k.name} href={k.href} aria-current={route.name === k.name ? "page" : undefined}>
              <i className="led" aria-hidden="true" /><span className="sk-code mono" aria-hidden="true">{k.code}</span>{k.label}
            </a>
          ))}
        </nav>
        <div className="bar-right">
          <span className="acq mono" title={STATIC ? "Static snapshot of the demo run" : "Live API"}>
            <i className="lamp" aria-hidden="true" />{STATIC ? "SNAPSHOT" : "LIVE"} · v0.2</span>
          <button type="button" className="guidekey" data-tour="guide-key" onClick={() => openGuide({ action: "drawer" })}
            aria-keyshortcuts="?" aria-label="Guide: open the service manual (shortcut ?)">
            <span className="balloon sm" aria-hidden="true">?</span>Guide
          </button>
          <ThemeToggle />
        </div>
      </header>
      <div id="main" tabIndex={-1} className="mainfocus">
        <div key={routeKey} className="boot">
          {route.name === "overview" && <Overview />}
          {route.name === "segments" && <Segments />}
          {route.name === "pruning" && <Pruning id={route.id} />}
          {route.name === "trace" && <TraceView id={route.id} />}
          {route.name === "missing" && <NotFound path={route.path} />}
        </div>
      </div>
      <footer className="plate">
        <div className="vents" aria-hidden="true" />
        <div className="plate-row">
          <span className="mono">CONTEXT PROFILER · MODEL CP-8200A · SER. 2611-0009</span>
          <span className="mono dim">Designed and built in Nairobi by Dior Wachira</span>
          <span className="plate-links">
            <a href="https://github.com/DiorWachira/context-profiler">Source</a>
            <a href="https://github.com/DiorWachira/context-profiler#method-in-one-paragraph">Method</a>
            <button type="button" className="linkish" onClick={() => openGuide({ action: "drawer" })}>Service manual</button>
          </span>
        </div>
      </footer>
      <Guide />
    </>
  );
}

function NotFound({ path }: { path: string }) {
  return (
    <main className="wrap">
      <section className="panel nosignal">
        <div className="screen grat nosig-screen" aria-hidden="true"><span className="mono">NO SIGNAL</span></div>
        <div className="nosig-copy">
          <span className="lbl">Error 404 · input not connected</span>
          <h1 className="title">Nothing is plugged into <code>#/{path}</code>.</h1>
          <p className="dim">The bench has three screens. Pick one, or open the service manual.</p>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <a className="btn pri" href="#/">Overview</a>
            <button type="button" className="btn" onClick={() => openGuide({ action: "drawer" })}>Service manual</button>
          </div>
        </div>
      </section>
    </main>
  );
}
