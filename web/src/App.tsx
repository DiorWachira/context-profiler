import { useEffect, useState } from "react";
import { Overview } from "./views/Overview";
import { TraceView } from "./views/TraceView";
import { Segments } from "./views/Segments";
import { Pruning } from "./views/Pruning";
import { STATIC } from "./api";

type Route =
  | { name: "overview" }
  | { name: "trace"; id: string }
  | { name: "segments" }
  | { name: "pruning"; id: number | null };

function parse(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  if (h.startsWith("trace/")) return { name: "trace", id: decodeURIComponent(h.slice(6)) };
  if (h === "segments") return { name: "segments" };
  if (h === "pruning") return { name: "pruning", id: null };
  if (h.startsWith("pruning/")) {
    const id = Number(h.slice(8));
    return { name: "pruning", id: Number.isInteger(id) && id > 0 ? id : null };
  }
  return { name: "overview" };
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
  }, [theme]);
  const shown = theme ?? systemTheme();
  const next: Theme = shown === "dark" ? "light" : "dark";
  return (
    <button type="button" className="btn sq theme" aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}
      onClick={() => {
        setTheme(next);
        try { localStorage.setItem(THEME_KEY, next); } catch { /* storage unavailable */ }
      }}>
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
        <circle cx="9" cy="9" r="7" /><path d="M9 2v14" />{shown === "dark" ? <path d="M9 2a7 7 0 0 1 0 14z" fill="currentColor" /> : <path d="M9 2a7 7 0 0 0 0 14z" fill="currentColor" />}
      </svg>
    </button>
  );
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => { setRoute(parse(location.hash)); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const cur = (n: Route["name"]) => (route.name === n ? "page" : undefined);
  return (
    <>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById("main")?.focus(); }}>
        Skip to content</a>
      <header className="bar">
        <a className="brand" href="#/" aria-label="Context Profiler overview">
          <svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
            <path d="M2 19h18M2 15h18" /><path d="M3 15V11h6v4M10 15V7h5v8M16 15v-3h3v3" /><path d="M2 3v2M7 3v2M12 3v2M17 3v2" />
          </svg>
          <b>Context Profiler</b>
        </a>
        <nav className="nav" aria-label="Screens">
          <a href="#/" aria-current={cur("overview")}>Overview</a>
          <a href="#/segments" aria-current={cur("segments")}>Segments</a>
          <a href="#/pruning" aria-current={cur("pruning")}>Pruning</a>
        </nav>
        <span className="runid mono">{STATIC ? "static demo snapshot · " : ""}ctxprof 0.2.0</span>
        <ThemeToggle />
      </header>
      <div id="main" tabIndex={-1} className="mainfocus">
        {route.name === "overview" && <Overview />}
        {route.name === "segments" && <Segments />}
        {route.name === "pruning" && <Pruning key={route.id ?? "latest"} id={route.id} />}
        {route.name === "trace" && <TraceView key={route.id} id={route.id} />}
      </div>
    </>
  );
}
