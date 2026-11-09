import { useEffect, useState } from "react";
import { Overview } from "./views/Overview";
import { TraceView } from "./views/TraceView";
import { Segments } from "./views/Segments";

type Route = { name: "overview" } | { name: "trace"; id: string } | { name: "segments" };

function parse(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  if (h.startsWith("trace/")) return { name: "trace", id: decodeURIComponent(h.slice(6)) };
  if (h === "segments") return { name: "segments" };
  return { name: "overview" };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const cur = (n: Route["name"]) => (route.name === n ? "page" : undefined);
  return (
    <>
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
        </nav>
        <span className="runid mono">ctxprof 0.1.0</span>
      </header>
      {route.name === "overview" && <Overview />}
      {route.name === "segments" && <Segments />}
      {route.name === "trace" && <TraceView key={route.id} id={route.id} />}
    </>
  );
}
