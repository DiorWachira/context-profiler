import { useEffect, useState } from "react";
import { THEME_EVENT } from "./heat";

export type Load<T> = { state: "loading" } | { state: "error"; error: string } | { state: "ok"; data: T };

export function useLoad<T>(fn: () => Promise<T>, deps: unknown[] = []): Load<T> {
  const [s, setS] = useState<Load<T>>({ state: "loading" });
  useEffect(() => {
    let alive = true;
    setS({ state: "loading" });
    fn().then(
      (data) => alive && setS({ state: "ok", data }),
      (e: unknown) => alive && setS({ state: "error", error: String(e) }),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}

/** A counter that changes whenever the colour theme changes. */
export function useThemeVersion(): number {
  const [n, setN] = useState(0);
  useEffect(() => {
    const bump = () => setN((x) => x + 1);
    const mq = window.matchMedia?.("(prefers-color-scheme: light)");
    window.addEventListener(THEME_EVENT, bump);
    mq?.addEventListener?.("change", bump);
    return () => { window.removeEventListener(THEME_EVENT, bump); mq?.removeEventListener?.("change", bump); };
  }, []);
  return n;
}

export function useReducedMotion(): boolean {
  const q = "(prefers-reduced-motion: reduce)";
  const [r, setR] = useState(() => window.matchMedia?.(q).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(q);
    const on = () => setR(mq.matches);
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, []);
  return r;
}
