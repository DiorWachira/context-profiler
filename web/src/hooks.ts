import { useEffect, useState } from "react";

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
