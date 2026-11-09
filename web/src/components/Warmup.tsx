// Loading state: the instrument warming up (a sweeping trace on an empty graticule).
export function Warmup({ label }: { label: string }) {
  return (
    <main className="wrap">
      <div className="panel warm" role="status" aria-live="polite">
        <div className="screen grat warm-screen" aria-hidden="true">
          <svg viewBox="0 0 400 60" preserveAspectRatio="none" className="warm-trace">
            <path d="M0 40 L60 40 L70 12 L80 52 L92 30 L110 40 L400 40" />
          </svg>
        </div>
        <span className="mono warm-lbl"><i className="lamp" aria-hidden="true" />{label}</span>
      </div>
    </main>
  );
}

export function Fault({ children }: { children: React.ReactNode }) {
  return (
    <main className="wrap">
      <div className="panel fault" role="alert">
        <span className="annun on">FAULT</span>
        <div>{children}</div>
      </div>
    </main>
  );
}
