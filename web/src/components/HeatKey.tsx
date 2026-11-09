import { BINS } from "../heat";

/** The one colour key in the app: seven influence bins plus "not scored". */
export function HeatKey() {
  const edges = ["0", ...BINS.map(String)];
  return (
    <span className="heatkey" role="img" aria-label="Colour key: influence from cold blue (near 0) to hot yellow (0.70 and above); hatched means not scored">
      <span className="lbl" aria-hidden="true">Influence</span>
      <span className="hk-band" aria-hidden="true">
        {edges.map((e, i) => <i key={e} className={`h${i}`}><span className="mono">{i === 0 ? "0" : e.replace(/^0/, "")}</span></i>)}
      </span>
      <span className="hk-pin" aria-hidden="true"><i className="hn" /><span className="mono">pinned</span></span>
    </span>
  );
}
