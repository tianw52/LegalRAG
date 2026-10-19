import { TAB_LABEL } from "../lib/tabs.js";

const ICON = { done: "✓", running: "●", queued: "○", failed: "✕" };
const READY_ORDER = ["sources", "timeline", "whatif", "relations"];

// Build progress on the Overview while a run is in flight (lib/progress.js):
// overall %, one bar per stage (width = its share of the work), and links to
// the tabs that are already usable.
export default function ProgressPanel({ progress, onOpen }) {
  const { stages, pct, running, ready, elapsed } = progress;
  const links = READY_ORDER.filter((id) => ready[id]);
  return (
    <section className="pp" aria-live="polite">
      <div className="pp-head">
        <span className="disp pp-pct">{pct}%</span>
        <div className="pp-now">
          <span className="pp-current">{running.length ? running.map((s) => s.label).join(" · ") : "Starting…"}</span>
          <span className="small muted">{elapsed} s so far · open views as they become ready</span>
        </div>
      </div>
      <div className="pp-stages">
        {stages.map((s) => (
          <div key={s.id} className={`pp-stage ${s.status}`} style={{ flex: `${s.w} 1 0` }}>
            <div className="pp-track"><i style={{ width: `${(s.fr * 100).toFixed(1)}%` }} /></div>
            <span className="pp-label">{ICON[s.status]} {s.short}</span>
            <span className="mono pp-status">
              {s.status === "done" ? `ready${s.seconds != null ? ` · ${Math.round(s.seconds)} s` : ""}` : s.status === "running" ? "working" : s.status}
            </span>
          </div>
        ))}
      </div>
      <div className="pp-ready">
        <span className="lbl">Ready now</span>
        {links.length ? (
          links.map((id) => (
            <button key={id} type="button" className="chip-dark" onClick={() => onOpen(id)}>
              {TAB_LABEL[id]} <span aria-hidden="true">→</span>
            </button>
          ))
        ) : (
          <span className="small muted">Nothing yet — sources come first.</span>
        )}
      </div>
    </section>
  );
}
