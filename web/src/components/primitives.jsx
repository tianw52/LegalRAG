// Small presentational building blocks shared across pages and tabs.
import { GROUP_COLOR, GROUP_LABEL, GROUP_ORDER } from "../lib/jurisdiction.js";

export function Dot({ color, size = 8 }) {
  return <i className="dot" style={{ width: size, height: size, background: color }} />;
}

// Legend of jurisdiction groups (local / federal / other / unknown).
export function JurisdictionLegend({ groups = GROUP_ORDER, labels = GROUP_LABEL }) {
  return (
    <div className="legend">
      {groups.map((g) => (
        <span key={g}>
          <Dot color={GROUP_COLOR[g]} />
          {labels[g]}
        </span>
      ))}
    </div>
  );
}

// Proportional horizontal bar: parts = [{value, color}].
export function SplitBar({ parts, height = 12, maxWidth }) {
  const shown = parts.filter((p) => p.value > 0);
  return (
    <div className="splitbar" style={{ height, maxWidth }} aria-hidden="true">
      {shown.map((p, i) => (
        <i key={i} style={{ flex: p.value, background: p.color }} />
      ))}
    </div>
  );
}

// Large toggle switch; `checked` = fact assumed true.
export function Switch({ checked, onChange, label }) {
  return (
    <label className="sw" title={label}>
      <input type="checkbox" aria-label={label} checked={checked} onChange={onChange} />
      <span />
    </label>
  );
}

// Text-only segmented control (underline marks the active option).
export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={`seg ${o.value === value ? "on" : ""}`}
          aria-pressed={o.value === value}
          onClick={() => onChange(o.value)}
        >
          {o.color && <Dot color={o.color} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Evidence ids rendered as links that open the passage in the Sources tab.
export function SourceLinks({ ids, onOpen }) {
  const shown = (ids || []).filter((id) => id !== "q");
  const fromQuestion = (ids || []).includes("q");
  if (!shown.length && !fromQuestion) return null;
  return (
    <span className="srcs">
      {fromQuestion && <span className="src muted">your question</span>}
      {shown.map((id) => (
        <button key={id} type="button" className="src" onClick={() => onOpen(id)}>
          {id}
        </button>
      ))}
    </span>
  );
}

export function Pending({ label = "Generating…" }) {
  return (
    <div className="pending" role="status">
      <span className="pulse" /> {label}
    </div>
  );
}
