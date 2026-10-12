import { useState } from "react";

const STATUS_LABEL = {
  stated: "You said",
  assumed: "Assumed",
  unknown: "Not mentioned",
};

// Pre-defined view: the facts that matter.  Switching a fact off ("what if
// this were different?") reveals what the sources say would change.
export default function FactToggle({ view, selection, onSelect }) {
  const { key_question, facts = [] } = view.spec;
  // unknown facts start "off" so their consequences are visible immediately
  const [off, setOff] = useState(() => new Set(facts.filter((f) => f.status === "unknown").map((f) => f.id)));

  function toggle(id) {
    setOff((cur) => {
      const next = new Set(cur);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const selectedKey = selection?.label;

  return (
    <section className="card view">
      <ViewHead view={view} />
      {key_question && <p className="key-question">{key_question}</p>}
      <ul className="facts">
        {facts.map((f) => {
          const isOff = off.has(f.id);
          return (
            <li
              key={f.id}
              className={`fact ${isOff ? "off" : ""} ${selectedKey === f.text ? "selected" : ""} ${
                f.evidence_ids.length ? "" : "ungrounded"
              }`}
            >
              <label className="switch" title={isOff ? "Assume this is true" : "What if this were different?"}>
                <input type="checkbox" checked={!isOff} onChange={() => toggle(f.id)} />
                <span />
              </label>
              <button
                className="fact-body"
                onClick={() =>
                  onSelect({ label: f.text, detail: f.why_it_matters, evidenceIds: f.evidence_ids })
                }
              >
                <span className={`tag status-${f.status}`}>{STATUS_LABEL[f.status]}</span>
                <span className="fact-text">{f.text}</span>
                {isOff ? (
                  <span className="if-changed">
                    <strong>If this is different:</strong> {f.if_changed}
                  </span>
                ) : (
                  <span className="why muted small">{f.why_it_matters}</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export function ViewHead({ view, extra }) {
  return (
    <div className="view-head">
      <div>
        <h3>{view.title}</h3>
        {view.description && <p className="muted small">{view.description}</p>}
      </div>
      <span className="spacer" />
      {extra}
      <span className={`tag source-${view.source}`}>
        {view.source === "agent" ? "Chosen for this question" : view.source === "llm" ? "Generated" : "From sources"}
      </span>
    </div>
  );
}
