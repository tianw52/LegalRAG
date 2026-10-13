import { useEffect, useState } from "react";
import { Dot, Pending, SourceLinks, Switch } from "../components/primitives.jsx";

const STATUS = {
  stated: { label: "You said", color: "var(--ink)" },
  assumed: { label: "Assumed", color: "var(--federal)" },
  unknown: { label: "Not mentioned", color: "var(--local)" },
};

// Facts that matter (fact_toggle view).  Flip a fact off to see what the
// sources say would change.  Facts the user never mentioned start flipped.
export default function WhatIfTab({ run, openEvidence }) {
  const view = run.views.fact_toggle;
  const facts = view?.spec?.facts || [];
  const [off, setOff] = useState(new Set());

  useEffect(() => {
    setOff(new Set(facts.filter((f) => f.status === "unknown").map((f) => f.id)));
  }, [view]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!view) {
    const failed = run.viewErrors?.fact_toggle;
    return (
      <section className="band">
        <h2 className="lbl band-label">Facts</h2>
        <div className="band-body">
          {failed ? <p className="error small">Couldn't build the facts: {failed}</p> : <Pending label="Extracting the facts that matter…" />}
        </div>
      </section>
    );
  }

  function toggle(id) {
    setOff((cur) => {
      const next = new Set(cur);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  return (
    <>
      <section className="band">
        <h2 className="lbl band-label">Key question</h2>
        <div className="band-body row wrap baseline spread">
          <p className="key-question">{view.spec.key_question}</p>
          <div className="legend">
            {Object.values(STATUS).map((s) => (
              <span key={s.label}><Dot color={s.color} />{s.label}</span>
            ))}
          </div>
        </div>
      </section>
      <section className="band">
        <div className="band-label">
          <h2 className="lbl">Facts · flip to test</h2>
          <p className="disp h-m accent">{off.size}<span className="small muted"> of {facts.length} changed</span></p>
        </div>
        <ul className="rows facts" aria-live="polite">
          {facts.map((f) => {
            const isOff = off.has(f.id);
            const st = STATUS[f.status] || STATUS.assumed;
            return (
              <li key={f.id} className={isOff ? "off" : ""}>
                <div className="fact-row">
                  <Switch checked={!isOff} onChange={() => toggle(f.id)} label={`What if this were different: ${f.text}`} />
                  <span className="fact-status lbl"><Dot color={st.color} />{st.label}</span>
                  <span className="fact-text">{f.text}</span>
                  <SourceLinks ids={f.evidence_ids} onOpen={openEvidence} />
                </div>
                {isOff ? (
                  <div className="fact-detail">
                    <span className="disp accent">If not →</span>
                    <span>{f.if_changed}</span>
                  </div>
                ) : (
                  <div className="fact-detail muted small">{f.why_it_matters}</div>
                )}
              </li>
            );
          })}
        </ul>
      </section>
    </>
  );
}
