import { Dot, Pending, SourceLinks, Switch, TabHead } from "../components/primitives.jsx";
import { FACT_STATUS, factIsOff, runFacts } from "../lib/facts.js";

// Facts that matter (fact_toggle view).  Flip a fact off to see what the
// sources say would change.  Facts the user never mentioned start flipped.
// Flips are stored as answers (run.answers), shared with the Overview.
export default function WhatIfTab({ run, openEvidence, setAnswer, onLook }) {
  const view = run.views.fact_toggle;
  const facts = runFacts(run);
  const topic = run.plan?.topic || "What-if";

  if (!view) {
    const failed = run.viewErrors?.fact_toggle;
    return (
      <>
        <TabHead title={topic} />
        <section className="band">
          <h2 className="lbl band-label">Facts</h2>
          <div className="band-body">
            {failed ? <p className="error small">Couldn't build the facts: {failed}</p> : <Pending label="Extracting the facts that matter…" />}
          </div>
        </section>
      </>
    );
  }

  const offCount = facts.filter((f) => factIsOff(f, run.answers)).length;

  return (
    <>
      <TabHead title={topic}>
        <span className="disp h-xl accent">{offCount}</span>
        <span className="muted">of {facts.length} facts<br />changed</span>
      </TabHead>
      <section className="band">
        <h2 className="lbl band-label">Key question</h2>
        <div className="band-body row wrap baseline spread">
          <p className="key-question">{view.spec.key_question}</p>
          <div className="legend">
            {Object.values(FACT_STATUS).map((s) => (
              <span key={s.label}><Dot color={s.color} />{s.label}</span>
            ))}
          </div>
        </div>
      </section>
      <section className="band">
        <h2 className="lbl band-label">Facts · flip to test</h2>
        <ul className="rows facts" aria-live="polite">
          {facts.map((f) => {
            const isOff = factIsOff(f, run.answers);
            const st = FACT_STATUS[f.status] || FACT_STATUS.assumed;
            return (
              <li key={f.id} className={isOff ? "off" : ""}>
                <div className="fact-row">
                  <Switch
                    checked={!isOff}
                    onChange={() => {
                      setAnswer(f.id, isOff ? "yes" : "no");
                      onLook(f.text);
                    }}
                    label={`What if this were different: ${f.text}`}
                  />
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
