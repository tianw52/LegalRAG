import { Pending, SourceLinks, TabHead } from "../components/primitives.jsx";
import { factIsOff, runFacts } from "../lib/facts.js";

// Table-formatted What-if (ported from the Demo's comparison scorecard,
// Demo/js/viz-renderers.js ComparisonRenderer): facts in rows, qualitative
// badge cells in columns.  Same data and answers as the Fact Breakdown tab
// (run.answers), so the two formats stay in sync for side-by-side comparison.

const STATED = {
  stated: { tone: "yes", sym: "✓", label: "You said" },
  assumed: { tone: "maybe", sym: "~", label: "We assumed" },
  unknown: { tone: "asked", sym: "?", label: "Not mentioned" },
};
const HOLDS = {
  on: { tone: "yes", sym: "✓", label: "True" },
  off: { tone: "no", sym: "✗", label: "What if not" },
};

function Badge({ tone, sym, label }) {
  return (
    <>
      <span className={`wt-sym wt-${tone}`} aria-hidden="true">{sym}</span>
      <span className="wt-val">{label}</span>
    </>
  );
}

export default function WhatIfTableTab({ run, openEvidence, setAnswer, onLook }) {
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
        <div className="band-body">
          <p className="key-question">{view.spec.key_question}</p>
        </div>
      </section>
      <section className="band">
        <h2 className="lbl band-label">What-if table</h2>
        <div className="band-body">
          <div className="wt" role="table" aria-label="What-if facts" aria-live="polite">
            <div className="wt-row wt-head" role="row">
              <span className="lbl" role="columnheader">Fact</span>
              <span className="lbl wt-c" role="columnheader">In your question</span>
              <span className="lbl wt-c" role="columnheader">Holds? (click)</span>
              <span className="lbl" role="columnheader">Effect</span>
            </div>
            {facts.map((f) => {
              const isOff = factIsOff(f, run.answers);
              const st = STATED[f.status] || STATED.assumed;
              const holds = isOff ? HOLDS.off : HOLDS.on;
              return (
                <div key={f.id} className={`wt-row ${isOff ? "off" : ""}`} role="row">
                  <div className="wt-fact" role="rowheader">
                    {f.text}
                    <SourceLinks ids={f.evidence_ids} onOpen={openEvidence} />
                  </div>
                  <div className={`wt-cell wt-c wt-bg-${st.tone}`} role="cell">
                    <Badge {...st} />
                  </div>
                  <div role="cell" className="wt-c">
                    <button
                      type="button"
                      className={`wt-cell wt-btn wt-bg-${holds.tone}`}
                      aria-pressed={isOff}
                      aria-label={`What if this were different: ${f.text}`}
                      onClick={() => {
                        setAnswer(f.id, isOff ? "yes" : "no");
                        onLook(f.text);
                      }}
                    >
                      <Badge {...holds} />
                    </button>
                  </div>
                  <div className="wt-effect" role="cell">
                    {isOff ? (
                      <><b className="accent">If not → </b>{f.if_changed}</>
                    ) : (
                      <span className="muted">{f.why_it_matters}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <p className="muted small wt-note">Qualitative labels only — no numeric scores. Click a "Holds?" cell to test the opposite.</p>
        </div>
      </section>
    </>
  );
}
