import { useState } from "react";
import { SourceLinks } from "../components/primitives.jsx";
import { GROUP_COLOR, GROUP_LABEL, GROUP_ORDER, jurisdictionGroup } from "../lib/jurisdiction.js";
import { runDocuments, yearSpan } from "../lib/stats.js";

// Decisions on a year axis, one lane per jurisdiction group; dot size = passages.
export default function TimelineTab({ run, province, openEvidence }) {
  const docs = runDocuments(run).filter((d) => d.year);
  const span = yearSpan(docs);
  const [sel, setSel] = useState(null);
  if (!span) return <section className="band"><p className="muted">No dated sources yet.</p></section>;

  const x = (y) => 3 + ((y - span.min) / Math.max(1, span.max - span.min)) * 94; // % within the lane
  const lanes = GROUP_ORDER.map((g) => ({ g, docs: docs.filter((d) => jurisdictionGroup(d.jurisdiction, province) === g) }))
    .filter((l) => l.docs.length);
  const selDoc = docs.find((d) => d.doc_id === sel);
  const step = span.max - span.min > 30 ? 10 : span.max - span.min > 10 ? 5 : span.max - span.min > 4 ? 2 : 1;
  const ticks = [];
  for (let t = Math.ceil(span.min / step) * step; t <= span.max; t += step) ticks.push(t);

  return (
    <>
      <section className="stat-strip">
        <div><span className="lbl">Oldest</span><span className="disp h-m">{span.min}</span></div>
        <div><span className="lbl">Newest</span><span className="disp h-m">{span.max}</span></div>
        <div><span className="lbl">Since {span.recentFrom}</span><span className="disp h-m">{span.recent}/{span.total}</span></div>
      </section>
      <section className="lanes-wrap">
        <div className="lanes">
          {lanes.map((l) => (
            <div key={l.g} className="lane">
              <div className="lane-label">
                <i className="dot big" style={{ background: GROUP_COLOR[l.g] }} />
                {GROUP_LABEL[l.g]}
                <span className="mono small muted">{l.docs.length}</span>
              </div>
              <div className="lane-track">
                {l.docs.map((d) => {
                  const size = 14 + Math.min(4, d.evidence_ids.length) * 6;
                  const on = d.doc_id === sel;
                  return (
                    <button
                      key={d.doc_id}
                      type="button"
                      className="tl-dot"
                      title={`${d.year} · ${d.case_name || d.citation}`}
                      aria-label={`${d.year} · ${d.case_name || d.citation}`}
                      aria-pressed={on}
                      onClick={() => setSel(on ? null : d.doc_id)}
                      style={{
                        left: `${x(d.year)}%`,
                        width: size,
                        height: size,
                        background: GROUP_COLOR[l.g],
                        borderColor: on ? "var(--ink)" : "var(--paper)",
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
          <div className="lane axis">
            <div className="lane-label"><span className="lbl">Dot size = passages</span></div>
            <div className="lane-track ticks">
              {ticks.map((t) => (
                <span key={t} style={{ left: `${x(t)}%` }} className="mono small muted">{t}</span>
              ))}
            </div>
          </div>
        </div>
      </section>
      {selDoc && (
        <section className="selected-doc" aria-live="polite">
          <span className="disp h-xl" style={{ color: GROUP_COLOR[jurisdictionGroup(selDoc.jurisdiction, province)] }}>{selDoc.year}</span>
          <div className="grow">
            <div className="h-doc">{selDoc.case_name || selDoc.citation}</div>
            <div className="mono small muted">{[selDoc.citation, selDoc.court].filter(Boolean).join(" · ")}</div>
          </div>
          <SourceLinks ids={selDoc.evidence_ids} onOpen={openEvidence} />
          <button type="button" className="btn primary small" onClick={() => openEvidence(selDoc.evidence_ids[0])}>Read passages</button>
        </section>
      )}
    </>
  );
}
