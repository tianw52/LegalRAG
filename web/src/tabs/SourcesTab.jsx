import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import HighlightedText, { passageMarks } from "../components/HighlightedText.jsx";
import { Pending, Segmented, TabHead } from "../components/primitives.jsx";
import { GROUP_COLOR, GROUP_ORDER, groupLabel, jurisdictionGroup } from "../lib/jurisdiction.js";
import { groupCounts, runDocuments, runPassages } from "../lib/stats.js";

const TYPES = [
  { value: "all", label: "All" },
  { value: "case", label: "Cases" },
  { value: "statute", label: "Statutes" },
];

// Where the sources come from (filter) · documents, most relevant first ·
// full-text reader of the focused passage, every retrieved passage in that
// document highlighted by its offsets.
export default function SourcesTab({ run, province, provinceLabel, focusId, setFocusId }) {
  const docs = runDocuments(run);
  const passages = runPassages(run);
  const [group, setGroup] = useState(null);
  const [type, setType] = useState("all");
  const [texts, setTexts] = useState({}); // doc_id → document (cache for this run)
  const [error, setError] = useState("");
  const markRef = useRef(null);

  const focus = passages.find((p) => p.evidence_id === focusId) || passages[0];
  const doc = focus ? texts[focus.doc_id] : null;

  useEffect(() => {
    if (!focus || texts[focus.doc_id]) return;
    setError("");
    api.document(focus.doc_id)
      .then((d) => setTexts((cur) => ({ ...cur, [focus.doc_id]: d })))
      .catch((e) => setError(e.message));
  }, [focus?.doc_id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [doc, focus?.evidence_id]);

  const topic = run.plan?.topic || "Sources";
  if (!passages.length) return <><TabHead title={topic} /><section className="band"><Pending label="Retrieving sources…" /></section></>;

  const groupOf = (d) => jurisdictionGroup(d.jurisdiction, province);
  const counts = groupCounts(docs, province);
  const shown = docs.filter((d) => (!group || groupOf(d) === group) && (type === "all" || (d.doc_type || "case") === type));
  const docPassages = focus ? passages.filter((p) => p.doc_id === focus.doc_id) : [];

  return (
    <>
      <TabHead title={topic}>
        <span className="lbl">{docs.length} documents · {passages.length} passages</span>
      </TabHead>
      <div className="sources">
        <aside className="src-filters">
          <div className="row baseline">
            <h2 className="lbl grow">Where they come from</h2>
            {group && <button type="button" className="link small" onClick={() => setGroup(null)}>Show all</button>}
          </div>
          <ul className="rows">
            {GROUP_ORDER.filter((g) => counts[g]).map((g) => (
              <li key={g}>
                <button
                  type="button"
                  className={`rowlink group-row ${group === g ? "on" : ""}`}
                  aria-pressed={group === g}
                  style={{ opacity: group && group !== g ? 0.5 : 1 }}
                  onClick={() => setGroup(group === g ? null : g)}
                >
                  <span className="disp group-count">{counts[g]}</span>
                  <span className="group-text">
                    <span className="inline group-name"><i className="dot" style={{ background: GROUP_COLOR[g] }} />{groupLabel(g, provinceLabel)}</span>
                    <span className="small muted">{courtsOf(docs.filter((d) => groupOf(d) === g))}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <h2 className="lbl">Type</h2>
          <Segmented label="Document type" options={TYPES} value={type} onChange={setType} />
        </aside>

        <section className="src-docs">
          <div className="row baseline src-docs-head">
            <h2 className="lbl grow">Documents · most relevant first</h2>
            <span className="lbl">{shown.length} of {docs.length}</span>
          </div>
          <ul className="rows doc-list">
            {shown.map((d) => {
              const on = focus?.doc_id === d.doc_id;
              const color = GROUP_COLOR[groupOf(d)];
              return (
                <li key={d.doc_id} className={on ? "open" : ""}>
                  <button type="button" className="rowlink doc-row" aria-expanded={on} onClick={() => setFocusId(d.evidence_ids[0])}>
                    <span className="doc-line">
                      <i className="dot big" style={{ background: color }} />
                      <span className="doc-name">{d.case_name || d.citation}</span>
                    </span>
                    <span className="doc-line sub">
                      <span className="mono small muted">{[d.doc_type === "statute" ? "Statute" : null, d.citation, d.court_abbrev || d.court, d.year].filter(Boolean).join(" · ")}</span>
                      <span className="pips" aria-label={`${d.evidence_ids.length} passages`}>
                        {d.evidence_ids.slice(0, 5).map((id) => <i key={id} style={{ background: color }} />)}
                      </span>
                    </span>
                  </button>
                  {on && d.evidence_ids.length > 1 && (
                    <div className="ev-chips" role="group" aria-label="Passages in this document">
                      {d.evidence_ids.map((id) => (
                        <button key={id} type="button" className={`ev ${id === focus.evidence_id ? "on" : ""}`} aria-pressed={id === focus.evidence_id} onClick={() => setFocusId(id)}>
                          {id}
                        </button>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </section>

        <article className="src-reader" aria-live="polite">
          {focus && (
            <header className="reader-head">
              <div className="grow">
                <span className="lbl inline">
                  <i className="dot" style={{ background: GROUP_COLOR[groupOf(focus)] }} />
                  {focus.doc_type === "statute" ? "Statute" : "Decision"} · {[focus.citation, focus.court, focus.year].filter(Boolean).join(" · ")} · {focus.evidence_id}
                </span>
                <h2 className="reader-title">{focus.case_name || focus.citation}</h2>
              </div>
              {focus.url && <a className="pill" href={focus.url} target="_blank" rel="noreferrer">CanLII ↗</a>}
            </header>
          )}
          {error && <p className="error small">{error}</p>}
          {!doc && !error && <Pending label="Loading document…" />}
          {doc && (
            <div className="reader">
              <div className="minimap" aria-hidden="true">
                {passageMarks(doc.text.length, docPassages).map((m) => (
                  <i key={m.id} style={{ top: `${m.top}%`, background: m.id === focus.evidence_id ? "var(--local)" : "var(--ink)" }} />
                ))}
              </div>
              <div className="reader-text">
                <HighlightedText text={doc.text} passages={docPassages} focusId={focus.evidence_id} markRef={markRef} />
              </div>
            </div>
          )}
        </article>
      </div>
    </>
  );
}

// "SKCA, SKKB, statutes" — the courts behind a jurisdiction group.
function courtsOf(docs) {
  const names = [...new Set(docs.map((d) => (d.doc_type === "statute" ? "statutes" : d.court_abbrev || d.court)).filter(Boolean))];
  return names.slice(0, 4).join(", ") + (names.length > 4 ? "…" : "");
}
