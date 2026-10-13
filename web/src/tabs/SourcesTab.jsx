import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import HighlightedText, { passageMarks } from "../components/HighlightedText.jsx";
import { Pending, Segmented } from "../components/primitives.jsx";
import { GROUP_COLOR, GROUP_LABEL, GROUP_ORDER, jurisdictionGroup } from "../lib/jurisdiction.js";
import { runPassages } from "../lib/stats.js";

// Passage list + full-text reader.  The reader loads the document of the
// focused passage and highlights every retrieved passage in it by offsets.
export default function SourcesTab({ run, province, focusId, setFocusId }) {
  const passages = runPassages(run);
  const [group, setGroup] = useState("all");
  const [docs, setDocs] = useState({}); // doc_id → document (cache for this run)
  const [error, setError] = useState("");
  const markRef = useRef(null);

  const focus = passages.find((p) => p.evidence_id === focusId) || passages[0];
  const doc = focus ? docs[focus.doc_id] : null;

  useEffect(() => {
    if (!focus || docs[focus.doc_id]) return;
    setError("");
    api.document(focus.doc_id)
      .then((d) => setDocs((cur) => ({ ...cur, [focus.doc_id]: d })))
      .catch((e) => setError(e.message));
  }, [focus?.doc_id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    markRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [doc, focus?.evidence_id]);

  if (!passages.length) return <section className="band"><Pending label="Retrieving sources…" /></section>;

  const groupOf = (p) => jurisdictionGroup(p.jurisdiction, province);
  const shown = passages.filter((p) => group === "all" || groupOf(p) === group);
  // Min–max scale: rerank scores are often all close to 1 (Qwen3-Reranker)
  const scores = passages.map((p) => p.rerank_score ?? 0);
  const lo = Math.min(...scores);
  const range = Math.max(1e-9, Math.max(...scores) - lo);
  const scaled = (s) => 10 + (((s ?? 0) - lo) / range) * 90;
  const docPassages = focus ? passages.filter((p) => p.doc_id === focus.doc_id) : [];

  return (
    <>
      <section className="tab-bar">
        <Segmented
          label="Filter by jurisdiction"
          value={group}
          onChange={setGroup}
          options={[
            { value: "all", label: "All" },
            ...GROUP_ORDER.filter((g) => passages.some((p) => groupOf(p) === g)).map((g) => ({
              value: g,
              label: GROUP_LABEL[g],
              color: GROUP_COLOR[g],
            })),
          ]}
        />
      </section>
      <div className="split">
        <section className="split-side list">
          <ol className="rows passage-list">
            {shown.map((p) => {
              const on = p.evidence_id === focus?.evidence_id;
              return (
                <li key={p.evidence_id}>
                  <button type="button" className={`rowlink passage-row ${on ? "on" : ""}`} aria-pressed={on} onClick={() => setFocusId(p.evidence_id)}>
                    <span className="mono passage-id">{p.evidence_id}</span>
                    <i className="dot" style={{ background: GROUP_COLOR[groupOf(p)] }} />
                    <span className="passage-name">{p.case_name || p.citation}</span>
                    <span className="score" aria-hidden="true"><i style={{ width: `${scaled(p.rerank_score)}%` }} /></span>
                  </button>
                </li>
              );
            })}
          </ol>
        </section>

        <article className="split-main reader-col" aria-live="polite">
          {focus && (
            <header className="reader-head">
              <div className="grow">
                <span className="lbl inline">
                  <i className="dot" style={{ background: GROUP_COLOR[groupOf(focus)] }} />
                  {focus.doc_type === "statute" ? "Statute" : "Decision"} · {[focus.citation, focus.court, focus.year].filter(Boolean).join(" · ")}
                </span>
                <h2 className="disp h-m">{focus.case_name || focus.citation}</h2>
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
