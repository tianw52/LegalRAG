import { useEffect, useRef } from "react";

// Right-hand column: every retrieved passage (e1..eN).  Items selected in any
// view highlight the passages they cite; clicking a passage opens its document.
export default function EvidencePanel({ evidence, selection, onClear, onOpen }) {
  const listRef = useRef(null);
  const passages = evidence.filter((e) => e.kind === "passage");
  const selected = new Set(selection?.evidenceIds || []);

  useEffect(() => {
    if (!selection) return;
    const first = listRef.current?.querySelector(".ev-card.hit");
    first?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [selection]);

  return (
    <div className="evidence-panel">
      <div className="lbl">Sources ({passages.length} passages)</div>
      {selection && (
        <div className="selection card">
          <div className="row">
            <strong>{selection.label}</strong>
            <span className="spacer" />
            <button className="btn ghost small" onClick={onClear}>
              Clear
            </button>
          </div>
          {selection.detail && <p className="small">{selection.detail}</p>}
          <p className="muted small">
            Cites{" "}
            {selection.evidenceIds
              .map((id) => (id === "q" ? "your question" : id))
              .join(", ")}
          </p>
        </div>
      )}
      {passages.length === 0 && <p className="muted small">Passages appear after you explore a question.</p>}
      <div className="ev-list" ref={listRef}>
        {passages.map((e) => (
          <button
            key={e.evidence_id}
            className={`ev-card ${selected.has(e.evidence_id) ? "hit" : ""} ${
              selection && !selected.has(e.evidence_id) ? "dim" : ""
            }`}
            onClick={() => onOpen(e.doc_id, [e.evidence_id])}
            title="Open the full document"
          >
            <div className="ev-head">
              <span className="ev-id">{e.evidence_id}</span>
              <span className="ev-cite">{e.citation}</span>
              {e.jurisdiction && <span className="tag">{e.jurisdiction}</span>}
            </div>
            {e.case_name && <div className="ev-name">{e.case_name}</div>}
            <div className="ev-text">{e.text.length > 420 ? e.text.slice(0, 420) + "…" : e.text}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
