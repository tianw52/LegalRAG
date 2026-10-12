import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// Full-text drawer for one document; retrieved passages are highlighted by
// their character offsets (chunks are stitched server-side in original order).
export default function DocumentReader({ docId, evidence, focusIds, onClose }) {
  const [doc, setDoc] = useState(null);
  const [error, setError] = useState("");
  const bodyRef = useRef(null);

  useEffect(() => {
    setDoc(null);
    setError("");
    api.document(docId).then(setDoc).catch((e) => setError(e.message));
  }, [docId]);

  useEffect(() => {
    if (!doc) return;
    const target =
      bodyRef.current?.querySelector("mark.focus") || bodyRef.current?.querySelector("mark");
    target?.scrollIntoView({ block: "center" });
  }, [doc]);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <div className="lbl">{doc?.doc_type === "statute" ? "Statute" : "Decision"}</div>
            <h2>{doc?.case_name || doc?.citation || "Loading…"}</h2>
            {doc && (
              <p className="muted small">
                {[doc.citation, doc.court, doc.jurisdiction, doc.year].filter(Boolean).join(" · ")}
                {doc.url && (
                  <>
                    {" · "}
                    <a href={doc.url} target="_blank" rel="noreferrer">
                      Open on CanLII
                    </a>
                  </>
                )}
              </p>
            )}
          </div>
          <button className="btn ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>
        {error && <div className="error">{error}</div>}
        {doc && (
          <div className="drawer-body" ref={bodyRef}>
            {highlight(doc.text, evidence, new Set(focusIds))}
          </div>
        )}
      </aside>
    </div>
  );
}

function highlight(text, evidence, focus) {
  const ranges = evidence
    .filter((e) => e.char_start != null && e.char_end != null)
    .map((e) => ({ start: e.char_start, end: e.char_end, id: e.evidence_id }))
    .sort((a, b) => a.start - b.start);
  const out = [];
  let pos = 0;
  ranges.forEach((r, i) => {
    const start = Math.max(r.start, pos);
    if (start >= r.end) return; // fully covered by a previous (overlapping) passage
    if (start > pos) out.push(text.slice(pos, start));
    out.push(
      <mark key={i} className={focus.has(r.id) ? "focus" : ""} title={r.id}>
        <sup>{r.id}</sup>
        {text.slice(start, r.end)}
      </mark>
    );
    pos = r.end;
  });
  out.push(text.slice(pos));
  return out;
}
