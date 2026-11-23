import { useState } from "react";
import { Segmented } from "./primitives.jsx";
import "./ComparativeEvidenceView.css";

export default function ComparativeEvidenceView({ categories, cases, passages, onOpenEvidence }) {
  const [selectedId, setSelectedId] = useState(null);
  const category = categories.find(({ id }) => id === selectedId) || categories[0];
  const passagesById = new Map(passages.map((passage) => [passage.evidence_id, passage]));
  const columns = cases.flatMap((entry) => {
    const relationship = entry.relationships.find(({ id }) => id === category?.id);
    if (!relationship) return [];
    const evidenceIds = [...new Set(relationship.evidenceIds || [])];
    const evidence = evidenceIds
      .map((id) => passagesById.get(id))
      .filter((passage) => passage?.doc_id === entry.id && typeof passage.text === "string" && passage.text.trim());
    return [{ entry, evidence, missing: evidence.length < evidenceIds.length }];
  });

  return (
    <section className="evidence-compare" aria-label="Comparative evidence">
      <h2 className="disp h-m">Comparative evidence</h2>
      <p className="small muted">Prototype · Uses phrase matches in retrieved passages.</p>
      {categories.length > 0 && (
        <Segmented label="Compare by relevance category" value={category.id} onChange={setSelectedId}
          options={categories.map(({ id, label, color }) => ({ value: id, label, color }))} />
      )}
      {cases.length === 0 ? <p>No cases retrieved.</p>
        : !category ? <p>No relevance categories available for this question.</p>
        : columns.length === 0 ? <p>No retrieved cases matched this category.</p>
        : (
          <div key={category.id} className="evidence-compare-columns" role="region" tabIndex={0}
            aria-label={`Cases for ${category.label}`}>
            {columns.map(({ entry, evidence, missing }) => {
              const metadata = [entry.doc.court || entry.doc.court_abbrev, entry.doc.year].filter(Boolean).join(" · ");
              return (
                <article className="evidence-compare-case" key={entry.id}>
                  <h3>{entry.name}</h3>
                  {metadata && <p className="mono small muted">{metadata}</p>}
                  {evidence.map(({ evidence_id: id, text }) => (
                    <div className="evidence-compare-passage" key={id}>
                      <button type="button" className="ev" aria-label={`Open passage ${id} in Sources Reader`}
                        onClick={() => onOpenEvidence(id)}>{id}</button>
                      <blockquote>{text}</blockquote>
                    </div>
                  ))}
                  {(evidence.length === 0 || missing) && (
                    <p className="small muted">{evidence.length === 0
                      ? "No supporting passage text is available for this relationship."
                      : "Some linked passages are unavailable."}</p>
                  )}
                </article>
              );
            })}
          </div>
        )}
    </section>
  );
}
