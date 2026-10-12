import { useMemo, useState } from "react";
import { ViewHead } from "./FactToggle.jsx";

const JURIS_LABEL = { federal: "Federal", unknown: "Unknown jurisdiction" };

export function jurisdictionGroup(code, province) {
  if (code && code === province) return "local";
  if (code === "federal") return "federal";
  if (!code || code === "unknown") return "unknown";
  return "other";
}

// Built-in view: retrieved documents grouped by jurisdiction, filterable.
export default function CaseLandscape({ view, province, selection, onSelect, onOpen }) {
  const docs = view.spec.documents || [];
  const [hiddenJuris, setHiddenJuris] = useState(new Set());
  const [docType, setDocType] = useState("all");

  const groups = useMemo(() => {
    const byJ = {};
    docs.forEach((d) => (byJ[d.jurisdiction || "unknown"] ||= []).push(d));
    const rank = (j) => ({ local: 0, federal: 1, other: 2, unknown: 3 })[jurisdictionGroup(j, province)];
    return Object.entries(byJ).sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]));
  }, [docs, province]);

  const selected = new Set(selection?.evidenceIds || []);

  function toggleJuris(j) {
    setHiddenJuris((cur) => {
      const next = new Set(cur);
      next.has(j) ? next.delete(j) : next.add(j);
      return next;
    });
  }

  const filters = (
    <div className="row wrap">
      {groups.map(([j, ds]) => (
        <button
          key={j}
          className={`chip small ${hiddenJuris.has(j) ? "" : "on"}`}
          onClick={() => toggleJuris(j)}
        >
          {JURIS_LABEL[j] || j} · {ds.length}
        </button>
      ))}
      <select className="small" value={docType} onChange={(e) => setDocType(e.target.value)}>
        <option value="all">Cases & statutes</option>
        <option value="case">Cases only</option>
        <option value="statute">Statutes only</option>
      </select>
    </div>
  );

  return (
    <section className="card view">
      <ViewHead view={view} />
      {filters}
      {groups
        .filter(([j]) => !hiddenJuris.has(j))
        .map(([j, ds]) => {
          const shown = ds.filter((d) => docType === "all" || d.doc_type === docType);
          if (!shown.length) return null;
          return (
            <div key={j} className={`juris-group g-${jurisdictionGroup(j, province)}`}>
              <div className="lbl">{JURIS_LABEL[j] || j}</div>
              <div className="doc-grid">
                {shown.map((d) => (
                  <div
                    key={d.doc_id}
                    className={`doc-card ${d.evidence_ids.some((id) => selected.has(id)) ? "hit" : ""}`}
                  >
                    <button
                      className="doc-main"
                      onClick={() =>
                        onSelect({
                          label: d.case_name || d.citation,
                          detail: [d.citation, d.court, d.year].filter(Boolean).join(" · "),
                          evidenceIds: d.evidence_ids,
                        })
                      }
                    >
                      <span className="doc-name">{d.case_name || d.citation}</span>
                      <span className="muted small">
                        {[d.citation, d.court || d.court_abbrev, d.year].filter(Boolean).join(" · ")}
                      </span>
                      <span className="small">
                        {d.doc_type === "statute" ? "Statute · " : ""}
                        {d.evidence_ids.length} relevant passage{d.evidence_ids.length > 1 ? "s" : ""}
                      </span>
                    </button>
                    <div className="doc-actions">
                      <button className="btn ghost small" onClick={() => onOpen(d.doc_id, d.evidence_ids)}>
                        Read
                      </button>
                      {d.url && (
                        <a className="btn ghost small" href={d.url} target="_blank" rel="noreferrer">
                          CanLII ↗
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
    </section>
  );
}
