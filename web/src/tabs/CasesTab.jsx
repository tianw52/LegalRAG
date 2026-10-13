import { useMemo, useState } from "react";
import ProvinceTiles from "../components/ProvinceTiles.jsx";
import { JurisdictionLegend, Segmented, SourceLinks, SplitBar } from "../components/primitives.jsx";
import { EXTRA_TILES, GROUP_COLOR, GROUP_ORDER, GROUP_SHORT, jurisdictionGroup } from "../lib/jurisdiction.js";
import { groupCounts, runDocuments } from "../lib/stats.js";

const TYPES = [
  { value: "all", label: "All" },
  { value: "case", label: "Cases" },
  { value: "statute", label: "Statutes" },
];

// Where the sources come from (tile map, click to filter) + the document list.
export default function CasesTab({ run, province, provinceLabel, openEvidence }) {
  const docs = runDocuments(run);
  const [juris, setJuris] = useState(null);
  const [type, setType] = useState("all");
  const [openDoc, setOpenDoc] = useState(null);

  const perJuris = useMemo(() => {
    const m = {};
    docs.forEach((d) => (m[d.jurisdiction || "unknown"] = (m[d.jurisdiction || "unknown"] || 0) + 1));
    return m;
  }, [docs]);
  const groups = groupCounts(docs, province);
  const shown = docs.filter(
    (d) => (!juris || (d.jurisdiction || "unknown") === juris) && (type === "all" || (d.doc_type || "case") === type)
  );

  return (
    <div className="split">
      <section className="split-side">
        <div className="row baseline">
          <h2 className="lbl grow">Where sources come from · click to filter</h2>
          {juris && <button type="button" className="link" onClick={() => setJuris(null)}>Show all</button>}
        </div>
        <ProvinceTiles
          ariaLabel="Filter by jurisdiction"
          extra={EXTRA_TILES}
          onPick={(code) => perJuris[code] && setJuris(juris === code ? null : code)}
          tileProps={(code) => {
            const count = perJuris[code] || 0;
            const color = GROUP_COLOR[jurisdictionGroup(code, province)];
            const on = juris === code;
            return {
              count: count || "",
              on,
              dim: !count,
              bg: count ? (on ? "var(--ink)" : color) : "var(--tile-empty)",
              fg: on ? "var(--paper)" : jurisdictionGroup(code, province) === "federal" && count ? "#fff" : "var(--ink)",
              border: on ? "1px solid var(--ink)" : "1px solid transparent",
            };
          }}
        />
        <SplitBar parts={GROUP_ORDER.map((g) => ({ value: groups[g], color: GROUP_COLOR[g] }))} />
        <span className="small muted">
          {GROUP_ORDER.filter((g) => groups[g]).map((g) => `${groups[g]} ${g === "local" ? provinceLabel : GROUP_SHORT[g]}`).join(" · ")}
        </span>
        <JurisdictionLegend />
      </section>

      <section className="split-main">
        <div className="row wrap">
          <Segmented label="Document type" options={TYPES} value={type} onChange={setType} />
          <span className="spacer" />
          <span className="lbl">{shown.length} of {docs.length} shown</span>
        </div>
        <ul className="rows">
          {shown.map((d) => {
            const on = openDoc === d.doc_id;
            const color = GROUP_COLOR[jurisdictionGroup(d.jurisdiction, province)];
            const n = d.evidence_ids.length;
            return (
              <li key={d.doc_id} className={on ? "open" : ""}>
                <button type="button" className="rowlink doc-row" aria-expanded={on} onClick={() => setOpenDoc(on ? null : d.doc_id)}>
                  <i className="dot big" style={{ background: color }} />
                  <span className="doc-name">{d.case_name || d.citation}</span>
                  <span className="mono small muted doc-meta">{[d.citation, d.court_abbrev || d.court, d.year].filter(Boolean).join(" · ")}</span>
                  <span className="pips" aria-label={`${n} passages`}>
                    {Array.from({ length: Math.min(n, 5) }, (_, i) => <i key={i} style={{ background: color }} />)}
                  </span>
                </button>
                {on && (
                  <div className="doc-detail">
                    <span className="lbl">{d.doc_type === "statute" ? "Statute" : "Decision"} · {n} passage{n > 1 ? "s" : ""}{d.court ? ` · ${d.court}` : ""}</span>
                    <SourceLinks ids={d.evidence_ids} onOpen={openEvidence} />
                    <span className="spacer" />
                    <button type="button" className="btn primary small" onClick={() => openEvidence(d.evidence_ids[0])}>Read passages</button>
                    {d.url && <a className="btn ghost small" href={d.url} target="_blank" rel="noreferrer">CanLII ↗</a>}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
