import { relevanceStatistics } from "../lib/relevanceStatistics.js";
import { Dot } from "./primitives.jsx";
import "./RelevanceStatistics.css";

const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });

function JurisdictionDistribution({ jurisdictions, total }) {
  let start = 0;
  const segments = jurisdictions.map(({ count, color }) => {
    const end = start + count;
    const gap = Math.min(0.3, count / total * 20);
    const fillEnd = end / total * 100 - gap;
    const segment = `${color} ${start / total * 100}% ${fillEnd}%`;
    start = end;
    return `${segment}, var(--paper) ${fillEnd}% ${end / total * 100}%`;
  });
  const background = jurisdictions.length === 1
    ? jurisdictions[0].color
    : `conic-gradient(${segments.join(", ")})`;

  return (
    <section>
      <h2 className="disp h-m">Jurisdiction Distribution</h2>
      {total === 0 ? <p className="small muted">No cases retrieved.</p> : (
        <div className="relevance-jurisdiction-chart">
          <div className="relevance-donut" style={{ background }}>
            <div><strong>{total}</strong><span className="lbl">{total === 1 ? "Case" : "Cases"}</span></div>
          </div>
          <ul className="rows relevance-jurisdiction-legend" aria-label="Cases by jurisdiction">
            {jurisdictions.map(({ code, label, color, count }) => (
              <li key={code}>
                <span className="inline"><Dot color={color} />{label}</span>
                <span className="mono small">{count} · {percent.format(count / total)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function CategoryDistribution({ categories, total }) {
  return (
    <section>
      <h2 className="disp h-m">Cases by Legal Category</h2>
      {categories.length === 0 ? <p className="small muted">No relevance categories available for this question.</p> : (
        <>
          <p className="small muted">Based on the existing prototype’s passage phrase matches.</p>
          <ul className="relevance-category-bars" aria-label="Unique case count by legal category">
            {categories.map(({ id, label, count, color }) => (
              <li key={id}>
                <div className="row spread">
                  <span>{label}</span><span className="mono small">{count}</span>
                </div>
                <div className="relevance-bar-track" aria-hidden="true">
                  <div style={{ background: color, width: `${total ? count / total * 100 : 0}%` }} />
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

export default function RelevanceStatistics({ cases, categories, province }) {
  const stats = relevanceStatistics({ cases, categories, province });
  return (
    <div className="relevance-statistics">
      <JurisdictionDistribution jurisdictions={stats.jurisdictions} total={stats.total} />
      <CategoryDistribution categories={stats.categories} total={stats.total} />
    </div>
  );
}
