import { ViewHead } from "./FactToggle.jsx";
import { jurisdictionGroup } from "./CaseLandscape.jsx";

const W = 800;
const PAD = 40;
const ROW = 18;

// Built-in view: retrieved decisions/statutes on a year axis, stacked per year.
export default function CaseTimeline({ view, province, selection, onSelect }) {
  const docs = view.spec.documents || [];
  if (!docs.length) return null;

  const years = docs.map((d) => d.year);
  const min = Math.min(...years);
  const max = Math.max(...years);
  const span = Math.max(1, max - min);
  const x = (y) => PAD + ((y - min) / span) * (W - 2 * PAD);

  const stack = {};
  const points = docs.map((d) => {
    const level = (stack[d.year] = (stack[d.year] || 0) + 1) - 1;
    return { d, cx: x(d.year), level };
  });
  const maxLevel = Math.max(...points.map((p) => p.level));
  const axisY = 30 + (maxLevel + 1) * ROW;
  const H = axisY + 36;
  const ticks = niceTicks(min, max);
  const selected = new Set(selection?.evidenceIds || []);

  return (
    <section className="card view">
      <ViewHead view={view} />
      <div className="legend row wrap small">
        <span><i className="dot g-local" /> Your province</span>
        <span><i className="dot g-federal" /> Federal</span>
        <span><i className="dot g-other" /> Other provinces</span>
        <span><i className="dot g-unknown" /> Unknown</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="timeline-svg" role="img" aria-label="Decisions by year">
        <line x1={PAD} x2={W - PAD} y1={axisY} y2={axisY} className="axis" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={axisY} y2={axisY + 5} className="axis" />
            <text x={x(t)} y={axisY + 20} textAnchor="middle" className="tick">
              {t}
            </text>
          </g>
        ))}
        {points.map(({ d, cx, level }) => {
          const hit = d.evidence_ids.some((id) => selected.has(id));
          return (
            <circle
              key={d.doc_id}
              cx={cx}
              cy={axisY - 12 - level * ROW}
              r={hit ? 8 : 6}
              className={`tl-dot g-${jurisdictionGroup(d.jurisdiction, province)} ${hit ? "hit" : ""}`}
              onClick={() =>
                onSelect({
                  label: d.case_name || d.citation,
                  detail: [d.citation, d.court, d.year].filter(Boolean).join(" · "),
                  evidenceIds: d.evidence_ids,
                })
              }
            >
              <title>{`${d.year} · ${d.case_name || d.citation}`}</title>
            </circle>
          );
        })}
      </svg>
    </section>
  );
}

function niceTicks(min, max) {
  const span = max - min;
  const step = span > 60 ? 20 : span > 30 ? 10 : span > 10 ? 5 : span > 4 ? 2 : 1;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(t);
  if (!ticks.length) ticks.push(min);
  return ticks;
}
