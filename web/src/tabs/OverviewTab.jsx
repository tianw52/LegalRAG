import { Pending, SplitBar } from "../components/primitives.jsx";
import { GROUP_COLOR, GROUP_ORDER, GROUP_SHORT } from "../lib/jurisdiction.js";
import { groupCounts, runCounts, runDocuments, runPassages, yearHistogram, yearSpan } from "../lib/stats.js";

// Orientation, key numbers, a list of the other tabs, and follow-up questions.
export default function OverviewTab({ run, province, provinceLabel, setTab, onExplore }) {
  const docs = runDocuments(run);
  const passages = runPassages(run);
  const counts = runCounts(run);
  const groups = groupCounts(docs, province);
  const span = yearSpan(docs);
  const cases = docs.filter((d) => d.doc_type !== "statute").length;
  const mapKinds = (run.plan?.views || []).map((v) => v.title);

  return (
    <>
      <section className="band">
        <h2 className="lbl band-label">In short</h2>
        <div className="band-body">
          {run.plan ? (
            <>
              <p className="lead">{run.plan.orientation}</p>
              {run.plan.extra_views_reason && <p className="small muted">{run.plan.extra_views_reason}</p>}
            </>
          ) : (
            <Pending label={run.done ? "No orientation available." : "Reading the sources…"} />
          )}
        </div>
      </section>

      {docs.length > 0 && (
        <section className="stats">
          <Stat label="Passages" value={passages.length} note="ranked by relevance">
            <ScoreBars scores={passages.map((p) => p.rerank_score ?? 0)} />
          </Stat>
          <Stat label="Documents" value={docs.length} note={`${cases} cases · ${docs.length - cases} statutes`}>
            <SplitBar maxWidth={220} parts={[{ value: cases, color: "var(--ink)" }, { value: docs.length - cases, color: GROUP_COLOR.other }]} />
          </Stat>
          <Stat
            label="Jurisdictions"
            value={new Set(docs.map((d) => d.jurisdiction)).size}
            note={GROUP_ORDER.filter((g) => groups[g]).map((g) => `${groups[g]} ${g === "local" ? provinceLabel : GROUP_SHORT[g]}`).join(" · ")}
          >
            <SplitBar maxWidth={220} parts={GROUP_ORDER.map((g) => ({ value: groups[g], color: GROUP_COLOR[g] }))} />
          </Stat>
          {span && (
            <Stat
              label="Span"
              value={span.min === span.max ? span.min : `${span.min}–${String(span.max).slice(-2)}`}
              note={`${span.recent} of ${span.total} since ${span.recentFrom}`}
            >
              <Spark values={yearHistogram(docs)} />
            </Stat>
          )}
        </section>
      )}

      <section className="band">
        <h2 className="lbl band-label">Views</h2>
        <ol className="rows">
          <ViewRow n={1} title="What-if" note={counts.facts ? `${counts.facts} facts · ${counts.factsUnknown} not mentioned` : "Building…"} onClick={() => setTab("whatif")} />
          <ViewRow n={2} title="Cases" note={`${counts.documents} documents by jurisdiction`} onClick={() => setTab("cases")} />
          <ViewRow n={3} title="Timeline" note={span ? `${span.min} to ${span.max}` : "—"} onClick={() => setTab("timeline")} />
          <ViewRow
            n={4}
            title="Maps"
            note={run.plan ? (mapKinds.length ? mapKinds.join(" · ") : "None needed for this question") : "Planning…"}
            onClick={() => setTab("maps")}
          />
          <ViewRow n={5} title="Sources" note={`${counts.passages} passages, read in context`} onClick={() => setTab("sources")} />
        </ol>
      </section>

      {run.plan?.follow_up_questions?.length > 0 && (
        <section className="band">
          <h2 className="lbl band-label">Explore next</h2>
          <ul className="rows">
            {run.plan.follow_up_questions.map((q) => (
              <li key={q}>
                <button type="button" className="rowlink follow" onClick={() => onExplore(q)}>
                  <span className="row-text">{q}</span>
                  <span className="go" aria-hidden="true">→</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function Stat({ label, value, note, children }) {
  return (
    <div className="stat">
      <span className="lbl">{label}</span>
      <span className="disp stat-value">{value}</span>
      {children}
      <span className="small muted">{note}</span>
    </div>
  );
}

function ViewRow({ n, title, note, onClick }) {
  return (
    <li>
      <button type="button" className="rowlink view-row" onClick={onClick}>
        <span className="num">{String(n).padStart(2, "0")}</span>
        <span className="disp h-m view-row-title">{title}</span>
        <span className="muted view-row-note">{note}</span>
        <span className="go" aria-hidden="true">→</span>
      </button>
    </li>
  );
}

function ScoreBars({ scores }) {
  const lo = Math.min(...scores);
  const range = Math.max(1e-9, Math.max(...scores) - lo);
  const w = 200 / Math.max(1, scores.length);
  return (
    <svg viewBox="0 0 200 28" className="spark" aria-hidden="true">
      {scores.map((s, i) => {
        const h = 4 + ((s - lo) / range) * 24;
        const color = i < scores.length / 4 ? "var(--ink)" : i < scores.length / 2 ? GROUP_COLOR.other : GROUP_COLOR.unknown;
        return <rect key={i} x={i * w} y={28 - h} width={Math.max(1, w - 2)} height={h} fill={color} />;
      })}
    </svg>
  );
}

function Spark({ values }) {
  const max = Math.max(1, ...values);
  const step = 200 / Math.max(1, values.length - 1);
  const points = values.map((v, i) => `${i * step},${26 - (v / max) * 23}`).join(" ");
  return (
    <svg viewBox="0 0 200 28" className="spark" aria-hidden="true">
      <polyline points={points} fill="none" stroke={GROUP_COLOR.federal} strokeWidth="2" />
    </svg>
  );
}
