import { useState } from "react";
import { Dot, Skeleton, SplitBar } from "../components/primitives.jsx";
import ProgressPanel from "../components/ProgressPanel.jsx";
import { FACT_STATUS, factIsOff, runFacts } from "../lib/facts.js";
import { evidenceIndex } from "../lib/exploration.js";
import { GROUP_COLOR, GROUP_ORDER, groupLabel, jurisdictionGroup } from "../lib/jurisdiction.js";
import { groupCounts, runDocuments, runPassages, yearSpan } from "../lib/stats.js";

const ANSWERS = [
  ["yes", "True"],
  ["no", "Not true"],
  ["unsure", "Not sure"],
];

// Overview: what was asked, what the sources say (topic, orientation, the most
// relevant sources), facts to narrow it down, previews of the other tabs and
// follow-up questions.  While building, a progress panel sits on top.
export default function OverviewTab(props) {
  const { run, province, provinceLabel, progress, setTab, onExplore } = props;
  const docs = runDocuments(run);
  const passages = runPassages(run);
  const groups = groupCounts(docs, province);
  const main = GROUP_ORDER.filter((g) => groups[g]).sort((a, b) => groups[b] - groups[a])[0];
  const span = yearSpan(docs);

  return (
    <>
      <AskedBand {...props} />
      {!run.done ? (
        <ProgressPanel progress={progress} onOpen={setTab} />
      ) : (
        <DoneLine {...props} />
      )}

      <section className="brief">
        <div className="brief-main">
          <span className="lbl">What the sources say</span>
          <h1 className="disp h-xl">{run.plan?.topic || (run.done ? "Exploration" : "Exploring…")}</h1>
          {run.plan ? (
            <p className="lead">{run.plan.orientation}</p>
          ) : run.viewErrors.plan ? (
            <p className="muted">No summary for this run ({run.viewErrors.plan}).</p>
          ) : (
            <Skeleton />
          )}
          {docs.length > 0 && (
            <p className="small muted">
              Based on {passages.length} passages from {docs.length} documents
              {main && ` · mostly ${groupLabel(main, provinceLabel)}`}
              {span && ` · ${span.min === span.max ? span.min : `${span.min}–${span.max}`}`}
            </p>
          )}
          <TopSources {...props} docs={docs} />
          {passages.length > 0 && (
            <button type="button" className="btn ghost small ask-cta" onClick={props.onAsk}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="M4 5h16v11H9l-5 4z" />
              </svg>
              Ask a question about these sources
            </button>
          )}
        </div>
        <NarrowDown {...props} />
      </section>

      <KeepExploring {...props} docs={docs} groups={groups} span={span} />

      {run.plan?.follow_up_questions?.length > 0 && (
        <section className="band">
          <h2 className="lbl band-label">You might also ask</h2>
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

function AskedBand({ run, provinceLabel, onExplore, canWiden, onTrace }) {
  const [full, setFull] = useState(false);
  const juris = run.query?.jurisdictions;
  const long = run.question.length > 240;
  return (
    <section className="asked">
      <span className="lbl">You asked</span>
      <div className="asked-body">
        <p className={`asked-q ${full || !long ? "" : "clamp-3"}`}>{run.question}</p>
        {long && (
          <button type="button" className="link small" onClick={() => setFull((f) => !f)}>{full ? "Show less" : "Show all"}</button>
        )}
        <div className="asked-meta">
          {run.query && (
            juris ? (
              <span className="inline">
                <Dot color={GROUP_COLOR.local} />{provinceLabel || juris[0]}
                <Dot color={GROUP_COLOR.federal} />Federal
              </span>
            ) : (
              <span>{run.query.relaxed ? "Few local sources, so all of Canada" : "All of Canada"}</span>
            )
          )}
          {run.done && canWiden && (
            <button type="button" className="link" onClick={() => onExplore(run.question, run.scope === "all" ? "profile" : "all")}>
              {run.scope === "all" ? "Focus on my province" : "Widen to all of Canada"}
            </button>
          )}
          {onTrace && <button type="button" className="link" onClick={onTrace}>Under the hood</button>}
        </div>
      </div>
    </section>
  );
}

function DoneLine({ run, progress, onExplore }) {
  const views = run.viewOrder.length;
  return (
    <div className="done-line">
      <span className="done-tick" aria-hidden="true">✓</span>
      {run.demo ? (
        <>
          <span>
            Example, prepared in advance · {views} views from {run.evidence.length - 1} passages · built in {Math.round(run.demo.totalS)} s with {run.demo.model}
          </span>
          <button type="button" className="link" onClick={() => onExplore(run.question)}>Run it live for my profile</button>
        </>
      ) : (
        <span>
          {run.stage === "error" ? "Stopped" : "All views ready"} · {views} views from {Math.max(0, run.evidence.length - 1)} passages · built in {progress.elapsed} s
        </span>
      )}
    </div>
  );
}

function TopSources({ run, docs, province, openEvidence, setTab }) {
  const byId = evidenceIndex(run);
  if (!docs.length) return run.done ? null : <Skeleton widths={[100, 100, 100]} />;
  return (
    <div className="top-sources">
      <div className="row baseline">
        <h2 className="lbl grow">Most relevant sources</h2>
        <button type="button" className="link small" onClick={() => setTab("sources")}>All {docs.length} →</button>
      </div>
      <ol className="rows">
        {docs.slice(0, 3).map((d) => {
          const first = byId[d.evidence_ids[0]];
          return (
            <li key={d.doc_id}>
              <button type="button" className="rowlink top-source" onClick={() => openEvidence(d.evidence_ids[0])}>
                <i className="dot big" style={{ background: GROUP_COLOR[jurisdictionGroup(d.jurisdiction, province)] }} />
                <span className="top-source-body">
                  <span className="top-source-name">{d.case_name || d.citation}</span>
                  <span className="mono small muted">
                    {[d.doc_type === "statute" ? "Statute" : "Decision", d.citation, d.court_abbrev || d.court, d.year].filter(Boolean).join(" · ")}
                  </span>
                  {first && <span className="clamp-2 small top-source-snippet">{first.text}</span>}
                </span>
                <span className="go" aria-hidden="true">→</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// Facts the user didn't state: answer them here (synced with the What-if tab).
function NarrowDown({ run, setAnswer, setTab }) {
  const facts = runFacts(run);
  const open = facts.filter((f) => f.status !== "stated");
  const answered = open.filter((f) => run.answers[f.id]).length;
  return (
    <aside className="narrow">
      <div className="row baseline">
        <span className="lbl grow">Help narrow it down</span>
        {open.length > 0 && <span className="mono small muted">{answered} of {open.length} answered</span>}
      </div>
      {!run.views.fact_toggle ? (
        run.viewErrors.fact_toggle ? (
          <p className="small muted">The facts couldn't be built: {run.viewErrors.fact_toggle}</p>
        ) : (
          <Skeleton widths={[90, 70, 84, 60]} />
        )
      ) : !open.length ? (
        <p className="muted">Your question already covers the facts these sources turn on.</p>
      ) : (
        <ul className="narrow-list">
          {open.slice(0, 4).map((f) => {
            const st = FACT_STATUS[f.status] || FACT_STATUS.assumed;
            const a = run.answers[f.id];
            return (
              <li key={f.id}>
                <span className="lbl inline"><Dot color={st.color} />{st.label}</span>
                <span className="narrow-q">{f.text}</span>
                <div className="answers" role="group" aria-label={f.text}>
                  {ANSWERS.map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      className={`ans ${a === key ? "on" : ""}`}
                      aria-pressed={a === key}
                      onClick={() => setAnswer(f.id, a === key ? undefined : key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                {a && a !== "unsure" && (
                  <p className="small narrow-effect">
                    {factIsOff(f, run.answers) ? <><b className="accent">If not → </b>{f.if_changed}</> : f.why_it_matters}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {facts.length > 0 && (
        <button type="button" className="link small" onClick={() => setTab("whatif")}>
          See all {facts.length} facts in What-if →
        </button>
      )}
    </aside>
  );
}

// Preview cards for the other tabs (locked while their stage is building).
function KeepExploring({ run, progress, setTab, docs, groups, span, province }) {
  const facts = runFacts(run);
  const unknown = facts.filter((f) => f.status === "unknown").length;
  const cases = docs.filter((d) => d.doc_type !== "statute").length;
  const maps = (run.plan?.views || []).map((v) => v.title);
  const cards = [
    {
      id: "whatif",
      title: "What-if",
      note: facts.length ? `${facts.length} facts · ${unknown} not mentioned` : null,
      art: <FactsArt facts={facts} answers={run.answers} />,
    },
    {
      id: "sources",
      title: "Sources",
      note: docs.length ? `${docs.length} documents · ${cases} cases · ${docs.length - cases} statutes` : null,
      art: <SplitBar height={18} parts={GROUP_ORDER.map((g) => ({ value: groups[g], color: GROUP_COLOR[g] }))} />,
    },
    {
      id: "timeline",
      title: "Timeline",
      note: span ? `${span.min}–${span.max} · ${span.recent} of ${span.total} since ${span.recentFrom}` : null,
      art: <TimelineArt docs={docs} span={span} province={province} />,
    },
    {
      id: "relations",
      title: "Relations",
      note: run.plan ? (maps.length ? maps.join(" · ") : "No extra maps needed here") : null,
      art: <RelationsArt n={maps.length} />,
    },
  ];
  return (
    <section className="band">
      <h2 className="lbl band-label">Keep exploring</h2>
      <div className="cards">
        {cards.map((c) => {
          const ready = progress.ready[c.id];
          return (
            <button key={c.id} type="button" className="card" disabled={!ready} onClick={() => setTab(c.id)}>
              <span className="card-art" aria-hidden="true">{ready && c.note ? c.art : <Skeleton widths={[70, 45]} />}</span>
              <span className="disp h-s">{c.title}</span>
              <span className="small muted">{ready ? c.note || "Nothing to show" : "Building…"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function FactsArt({ facts, answers }) {
  return (
    <span className="facts-art">
      {facts.slice(0, 8).map((f) => (
        <i key={f.id} className={factIsOff(f, answers) ? "off" : ""} />
      ))}
    </span>
  );
}

function TimelineArt({ docs, span, province }) {
  if (!span) return null;
  const per = {};
  return (
    <svg viewBox="0 0 200 44" className="card-svg">
      <line x1="0" y1="40" x2="200" y2="40" stroke="var(--ink)" strokeWidth="1" />
      {docs.filter((d) => d.year).map((d) => {
        const x = 8 + ((d.year - span.min) / Math.max(1, span.max - span.min)) * 184;
        const k = Math.round(x / 10);
        const level = (per[k] = (per[k] || 0) + 1) - 1;
        return level < 4 ? (
          <circle key={d.doc_id} cx={x} cy={33 - level * 9} r="4" fill={GROUP_COLOR[jurisdictionGroup(d.jurisdiction, province)]} />
        ) : null;
      })}
    </svg>
  );
}

function RelationsArt({ n }) {
  const spokes = [[30, 10], [28, 36], [170, 8], [172, 34], [100, 4]].slice(0, Math.max(2, n + 2));
  return (
    <svg viewBox="0 0 200 44" className="card-svg">
      {spokes.map(([x, y], i) => <line key={i} x1="100" y1="22" x2={x} y2={y} className="ln" />)}
      <circle cx="100" cy="22" r="9" fill="var(--local)" />
      {spokes.map(([x, y], i) => <circle key={`c${i}`} cx={x} cy={y} r="4.5" fill={i % 2 ? "var(--federal)" : "var(--ink)"} />)}
    </svg>
  );
}
