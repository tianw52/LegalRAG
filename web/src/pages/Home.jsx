import { useEffect, useState } from "react";
import { api } from "../api.js";
import { Dot, JurisdictionLegend } from "../components/primitives.jsx";
import { GROUP_COLOR, jurisdictionGroup, provinceCode } from "../lib/jurisdiction.js";
import { runCounts, runDocuments, yearSpan } from "../lib/stats.js";

// Home: ask a question, try a benchmark question, revisit this session's explorations.
export default function Home({ header, profile, provinceLabel, allCanada, runs, onExplore, onOpenRun }) {
  const [question, setQuestion] = useState("");
  const [examples, setExamples] = useState([]);

  useEffect(() => {
    api.examples(profile.province).then(setExamples).catch(() => setExamples([]));
  }, [profile.province]);

  function submit(e) {
    e.preventDefault();
    if (question.trim().length >= 3) onExplore(question.trim());
  }

  return (
    <div className="page">
      {header}
      <main>
        <section className="hero">
          <div className="hero-text">
            <span className="lbl">Canadian case law &amp; statutes, mapped</span>
            <h1 className="disp h-xxl">See the law around your question</h1>
            <form onSubmit={submit} className="ask">
              <label htmlFor="q" className="lbl">Your question</label>
              <div className="underline-input big">
                <textarea
                  id="q"
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) submit(e);
                  }}
                  placeholder="Describe a situation or ask about Canadian law"
                />
                <button className="go-btn" aria-label="Explore" disabled={question.trim().length < 3}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
                    <path d="M5 12h14M13 6l6 6-6 6" />
                  </svg>
                </button>
              </div>
              <div className="scope-line">
                {allCanada ? (
                  <span><Dot color={GROUP_COLOR.other} />All of Canada — every province + federal sources</span>
                ) : (
                  <>
                    <span><Dot color={GROUP_COLOR.local} />{provinceLabel}</span>
                    <span><Dot color={GROUP_COLOR.federal} />+ federal sources</span>
                  </>
                )}
              </div>
            </form>
          </div>
          <div className="hero-art">
            <HeroGraph />
            <JurisdictionLegend />
          </div>
        </section>

        {examples.length > 0 && (
          <section className="band">
            <h2 className="lbl band-label">Try a benchmark question</h2>
            <ol className="rows">
              {examples.map((ex, i) => {
                const code = provinceCode(ex.province);
                return (
                  <li key={ex.query_id}>
                    <button type="button" className="rowlink" onClick={() => onExplore(ex.text)}>
                      <span className="num">{String(i + 1).padStart(2, "0")}</span>
                      <span className="code" style={{ color: code === profile.province ? "var(--accent-ink)" : "var(--muted)" }}>
                        {code || "—"}
                      </span>
                      <span className="row-text clamp-2">{ex.text}</span>
                      <span className="go" aria-hidden="true">→</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </section>
        )}

        {runs.length > 0 && (
          <section className="band" id="library">
            <h2 className="lbl band-label">Your explorations · {runs.length}</h2>
            <div className="run-cards">
              {runs.slice().reverse().map((r) => (
                <RunCard key={r.id} run={r} province={profile.province} onOpen={() => onOpenRun(r.id)} />
              ))}
            </div>
          </section>
        )}
      </main>
      <footer className="footer lbl">Organises retrieved legal sources for exploration. Not legal advice.</footer>
    </div>
  );
}

function RunCard({ run, province, onOpen }) {
  const docs = runDocuments(run);
  const counts = runCounts(run);
  const span = yearSpan(docs);
  const views = run.viewOrder.length;
  return (
    <button type="button" className="run-card rowlink" onClick={onOpen}>
      <svg viewBox="0 0 300 100" className="run-card-art" aria-hidden="true">
        <line x1="0" y1="86" x2="300" y2="86" stroke="var(--ink)" strokeWidth="1" />
        {span &&
          stackDots(docs, span).map((d, i) => (
            <circle key={i} cx={d.x} cy={d.y} r={6} fill={GROUP_COLOR[jurisdictionGroup(d.jurisdiction, province)]} />
          ))}
      </svg>
      <span className="disp h-s">{run.plan?.topic || (run.done ? "Exploration" : "Exploring…")}</span>
      <span className="clamp-2 small muted">{run.question}</span>
      <span className="lbl">{counts.documents} sources · {views} views{run.done ? "" : " · running"}</span>
    </button>
  );
}

function stackDots(docs, span) {
  const perX = {};
  return docs
    .filter((d) => d.year)
    .map((d) => {
      const x = 15 + ((d.year - span.min) / Math.max(1, span.max - span.min)) * 270;
      const key = Math.round(x / 14);
      const level = (perX[key] = (perX[key] || 0) + 1) - 1;
      return { x, y: 72 - level * 16, jurisdiction: d.jurisdiction };
    })
    .filter((d) => d.y > 6);
}

// Decorative network: a question at the centre, sources around it.
function HeroGraph() {
  const hub = [260, 200];
  const spokes = [
    [150, 110, 16, "local"], [180, 290, 12, "local"], [380, 120, 18, "federal"],
    [400, 270, 14, "federal"], [290, 340, 11, "other"],
  ];
  const leaves = [
    [150, 110, 80, 60, 7, "other"], [150, 110, 60, 170, 8, "local"], [180, 290, 90, 320, 7, "unknown"],
    [380, 120, 460, 70, 8, "federal"], [380, 120, 300, 50, 6, "unknown"], [400, 270, 480, 220, 7, "other"],
    [400, 270, 460, 340, 9, "federal"], [290, 340, 200, 370, 6, "unknown"],
  ];
  return (
    <svg viewBox="0 0 520 400" className="hero-graph" aria-hidden="true">
      {spokes.map(([x, y], i) => (
        <line key={`s${i}`} x1={hub[0]} y1={hub[1]} x2={x} y2={y} className="ln-hot" />
      ))}
      {leaves.map(([x1, y1, x2, y2], i) => (
        <line key={`l${i}`} x1={x1} y1={y1} x2={x2} y2={y2} className="ln" />
      ))}
      <circle cx={hub[0]} cy={hub[1]} r="44" fill={GROUP_COLOR.local} />
      {spokes.map(([x, y, r, g], i) => (
        <circle key={`sc${i}`} cx={x} cy={y} r={r} fill={GROUP_COLOR[g]} />
      ))}
      {leaves.map(([, , x, y, r, g], i) => (
        <circle key={`lc${i}`} cx={x} cy={y} r={r} fill={GROUP_COLOR[g]} />
      ))}
    </svg>
  );
}
