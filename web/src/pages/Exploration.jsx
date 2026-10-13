import { useState } from "react";
import { Dot, Pending } from "../components/primitives.jsx";
import TracePanel from "../components/TracePanel.jsx";
import { GROUP_COLOR } from "../lib/jurisdiction.js";
import { runCounts } from "../lib/stats.js";
import OverviewTab from "../tabs/OverviewTab.jsx";
import WhatIfTab from "../tabs/WhatIfTab.jsx";
import CasesTab from "../tabs/CasesTab.jsx";
import TimelineTab from "../tabs/TimelineTab.jsx";
import MapsTab from "../tabs/MapsTab.jsx";
import SourcesTab from "../tabs/SourcesTab.jsx";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "whatif", label: "What-if", count: "facts" },
  { id: "cases", label: "Cases", count: "documents" },
  { id: "timeline", label: "Timeline" },
  { id: "maps", label: "Maps", count: "maps" },
  { id: "sources", label: "Sources", count: "passages" },
];

// One exploration, split into tabs.  Every tab reads the same run state
// (lib/exploration.js); "source" links jump to the Sources tab's reader.
export default function Exploration({ run, profile, provinceLabel, allCanada, renderHeader, onExplore }) {
  const [tab, setTab] = useState("overview");
  const [focusId, setFocusId] = useState(null); // evidence id shown in the Sources reader
  const [showTrace, setShowTrace] = useState(false);
  const counts = runCounts(run);

  function openEvidence(id) {
    setFocusId(id);
    setTab("sources");
    window.scrollTo({ top: 0 });
  }

  const shared = { run, province: profile.province, provinceLabel, openEvidence, setTab, onExplore };

  const tabs = (
    <nav className="tabs" aria-label="Exploration views">
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`tab ${tab === t.id ? "on" : ""}`}
          aria-current={tab === t.id ? "page" : undefined}
          onClick={() => setTab(t.id)}
        >
          {t.label}
          {t.count && counts[t.count] > 0 && <sup>{counts[t.count]}</sup>}
        </button>
      ))}
    </nav>
  );

  return (
    <div className="page">
      {renderHeader(tabs)}
      <TitleBand
        run={run}
        provinceLabel={provinceLabel}
        onScope={allCanada ? null : (s) => onExplore(run.question, s)}
        onTrace={() => setShowTrace(true)}
      />
      {run.errors.length > 0 && (
        <div className="band-errors">
          {run.errors.map((e, i) => (
            <p key={i} className="error small">{e}</p>
          ))}
        </div>
      )}
      <main>
        {tab === "overview" && <OverviewTab {...shared} />}
        {tab === "whatif" && <WhatIfTab {...shared} />}
        {tab === "cases" && <CasesTab {...shared} />}
        {tab === "timeline" && <TimelineTab {...shared} />}
        {tab === "maps" && <MapsTab {...shared} />}
        {tab === "sources" && <SourcesTab {...shared} focusId={focusId} setFocusId={setFocusId} />}
      </main>
      <footer className="footer lbl">Organises retrieved legal sources for exploration. Not legal advice.</footer>
      {showTrace && run.traceId && (
        <TracePanel traceId={run.traceId} live={!run.done} onClose={() => setShowTrace(false)} />
      )}
    </div>
  );
}

function TitleBand({ run, provinceLabel, onScope, onTrace }) {
  const q = run.query;
  const views = run.viewOrder.length;
  const built = q
    ? `Understood → ${q.retrieved ?? "?"} retrieved → ${run.evidence.length ? run.evidence.length - 1 : 0} kept → ${views} views`
    : "";
  return (
    <section className="title-band">
      <h1 className="disp h-xl">{run.plan?.topic || (run.done ? "Exploration" : "Exploring…")}</h1>
      <dl className="kv">
        <dt className="lbl">Question</dt>
        <dd className="clamp-3">{run.question}</dd>
        <dt className="lbl">Scope</dt>
        <dd>
          {q?.jurisdictions ? (
            <span className="inline">
              <Dot color={GROUP_COLOR.local} />{provinceLabel}
              <Dot color={GROUP_COLOR.federal} />Federal
            </span>
          ) : (
            <span>{q?.relaxed ? "Few local sources — all of Canada" : "All of Canada"}</span>
          )}
          {run.done && onScope && (
            <>
              {" · "}
              <button type="button" className="link" onClick={() => onScope(run.scope === "all" ? "profile" : "all")}>
                {run.scope === "all" ? `focus on ${provinceLabel}` : "all of Canada"}
              </button>
            </>
          )}
        </dd>
        <dt className="lbl">Built</dt>
        <dd>
          {run.done ? built : <Pending label={run.message || "Working…"} />}
          {run.traceId && (
            <>
              {" · "}
              <button type="button" className="link" onClick={onTrace}>under the hood</button>
            </>
          )}
        </dd>
      </dl>
    </section>
  );
}
