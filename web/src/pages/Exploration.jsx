import { useEffect, useMemo, useState } from "react";
import ChatPanel from "../components/ChatPanel.jsx";
import TracePanel from "../components/TracePanel.jsx";
import { evidenceIndex } from "../lib/exploration.js";
import { runProgress } from "../lib/progress.js";
import { runCounts } from "../lib/stats.js";
import { TAB_LABEL, TABS } from "../lib/tabs.js";
import OverviewTab from "../tabs/OverviewTab.jsx";
import RelationsTab from "../tabs/RelationsTab.jsx";
import SourcesTab from "../tabs/SourcesTab.jsx";
import TimelineTab from "../tabs/TimelineTab.jsx";
import WhatIfTab from "../tabs/WhatIfTab.jsx";

// Ask-panel suggestions per tab; `sel` = what is selected there (if anything)
const SUGGESTIONS = {
  overview: () => ["Explain the summary simply", "What is still unclear?"],
  whatif: (sel) => (sel ? ["What if this is not true?", "Which sources discuss this?"] : ["Which facts matter most?", "What facts are missing?"]),
  sources: (sel) => ["Summarise this document", sel ? `Explain passage ${sel.split(" · ").pop()}` : "Which source is most relevant?"],
  timeline: (sel) => ["How has the law changed over time?", sel ? `Summarise the ${sel.split(" · ")[0]} decision` : "Which decision is most recent?"],
  relations: () => ["Explain this in plain words", "Walk me through it step by step"],
};

// One exploration, split into tabs.  Every tab reads the same run state
// (lib/exploration.js); "source" links jump to the Sources tab's reader.
// The Ask panel (chat) sits beside every tab and follows what you look at.
export default function Exploration({ run, profile, options, userProvince, renderHeader, onExplore, updateRun }) {
  const [tab, setTab] = useState("overview");
  const [focusId, setFocusId] = useState(null); // evidence id shown in the Sources reader
  const [selections, setSelections] = useState({}); // tab → label of the selected item
  const [showTrace, setShowTrace] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [now, setNow] = useState(Date.now());

  // Re-render while building so progress estimates move
  useEffect(() => {
    if (run.done) return;
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, [run.done]);

  const progress = runProgress(run, now);
  const counts = runCounts(run);
  const evidenceById = useMemo(() => evidenceIndex(run), [run.evidence]); // eslint-disable-line react-hooks/exhaustive-deps
  // Colours are relative to the province this exploration focused on
  // (a demo's own province, else the user's).
  const province = run.province || userProvince;
  const provinceLabel = options.provinces.find((p) => p.code === province)?.label || null;

  function openTab(id) {
    setTab(id);
    window.scrollTo({ top: 0 });
  }
  function openEvidence(id) {
    if (!evidenceById[id]) return;
    setFocusId(id);
    openTab("sources");
  }
  const onLook = (label) => setSelections((s) => (s[tab] === label ? s : { ...s, [tab]: label }));

  const focus = evidenceById[focusId] || run.evidence.find((e) => e.kind === "passage");
  const sel = tab === "sources" && focus ? `${focus.case_name || focus.citation} · ${focus.evidence_id}` : selections[tab];
  const context = `${TAB_LABEL[tab]} · ${sel || (tab === "overview" ? "your question" : "the whole tab")}`;
  const asked = run.chat.threads.reduce((n, t) => n + t.msgs.filter((m) => m.role === "user").length, 0);

  const tabs = (
    <nav className="tabs" aria-label="Exploration views">
      {TABS.map((t) => {
        const ready = progress.ready[t.id];
        return (
          <button
            key={t.id}
            type="button"
            className={`tab ${tab === t.id ? "on" : ""}`}
            aria-current={tab === t.id ? "page" : undefined}
            disabled={!ready}
            title={ready ? undefined : "Still building…"}
            onClick={() => openTab(t.id)}
          >
            {t.label}
            {t.count && counts[t.count] > 0 && <sup>{counts[t.count]}</sup>}
          </button>
        );
      })}
    </nav>
  );
  const askButton = (
    <button type="button" className={`pill ask-btn ${chatOpen ? "on" : ""}`} aria-expanded={chatOpen} onClick={() => setChatOpen((o) => !o)}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M4 5h16v11H9l-5 4z" />
      </svg>
      Ask{asked > 0 && <span className="mono small">{asked}</span>}
    </button>
  );

  const shared = {
    run, province, provinceLabel, progress, openEvidence, onExplore, onLook,
    setTab: openTab,
    onTrace: run.traceId ? () => setShowTrace(true) : null,
    setAnswer: (factId, value) => updateRun((r) => ({ ...r, answers: { ...r.answers, [factId]: value } })),
    canWiden: !!userProvince,
  };

  return (
    <div className="page">
      {renderHeader(tabs, askButton)}
      {!run.done && (
        <div className="progress run-progress" role="progressbar" aria-label="Exploration progress" aria-valuenow={progress.pct} aria-valuemin={0} aria-valuemax={100}>
          <i style={{ width: `${progress.pct}%` }} />
        </div>
      )}
      <div className="explore-body">
        <div className="explore-main">
          {run.errors.length > 0 && (
            <div className="band-errors">
              {run.errors.map((e, i) => (
                <p key={i} className="error small">{e}</p>
              ))}
            </div>
          )}
          <main>
            {tab === "overview" && <OverviewTab {...shared} onAsk={() => setChatOpen(true)} />}
            {tab === "whatif" && <WhatIfTab {...shared} />}
            {tab === "sources" && <SourcesTab {...shared} focusId={focus?.evidence_id} setFocusId={setFocusId} />}
            {tab === "timeline" && <TimelineTab {...shared} />}
            {tab === "relations" && <RelationsTab {...shared} />}
          </main>
          <footer className="footer lbl">Organises retrieved legal sources for exploration. Not legal advice.</footer>
        </div>
        {chatOpen && (
          <ChatPanel
            run={run}
            userId={profile.user_id}
            tab={tab}
            context={context}
            suggestions={SUGGESTIONS[tab](sel)}
            updateRun={updateRun}
            onOpenEvidence={openEvidence}
            onGoTab={openTab}
            onClose={() => setChatOpen(false)}
          />
        )}
      </div>
      {showTrace && run.traceId && (
        <TracePanel traceId={run.traceId} live={!run.done} onClose={() => setShowTrace(false)} />
      )}
    </div>
  );
}
