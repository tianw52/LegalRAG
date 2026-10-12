import { useEffect, useMemo, useRef, useState } from "react";
import { api, exploreStream } from "../api.js";
import QuestionBox from "./QuestionBox.jsx";
import EvidencePanel from "./EvidencePanel.jsx";
import DocumentReader from "./DocumentReader.jsx";
import CaseLandscape from "../views/CaseLandscape.jsx";
import CaseTimeline from "../views/CaseTimeline.jsx";
import FactToggle from "../views/FactToggle.jsx";
import AdhocView from "../views/AdhocView.jsx";

const EMPTY_RUN = {
  stage: "",
  message: "",
  query: null,
  evidence: [],
  views: {},
  viewOrder: [],
  plan: null,
  errors: [],
  done: false,
};

// Folds one NDJSON event from /api/explore into the exploration state.
function applyEvent(run, ev) {
  switch (ev.event) {
    case "status":
      return { ...run, stage: ev.stage, message: ev.message };
    case "query":
      return { ...run, query: ev };
    case "evidence":
      return { ...run, evidence: ev.items };
    case "view": {
      const v = ev.view;
      const viewOrder = run.viewOrder.includes(v.view_id) ? run.viewOrder : [...run.viewOrder, v.view_id];
      return { ...run, views: { ...run.views, [v.view_id]: v }, viewOrder };
    }
    case "plan":
      return { ...run, plan: ev };
    case "view_error":
      return { ...run, errors: [...run.errors, `${ev.view_id}: ${ev.message}`] };
    case "error":
      return { ...run, errors: [...run.errors, ev.message], done: true, stage: "error" };
    case "done":
      return { ...run, done: true, stage: "done", message: "", timings: ev.timings };
    default:
      return run;
  }
}

export default function Workspace({ profile, options, onEditProfile, onLogout }) {
  const [runs, setRuns] = useState([]); // exploration trail, newest last
  const [activeId, setActiveId] = useState(null);
  const [selection, setSelection] = useState(null); // {label, detail, evidenceIds}
  const [reader, setReader] = useState(null); // {docId, focusIds}
  const [examples, setExamples] = useState([]);
  const abortRef = useRef(null);

  const provinceLabel =
    options.provinces.find((p) => p.code === profile.province)?.label || "Canada";
  const active = runs.find((r) => r.id === activeId) || null;

  useEffect(() => {
    api.examples(profile.province).then(setExamples).catch(() => setExamples([]));
  }, [profile.province]);

  const evidenceById = useMemo(() => {
    const map = {};
    (active?.evidence || []).forEach((e) => (map[e.evidence_id] = e));
    return map;
  }, [active]);

  async function explore(question, scope = "profile") {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const id = `${Date.now()}`;
    setRuns((rs) => [...rs, { id, question, scope, ...EMPTY_RUN }]);
    setActiveId(id);
    setSelection(null);
    const update = (fn) => setRuns((rs) => rs.map((r) => (r.id === id ? fn(r) : r)));
    try {
      await exploreStream(
        { question, userId: profile.user_id, scope },
        (ev) => update((r) => applyEvent(r, ev)),
        controller.signal
      );
    } catch (e) {
      if (e.name !== "AbortError") update((r) => applyEvent(r, { event: "error", message: e.message }));
    }
  }

  function select(sel) {
    setSelection(sel && sel.evidenceIds?.length ? sel : null);
  }

  function openDocument(docId, focusIds = []) {
    setReader({ docId, focusIds });
  }

  const ordered = (active?.viewOrder || []).map((id) => active.views[id]);
  const landscape = active?.views.case_landscape;
  const timeline = active?.views.case_timeline;
  const facts = active?.views.fact_toggle;
  const adhoc = ordered.filter((v) => v.source === "agent");

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand small">LegalRAG Explore</span>
        <span className="spacer" />
        <span className="profile-chip" title="Your profile">
          {profile.display_name} · {provinceLabel}
        </span>
        <button className="btn ghost" onClick={onEditProfile}>
          Edit profile
        </button>
        <button className="btn ghost" onClick={onLogout}>
          Log out
        </button>
      </header>

      <div className="workspace">
        <aside className="trail">
          <div className="lbl">Your explorations</div>
          {runs.length === 0 && <p className="muted small">Questions you explore appear here.</p>}
          {runs
            .slice()
            .reverse()
            .map((r) => (
              <button
                key={r.id}
                className={`trail-item ${r.id === activeId ? "on" : ""}`}
                onClick={() => {
                  setActiveId(r.id);
                  setSelection(null);
                }}
                title={r.question}
              >
                <span className="trail-topic">{r.plan?.topic || "Exploring…"}</span>
                <span className="trail-q">{r.question}</span>
              </button>
            ))}
        </aside>

        <main className="main">
          <QuestionBox
            onSubmit={(q) => explore(q)}
            busy={active && !active.done}
            examples={runs.length === 0 ? examples : []}
          />

          {active && (
            <>
              <RunHeader run={active} provinceLabel={provinceLabel} onScope={(s) => explore(active.question, s)} />

              {active.plan && (
                <section className="orientation card">
                  <div className="lbl">{active.plan.topic}</div>
                  <p>{active.plan.orientation}</p>
                </section>
              )}

              {active.errors.length > 0 && (
                <div className="error">
                  {active.errors.map((e, i) => (
                    <div key={i}>{e}</div>
                  ))}
                </div>
              )}

              {facts ? (
                <FactToggle view={facts} selection={selection} onSelect={select} />
              ) : (
                !active.done && active.evidence.length > 0 && <Placeholder title="Facts that matter" />
              )}

              {adhoc.map((v) => (
                <AdhocView key={v.view_id} view={v} evidenceById={evidenceById} onSelect={select} />
              ))}
              {!active.done && active.plan && adhoc.length < active.plan.views.length &&
                active.plan.views.slice(adhoc.length).map((pv, i) => (
                  <Placeholder key={i} title={pv.title} />
                ))}

              {landscape && (
                <CaseLandscape
                  view={landscape}
                  province={profile.province}
                  selection={selection}
                  onSelect={select}
                  onOpen={openDocument}
                />
              )}
              {timeline && (
                <CaseTimeline
                  view={timeline}
                  province={profile.province}
                  selection={selection}
                  onSelect={select}
                />
              )}

              {active.plan?.follow_up_questions?.length > 0 && (
                <section className="card followups">
                  <div className="lbl">Explore next</div>
                  <div className="chips">
                    {active.plan.follow_up_questions.map((q) => (
                      <button key={q} className="chip" onClick={() => explore(q)}>
                        {q}
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </main>

        <aside className="evidence-col">
          <EvidencePanel
            evidence={active?.evidence || []}
            selection={selection}
            onClear={() => setSelection(null)}
            onOpen={openDocument}
          />
        </aside>
      </div>

      {reader && (
        <DocumentReader
          docId={reader.docId}
          evidence={(active?.evidence || []).filter((e) => e.doc_id === reader.docId)}
          focusIds={reader.focusIds}
          onClose={() => setReader(null)}
        />
      )}

      <footer className="disclaimer">
        This tool organises retrieved legal sources for exploration. It is not legal advice.
      </footer>
    </div>
  );
}

function RunHeader({ run, provinceLabel, onScope }) {
  const q = run.query;
  return (
    <section className="run-header">
      {!run.done && <div className="status">{run.message || "Working…"}</div>}
      {q && (
        <div className="row wrap">
          <span className="pill">
            {q.jurisdictions
              ? `Sources: ${provinceLabel} + federal`
              : q.relaxed
              ? "Few local sources — showing all of Canada"
              : "Sources: all of Canada"}
          </span>
          {run.done && (
            <button
              className="btn ghost small"
              onClick={() => onScope(run.scope === "all" ? "profile" : "all")}
            >
              {run.scope === "all" ? `Focus on ${provinceLabel}` : "Search all of Canada"}
            </button>
          )}
          {q.keywords?.length > 0 && (
            <span className="muted small">Key terms: {q.keywords.join(", ")}</span>
          )}
        </div>
      )}
    </section>
  );
}

function Placeholder({ title }) {
  return (
    <section className="card placeholder">
      <div className="lbl">{title}</div>
      <div className="shimmer" />
    </section>
  );
}
