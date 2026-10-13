// Client-side state for one exploration run, built by folding the NDJSON
// events from POST /api/explore (see legalrag/explore/engine.py for the protocol).

export const EMPTY_RUN = {
  traceId: null,   // GET /api/traces/{traceId} — "Under the hood"
  stage: "",       // formulating | retrieving | building | done | error
  message: "",     // human-readable status
  query: null,     // {reformulated, keywords, jurisdictions, relaxed, scope}
  evidence: [],    // [{evidence_id, kind, text, citation, ...}] — q + e1..eN
  views: {},       // view_id → View envelope
  viewOrder: [],   // view_ids in arrival order
  plan: null,      // {topic, orientation, views, follow_up_questions}
  errors: [],      // fatal error messages (formulator fallback, stream errors)
  viewErrors: {},  // view_id → message for views that failed or timed out
  done: false,
};

export function newRun(question, scope) {
  return { id: `${Date.now()}`, question, scope, ...EMPTY_RUN };
}

export function applyEvent(run, ev) {
  switch (ev.event) {
    case "trace":
      return { ...run, traceId: ev.trace_id };
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
      // Per-view failures are shown where the view would appear; the
      // formulator fallback is a run-level notice.
      return ev.view_id === "formulator"
        ? { ...run, errors: [...run.errors, ev.message] }
        : { ...run, viewErrors: { ...run.viewErrors, [ev.view_id]: ev.message } };
    case "error":
      return { ...run, errors: [...run.errors, ev.message], done: true, stage: "error" };
    case "done":
      return { ...run, done: true, stage: "done", message: "" };
    default:
      return run;
  }
}

export function evidenceIndex(run) {
  const map = {};
  (run?.evidence || []).forEach((e) => (map[e.evidence_id] = e));
  return map;
}
