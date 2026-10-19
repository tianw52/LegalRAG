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
  marks: {},       // milestone → ms timestamp (lib/progress.js): retrieving, evidence, fact_toggle, plan, relations, done
  demo: null,      // {builtAt, model, totalS} when replayed from a precomputed demo
  answers: {},     // fact id → "yes" | "no" | "unsure" (Overview "narrow it down" ↔ What-if toggles)
  chat: { threads: [], cur: null }, // Ask panel: [{id, title, msgs: [{role, text, ...}]}]
};

export function newRun(question, scope, province) {
  return { id: `${Date.now()}`, question, scope, province, startedAt: Date.now(), ...EMPTY_RUN };
}

// `at` = when the event arrived (ms); progress timing is measured client-side.
export function applyEvent(run, ev, at = Date.now()) {
  const mark = (name, extra) => ({ ...run, ...extra, marks: { ...run.marks, [name]: run.marks[name] ?? at } });
  switch (ev.event) {
    case "trace":
      return { ...run, traceId: ev.trace_id };
    case "status":
      return mark(ev.stage, { stage: ev.stage, message: ev.message });
    case "query":
      return { ...run, query: ev };
    case "evidence":
      return mark("evidence", { evidence: ev.items });
    case "view": {
      const v = ev.view;
      const viewOrder = run.viewOrder.includes(v.view_id) ? run.viewOrder : [...run.viewOrder, v.view_id];
      return settleRelations(mark(v.view_id, { views: { ...run.views, [v.view_id]: v }, viewOrder }), at);
    }
    case "plan":
      return settleRelations(mark("plan", { plan: ev }), at);
    case "view_error":
      // Per-view failures are shown where the view would appear; the
      // formulator fallback is a run-level notice.
      return ev.view_id === "formulator"
        ? { ...run, errors: [...run.errors, ev.message] }
        : settleRelations(mark(ev.view_id, { viewErrors: { ...run.viewErrors, [ev.view_id]: ev.message } }), at);
    case "error":
      return mark("done", { errors: [...run.errors, ev.message], done: true, stage: "error" });
    case "done":
      return mark("done", { done: true, stage: "done", message: "" });
    default:
      return run;
  }
}

// "relations" is reached when every view the planner chose has arrived or failed.
function settleRelations(run, at) {
  if (!run.plan || run.marks.relations) return run;
  const pending = run.plan.views.some((_, i) => !run.views[`adhoc_${i}`] && !run.viewErrors[`adhoc_${i}`]);
  return pending ? run : { ...run, marks: { ...run.marks, relations: at } };
}

// A precomputed demo (GET /api/demos/{id}) replayed into a finished run.
export function demoRun(bundle) {
  const run = newRun(bundle.question, bundle.scope, bundle.kind === "local" ? bundle.province : null);
  const replayed = bundle.events.reduce((r, ev) => applyEvent(r, ev, r.startedAt), run);
  return { ...replayed, demo: { builtAt: bundle.built_at, model: bundle.model, totalS: bundle.total_s, queryId: bundle.query_id } };
}

export function evidenceIndex(run) {
  const map = {};
  (run?.evidence || []).forEach((e) => (map[e.evidence_id] = e));
  return map;
}
