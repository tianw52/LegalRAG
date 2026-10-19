// Build progress of one run (Overview progress panel, locked tabs).
// Stages follow the engine (legalrag/explore/engine.py); facts and summary
// run in parallel, relation maps start once the planner has chosen them.
// While a stage runs its bar is an estimate from `expect` (seconds); it only
// completes when the real event arrives.

export const STAGES = [
  { id: "understand", label: "Understanding your question", short: "Understand question", w: 10, expect: 8 },
  { id: "sources", label: "Finding and ranking sources", short: "Find sources", w: 30, expect: 30, opens: ["sources", "timeline"] },
  { id: "facts", label: "Writing the What-if facts", short: "What-if", w: 25, expect: 45, opens: ["whatif"] },
  { id: "summary", label: "Writing the summary and follow-ups", short: "Summary", w: 15, expect: 45, opens: ["overview"] },
  { id: "relations", label: "Drawing the relation maps", short: "Relations", w: 20, expect: 60, opens: ["relations"] },
];

// [start, end] milestones per stage (keys of run.marks; "start" = run.startedAt)
const SPANS = {
  understand: ["start", "retrieving"],
  sources: ["retrieving", "evidence"],
  facts: ["evidence", "fact_toggle"],
  summary: ["evidence", "plan"],
  relations: ["plan", "relations"],
};
const FAILURE = { facts: "fact_toggle", summary: "plan" };

export function runProgress(run, now = Date.now()) {
  const at = (m) => (m === "start" ? run.startedAt : run.marks[m]);
  let total = 0;
  const stages = STAGES.map((st) => {
    const [from, to] = SPANS[st.id];
    const start = at(from);
    let end = at(to);
    // A run that ended (done/error) settles stages whose event never came
    if (!end && run.done) end = run.marks.done;
    const failed = !!(FAILURE[st.id] && run.viewErrors[FAILURE[st.id]]) || (run.done && !at(to) && st.id !== "relations");
    let fr = 0;
    if (end) fr = 1;
    else if (start) fr = Math.min(0.92, 1 - Math.exp(-(now - start) / 1000 / st.expect));
    total += st.w * fr;
    const status = end ? (failed ? "failed" : "done") : start ? "running" : "queued";
    return { ...st, status, fr, seconds: end && start ? (end - start) / 1000 : null };
  });
  const by = Object.fromEntries(stages.map((s) => [s.id, s]));
  const settled = (id) => by[id].status === "done" || by[id].status === "failed";
  return {
    stages,
    pct: run.done ? 100 : Math.min(99, Math.round(total)),
    elapsed: Math.round(((run.marks.done || now) - run.startedAt) / 1000),
    running: stages.filter((s) => s.status === "running"),
    ready: {
      overview: true,
      sources: settled("sources"),
      timeline: settled("sources"),
      whatif: settled("facts"),
      relations: settled("summary"), // the tab lists the planned maps and builds them in place
    },
  };
}
