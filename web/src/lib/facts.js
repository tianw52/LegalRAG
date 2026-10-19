// What-if facts (fact_toggle view) and the user's answers to them.
// run.answers[factId] = "yes" | "no" | "unsure" is shared by the Overview's
// "Help narrow it down" and the What-if toggles.

export const FACT_STATUS = {
  stated: { label: "You said", color: "var(--ink)" },
  assumed: { label: "We assumed", color: "var(--federal)" },
  unknown: { label: "Not mentioned", color: "var(--local)" },
};

export function runFacts(run) {
  return run?.views?.fact_toggle?.spec?.facts || [];
}

// Off = "what if this were not true".  Unanswered facts the user never
// mentioned start off, so their consequence is visible.
export function factIsOff(fact, answers) {
  const a = answers[fact.id];
  return a === "no" || ((a == null || a === "unsure") && fact.status === "unknown");
}
