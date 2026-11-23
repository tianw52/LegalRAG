import { runDocuments, runPassages } from "./stats.js";

const COLORS = ["#275b92", "#a04b29", "#347052", "#80558b", "#856317", "#a13c61", "#326f76", "#555b9c"];

// Explicit legal phrases also cover runs where query formulation fell back.
const ISSUE_PHRASES = [
  "duty of care", "standard of care", "causation", "contributory negligence",
  "psychological damages", "mitigation of damages", "wrongful dismissal",
  "constructive dismissal", "reasonable notice", "just cause", "employment insurance",
  "breach of contract", "unjust enrichment", "limitation period", "judicial review",
  "procedural fairness", "human rights", "child support", "spousal support",
];

function normalize(text) {
  return ` ${(text || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim()} `;
}

export function caseRelevance(run) {
  const question = run.question || run.evidence?.find((item) => item.kind === "question")?.text || "";
  const normalizedQuestion = normalize(question);
  const categories = [...new Set([
    ...ISSUE_PHRASES,
    ...(run.query?.keywords || []).map((keyword) => normalize(keyword).trim())
      .filter((keyword) => keyword.includes(" ")),
  ])]
    .filter((phrase) => normalizedQuestion.includes(normalize(phrase)))
    .sort((a, b) => normalizedQuestion.indexOf(normalize(a)) - normalizedQuestion.indexOf(normalize(b)) || b.length - a.length)
    .slice(0, COLORS.length)
    .map((label, index) => ({ id: label, label, color: COLORS[index] }));

  const documents = new Map(runDocuments(run).map((doc) => [doc.doc_id, doc]));
  const cases = new Map();
  for (const passage of runPassages(run)) {
    const { doc_id: id, evidence_id: evidenceId } = passage;
    const doc = documents.get(id) || passage;
    if (!id || (doc.doc_type || passage.doc_type || "case") !== "case") continue;
    if (!cases.has(id)) {
      cases.set(id, {
        id, doc, name: doc.case_name || doc.citation || id,
        evidenceIds: new Set(), matches: new Map(),
      });
    }
    if (!evidenceId) continue;
    const entry = cases.get(id);
    entry.evidenceIds.add(evidenceId);
    const text = normalize(passage.text);
    for (const category of categories) {
      if (!text.includes(normalize(category.label))) continue;
      if (!entry.matches.has(category.id)) entry.matches.set(category.id, new Set());
      entry.matches.get(category.id).add(evidenceId);
    }
  }
  return {
    question,
    categories,
    cases: [...cases.values()].map(({ evidenceIds, matches, ...entry }) => ({
      ...entry,
      evidenceIds: [...evidenceIds],
      relationships: categories.filter(({ id }) => matches.has(id)).map((category) => ({
        ...category, evidenceIds: [...matches.get(category.id)],
      })),
    })),
  };
}
