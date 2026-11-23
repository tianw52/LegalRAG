import assert from "node:assert/strict";
import test from "node:test";
import { caseRelevance } from "../src/lib/caseRelevance.js";
import { relevanceStatistics } from "../src/lib/relevanceStatistics.js";

function passage(evidence_id, doc_id, text, doc_type = "case") {
  return { kind: "passage", evidence_id, doc_id, text, doc_type, case_name: doc_id };
}

test("uses the question for categories and case passages for independently cited edges", () => {
  const graph = caseRelevance({
    question: "What is the duty of care, and how is CAUSATION evaluated?",
    query: { keywords: ["duty of care", "employment insurance"] },
    evidence: [
      { kind: "question", evidence_id: "q", text: "Duty of care and causation" },
      passage("e1", "Case A", "The duty-of-care analysis also considers causation."),
      passage("e2", "Case A", "The duty of care applies."),
      passage("e3", "Case B", "The causation analysis differs."),
      passage("e4", "Case C", "The word is precausation."),
      passage("e5", "Statute A", "Duty of care and causation", "statute"),
      passage(undefined, "Case C", "Duty of care"),
    ],
  });
  assert.deepEqual(graph.categories.map(({ id }) => id), ["duty of care", "causation"]);
  assert.equal(new Set(graph.categories.map(({ color }) => color)).size, 2);
  assert.deepEqual(graph.cases.map(({ id }) => id), ["Case A", "Case B", "Case C"]);
  assert.deepEqual(graph.cases.map(({ relationships }) => relationships.map(({ id, evidenceIds }) => [id, evidenceIds])), [
    [["duty of care", ["e1", "e2"]], ["causation", ["e1"]]],
    [["causation", ["e3"]]],
    [],
  ]);
});

test("statistics counts each case once per jurisdiction and once per shared category", () => {
  const question = "How do duty of care, causation, and child support apply?";
  const evidence = [
    { ...passage("e1", "Case A", "Duty of care and causation."), jurisdiction: "BC" },
    { ...passage("e2", "Case A", "Causation again."), jurisdiction: "BC" },
    { ...passage("e3", "Case B", "Causation."), jurisdiction: "federal" },
    { ...passage("e4", "Case C", "Other facts."), jurisdiction: "ON" },
    passage("e5", "Case D", "Other facts."),
    { ...passage("e6", "Statute A", "Causation.", "statute"), jurisdiction: "BC" },
  ];
  const relevance = caseRelevance({ question, evidence });
  const stats = relevanceStatistics({ ...relevance, province: "BC" });
  assert.equal(stats.total, 4);
  assert.deepEqual(Object.fromEntries(stats.jurisdictions.map(({ code, count }) => [code, count])), {
    BC: 1, ON: 1, federal: 1, unknown: 1,
  });
  assert.deepEqual(stats.categories, relevance.categories.map((category, index) => ({
    ...category, count: [1, 2, 0][index],
  })));
  const changed = relevanceStatistics(caseRelevance({ question: "What is child support?", evidence: [] }));
  assert.equal(changed.total, 0);
  assert.deepEqual(changed.jurisdictions, []);
  assert.deepEqual(changed.categories.map(({ id, count }) => [id, count]), [["child support", 0]]);
});

test("recalculates from the current question and handles changing or empty retrieval results", () => {
  const evidence = [passage("e1", "Case A", "Causation and shareholder oppression.")];
  assert.equal(caseRelevance({ question: "What is causation?", evidence }).cases[0].relationships[0].id, "causation");
  const updated = caseRelevance({
    question: "What is shareholder oppression?",
    query: { keywords: ["shareholder oppression"] },
    evidence: [...evidence, passage("e2", "Case B", "Unrelated text.")],
  });
  assert.deepEqual(updated.categories.map(({ id }) => id), ["shareholder oppression"]);
  assert.deepEqual(updated.cases.map(({ relationships }) => relationships.length), [1, 0]);
  assert.deepEqual(caseRelevance({ question: "What is causation?", evidence: [] }).cases, []);
  assert.deepEqual(caseRelevance({ question: "Can I get help?", evidence }).categories, []);
});
