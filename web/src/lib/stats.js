// Derived numbers for one exploration run (Overview stats, tab counts,
// timeline lanes).  Documents come from the server's case_landscape view.
import { GROUP_ORDER, jurisdictionGroup } from "./jurisdiction.js";

export function runDocuments(run) {
  return run?.views?.case_landscape?.spec?.documents || [];
}

export function runPassages(run) {
  return (run?.evidence || []).filter((e) => e.kind === "passage");
}

export function groupCounts(docs, province) {
  const counts = Object.fromEntries(GROUP_ORDER.map((g) => [g, 0]));
  docs.forEach((d) => (counts[jurisdictionGroup(d.jurisdiction, province)] += 1));
  return counts;
}

export function yearSpan(docs) {
  const years = docs.map((d) => d.year).filter(Boolean);
  if (!years.length) return null;
  const min = Math.min(...years);
  const max = Math.max(...years);
  // "recent" = the last third of the span (at least the last 5 years)
  const recentFrom = Math.max(min, max - Math.max(5, Math.round((max - min) / 3)));
  return { min, max, recentFrom, recent: years.filter((y) => y >= recentFrom).length, total: years.length };
}

// Counts per year bucket, for small sparklines.
export function yearHistogram(docs, buckets = 9) {
  const span = yearSpan(docs);
  if (!span) return [];
  const width = Math.max(1, span.max - span.min + 1) / buckets;
  const counts = Array(buckets).fill(0);
  docs.forEach((d) => {
    if (d.year) counts[Math.min(buckets - 1, Math.floor((d.year - span.min) / width))] += 1;
  });
  return counts;
}

export function runCounts(run) {
  const facts = run?.views?.fact_toggle?.spec?.facts || [];
  const maps = (run?.viewOrder || []).filter((id) => run.views[id]?.source === "agent");
  return {
    facts: facts.length,
    factsUnknown: facts.filter((f) => f.status === "unknown").length,
    documents: runDocuments(run).length,
    passages: runPassages(run).length,
    maps: maps.length,
  };
}
