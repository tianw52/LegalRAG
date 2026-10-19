// The exploration's tabs (header order).  `count` = key of lib/stats.js runCounts().
export const TABS = [
  { id: "overview", label: "Overview" },
  { id: "whatif", label: "What-if", count: "facts" },
  { id: "sources", label: "Sources", count: "documents" },
  { id: "timeline", label: "Timeline" },
  { id: "relations", label: "Relations", count: "maps" },
];
export const TAB_LABEL = Object.fromEntries(TABS.map((t) => [t.id, t.label]));
