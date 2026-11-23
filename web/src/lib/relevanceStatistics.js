import { GROUP_COLOR, PROVINCE_TILES } from "./jurisdiction.js";

const PROVINCE_COLORS = [
  "var(--other)", "#54746b", "#986447", "#636ca1", "#967437", "#8d5874",
  "#527b91", "#697441", "#a35d58", "#796598", "#748d54", "#a17583", "#697484",
];

export function relevanceStatistics({ cases, categories, province }) {
  const jurisdictions = [
    ...PROVINCE_TILES.map(({ code, name }, index) => ({
      code, label: name, color: code === province ? GROUP_COLOR.local : PROVINCE_COLORS[index],
    })),
    { code: "federal", label: "Federal", color: GROUP_COLOR.federal },
    { code: "unknown", label: "Unknown", color: GROUP_COLOR.unknown },
  ];
  const jurisdictionCounts = new Map(jurisdictions.map(({ code }) => [code, 0]));
  const categoryCounts = new Map(categories.map(({ id }) => [id, 0]));
  const uniqueCases = new Map(cases.map((entry) => [entry.id, entry]));

  for (const { doc, relationships } of uniqueCases.values()) {
    const code = jurisdictionCounts.has(doc.jurisdiction) ? doc.jurisdiction : "unknown";
    jurisdictionCounts.set(code, jurisdictionCounts.get(code) + 1);
    for (const id of new Set(relationships.map(({ id }) => id))) {
      if (categoryCounts.has(id)) categoryCounts.set(id, categoryCounts.get(id) + 1);
    }
  }

  return {
    total: uniqueCases.size,
    jurisdictions: jurisdictions.map((entry) => ({ ...entry, count: jurisdictionCounts.get(entry.code) }))
      .filter(({ count }) => count > 0),
    categories: categories.map((category) => ({ ...category, count: categoryCounts.get(category.id) })),
  };
}
