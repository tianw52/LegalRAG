// Jurisdiction helpers shared by pages and tabs.  Codes come from
// legalrag/ingestion/canlegal_courts.yaml (two-letter province codes, "federal", "unknown").

// Groups relative to the user's province; colours are CSS tokens from theme.css.
export const GROUP_ORDER = ["local", "federal", "other", "unknown"];
export const GROUP_COLOR = {
  local: "var(--local)",
  federal: "var(--federal)",
  other: "var(--other)",
  unknown: "var(--unknown)",
};
export const GROUP_LABEL = {
  local: "Your province",
  federal: "Federal",
  other: "Other provinces",
  unknown: "Unknown",
};

export function jurisdictionGroup(code, province) {
  if (code && code === province) return "local";
  if (code === "federal") return "federal";
  if (!code || code === "unknown") return "unknown";
  return "other";
}

// Tile map of Canada on an 8-column grid (roughly geographic).
export const PROVINCE_TILES = [
  { code: "YT", label: "YT", name: "Yukon", col: 2, row: 1 },
  { code: "NT", label: "NT", name: "Northwest Territories", col: 3, row: 1 },
  { code: "NU", label: "NU", name: "Nunavut", col: 4, row: 1 },
  { code: "BC", label: "BC", name: "British Columbia", col: 1, row: 2 },
  { code: "AB", label: "AB", name: "Alberta", col: 2, row: 2 },
  { code: "SK", label: "SK", name: "Saskatchewan", col: 3, row: 2 },
  { code: "MB", label: "MB", name: "Manitoba", col: 4, row: 2 },
  { code: "ON", label: "ON", name: "Ontario", col: 5, row: 2 },
  { code: "QC", label: "QC", name: "Quebec", col: 6, row: 2 },
  { code: "NL", label: "NL", name: "Newfoundland and Labrador", col: 7, row: 2 },
  { code: "NB", label: "NB", name: "New Brunswick", col: 6, row: 3 },
  { code: "PE", label: "PE", name: "Prince Edward Island", col: 7, row: 3 },
  { code: "NS", label: "NS", name: "Nova Scotia", col: 8, row: 3 },
];

// Group label with the exploration's province named ("Saskatchewan", not "Your province").
export function groupLabel(group, provinceLabel) {
  return group === "local" && provinceLabel ? provinceLabel : GROUP_LABEL[group];
}
