// A "selection" links a clicked view item to the evidence passages it cites.
// Shape: { label, detail, evidenceIds }.  Maps tab shows it beside the diagram.

export function makeSelection(label, detail, evidenceIds) {
  return evidenceIds?.length ? { label, detail: detail || "", evidenceIds } : null;
}
