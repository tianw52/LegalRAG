// Full document text with retrieved passages marked by character offsets.
// passages: [{evidence_id, char_start, char_end}]; overlapping ranges are merged.
export default function HighlightedText({ text, passages, focusId, markRef }) {
  const ranges = passages
    .filter((p) => p.char_start != null && p.char_end != null)
    .map((p) => ({ start: p.char_start, end: p.char_end, id: p.evidence_id }))
    .sort((a, b) => a.start - b.start);
  const out = [];
  let pos = 0;
  ranges.forEach((r, i) => {
    const start = Math.max(r.start, pos);
    if (start >= r.end) return; // fully inside a previous passage
    if (start > pos) out.push(text.slice(pos, start));
    const focus = r.id === focusId;
    out.push(
      <mark key={i} className={focus ? "focus" : ""} ref={focus ? markRef : undefined}>
        <span className="mark-id">{r.id}</span>
        {text.slice(start, r.end)}
      </mark>
    );
    pos = r.end;
  });
  out.push(text.slice(pos));
  return <>{out}</>;
}

// Positions (0–100 %) of passages within a document, for the reader minimap.
export function passageMarks(textLength, passages) {
  if (!textLength) return [];
  return passages
    .filter((p) => p.char_start != null)
    .map((p) => ({ id: p.evidence_id, top: (p.char_start / textLength) * 100 }));
}
