import { PROVINCE_TILES } from "../lib/jurisdiction.js";

// Tile map of Canada (8-column grid, roughly geographic).  Used for picking a
// province in onboarding and for filtering sources by jurisdiction.
//   tileProps(code) → { on, bg, fg, border, count }   (count optional)
//   extra: additional tiles, e.g. [{code: "federal", label: "FED", col, row}]
export default function ProvinceTiles({ tileProps, onPick, extra = [], ariaLabel }) {
  const tiles = [...PROVINCE_TILES, ...extra];
  return (
    <div className="tiles" role="group" aria-label={ariaLabel}>
      {tiles.map((t) => {
        const p = tileProps(t.code);
        return (
          <button
            key={t.code}
            type="button"
            className="tile"
            aria-label={`${t.name || t.label}${p.count != null ? `: ${p.count}` : ""}`}
            aria-pressed={!!p.on}
            onClick={() => onPick(t.code)}
            style={{
              gridColumn: t.col,
              gridRow: t.row,
              background: p.bg,
              color: p.fg,
              border: p.border,
              opacity: p.dim ? 0.45 : 1,
            }}
          >
            <span className="tile-code">{t.label}</span>
            {p.count != null && <span className="disp tile-count">{p.count}</span>}
          </button>
        );
      })}
    </div>
  );
}
