import { PROVINCE_TILES } from "../lib/jurisdiction.js";

// Tile map of Canada (8-column grid, roughly geographic), for picking a
// province in onboarding.   tileProps(code) → { on, bg, fg, border }
export default function ProvinceTiles({ tileProps, onPick, ariaLabel }) {
  return (
    <div className="tiles" role="group" aria-label={ariaLabel}>
      {PROVINCE_TILES.map((t) => {
        const p = tileProps(t.code);
        return (
          <button
            key={t.code}
            type="button"
            className="tile"
            aria-label={t.name}
            aria-pressed={!!p.on}
            onClick={() => onPick(t.code)}
            style={{ gridColumn: t.col, gridRow: t.row, background: p.bg, color: p.fg, border: p.border }}
          >
            <span className="tile-code">{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}
