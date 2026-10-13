import { useMemo, useState } from "react";
import DemoViz from "../components/DemoViz.jsx";
import { Pending, Segmented, SourceLinks } from "../components/primitives.jsx";
import { evidenceIndex } from "../lib/exploration.js";

// Agent-chosen views (0–3, decided by the planner), one at a time.
export default function MapsTab({ run, openEvidence }) {
  const planned = run.plan?.views || [];
  const built = run.viewOrder.map((id) => run.views[id]).filter((v) => v.source === "agent");
  const [current, setCurrent] = useState(null);
  const [selection, setSelection] = useState(null);
  const evidenceById = useMemo(() => evidenceIndex(run), [run.evidence]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!run.plan) {
    return <section className="band"><Pending label={run.done ? "No plan was produced." : "Deciding which maps would help…"} /></section>;
  }
  if (!planned.length) {
    return (
      <section className="band">
        <h2 className="lbl band-label">Maps</h2>
        <div className="band-body">
          <p className="lead">No extra maps for this question.</p>
          <p className="muted">{run.plan.extra_views_reason}</p>
        </div>
      </section>
    );
  }

  const options = planned.map((pv, i) => ({ value: `adhoc_${i}`, label: pv.title }));
  const activeId = current && options.some((o) => o.value === current) ? current : options[0].value;
  const index = Number(activeId.split("_")[1]);
  const view = built.find((v) => v.view_id === activeId);
  const error = run.viewErrors?.[activeId];
  const hidden = view?.audit?.unbacked_items || 0;

  return (
    <>
      <section className="tab-bar">
        <Segmented
          label="Map"
          options={options}
          value={activeId}
          onChange={(v) => {
            setCurrent(v);
            setSelection(null);
          }}
        />
        <span className="lbl">Chosen for this question</span>
      </section>
      <div className="split">
        <section className="split-main wide">
          <p className="muted">{planned[index]?.purpose}</p>
          {view ? (
            <DemoViz spec={view.spec} evidenceById={evidenceById} onSelect={setSelection} />
          ) : error ? (
            <p className="error small">This map couldn't be built: {error}</p>
          ) : (
            <Pending label="Building this map…" />
          )}
          {hidden > 0 && (
            <p className="small muted">{hidden} item{hidden > 1 ? "s" : ""} not backed by a retrieved source {hidden > 1 ? "were" : "was"} hidden.</p>
          )}
        </section>
        <aside className="split-side right" aria-live="polite">
          {selection ? (
            <>
              <span className="lbl">Selected</span>
              <span className="disp h-s">{selection.label}</span>
              {selection.detail && <p className="muted">{selection.detail}</p>}
              <SourceLinks ids={selection.evidenceIds} onOpen={openEvidence} />
            </>
          ) : (
            <>
              <span className="lbl">Why this map</span>
              <p className="muted">{planned[index]?.why_useful || planned[index]?.focus}</p>
              <p className="small muted">Click any item to see the passages behind it.</p>
            </>
          )}
        </aside>
      </div>
    </>
  );
}
