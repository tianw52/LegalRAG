import { useEffect, useRef } from "react";
import "../vendor/viz-renderers.js"; // sets window.VizRenderers (Demo renderers)
import { makeSelection } from "../lib/selection.js";

// Renders an agent-built view spec with the Demo's deterministic renderers
// (network, matrix, decision/statute tree, flow, timeline).  Items without
// evidence are hidden by the renderers.  onSelect receives a selection
// {label, detail, evidenceIds} (see lib/selection.js).
export default function DemoViz({ spec, evidenceById, onSelect }) {
  const mountRef = useRef(null);
  // Keep the latest callback without re-rendering the diagram (which would reset its selection)
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !window.VizRenderers) return;
    const demoEvidence = {};
    Object.values(evidenceById).forEach((e) => {
      demoEvidence[e.evidence_id] = { evidence_id: e.evidence_id, passage: e.text, citation: e.citation };
    });
    try {
      window.VizRenderers.renderPrimaryView(mount, spec, demoEvidence, {
        onSelect: (payload) => {
          const item = payload.item || {};
          onSelectRef.current?.(
            makeSelection(
              item.label || item.event || item.location_name || item.relation ||
                (item.row && item.column ? `${item.row} × ${item.column}` : item.id),
              item.description || item.detail || (item.value != null ? String(item.value) : ""),
              payload.evidence_ids || item.evidence_ids
            )
          );
        },
      });
    } catch (err) {
      mount.innerHTML = `<div class="vr-error">Render error: ${String(err.message)}</div>`;
    }
    return () => window.VizRenderers.clearMount(mount);
  }, [spec, evidenceById]);

  return <div className="viz-stage" ref={mountRef} />;
}
