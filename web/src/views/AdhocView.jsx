import { useEffect, useRef } from "react";
import "../vendor/viz-renderers.js"; // sets window.VizRenderers (Demo renderers)
import { ViewHead } from "./FactToggle.jsx";

// Agent-chosen view (concept map, comparison, legal test, …) rendered with the
// Demo's deterministic renderers.  The renderers hide items without evidence.
export default function AdhocView({ view, evidenceById, onSelect }) {
  const mountRef = useRef(null);
  // Keep the latest callback without re-rendering the diagram (which would reset its selection)
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !window.VizRenderers) return;
    // Demo renderers expect {evidence_id, passage, citation}
    const demoEvidence = {};
    Object.values(evidenceById).forEach((e) => {
      demoEvidence[e.evidence_id] = { evidence_id: e.evidence_id, passage: e.text, citation: e.citation };
    });
    try {
      window.VizRenderers.renderPrimaryView(mount, view.spec, demoEvidence, {
        onSelect: (payload) => {
          const item = payload.item || {};
          onSelectRef.current({
            label:
              item.label || item.event || item.location_name || item.relation ||
              (item.row && item.column ? `${item.row} × ${item.column}` : item.id),
            detail: item.description || item.detail || (item.value != null ? String(item.value) : ""),
            evidenceIds: payload.evidence_ids || item.evidence_ids || [],
          });
        },
      });
    } catch (err) {
      mount.innerHTML = `<div class="vr-error">Render error: ${String(err.message)}</div>`;
    }
    return () => window.VizRenderers.clearMount(mount);
  }, [view, evidenceById]);

  const hidden = view.audit?.unbacked_items || 0;
  return (
    <section className="card view">
      <ViewHead view={view} />
      <div className="viz-stage" ref={mountRef} />
      {hidden > 0 && (
        <p className="muted small">
          {hidden} item{hidden > 1 ? "s" : ""} not backed by a retrieved source {hidden > 1 ? "were" : "was"} hidden.
        </p>
      )}
    </section>
  );
}
