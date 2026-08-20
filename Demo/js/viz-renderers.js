/**
 * Deterministic LegalRAG visualization renderers — no CDN, no model-generated code.
 */
(function (global) {
  "use strict";

  /* ─── colour palettes ───────────────────────────────────────────────── */
  const ENTITY_COLORS = {
    plaintiff:   { fill: "#0f766e", ring: "#99f6e4",  label: "Plaintiff"      },
    defendant:   { fill: "#b45309", ring: "#fde68a",  label: "Defendant"      },
    third_party: { fill: "#475569", ring: "#cbd5e1",  label: "Third party"    },
    actor:       { fill: "#334155", ring: "#cbd5e1",  label: "Actor"          },
    doctrine:    { fill: "#1d4ed8", ring: "#bfdbfe",  label: "Doctrine / issue" },
    party:       { fill: "#0f766e", ring: "#99f6e4",  label: "Party"          },
    entity:      { fill: "#334155", ring: "#e2e8f0",  label: "Entity"         },
  };

  const TIMELINE_CAT = {
    preparation: { color: "#0f766e", bg: "#ccfbf1", label: "Preparation" },
    entry:       { color: "#b45309", bg: "#ffedd5", label: "Entry"        },
    retreat:     { color: "#7c3aed", bg: "#ede9fe", label: "Retreat"      },
    legal:       { color: "#1d4ed8", bg: "#dbeafe", label: "Legal theory" },
    event:       { color: "#334155", bg: "#e2e8f0", label: "Event"        },
  };

  const FLOW_CAT = {
    manufacturer: { fill: "#1d4ed8", stroke: "#93c5fd", bg: "#eff6ff", label: "Manufacturer"        },
    store:        { fill: "#b45309", stroke: "#fcd34d", bg: "#fffbeb", label: "Store / third party"  },
    retail:       { fill: "#0f766e", stroke: "#6ee7b7", bg: "#ecfdf5", label: "Retail"               },
    plaintiff:    { fill: "#7c3aed", stroke: "#c4b5fd", bg: "#f5f3ff", label: "End user / plaintiff" },
    location:     { fill: "#334155", stroke: "#94a3b8", bg: "#f1f5f9", label: "Stage"                },
  };

  const TREE_COLORS = {
    grant:       { fill: "#7c3aed", stroke: "#c4b5fd", text: "#fff" },
    right:       { fill: "#0f766e", stroke: "#6ee7b7", text: "#fff" },
    condition:   { fill: "#92400e", stroke: "#fcd34d", text: "#fff" },
    restriction: { fill: "#dc2626", stroke: "#fca5a5", text: "#fff" },
    exception:   { fill: "#1e40af", stroke: "#93c5fd", text: "#fff" },
    default:     { fill: "#334155", stroke: "#94a3b8", text: "#fff" },
  };

  const DT_COLORS = {
    question:    { fill: "#1e3a5f", stroke: "#93c5fd", text: "#fff" },
    allowed:     { fill: "#0f766e", stroke: "#6ee7b7", text: "#fff" },
    conditional: { fill: "#92400e", stroke: "#fcd34d", text: "#fff" },
    prohibited:  { fill: "#dc2626", stroke: "#fca5a5", text: "#fff" },
    outcome:     { fill: "#334155", stroke: "#94a3b8", text: "#fff" },
    default:     { fill: "#475569", stroke: "#94a3b8", text: "#fff" },
  };

  const CMP_TONE = {
    yes:     { bg: "#ecfdf5", border: "#6ee7b7", fg: "#065f46", sym: "✓" },
    no:      { bg: "#fef2f2", border: "#fca5a5", fg: "#991b1b", sym: "✗" },
    maybe:   { bg: "#fffbeb", border: "#fcd34d", fg: "#92400e", sym: "~" },
    offered: { bg: "#eff6ff", border: "#93c5fd", fg: "#1e40af", sym: "→" },
    asked:   { bg: "#eff6ff", border: "#93c5fd", fg: "#1e40af", sym: "?" },
    missing: { bg: "#f1f5f9", border: "#e2e8f0", fg: "#475569", sym: "—" },
    "n/a":   { bg: "#f1f5f9", border: "#e2e8f0", fg: "#475569", sym: "—" },
    fails:   { bg: "#fef2f2", border: "#fca5a5", fg: "#991b1b", sym: "✗" },
    likely:  { bg: "#ecfdf5", border: "#6ee7b7", fg: "#065f46", sym: "✓" },
  };

  /* ─── utilities ─────────────────────────────────────────────────────── */
  const mountState = new WeakMap();

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function short(s, n) {
    s = String(s == null ? "" : s).replace(/\s+/g, " ").trim();
    n = n || 18;
    return s.length <= n ? s : s.slice(0, n - 1) + "\u2026";
  }

  /* Break s into lines of at most maxChars characters. */
  function wrapSvgText(s, maxChars) {
    s = String(s || "").replace(/\s+/g, " ").trim();
    const words = s.split(" ");
    const lines = [];
    let cur = "";
    words.forEach(function (w) {
      if ((cur ? cur + " " + w : w).length <= maxChars) {
        cur = cur ? cur + " " + w : w;
      } else {
        if (cur) lines.push(cur);
        cur = w.slice(0, maxChars);
      }
    });
    if (cur) lines.push(cur);
    return lines.slice(0, 3);
  }

  function hasEvidence(item) {
    return Array.isArray(item.evidence_ids) && item.evidence_ids.length > 0;
  }

  function clearMount(el) {
    el.innerHTML = "";
    mountState.set(el, { onSelect: (mountState.get(el) || {}).onSelect });
  }

  function fireSelect(mountEl, payload) {
    const st = mountState.get(mountEl) || {};
    if (typeof st.onSelect === "function") st.onSelect(payload);
  }

  function applySelection(root, selectedId) {
    if (!root) return;
    root.querySelectorAll("[data-viz-id]").forEach(function (node) {
      const id = node.getAttribute("data-viz-id");
      node.classList.toggle("is-selected", !!(selectedId && id === selectedId));
      node.classList.toggle("is-dimmed",   !!(selectedId && id !== selectedId));
    });
  }

  /* ─── top-level dispatch ────────────────────────────────────────────── */
  function renderPrimaryView(mountEl, view, evidenceById, opts) {
    const options = opts || {};
    clearMount(mountEl);
    const st = mountState.get(mountEl) || {};
    if (typeof options.onSelect === "function") st.onSelect = options.onSelect;
    mountState.set(mountEl, st);
    mountEl._vizApplySelection = function (id) { applySelection(mountEl, id); };
    mountEl._vizClearSelection = function ()   { applySelection(mountEl, null); };

    const RENDERERS = {
      timeline:      TimelineRenderer,
      map:           FlowchartRenderer,
      network:       NetworkRenderer,
      matrix:        ComparisonRenderer,
      statute_tree:  StatuteTreeRenderer,
      decision_tree: DecisionTreeRenderer,
      evidence_view: EvidenceRenderer,
      bar_chart:     disabled,
      grouped_bar_chart: disabled,
      line_chart:    disabled,
    };

    const fn = RENDERERS[view && view.type];
    if (!fn) {
      mountEl.innerHTML = '<div class="vr-error">Unsupported view: ' + esc(String((view || {}).type)) + "</div>";
      return;
    }
    fn(mountEl, view, evidenceById || {}, options);
  }

  function disabled(el) {
    el.innerHTML = '<div class="vr-error">Numeric charts are disabled in this demo.</div>';
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q4  HORIZONTAL SVG TIMELINE
     Track line, numbered circles, cards alternating above/below.
  ══════════════════════════════════════════════════════════════════════ */
  function TimelineRenderer(el, view) {
    const items = (view.items || []).filter(hasEvidence);
    if (!items.length) {
      el.innerHTML = '<div class="vr-error">No evidence-backed events.</div>';
      return;
    }

    const n      = items.length;
    const W      = 760;
    const TY     = 150;            /* track y */
    const CARD_W = Math.min(160, Math.floor((W - 40) / n) - 16);
    const CARD_H = 104;
    const H      = TY + CARD_H + 44; /* enough for below-track cards */
    const R_OUTER = 20, R_INNER = 13;

    /* x positions: evenly spaced */
    const xs = items.map(function (_, i) {
      return Math.round(40 + (i + 0.5) * ((W - 80) / n));
    });

    /* build svg */
    let svg = '<svg viewBox="0 0 ' + W + ' ' + H +
      '" role="img" aria-label="Chronological timeline" style="width:100%;height:auto;display:block;overflow:visible">';

    /* arrow marker */
    svg += '<defs>' +
      '<marker id="tlArr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/></marker>' +
      '</defs>';

    /* track */
    svg += '<line x1="30" y1="' + TY + '" x2="' + (W - 30) + '" y2="' + TY + '" ' +
      'stroke="#e2e8f0" stroke-width="4" stroke-linecap="round"/>';

    /* segment arrows */
    for (var i = 0; i < n - 1; i++) {
      svg += '<line x1="' + (xs[i] + R_INNER + 2) + '" y1="' + TY + '" ' +
        'x2="' + (xs[i + 1] - R_INNER - 2) + '" y2="' + TY + '" ' +
        'stroke="#94a3b8" stroke-width="2" marker-end="url(#tlArr)"/>';
    }

    /* cards + circles */
    items.forEach(function (item, i) {
      const meta  = TIMELINE_CAT[item.category] || TIMELINE_CAT.event;
      const vid   = item.id || ("t" + i);
      const above = i % 2 === 0;
      const cx    = xs[i];
      const cardX = cx - CARD_W / 2;
      const cardY = above ? TY - 24 - CARD_H : TY + 24;

      /* connector dashed line */
      const lineY1 = above ? cardY + CARD_H : cardY;
      const lineY2 = above ? TY - R_INNER : TY + R_INNER;
      svg += '<line x1="' + cx + '" y1="' + lineY1 + '" x2="' + cx + '" y2="' + lineY2 + '" ' +
        'stroke="' + meta.color + '" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.6"/>';

      /* outer glow ring */
      svg += '<circle cx="' + cx + '" cy="' + TY + '" r="' + R_OUTER + '" fill="' + meta.color + '" opacity="0.12"/>';

      /* card group — interactive */
      svg += '<g class="vr-svg-node" data-viz-id="' + esc(vid) + '" tabindex="0" role="button" aria-label="' +
        esc(item.event || "") + '" style="cursor:pointer">';

      /* card shadow (simple offset rect) */
      svg += '<rect x="' + (cardX + 2) + '" y="' + (cardY + 2) + '" width="' + CARD_W + '" height="' + CARD_H +
        '" rx="10" fill="#0001" opacity="0.06"/>';

      /* card background */
      svg += '<rect x="' + cardX + '" y="' + cardY + '" width="' + CARD_W + '" height="' + CARD_H +
        '" rx="10" fill="#fff" stroke="' + meta.color + '" stroke-width="1.5"/>';

      /* coloured top bar */
      svg += '<rect x="' + cardX + '" y="' + cardY + '" width="' + CARD_W + '" height="22" rx="10" fill="' + meta.color + '"/>';
      svg += '<rect x="' + cardX + '" y="' + (cardY + 12) + '" width="' + CARD_W + '" height="10" fill="' + meta.color + '"/>';

      /* category label in top bar */
      svg += '<text x="' + (cardX + CARD_W / 2) + '" y="' + (cardY + 15) + '" text-anchor="middle" ' +
        'font-size="9" font-weight="700" font-family="system-ui" fill="#fff" letter-spacing="0.06em">' +
        esc(meta.label.toUpperCase()) + '</text>';

      /* event title (wrapped) */
      const titleLines = wrapSvgText(item.event || "", Math.floor(CARD_W / 7));
      titleLines.forEach(function (line, li) {
        svg += '<text x="' + (cardX + CARD_W / 2) + '" y="' + (cardY + 37 + li * 15) + '" ' +
          'text-anchor="middle" font-size="11" font-weight="700" font-family="system-ui" fill="#0f172a">' +
          esc(line) + '</text>';
      });

      /* stage label at bottom */
      svg += '<text x="' + (cardX + CARD_W / 2) + '" y="' + (cardY + CARD_H - 8) + '" text-anchor="middle" ' +
        'font-size="9" font-family="system-ui" fill="#94a3b8">' +
        esc(short(item.date_label || "", 22)) + '</text>';

      svg += '</g>';

      /* circle on track (drawn after card so it's on top) */
      svg += '<circle cx="' + cx + '" cy="' + TY + '" r="' + R_INNER + '" fill="' + meta.color + '"/>';
      svg += '<text x="' + cx + '" y="' + (TY + 4) + '" text-anchor="middle" ' +
        'font-size="11" font-weight="800" font-family="system-ui" fill="#fff" pointer-events="none">' + (i + 1) + '</text>';
    });

    svg += '</svg>';

    /* legend + note */
    const used = {};
    items.forEach(function (item) { used[item.category || "event"] = true; });
    let legendHtml = '<div class="vr-legend">';
    Object.keys(used).forEach(function (c) {
      const m = TIMELINE_CAT[c] || TIMELINE_CAT.event;
      legendHtml += '<span class="vr-legend-item"><i style="background:' + m.color + '"></i>' + esc(m.label) + '</span>';
    });
    legendHtml += '</div>';

    const wrap = document.createElement("div");
    wrap.className = "vr-timeline-svg-wrap";
    wrap.innerHTML = legendHtml + svg +
      '<p class="vr-muted" style="text-align:center;margin-top:8px">Sequence order from the hypo — no calendar dates invented. Click a stage to see evidence.</p>';
    el.appendChild(wrap);

    /* click / keyboard handlers */
    const byId = {};
    items.forEach(function (item, i) { byId[item.id || ("t" + i)] = item; });

    wrap.querySelectorAll(".vr-svg-node").forEach(function (g) {
      function activate() {
        const vid = g.getAttribute("data-viz-id");
        const item = byId[vid];
        if (!item) return;
        applySelection(el, vid);
        fireSelect(el, { kind: "timeline", item: item, evidence_ids: item.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
      });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q5  SVG LEFT-TO-RIGHT FLOWCHART
     Rounded boxes for each stage, labelled arrows between them.
  ══════════════════════════════════════════════════════════════════════ */
  function FlowchartRenderer(el, view) {
    const items = (view.items || []).filter(hasEvidence);
    const edges = (view.flow_edges || []).filter(hasEvidence);
    if (!items.length) {
      el.innerHTML = '<div class="vr-error">No evidence-backed stages.</div>';
      return;
    }

    const BW = 162, BH = 100, GAP = 52;
    const PAD = 20;
    const W   = PAD + items.length * BW + (items.length - 1) * GAP + PAD;
    const H   = BH + 56; /* box + top/bottom padding */
    const BY  = 28;       /* box top y */

    /* x start for each box */
    const xs = items.map(function (_, i) { return PAD + i * (BW + GAP); });

    /* build index for quick lookup */
    const idxById = {};
    items.forEach(function (item, i) { idxById[item.id] = i; });

    let svg = '<svg viewBox="0 0 ' + W + ' ' + H +
      '" role="img" aria-label="Chain-of-control diagram" style="width:100%;height:auto;display:block;overflow:visible">';

    /* arrow marker */
    svg += '<defs>' +
      '<marker id="fcArr" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">' +
      '<path d="M0,0 L0,6 L10,3 z" fill="#64748b"/></marker>' +
      '</defs>';

    /* edges (draw before boxes) */
    edges.forEach(function (edge) {
      const si = idxById[edge.source];
      const ti = idxById[edge.target];
      if (si == null || ti == null) return;
      const x1 = xs[si] + BW;
      const x2 = xs[ti];
      const my = BY + BH / 2;
      const mx = (x1 + x2) / 2;
      svg += '<line x1="' + x1 + '" y1="' + my + '" x2="' + x2 + '" y2="' + my + '" ' +
        'stroke="#94a3b8" stroke-width="2" marker-end="url(#fcArr)"/>';
      /* relation label */
      const label = short(edge.relation || "", 20);
      const tw = Math.max(60, label.length * 6.5);
      svg += '<rect x="' + (mx - tw / 2) + '" y="' + (my - 11) + '" width="' + tw + '" height="16" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>';
      svg += '<text x="' + mx + '" y="' + (my + 2) + '" text-anchor="middle" ' +
        'font-size="9.5" font-family="system-ui" fill="#334155" font-weight="600">' + esc(label) + '</text>';
    });

    /* boxes */
    items.forEach(function (item, i) {
      const cat  = (item.category || "location").toLowerCase();
      const meta = FLOW_CAT[cat] || FLOW_CAT.location;
      const vid  = item.id || ("c" + i);
      const bx   = xs[i];

      svg += '<g class="vr-svg-node" data-viz-id="' + esc(vid) + '" tabindex="0" role="button" ' +
        'aria-label="' + esc(item.location_name || "") + '" style="cursor:pointer">';

      /* shadow */
      svg += '<rect x="' + (bx + 2) + '" y="' + (BY + 2) + '" width="' + BW + '" height="' + BH + '" rx="12" fill="#0001" opacity="0.08"/>';

      /* box background */
      svg += '<rect x="' + bx + '" y="' + BY + '" width="' + BW + '" height="' + BH + '" ' +
        'rx="12" fill="' + meta.bg + '" stroke="' + meta.stroke + '" stroke-width="2"/>';

      /* left accent bar */
      svg += '<rect x="' + bx + '" y="' + BY + '" width="6" height="' + BH + '" rx="6" fill="' + meta.fill + '"/>';
      svg += '<rect x="' + (bx + 3) + '" y="' + BY + '" width="3" height="' + BH + '" fill="' + meta.fill + '"/>';

      /* stage badge */
      svg += '<circle cx="' + (bx + 22) + '" cy="' + (BY + 20) + '" r="11" fill="' + meta.fill + '"/>';
      svg += '<text x="' + (bx + 22) + '" y="' + (BY + 24) + '" text-anchor="middle" ' +
        'font-size="10" font-weight="800" font-family="system-ui" fill="#fff">' + (i + 1) + '</text>';

      /* category label */
      svg += '<text x="' + (bx + 38) + '" y="' + (BY + 16) + '" ' +
        'font-size="8.5" font-weight="700" font-family="system-ui" fill="' + meta.fill + '" letter-spacing="0.05em">' +
        esc(meta.label.toUpperCase()) + '</text>';

      /* location name (main label) */
      const nameLines = wrapSvgText(item.location_name || "", 18);
      nameLines.forEach(function (line, li) {
        svg += '<text x="' + (bx + 12) + '" y="' + (BY + 40 + li * 16) + '" ' +
          'font-size="12" font-weight="700" font-family="system-ui" fill="#0f172a">' + esc(line) + '</text>';
      });

      /* description (short) */
      const descLines = wrapSvgText(item.description || "", 22).slice(0, 2);
      descLines.forEach(function (line, li) {
        svg += '<text x="' + (bx + 12) + '" y="' + (BY + BH - 22 + li * 13) + '" ' +
          'font-size="9.5" font-family="system-ui" fill="#475569">' + esc(line) + '</text>';
      });

      svg += '</g>';
    });

    svg += '</svg>';

    /* legend */
    const used = {};
    items.forEach(function (item) { used[(item.category || "location").toLowerCase()] = true; });
    let legendHtml = '<div class="vr-legend">';
    Object.keys(used).forEach(function (c) {
      const m = FLOW_CAT[c] || FLOW_CAT.location;
      legendHtml += '<span class="vr-legend-item"><i style="background:' + m.fill + '"></i>' + esc(m.label) + '</span>';
    });
    legendHtml += '</div>';

    const wrap = document.createElement("div");
    wrap.className = "vr-flowchart-wrap";
    wrap.innerHTML = '<div class="vr-diagram-title">Chain-of-Control — evidence-grounded custody sequence</div>' +
      legendHtml + svg +
      '<p class="vr-muted" style="margin-top:8px">Direction of control/custody from the evidence — not a geographic map. Click a stage to see the citation.</p>';
    el.appendChild(wrap);

    /* handlers */
    const byId = {};
    items.forEach(function (item, i) { byId[item.id || ("c" + i)] = item; });

    wrap.querySelectorAll(".vr-svg-node").forEach(function (g) {
      function activate() {
        const vid  = g.getAttribute("data-viz-id");
        const item = byId[vid];
        if (!item) return;
        applySelection(el, vid);
        fireSelect(el, { kind: "chain-of-control", item: item, evidence_ids: item.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
      });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q2  VISUAL COMPARISON SCORECARD
     Criteria in rows, columns side-by-side — large visual badges, no table.
  ══════════════════════════════════════════════════════════════════════ */
  function ComparisonRenderer(el, view) {
    const items = (view.items || []).filter(hasEvidence);
    if (!items.length) {
      el.innerHTML = '<div class="vr-error">No evidence-backed comparison items.</div>';
      return;
    }

    /* collect unique rows and columns preserving order */
    const rows = [], cols = [];
    items.forEach(function (item) {
      if (rows.indexOf(item.row) < 0) rows.push(item.row);
      if (cols.indexOf(item.column) < 0) cols.push(item.column);
    });

    /* lookup helper */
    function cell(r, c) {
      return items.find(function (i) { return i.row === r && i.column === c; });
    }

    /* tone helper */
    function tone(val) {
      const k = String(val || "").toLowerCase().replace(/[^a-z\/]+/g, "");
      return CMP_TONE[k] || CMP_TONE[String(val || "").toLowerCase()] || { bg: "#f8fafc", border: "#e2e8f0", fg: "#0f172a", sym: "?" };
    }

    const root = document.createElement("div");
    root.className = "vr-comparison";

    /* ── header ── */
    const header = document.createElement("div");
    header.className = "vr-cmp-header";
    header.innerHTML = '<div class="vr-cmp-criterion-label"></div>';
    cols.forEach(function (c) {
      header.innerHTML += '<div class="vr-cmp-col-head">' + esc(c) + '</div>';
    });
    root.appendChild(header);

    /* ── rows ── */
    rows.forEach(function (r, ri) {
      const row = document.createElement("div");
      row.className = "vr-cmp-row" + (ri % 2 === 0 ? "" : " vr-cmp-row-alt");

      /* criterion label */
      const crit = document.createElement("div");
      crit.className = "vr-cmp-criterion-label";
      crit.textContent = r;
      row.appendChild(crit);

      /* cells */
      cols.forEach(function (c) {
        const item = cell(r, c);
        const btn  = document.createElement("button");
        btn.type   = "button";
        btn.className = "vr-cmp-cell";
        if (item) {
          const t   = tone(item.value);
          btn.setAttribute("data-viz-id", item.id);
          btn.setAttribute("aria-label", r + " — " + c + ": " + item.value);
          btn.style.background    = t.bg;
          btn.style.borderColor   = t.border;
          btn.style.color         = t.fg;
          /* symbol badge + value */
          btn.innerHTML =
            '<span class="vr-cmp-sym" style="background:' + t.border + ';color:' + t.fg + '">' +
            esc(item.symbol || t.sym) + '</span>' +
            '<span class="vr-cmp-val">' + esc(String(item.value)) + '</span>';
          btn.addEventListener("click", function () {
            applySelection(el, item.id);
            fireSelect(el, { kind: "matrix", item: item, evidence_ids: item.evidence_ids || [] });
          });
          btn.addEventListener("keydown", function (e) {
            if (e.key === "Enter" || e.key === " ") { e.preventDefault(); btn.click(); }
          });
        } else {
          btn.innerHTML = '<span class="vr-cmp-val" style="color:#cbd5e1">—</span>';
          btn.disabled  = true;
        }
        row.appendChild(btn);
      });

      root.appendChild(row);
    });

    /* legend note */
    const note = document.createElement("p");
    note.className = "vr-muted";
    note.style.marginTop = "10px";
    note.textContent = "Qualitative labels only — no numeric scores. Click a cell to see the supporting passage.";
    root.appendChild(note);

    el.appendChild(root);
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q1 / Q3  HIERARCHICAL SVG NETWORK
  ══════════════════════════════════════════════════════════════════════ */
  function NetworkRenderer(el, view) {
    const net   = view.network || { nodes: [], edges: [] };
    const nodes = (net.nodes || []).filter(hasEvidence);
    const nodeIds = {};
    nodes.forEach(function (n) { nodeIds[n.id] = true; });
    const edges = (net.edges || []).filter(function (e) {
      return hasEvidence(e) && nodeIds[e.source] && nodeIds[e.target];
    });

    const wrap = document.createElement("div");
    wrap.className = "vr-network";

    /* legend */
    const legend = document.createElement("div");
    legend.className = "vr-legend";
    const used = {};
    nodes.forEach(function (n) { used[(n.entity_type || "entity").toLowerCase()] = true; });
    Object.keys(used).forEach(function (t) {
      const m = ENTITY_COLORS[t] || ENTITY_COLORS.entity;
      const chip = document.createElement("span");
      chip.className = "vr-legend-item";
      chip.innerHTML = '<i style="background:' + m.fill + '"></i>' + esc(m.label);
      legend.appendChild(chip);
    });
    wrap.appendChild(legend);

    /* positions by layer */
    const layers = {};
    nodes.forEach(function (n, i) {
      const L = n.layer != null ? Number(n.layer) : i;
      (layers[L] = layers[L] || []).push(n);
    });
    const layerKeys = Object.keys(layers).map(Number).sort(function (a, b) { return a - b; });
    const W = 720, rowH = 110;
    const H = Math.max(300, layerKeys.length * rowH + 40);
    const pos = {};
    layerKeys.forEach(function (L, li) {
      const row = layers[L];
      const y   = 40 + li * rowH;
      row.forEach(function (n, i) {
        pos[n.id] = { x: ((i + 1) * W) / (row.length + 1), y: y };
      });
    });

    /* edges */
    let edgesSvg = "";
    edges.forEach(function (e, i) {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      const mx   = (a.x + b.x) / 2;
      const my   = (a.y + b.y) / 2;
      const label = short(e.relation || "", 22);
      const tw   = Math.max(54, label.length * 6.4);
      const eid  = "edge-" + i;
      edgesSvg +=
        '<g class="vr-svg-edge" data-viz-id="' + eid + '" tabindex="0" style="cursor:pointer">' +
        '<line x1="' + a.x + '" y1="' + a.y + '" x2="' + b.x + '" y2="' + b.y +
        '" stroke="#94a3b8" stroke-width="2" marker-end="url(#netArr)"/>' +
        '<rect x="' + (mx - tw / 2) + '" y="' + (my - 10) + '" width="' + tw + '" height="18" rx="9" fill="#fff" stroke="#e2e8f0"/>' +
        '<text x="' + mx + '" y="' + (my + 3) + '" text-anchor="middle" font-size="10" font-family="system-ui" fill="#334155">' +
        esc(label) + '</text></g>';
    });

    /* nodes */
    let nodesSvg = "";
    nodes.forEach(function (n) {
      const p    = pos[n.id];
      const t    = (n.entity_type || "entity").toLowerCase();
      const meta = ENTITY_COLORS[t] || ENTITY_COLORS.entity;
      const label = short(n.label || n.id, 16);
      nodesSvg +=
        '<g class="vr-svg-node" data-viz-id="' + esc(n.id) + '" tabindex="0" style="cursor:pointer">' +
        '<circle cx="' + p.x + '" cy="' + p.y + '" r="24" fill="' + meta.fill + '" stroke="' + meta.ring + '" stroke-width="3"/>' +
        '<text x="' + p.x + '" y="' + (p.y + 40) + '" text-anchor="middle" font-size="12" font-weight="700" font-family="system-ui" fill="#0f172a">' +
        esc(label) + '</text></g>';
    });

    const box = document.createElement("div");
    box.className = "vr-network-canvas";
    box.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Relationship graph" style="width:100%;height:auto;display:block">' +
      '<defs><marker id="netArr" markerWidth="10" markerHeight="10" refX="20" refY="3" orient="auto">' +
      '<path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker></defs>' +
      edgesSvg + nodesSvg + '</svg>';
    wrap.appendChild(box);
    el.appendChild(wrap);

    /* handlers */
    const byId = {};
    nodes.forEach(function (n) { byId[n.id] = n; });

    box.querySelectorAll(".vr-svg-node").forEach(function (g) {
      function activate() {
        const id = g.getAttribute("data-viz-id");
        const n  = byId[id];
        if (!n) return;
        applySelection(el, id);
        fireSelect(el, { kind: "network-node", item: n, evidence_ids: n.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
    });

    box.querySelectorAll(".vr-svg-edge").forEach(function (g, i) {
      function activate() {
        const edge = edges[i];
        if (!edge) return;
        applySelection(el, g.getAttribute("data-viz-id"));
        fireSelect(el, { kind: "network-edge", item: edge, evidence_ids: edge.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════
     SHARED TREE LAYOUT  — used by StatuteTreeRenderer & DecisionTreeRenderer
  ══════════════════════════════════════════════════════════════════════ */
  function buildTreeLayout(items, NW, NH, GX, GY, PADX, PADY) {
    var childrenOf = {};
    var rootIds    = [];
    items.forEach(function (item) { childrenOf[item.id] = []; });
    items.forEach(function (item) {
      if (item.parent && childrenOf[item.parent]) {
        childrenOf[item.parent].push(item.id);
      } else if (!item.parent) {
        rootIds.push(item.id);
      }
    });

    function subtreeW(id) {
      var kids = childrenOf[id] || [];
      if (!kids.length) return NW;
      var total = kids.reduce(function (s, k) { return s + subtreeW(k) + GX; }, -GX);
      return Math.max(NW, total);
    }

    var positions = {};
    function assignPos(id, cx, y) {
      positions[id] = { x: Math.round(cx), y: Math.round(y) };
      var kids = childrenOf[id] || [];
      if (!kids.length) return;
      var totalW = kids.reduce(function (s, k) { return s + subtreeW(k) + GX; }, -GX);
      var x = cx - totalW / 2;
      kids.forEach(function (k) {
        var sw = subtreeW(k);
        assignPos(k, x + sw / 2, y + NH + GY);
        x += sw + GX;
      });
    }

    var rootId   = rootIds[0] || (items[0] && items[0].id);
    var totalW   = subtreeW(rootId);
    var canvasW  = Math.max(560, totalW + 2 * PADX);
    assignPos(rootId, canvasW / 2, PADY);
    var maxY = 0;
    Object.keys(positions).forEach(function (k) { if (positions[k].y > maxY) maxY = positions[k].y; });
    var canvasH = maxY + NH + PADY;

    return { positions: positions, childrenOf: childrenOf, canvasW: canvasW, canvasH: canvasH };
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q6  STATUTE TREE — rights / obligations hierarchy
     Rounded-rect nodes colored by category; bezier edges; click for details.
  ══════════════════════════════════════════════════════════════════════ */
  function StatuteTreeRenderer(el, view) {
    var items = (view.items || []).filter(hasEvidence);
    if (!items.length) {
      el.innerHTML = '<div class="vr-error">No evidence-backed items.</div>';
      return;
    }

    var NW = 110, NH = 54, GX = 14, GY = 66, PADX = 44, PADY = 34;
    var layout = buildTreeLayout(items, NW, NH, GX, GY, PADX, PADY);
    var pos = layout.positions, canvasW = layout.canvasW, canvasH = layout.canvasH;

    var byId = {};
    items.forEach(function (item) { byId[item.id] = item; });

    var edgesSvg = "";
    items.forEach(function (item) {
      if (!item.parent) return;
      var pp = pos[item.parent], cp = pos[item.id];
      if (!pp || !cp) return;
      var x1 = pp.x, y1 = pp.y + NH, x2 = cp.x, y2 = cp.y;
      var cy = y1 + (y2 - y1) * 0.5;
      edgesSvg += '<path d="M' + x1 + ',' + y1 + ' C' + x1 + ',' + cy + ' ' + x2 + ',' + cy + ' ' + x2 + ',' + y2 + '" ' +
        'fill="none" stroke="#94a3b8" stroke-width="1.8"/>';
    });

    var nodesSvg = "";
    items.forEach(function (item) {
      var p = pos[item.id];
      if (!p) return;
      var meta = TREE_COLORS[item.category || "default"] || TREE_COLORS.default;
      var lines = wrapSvgText(item.label || item.id, 13);
      var lineH = 14, textH = lines.length * lineH;
      var textY = p.y + (NH - textH) / 2 + lineH - 1;
      nodesSvg +=
        '<g class="vr-svg-node" data-viz-id="' + esc(item.id) + '" tabindex="0" style="cursor:pointer">' +
        '<rect x="' + (p.x - NW / 2) + '" y="' + p.y + '" width="' + NW + '" height="' + NH + '" rx="7" ' +
        'fill="' + meta.fill + '" stroke="' + meta.stroke + '" stroke-width="2"/>';
      lines.forEach(function (line, i) {
        nodesSvg += '<text x="' + p.x + '" y="' + (textY + i * lineH) + '" text-anchor="middle" ' +
          'font-size="10.5" font-weight="600" font-family="system-ui,sans-serif" fill="' + meta.text + '">' +
          esc(line) + '</text>';
      });
      nodesSvg += '</g>';
    });

    var cats = [];
    items.forEach(function (item) {
      var c = item.category || "default";
      if (cats.indexOf(c) < 0) cats.push(c);
    });

    var wrap = document.createElement("div");
    wrap.className = "vr-statute-tree-wrap";
    wrap.innerHTML =
      '<svg viewBox="0 0 ' + canvasW + ' ' + canvasH + '" role="img" aria-label="Rights hierarchy tree" ' +
      'style="width:100%;height:auto;display:block;overflow:visible">' +
      edgesSvg + nodesSvg + '</svg>';

    var legendHtml = '<div class="vr-tree-legend">';
    cats.forEach(function (cat) {
      var meta  = TREE_COLORS[cat] || TREE_COLORS.default;
      var entry = (view.legend || []).find(function (l) { return l.key === cat; });
      legendHtml += '<span class="vr-tree-leg-item"><span class="vr-tree-leg-swatch" ' +
        'style="background:' + meta.fill + ';border-color:' + meta.stroke + '"></span>' +
        esc((entry && entry.label) || cat) + '</span>';
    });
    legendHtml += '</div>';
    wrap.insertAdjacentHTML("beforeend", legendHtml);

    if (view.limited_evidence) {
      wrap.insertAdjacentHTML("beforeend",
        '<p class="vr-limited-note">\u26a0 Limited evidence \u2014 some relationships inferred from query.</p>');
    }
    el.appendChild(wrap);

    wrap.querySelectorAll(".vr-svg-node").forEach(function (g) {
      function activate() {
        var id = g.getAttribute("data-viz-id");
        var item = byId[id];
        if (!item) return;
        applySelection(el, id);
        fireSelect(el, { kind: "tree-node", item: item, evidence_ids: item.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
      });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════
     Q8  DECISION TREE — legal analysis branching with labeled edges
     Rounded-rect nodes; outcome colors (allowed/prohibited); click for details.
  ══════════════════════════════════════════════════════════════════════ */
  function DecisionTreeRenderer(el, view) {
    var items = (view.items || []).filter(hasEvidence);
    if (!items.length) {
      el.innerHTML = '<div class="vr-error">No evidence-backed items.</div>';
      return;
    }

    var NW = 134, NH = 60, GX = 20, GY = 80, PADX = 50, PADY = 38;
    var layout = buildTreeLayout(items, NW, NH, GX, GY, PADX, PADY);
    var pos = layout.positions, canvasW = layout.canvasW, canvasH = layout.canvasH;

    var byId = {};
    items.forEach(function (item) { byId[item.id] = item; });

    var edgesSvg =
      '<defs><marker id="dtArr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/></marker></defs>';

    items.forEach(function (item) {
      if (!item.parent) return;
      var pp = pos[item.parent], cp = pos[item.id];
      if (!pp || !cp) return;
      var x1 = pp.x, y1 = pp.y + NH, x2 = cp.x, y2 = cp.y;
      var cy = y1 + (y2 - y1) * 0.48;

      edgesSvg += '<path d="M' + x1 + ',' + y1 + ' C' + x1 + ',' + cy + ' ' + x2 + ',' + cy + ' ' + x2 + ',' + y2 + '" ' +
        'fill="none" stroke="#94a3b8" stroke-width="2" marker-end="url(#dtArr)"/>';

      var label = item.edge_label || "";
      if (label) {
        var mx = (x1 + x2) / 2;
        var my = y1 + (y2 - y1) * 0.44;
        var lines = wrapSvgText(label, 15);
        var tw = Math.max(56, Math.max.apply(null, lines.map(function (l) { return l.length; })) * 6.2 + 14);
        var th = lines.length * 13 + 8;
        edgesSvg += '<rect x="' + (mx - tw / 2) + '" y="' + (my - th / 2) + '" width="' + tw + '" height="' + th +
          '" rx="5" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>';
        lines.forEach(function (line, li) {
          edgesSvg += '<text x="' + mx + '" y="' + (my - th / 2 + 10 + li * 13) + '" text-anchor="middle" ' +
            'font-size="9.5" font-family="system-ui,sans-serif" fill="#475569">' + esc(line) + '</text>';
        });
      }
    });

    var nodesSvg = "";
    items.forEach(function (item) {
      var p = pos[item.id];
      if (!p) return;
      var colorKey = item.outcome || item.node_type || "default";
      var meta = DT_COLORS[colorKey] || DT_COLORS.default;
      var lines = wrapSvgText(item.label || item.id, 15);
      var lineH = 14, textH = lines.length * lineH;
      var textY = p.y + (NH - textH) / 2 + lineH - 1;

      nodesSvg +=
        '<g class="vr-svg-node" data-viz-id="' + esc(item.id) + '" tabindex="0" style="cursor:pointer">' +
        '<rect x="' + (p.x - NW / 2) + '" y="' + p.y + '" width="' + NW + '" height="' + NH + '" rx="8" ' +
        'fill="' + meta.fill + '" stroke="' + meta.stroke + '" stroke-width="2.5"/>';
      lines.forEach(function (line, i) {
        nodesSvg += '<text x="' + p.x + '" y="' + (textY + i * lineH) + '" text-anchor="middle" ' +
          'font-size="11" font-weight="600" font-family="system-ui,sans-serif" fill="' + meta.text + '">' +
          esc(line) + '</text>';
      });
      nodesSvg += '</g>';
    });

    var wrap = document.createElement("div");
    wrap.className = "vr-decision-tree-wrap";
    wrap.innerHTML =
      '<svg viewBox="0 0 ' + canvasW + ' ' + canvasH + '" role="img" aria-label="Legal decision tree" ' +
      'style="width:100%;height:auto;display:block;overflow:visible">' +
      edgesSvg + nodesSvg + '</svg>';

    var LEGEND_ENTRIES = [
      { key: "question",   label: "Decision point" },
      { key: "allowed",    label: "Permitted"      },
      { key: "prohibited", label: "Prohibited"     },
    ];
    var legendHtml = '<div class="vr-tree-legend">';
    LEGEND_ENTRIES.forEach(function (entry) {
      var meta = DT_COLORS[entry.key] || DT_COLORS.default;
      legendHtml += '<span class="vr-tree-leg-item"><span class="vr-tree-leg-swatch" ' +
        'style="background:' + meta.fill + ';border-color:' + meta.stroke + '"></span>' +
        esc(entry.label) + '</span>';
    });
    legendHtml += '</div>';
    wrap.insertAdjacentHTML("beforeend", legendHtml);
    el.appendChild(wrap);

    wrap.querySelectorAll(".vr-svg-node").forEach(function (g) {
      function activate() {
        var id = g.getAttribute("data-viz-id");
        var item = byId[id];
        if (!item) return;
        applySelection(el, id);
        fireSelect(el, { kind: "tree-node", item: item, evidence_ids: item.evidence_ids || [] });
      }
      g.addEventListener("click", activate);
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }
      });
    });
  }

  /* ── Evidence fallback ──────────────────────────────────────────────── */
  function EvidenceRenderer(el, view) {
    const items = (view.items || []).filter(hasEvidence);
    const root  = document.createElement("div");
    root.className = "vr-evidence-board";
    items.forEach(function (item, idx) {
      const card = document.createElement("button");
      card.type  = "button";
      card.className = "vr-ev-card";
      card.setAttribute("data-viz-id", item.id || "e" + idx);
      card.innerHTML =
        '<div class="vr-ev-title">' + esc(item.title || "") + '</div>' +
        '<div class="vr-ev-text">' + esc(item.text || "") + '</div>';
      card.addEventListener("click", function () {
        applySelection(el, item.id || "e" + idx);
        fireSelect(el, { kind: "evidence", item: item, evidence_ids: item.evidence_ids || [] });
      });
      root.appendChild(card);
    });
    el.appendChild(root);
  }

  /* ─── exports ───────────────────────────────────────────────────────── */
  global.VizRenderers = {
    renderPrimaryView: renderPrimaryView,
    setSelectHandler:  function (fn) { /* legacy shim */ },
    clearMount:        clearMount,
    applySelection:    applySelection,
    ALLOWED: ["timeline", "map", "network", "matrix", "statute_tree", "decision_tree", "evidence_view"],
  };

})(window);
