(function () {
  "use strict";

  function showBootError(err) {
    var workspace = document.getElementById("workspace");
    var msg = err && err.stack ? err.stack : String(err);
    if (workspace) {
      workspace.innerHTML =
        '<div class="error"><strong>Demo failed to load.</strong><pre style="white-space:pre-wrap">' +
        String(msg)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;") +
        "</pre></div>";
    }
    if (window.console && console.error) console.error(err);
  }

  try {
    function esc(s) {
      return String(s == null ? "" : s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }
    function truncate(s, n) {
      s = String(s || "");
      return s.length > n ? s.slice(0, n) + "…" : s;
    }
    function conciseAnswer(text, maxLen) {
      text = String(text || "").replace(/\s+/g, " ").trim();
      if (!text) return "(missing)";
      maxLen = maxLen || 420;
      if (text.length <= maxLen) return text;
      var cut = text.slice(0, maxLen);
      var stop = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("; "));
      if (stop > maxLen * 0.55) return cut.slice(0, stop + 1);
      return cut.replace(/\s+\S*$/, "") + "…";
    }

    var VIEW_LABEL = {
      evidence_view:  "Evidence cards",
      timeline:       "Chronological timeline",
      network:        "Evidence-backed relationship graph",
      map:            "Data flow diagram",
      matrix:         "Qualitative comparison matrix",
      statute_tree:   "Rights hierarchy tree",
      decision_tree:  "Legal decision tree",
    };

    var TYPE_WHY = {
      temporal:    "Temporal narrative → chronological sequence grounded in the hypo and answer.",
      geographic:  "Control/custody narrative → chain-of-control diagram (not a geographic map).",
      comparison:  "Comparison question → qualitative matrix of what is / is not established.",
      relational:  "Party/doctrine relations → hierarchical relationship graph with evidence links.",
      contractual: "Contract structure → rights hierarchy tree showing granted rights and conditions.",
      analytical:  "Multi-step legal test → decision tree showing permitted and prohibited paths.",
      process:     "Process/flow question → data flow diagram showing custody and handoffs.",
    };

    var workspace = document.getElementById("workspace");
    var statusEl = document.getElementById("load-status");
    var nav = document.getElementById("q-nav");
    var bundle = window.ANSWER_VIZ_BUNDLE;
    var renderErrors = [];
    var currentMount = null;
    var currentSpec = null;

    if (!workspace) throw new Error("Missing #workspace element");
    workspace.innerHTML = "";

    if (!bundle || !bundle.queries || !bundle.queries.length) {
      workspace.innerHTML =
        '<div class="error">Could not load demo data (<code>data/answer_viz/bundle.js</code>). Serve this folder with <code>python3 -m http.server 8765</code> and open <code>http://127.0.0.1:8765/answer-viz.html</code>.</div>';
      if (statusEl) statusEl.textContent = "Bundle missing";
      return;
    }
    if (!window.VizRenderers) {
      workspace.innerHTML = '<div class="error">viz-renderers.js failed to load.</div>';
      if (statusEl) statusEl.textContent = "Renderers missing";
      return;
    }
    if (!nav) throw new Error("Missing #q-nav element");

    var byId = {};
    bundle.queries.forEach(function (spec) {
      byId[String(spec.query_id)] = spec;
    });

    function primaryLabel(spec) {
      var t = (spec.primary_view || {}).type;
      return VIEW_LABEL[t] || t || "Visualization";
    }

    function countVisualItems(view) {
      if (!view) return 0;
      if (view.type === "network") {
        var n = (view.network || {}).nodes || [];
        var e = (view.network || {}).edges || [];
        return n.filter(hasEv).length + e.filter(hasEv).length;
      }
      return (view.items || []).filter(hasEv).length;
    }
    function hasEv(item) {
      return Array.isArray(item.evidence_ids) && item.evidence_ids.length > 0;
    }

    function fillList(root, sel, arr, empty) {
      var ul = root.querySelector(sel);
      if (!ul) return;
      var items = arr || [];
      if (!items.length) {
        ul.innerHTML = "<li>" + esc(empty) + "</li>";
        return;
      }
      ul.innerHTML = "";
      items.forEach(function (t) {
        var li = document.createElement("li");
        li.textContent = t;
        ul.appendChild(li);
      });
    }

    function renderQuery(qid) {
      var spec = byId[String(qid)];
      if (!spec) return;
      currentSpec = spec;
      var src = spec._source || {};
      var primary = spec.primary_view || {};
      var pLabel = primaryLabel(spec);
      var limited = !!spec.limited_evidence;
      var answer = conciseAnswer(src.generated_answer || spec.fallback_text || "");

      [].slice.call(nav.querySelectorAll("button")).forEach(function (btn) {
        btn.classList.toggle("active", btn.getAttribute("data-qid") === String(qid));
      });

      workspace.innerHTML =
        '<section class="card" id="query-' +
        esc(String(qid)) +
        '">' +
        '<div class="card-head">' +
        "<strong style='font-family:Georgia,serif'>Query " +
        esc(String(qid)) +
        "</strong>" +
        '<span class="badge ink">' +
        esc(spec.query_type || "—") +
        "</span>" +
        '<span class="badge blue">' +
        esc(pLabel) +
        "</span>" +
        (limited
          ? '<span class="badge warn">Limited Evidence</span>'
          : '<span class="badge ok">Evidence-backed</span>') +
        '<span style="font-size:13.5px;font-weight:700;color:#334155">' +
        esc(spec.title || "") +
        "</span>" +
        "</div>" +
        '<div class="layout">' +
        '<aside class="pane">' +
        (limited
          ? '<div class="limited-banner"><strong>Limited Evidence</strong>Retrieved documents do not support a confident ruling. Supported facts are separated from what is not established.</div>'
          : "") +
        '<div class="lbl">Query</div><p class="body">' +
        esc(truncate(src.query || "(missing)", 900)) +
        "</p>" +
        '<div class="answer-box"><div class="lbl">Gemini answer</div><p class="body">' +
        esc(answer) +
        "</p></div>" +
        '<div class="fact-grid">' +
        '<div class="fact-box yes"><h5>Supported by retrieved evidence</h5><ul class="est-list"></ul></div>' +
        '<div class="fact-box no"><h5>Not established by available evidence</h5><ul class="not-list"></ul></div>' +
        "</div>" +
        "</aside>" +
        '<div class="pane">' +
        '<div class="viz-head"><strong style="font-size:14px">Primary visualization</strong>' +
        '<span style="font-size:11px;font-weight:700;color:#64748b">' +
        esc(pLabel) +
        " · " +
        countVisualItems(primary) +
        " items</span></div>" +
        '<div class="viz-stage"><div class="viz-mount primary-mount"></div></div>' +
        '<div class="selection-panel selection" id="selection-panel">Select a visual element to inspect explanation and evidence.</div>' +
        "</div>" +
        '<aside class="pane">' +
        "<strong style='font-size:13px;display:block;margin-bottom:8px'>Details / evidence</strong>" +
        '<div class="evidence-list" style="max-height:36rem;overflow:auto"></div>' +
        "</aside>" +
        "</div>" +
        '<details class="addl"><summary>Additional details</summary><div class="addl-mount"></div></details>' +
        '<details class="about"><summary>About this visualization</summary>' +
        "<p style='margin:8px 0 4px'><strong>Why this format:</strong> " +
        esc(TYPE_WHY[spec.query_type] || "Query-dependent structured view.") +
        "</p>" +
        "<p style='margin:4px 0;color:#64748b'>" +
        esc(spec.summary || "") +
        "</p>" +
        '<ul class="lim-list" style="margin:6px 0 0;padding-left:18px;color:#9a3412"></ul>' +
        "</details>" +
        "</section>";

      var root = workspace.querySelector(".card");
      fillList(root, ".est-list", spec.established_facts, "See query and evidence panel.");
      fillList(root, ".not-list", spec.not_established, "None flagged.");
      // Filter leftover score/chart wording from older limitations
      var lims = (spec.limitations || []).filter(function (t) {
        return !/score|percent|ranking|confidence|chart are didactic/i.test(String(t));
      });
      fillList(root, ".lim-list", lims, "None reported.");

      var evidenceById = {};
      var evList = root.querySelector(".evidence-list");
      (spec.evidence_links || []).forEach(function (e) {
        evidenceById[e.evidence_id] = e;
        var card = document.createElement("div");
        card.className = "ev-side";
        card.setAttribute("data-eid", e.evidence_id);
        var cite = (e.citation || e.document_id || "").split("/").pop() || "";
        card.innerHTML =
          '<div style="font-weight:700;color:#0f766e;margin-bottom:3px">' +
          esc(e.evidence_id) +
          " · " +
          esc(truncate(cite, 28)) +
          "</div>" +
          '<div style="white-space:pre-wrap;color:#334155">' +
          esc(truncate(e.passage, 360)) +
          "</div>";
        evList.appendChild(card);
      });

      var selection = root.querySelector(".selection");
      currentMount = root.querySelector(".primary-mount");

      function highlightEvidence(ids) {
        var set = {};
        (ids || []).forEach(function (id) {
          set[id] = true;
        });
        evList.querySelectorAll("[data-eid]").forEach(function (node) {
          node.classList.toggle(
            "evidence-hit",
            !!set[node.getAttribute("data-eid")]
          );
        });
      }

      function clearSelection() {
        if (currentMount && currentMount._vizClearSelection) {
          currentMount._vizClearSelection();
        }
        highlightEvidence([]);
        selection.innerHTML =
          "Select a visual element to inspect explanation and evidence.";
      }

      function showSelection(payload) {
        var item = payload.item || {};
        var eids = payload.evidence_ids || [];
        var label =
          item.event ||
          item.label ||
          item.location_name ||
          item.title ||
          (item.row && item.column
            ? item.row + " × " + item.column
            : item.relation || item.id || "");
        var detail =
          item.description ||
          item.text ||
          item.relation ||
          (item.value != null ? String(item.value) : "") ||
          "";
        var passages = eids
          .map(function (id) {
            return evidenceById[id];
          })
          .filter(Boolean);

        var html =
          "<div style='display:grid;gap:4px'>" +
          "<div><strong>Selected:</strong> " +
          esc(payload.kind) +
          "</div>" +
          "<div><strong>Label:</strong> " +
          esc(label) +
          "</div>" +
          "<div><strong>Explanation:</strong> " +
          esc(detail) +
          "</div>";
        if (eids.length) {
          html +=
            "<div><strong>Evidence IDs:</strong> " + esc(eids.join(", ")) + "</div>";
          passages.forEach(function (p) {
            html +=
              '<div style="margin-top:6px;padding:8px;border-radius:8px;background:#f8fafc;border:1px solid #e2e8f0">' +
              "<div style='font-weight:700;color:#0f766e;margin-bottom:3px'>" +
              esc(p.evidence_id) +
              "</div>" +
              '<div style="white-space:pre-wrap;font-size:12px">' +
              esc(truncate(p.passage, 420)) +
              "</div></div>";
          });
        }
        html +=
          '<div class="sel-actions"><button type="button" class="btn-clear" id="btn-clear-sel">Clear selection</button></div></div>';
        selection.innerHTML = html;
        highlightEvidence(eids);
        var clearBtn = selection.querySelector("#btn-clear-sel");
        if (clearBtn) clearBtn.addEventListener("click", clearSelection);
      }

      try {
        window.VizRenderers.renderPrimaryView(
          currentMount,
          primary,
          evidenceById,
          {
            onSelect: showSelection,
            diagramLabel: pLabel,
          }
        );
      } catch (err) {
        renderErrors.push({ qid: String(qid), where: "primary", message: err.message });
        currentMount.innerHTML =
          '<div class="error">Render error: ' + esc(err.message) + "</div>";
      }

      // Collapsed additional details (secondary material only)
      var addlHost = root.querySelector(".addl-mount");
      var addl = spec.additional_details;
      if (addl && addl.type) {
        try {
          window.VizRenderers.renderPrimaryView(addlHost, addl, evidenceById, {
            onSelect: showSelection,
          });
        } catch (err) {
          renderErrors.push({
            qid: String(qid),
            where: "additional",
            message: err.message,
          });
          addlHost.innerHTML =
            '<div class="error">' + esc(err.message) + "</div>";
        }
      } else {
        root.querySelector("details.addl").style.display = "none";
      }

      if (statusEl) {
        statusEl.textContent =
          "Showing Q" +
          qid +
          " · " +
          pLabel +
          " · " +
          countVisualItems(primary) +
          " evidence-backed items";
      }
    }

    // Build Q1–Q10 selector
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].forEach(function (qid) {
      if (!byId[String(qid)]) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.setAttribute("data-qid", String(qid));
      btn.textContent = "Q" + qid;
      btn.addEventListener("click", function () {
        renderQuery(qid);
      });
      nav.appendChild(btn);
    });

    window.__ANSWER_VIZ_RENDER_ERRORS__ = renderErrors;
    window.__ANSWER_VIZ_SELECT__ = function (qid) {
      renderQuery(qid);
    };

    renderQuery(1);
  } catch (err) {
    showBootError(err);
  }
})();
