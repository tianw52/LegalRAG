#!/usr/bin/env python3
"""
LegalBenchRAG Eval Viewer — stdlib only, no extra installs.

Usage:
    python scripts/eval_viewer.py [PORT]          # default 8765
    python scripts/eval_viewer.py 9000

Then open http://<lab-machine-ip>:PORT in your browser.
VPN users can access directly — no port-forwarding needed.
"""
from __future__ import annotations
import http.server
import json
import pathlib
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

BASE           = pathlib.Path(__file__).resolve().parent.parent
CORPUS_DIR     = BASE / "data/legalbenchrag-mini/corpus"
LOGS_DIR       = BASE / "logs/eval/legalbenchrag-mini"
BENCHMARKS_DIR = BASE / "data/legalbenchrag-mini/benchmarks"

_trace_cache:  dict[tuple, list]          = {}
_ds_offsets:   dict[tuple, dict[str,int]] = {}  # (model,embedder) → {dataset: min_query_idx}
_corpus_cache: dict[str, str]             = {}
_gt_cache:     dict[str, list]            = {}  # dataset → list of test dicts


# ── data helpers ──────────────────────────────────────────────────────────────

def list_models() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir():
            continue
        embs = sorted(
            f.stem.removeprefix("lbr_hier_")
            for f in d.glob("lbr_hier_*.jsonl")
        )
        if embs:
            result[d.name] = embs
    return result


def _load_trace(model: str, embedder: str) -> list[dict]:
    key = (model, embedder)
    if key not in _trace_cache:
        path = LOGS_DIR / model / f"lbr_hier_{embedder}.jsonl"
        rows: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "query" in obj:
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
        # Compute min query_idx per dataset so reformulated-query traces can
        # still look up ground-truth snippets by position (not by text match).
        ds_min: dict[str, int] = {}
        for row in rows:
            ds = row.get("tags", ["unknown"])[0]
            qi = row.get("query_idx", 1)
            if ds not in ds_min or qi < ds_min[ds]:
                ds_min[ds] = qi
        _trace_cache[key] = rows
        _ds_offsets[key]  = ds_min
    return _trace_cache[key]


def _gt_tests(dataset: str) -> list[dict]:
    """Indexed list of benchmark tests for a dataset."""
    if dataset not in _gt_cache:
        path = BENCHMARKS_DIR / f"{dataset}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        _gt_cache[dataset] = data["tests"]
    return _gt_cache[dataset]


def _corpus_text(file: str, s: int, e: int) -> str:
    if file not in _corpus_cache:
        p = CORPUS_DIR / file
        try:
            _corpus_cache[file] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _corpus_cache[file] = ""
    return _corpus_cache[file][s:e]


# ── API response builders ─────────────────────────────────────────────────────

def api_models() -> dict:
    return list_models()


def api_queries(model: str, embedder: str, dataset: str | None) -> list:
    rows = _load_trace(model, embedder)
    out = []
    for i, e in enumerate(rows):
        tags = e.get("tags", [])
        ds = tags[0] if tags else "unknown"
        if dataset and ds != dataset:
            continue
        mbk  = e.get("metrics_by_k", [])
        last = mbk[-1] if mbk else {}
        out.append({
            "idx":             i,
            "dataset":         ds,
            "query":           e["query"][:200],
            "n_gt_snippets":   last.get("n_gt_snippets", 0),
            "n_gt_hit":        last.get("n_gt_hit", 0),
            "char_recall_max": round(last.get("char_recall", 0), 4),
            "k_values":        [m["k"] for m in mbk],
            "char_recalls":    [round(m.get("char_recall", 0), 4) for m in mbk],
        })
    return out


def api_query(model: str, embedder: str, idx: int) -> dict:
    rows = _load_trace(model, embedder)
    e    = rows[idx]
    tags = e.get("tags", [])
    ds   = tags[0] if tags else "unknown"

    # Use query_idx (1-based, global) to find the matching benchmark test.
    # This works for reformulated-query models whose `query` text doesn't match
    # the original benchmark text.
    query_idx  = e.get("query_idx", 1)
    ds_offset  = _ds_offsets.get((model, embedder), {}).get(ds, 1)
    test_idx   = query_idx - ds_offset
    tests      = _gt_tests(ds)
    test       = tests[test_idx] if 0 <= test_idx < len(tests) else {}
    snippets   = test.get("snippets", [])

    # Hit map from last-K metrics — covers all 64 retrieved chunks
    mbk  = e.get("metrics_by_k", [])
    last = mbk[-1] if mbk else {}
    hit_by_id: dict[str, dict] = {
        c["chunk_id"]: c for c in last.get("chunk_hits", [])
    }

    retrieved = []
    for chunk in e.get("retrieved_all", []):
        cid   = chunk["chunk_id"]
        hinfo = hit_by_id.get(cid, {})
        text  = _corpus_text(chunk["file"], chunk["char_start"], chunk["char_end"])
        retrieved.append({
            "rank":        chunk["rank"],
            "file":        chunk["file"],
            "char_start":  chunk["char_start"],
            "char_end":    chunk["char_end"],
            "score":       chunk["score"],
            "text":        text,
            "is_hit":      hinfo.get("is_chunk_hit", False),
            "gt_overlaps": hinfo.get("gt_overlaps", []),
        })

    return {
        "dataset":        ds,
        "query":          e["query"],
        "original_query": test.get("query", ""),   # original benchmark query
        "ground_truth":   snippets,
        "total_gt_chars": e.get("total_gt_chars", 0),
        "retrieved":      retrieved,
        "metrics_by_k":   mbk,
    }


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LegalBenchRAG Eval Viewer</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  html, body { height: 100%; margin: 0; }
  mark.gt { background: #fef08a; border-radius: 2px; padding: 0 1px; }
  .chunk-hit  { background: #f0fdf4; border-color: #86efac; }
  .chunk-miss { background: #f9fafb; border-color: #e5e7eb; }
  .q-item { border-left: 3px solid transparent; }
  .q-item.active { background: #eff6ff; border-left-color: #3b82f6; }
  .q-item:hover:not(.active) { background: #f8fafc; }
  pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; }
  .line-clamp-2 {
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
  }
  .chart-box { position: relative; height: 220px; }
</style>
</head>
<body class="h-screen flex flex-col bg-gray-100 text-gray-900" style="font-size:13px">

<!-- ── header ── -->
<header class="bg-white border-b px-4 py-2 flex flex-wrap items-center gap-2 flex-shrink-0 shadow-sm">
  <span class="font-bold text-gray-800 mr-1">⚖ LegalBenchRAG Eval Viewer</span>

  <label class="text-gray-500 text-xs">Query model</label>
  <select id="sel-model" class="border rounded px-2 py-1 bg-white text-xs"></select>

  <label class="text-gray-500 text-xs">Embedder</label>
  <select id="sel-embedder" class="border rounded px-2 py-1 bg-white text-xs"></select>

  <label class="text-gray-500 text-xs">Dataset</label>
  <select id="sel-dataset" class="border rounded px-2 py-1 bg-white text-xs">
    <option value="">All</option>
    <option value="contractnli">contractnli</option>
    <option value="cuad">cuad</option>
    <option value="maud">maud</option>
    <option value="privacy_qa">privacy_qa</option>
  </select>

  <input id="search" type="text" placeholder="Filter queries…"
         class="border rounded px-2 py-1 text-xs w-52">

  <div id="stats" class="ml-auto text-xs text-gray-400"></div>
</header>

<!-- ── main ── -->
<div class="flex-1 flex overflow-hidden">

  <!-- query list -->
  <aside class="w-72 flex-shrink-0 bg-white border-r flex flex-col overflow-hidden">
    <div id="list-hdr" class="px-3 py-1.5 text-xs text-gray-500 border-b bg-gray-50 flex-shrink-0">
      Loading…
    </div>
    <div id="query-list" class="overflow-y-auto flex-1"></div>
  </aside>

  <!-- detail -->
  <main id="detail" class="flex-1 overflow-y-auto">
    <div class="flex items-center justify-center h-full text-gray-400 text-sm">
      Select a query from the list
    </div>
  </main>

</div>

<script>
// ── state ─────────────────────────────────────────────────────────────────────
let curModel = '', curEmbedder = '', curDataset = '';
let allQueries = [], filteredQueries = [];
let _charts = {};

const $ = id => document.getElementById(id);
const esc = s => String(s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

// ── colours ───────────────────────────────────────────────────────────────────
const DS_COLORS = {
  contractnli: 'bg-blue-100 text-blue-700',
  cuad:        'bg-purple-100 text-purple-700',
  maud:        'bg-teal-100 text-teal-700',
  privacy_qa:  'bg-rose-100 text-rose-700',
};
function dsBadge(ds) {
  const cls = DS_COLORS[ds] || 'bg-gray-100 text-gray-600';
  return `<span class="px-1.5 py-0.5 rounded text-xs font-medium ${cls}">${esc(ds)}</span>`;
}
function recallColor(v) {
  if (v >= 0.8) return '#22c55e';
  if (v >= 0.5) return '#84cc16';
  if (v >= 0.2) return '#f59e0b';
  if (v > 0)   return '#f97316';
  return '#ef4444';
}

// ── init ──────────────────────────────────────────────────────────────────────
async function init() {
  const models = await get('/api/models');
  const mSel = $('sel-model'), eSel = $('sel-embedder');

  Object.keys(models).forEach(m => {
    const o = document.createElement('option');
    o.value = o.textContent = m;
    mSel.appendChild(o);
  });

  function rebuildEmbedders() {
    eSel.innerHTML = '';
    (models[mSel.value] || []).forEach(e => {
      const o = document.createElement('option');
      o.value = o.textContent = e;
      eSel.appendChild(o);
    });
  }

  mSel.addEventListener('change', () => { rebuildEmbedders(); loadQueries(); });
  eSel.addEventListener('change', loadQueries);
  $('sel-dataset').addEventListener('change', loadQueries);
  $('search').addEventListener('input', filterAndRender);

  rebuildEmbedders();
  await loadQueries();
}

// ── query list ────────────────────────────────────────────────────────────────
async function loadQueries() {
  curModel    = $('sel-model').value;
  curEmbedder = $('sel-embedder').value;
  curDataset  = $('sel-dataset').value;
  if (!curModel || !curEmbedder) return;

  $('query-list').innerHTML = '<div class="p-4 text-gray-400 text-xs">Loading…</div>';
  $('list-hdr').textContent = 'Loading…';

  const qs = new URLSearchParams({ model: curModel, embedder: curEmbedder, dataset: curDataset });
  allQueries = await get(`/api/queries?${qs}`);
  filterAndRender();
}

function filterAndRender() {
  const q = $('search').value.toLowerCase().trim();
  filteredQueries = q
    ? allQueries.filter(e => e.query.toLowerCase().includes(q))
    : allQueries;

  const n = filteredQueries.length;
  const avg = n
    ? (filteredQueries.reduce((s,e) => s + e.char_recall_max, 0) / n).toFixed(3) : '–';

  $('stats').textContent = `${n} queries · avg R@max = ${avg}`;
  $('list-hdr').textContent = `${n} quer${n === 1 ? 'y' : 'ies'}`;

  const list = $('query-list');
  list.innerHTML = '';
  const frag = document.createDocumentFragment();
  filteredQueries.forEach(q => frag.appendChild(makeItem(q)));
  list.appendChild(frag);
}

function makeItem(q) {
  const div = document.createElement('div');
  div.className = 'q-item px-3 py-2 border-b cursor-pointer transition-colors';
  div.dataset.idx = q.idx;

  const hit = q.n_gt_hit, total = q.n_gt_snippets;
  const hitHtml = hit === total
    ? `<span class="text-green-600 font-medium text-xs">✓ ${hit}/${total}</span>`
    : hit > 0
    ? `<span class="text-amber-600 font-medium text-xs">~ ${hit}/${total}</span>`
    : `<span class="text-red-500 font-medium text-xs">✗ 0/${total}</span>`;

  const r = q.char_recall_max;
  div.innerHTML = `
    <div class="flex items-center gap-1.5 mb-1">
      ${dsBadge(q.dataset)} ${hitHtml}
      <span class="ml-auto text-xs text-gray-400">R@max&thinsp;${r.toFixed(3)}</span>
    </div>
    <p class="text-xs text-gray-700 leading-snug line-clamp-2">${esc(q.query)}</p>
    <div class="mt-1.5 flex items-center gap-1.5">
      <div class="flex-1 bg-gray-100 rounded-full h-1" style="overflow:hidden">
        <div style="width:${(r*100).toFixed(1)}%;height:100%;background:${recallColor(r)};border-radius:9999px"></div>
      </div>
      <span class="text-gray-400" style="font-size:10px">idx ${q.idx}</span>
    </div>`;

  div.addEventListener('click', () => {
    document.querySelectorAll('.q-item').forEach(el =>
      el.classList.toggle('active', parseInt(el.dataset.idx) === q.idx));
    loadDetail(q.idx);
  });
  return div;
}

// ── detail ────────────────────────────────────────────────────────────────────
function destroyCharts() {
  Object.values(_charts).forEach(c => c.destroy());
  _charts = {};
}

async function loadDetail(idx) {
  destroyCharts();
  const panel = $('detail');
  panel.innerHTML = '<div class="p-8 text-gray-400 text-sm">Loading…</div>';
  try {
    const qs = new URLSearchParams({ model: curModel, embedder: curEmbedder, idx });
    const data = await get(`/api/query?${qs}`);
    panel.innerHTML = renderDetail(data);
    renderCharts(data.metrics_by_k);
  } catch(e) {
    panel.innerHTML = `<div class="p-8 text-red-500">Error: ${esc(e.message)}</div>`;
  }
}

// ── Chart.js line charts ──────────────────────────────────────────────────────
function makeChartCfg(labels, values, title, yLabel, color) {
  return {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: color,
        backgroundColor: color + '18',
        pointBackgroundColor: color,
        pointBorderColor: '#fff',
        pointBorderWidth: 1.5,
        pointRadius: 6,
        pointHoverRadius: 8,
        borderWidth: 2.5,
        tension: 0,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: title,
          color: '#1f2937',
          font: { size: 12, weight: 'bold', family: "'Georgia', serif" },
          padding: { bottom: 6 }
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'K (Retrieval Depth)',
            color: '#374151',
            font: { weight: 'bold', size: 11 }
          },
          grid: { color: 'rgba(0,0,0,0.10)' },
          ticks: { color: '#6b7280', font: { size: 10 } }
        },
        y: {
          min: 0,
          title: {
            display: true,
            text: yLabel,
            color: '#374151',
            font: { weight: 'bold', size: 11 }
          },
          grid: { color: 'rgba(0,0,0,0.10)' },
          ticks: { color: '#6b7280', font: { size: 10 } }
        }
      }
    }
  };
}

function renderCharts(mbk) {
  if (!mbk || !mbk.length) return;
  const ks = mbk.map(m => `K=${m.k}`);

  const specs = [
    { id: 'ch-cr', vals: mbk.map(m => m.char_recall),    title: 'Char Recall @ K',    yLabel: 'Char Recall',    color: '#2563eb' },
    { id: 'ch-cp', vals: mbk.map(m => m.char_precision), title: 'Char Precision @ K', yLabel: 'Char Precision', color: '#0891b2' },
    { id: 'ch-xr', vals: mbk.map(m => m.chunk_recall),   title: 'Chunk Recall @ K',   yLabel: 'Chunk Recall',   color: '#7c3aed' },
    { id: 'ch-xp', vals: mbk.map(m => m.chunk_precision),title: 'Chunk Precision @ K',yLabel: 'Chunk Precision',color: '#a21caf' },
  ];

  for (const { id, vals, title, yLabel, color } of specs) {
    const canvas = document.getElementById(id);
    if (!canvas) continue;
    _charts[id] = new Chart(canvas.getContext('2d'), makeChartCfg(ks, vals, title, yLabel, color));
  }
}

// ── text helpers ──────────────────────────────────────────────────────────────
function highlightText(text, gtOverlaps, charStart) {
  if (!gtOverlaps || !gtOverlaps.length) return esc(text);
  const spans = gtOverlaps
    .map(o => [Math.max(0, o.overlap_span[0]-charStart), Math.min(text.length, o.overlap_span[1]-charStart)])
    .filter(([s,e]) => s < e)
    .sort((a,b) => a[0]-b[0]);
  const merged = [];
  for (const [s,e] of spans) {
    if (merged.length && s <= merged[merged.length-1][1])
      merged[merged.length-1][1] = Math.max(merged[merged.length-1][1], e);
    else merged.push([s,e]);
  }
  let html = '', pos = 0;
  for (const [s,e] of merged) {
    html += esc(text.slice(pos,s));
    html += `<mark class="gt">${esc(text.slice(s,e))}</mark>`;
    pos = e;
  }
  return html + esc(text.slice(pos));
}

// ── detail renderer ───────────────────────────────────────────────────────────
function renderDetail(data) {
  // Ground truth
  const gtHtml = data.ground_truth.length
    ? data.ground_truth.map(s => `
      <div class="border border-green-300 rounded-lg p-3 bg-green-50">
        <div class="text-xs text-gray-500 font-mono mb-1.5">${esc(s.file_path||'')} &nbsp;[${(s.span||[]).join(' : ')}]</div>
        <pre class="text-gray-800 leading-relaxed">${esc(s.answer||'')}</pre>
      </div>`).join('')
    : '<p class="text-xs text-gray-400">(no benchmark answer found)</p>';

  // Original query row (only shown when different from displayed query)
  const origRow = data.original_query && data.original_query !== data.query
    ? `<div class="mt-2 text-xs text-gray-500">
         <span class="font-medium">Original benchmark query:</span>
         <span class="italic">${esc(data.original_query)}</span>
       </div>`
    : '';

  // Retrieved chunks
  const nHits = data.retrieved.filter(c => c.is_hit).length;
  const chunksHtml = data.retrieved.map(c => {
    const badge = c.is_hit
      ? '<span class="text-xs font-semibold text-green-700 bg-green-100 px-1.5 py-0.5 rounded">HIT</span>'
      : '<span class="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">miss</span>';
    const overlapInfo = (c.gt_overlaps||[]).map(o =>
      `<span class="text-xs text-green-700">⟵ ${o.overlap_chars} ch overlap (${o.overlap_pct_of_gt.toFixed(1)}% of GT · ${o.overlap_pct_of_chunk.toFixed(1)}% of chunk)</span>`
    ).join(' ');
    const highlighted = highlightText(c.text, c.gt_overlaps, c.char_start);
    return `
    <div class="border rounded-lg p-3 ${c.is_hit ? 'chunk-hit' : 'chunk-miss'}">
      <div class="flex flex-wrap items-center gap-1.5 mb-1.5">
        <span class="font-bold text-gray-500 w-6">#${c.rank}</span>
        ${badge}
        <span class="font-mono text-gray-600 text-xs truncate max-w-xs">${esc(c.file)}</span>
        <span class="text-gray-400 text-xs">[${c.char_start} : ${c.char_end}]</span>
        <span class="ml-auto font-mono text-xs text-gray-500">score ${c.score.toFixed(6)}</span>
      </div>
      ${overlapInfo ? `<div class="mb-1.5">${overlapInfo}</div>` : ''}
      <pre class="text-gray-700 border-t pt-2 mt-1">${highlighted}</pre>
    </div>`;
  }).join('');

  return `
  <div class="p-6 max-w-5xl mx-auto space-y-6 pb-16">

    <!-- Query -->
    <section>
      <div class="flex items-center gap-2 mb-2">
        <h2 class="font-semibold text-gray-700">Query</h2>
        ${dsBadge(data.dataset)}
      </div>
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-gray-800 leading-relaxed">
        ${esc(data.query)}
      </div>
      ${origRow}
    </section>

    <!-- Ground Truth -->
    <section>
      <h2 class="font-semibold text-gray-700 mb-2">
        Ground Truth
        <span class="font-normal text-xs text-gray-400 ml-1">${data.total_gt_chars} chars total</span>
      </h2>
      <div class="space-y-2">${gtHtml}</div>
    </section>

    <!-- Metrics charts (2×2) -->
    <section>
      <h2 class="font-semibold text-gray-700 mb-3">Eval Metrics @ K</h2>
      <div class="grid grid-cols-2 gap-4">
        <div class="bg-white border rounded-lg p-3 chart-box"><canvas id="ch-cr"></canvas></div>
        <div class="bg-white border rounded-lg p-3 chart-box"><canvas id="ch-cp"></canvas></div>
        <div class="bg-white border rounded-lg p-3 chart-box"><canvas id="ch-xr"></canvas></div>
        <div class="bg-white border rounded-lg p-3 chart-box"><canvas id="ch-xp"></canvas></div>
      </div>
      <div class="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs text-gray-500">
        ${data.metrics_by_k.map(m =>
          `<span>K=${m.k}: cR=${m.char_recall.toFixed(3)} cP=${m.char_precision.toFixed(4)} xR=${m.chunk_recall.toFixed(3)} xP=${m.chunk_precision.toFixed(4)}</span>`
        ).join('')}
      </div>
    </section>

    <!-- Retrieved Chunks -->
    <section>
      <h2 class="font-semibold text-gray-700 mb-2">
        Top-${data.retrieved.length} Retrieved Chunks
        <span class="font-normal text-xs text-gray-400 ml-1">
          — ${nHits} hit${nHits !== 1 ? 's' : ''} (GT overlap highlighted in yellow)
        </span>
      </h2>
      <div class="space-y-2">${chunksHtml}</div>
    </section>

  </div>`;
}

// ── boot ──────────────────────────────────────────────────────────────────────
init().catch(err => {
  document.body.innerHTML =
    `<div class="p-8 text-red-500">Failed to initialise: ${esc(err.message)}</div>`;
});
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────────

_models_cache: dict | None = None


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        global _models_cache
        parsed = urllib.parse.urlparse(self.path)
        p      = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        try:
            if p == "/":
                self._send(200, "text/html; charset=utf-8", HTML.encode())

            elif p == "/api/models":
                if _models_cache is None:
                    _models_cache = list_models()
                self._json(_models_cache)

            elif p == "/api/queries":
                data = api_queries(
                    params["model"],
                    params["embedder"],
                    params.get("dataset") or None,
                )
                self._json(data)

            elif p == "/api/query":
                data = api_query(
                    params["model"],
                    params["embedder"],
                    int(params["idx"]),
                )
                self._json(data)

            else:
                self.send_error(404)

        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self._send(status, "application/json", body)

    def _send(self, status: int, ct: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Eval viewer →  {url}")
    # print("Lab machine?   ssh -L 8765:localhost:8765 twa174@secb1010u-d11")
    # print("Stop:          Ctrl-C\n")
    threading.Timer(0.6, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
