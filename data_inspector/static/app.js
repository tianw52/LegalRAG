// ── state ─────────────────────────────────────────────────────────────────────
const DS_ID = new URLSearchParams(location.search).get('ds') || 'legalbenchrag-mini';

let curModel = '', curEmbedder = '', curDataset = '', curChunker = 'hier';
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
  const [datasets, models, chunkers] = await Promise.all([
    get('/api/datasets'),
    get(`/api/models?ds=${DS_ID}`),
    get(`/api/chunkers?ds=${DS_ID}`),
  ]);

  // Header title + benchmark filter
  const dsMeta = datasets.find(d => d.id === DS_ID);
  if (dsMeta) {
    $('hdr-title').textContent = `⚖ ${dsMeta.label} Inspector`;
    const bSel = $('sel-dataset');
    for (const name of dsMeta.benchmark_names) {
      const o = document.createElement('option');
      o.value = o.textContent = name;
      bSel.appendChild(o);
    }
  }

  // Chunker filter — hide wrap if only one option
  const cSel = $('sel-chunker');
  chunkers.forEach(c => {
    const o = document.createElement('option');
    o.value = o.textContent = c;
    cSel.appendChild(o);
  });
  $('chunker-wrap').style.display = chunkers.length > 1 ? '' : 'none';

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
  cSel.addEventListener('change', loadQueries);
  $('sel-dataset').addEventListener('change', loadQueries);
  $('search').addEventListener('input', filterAndRender);
  $('sel-sort').addEventListener('change', filterAndRender);
  $('sel-sort-dir').addEventListener('change', filterAndRender);

  rebuildEmbedders();
  await loadQueries();
}

// ── query list ────────────────────────────────────────────────────────────────
async function loadQueries() {
  curModel    = $('sel-model').value;
  curEmbedder = $('sel-embedder').value;
  curChunker  = $('sel-chunker').value || 'hier';
  curDataset  = $('sel-dataset').value;
  if (!curModel || !curEmbedder) return;

  $('query-list').innerHTML = '<div class="p-4 text-gray-400 text-xs">Loading…</div>';
  $('list-hdr').textContent = 'Loading…';

  const qs = new URLSearchParams({
    ds: DS_ID, model: curModel, embedder: curEmbedder,
    chunker: curChunker, dataset: curDataset,
  });

  allQueries = await get(`/api/queries?${qs}`);
  filterAndRender();
}

function sortedQueries(queries) {
  const field = $('sel-sort').value;
  const asc   = $('sel-sort-dir').value === 'asc';
  return [...queries].sort((a, b) => {
    const diff = a[field] - b[field];
    return asc ? diff : -diff;
  });
}

function filterAndRender() {
  const q = $('search').value.toLowerCase().trim();
  const filtered = q
    ? allQueries.filter(e => e.query.toLowerCase().includes(q))
    : allQueries;
  filteredQueries = sortedQueries(filtered);

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

  const cr = q.char_recall_max;
  const xr = q.chunk_recall_max;
  div.innerHTML = `
    <div class="flex items-center gap-1.5 mb-1">
      ${dsBadge(q.dataset)} ${hitHtml}
      <span class="ml-auto text-xs text-gray-400">cR&thinsp;${cr.toFixed(3)}&ensp;xR&thinsp;${xr.toFixed(3)}</span>
    </div>
    <p class="text-xs text-gray-700 leading-snug line-clamp-2">${esc(q.query)}</p>
    <div class="mt-1.5 flex items-center gap-1.5">
      <div class="flex-1 bg-gray-100 rounded-full h-1" style="overflow:hidden">
        <div style="width:${(cr*100).toFixed(1)}%;height:100%;background:${recallColor(cr)};border-radius:9999px"></div>
      </div>
      <span class="text-gray-400" style="font-size:10px">idx&thinsp;${q.idx}</span>
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
    const qs = new URLSearchParams({ ds: DS_ID, model: curModel, embedder: curEmbedder, chunker: curChunker, idx });
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
          callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(4)}` }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'K (Retrieval Depth)', color: '#374151', font: { weight: 'bold', size: 11 } },
          grid:  { color: 'rgba(0,0,0,0.10)' },
          ticks: { color: '#6b7280', font: { size: 10 } }
        },
        y: {
          min: 0,
          title: { display: true, text: yLabel, color: '#374151', font: { weight: 'bold', size: 11 } },
          grid:  { color: 'rgba(0,0,0,0.10)' },
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
    { id: 'ch-cr', vals: mbk.map(m => m.char_recall),     title: 'Char Recall @ K',     yLabel: 'Char Recall',     color: '#2563eb' },
    { id: 'ch-cp', vals: mbk.map(m => m.char_precision),  title: 'Char Precision @ K',  yLabel: 'Char Precision',  color: '#0891b2' },
    { id: 'ch-xr', vals: mbk.map(m => m.chunk_recall),    title: 'Chunk Recall @ K',    yLabel: 'Chunk Recall',    color: '#7c3aed' },
    { id: 'ch-xp', vals: mbk.map(m => m.chunk_precision), title: 'Chunk Precision @ K', yLabel: 'Chunk Precision', color: '#a21caf' },
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
  const gtHtml = data.ground_truth.length
    ? data.ground_truth.map(s => `
      <div class="border border-green-300 rounded-lg p-3 bg-green-50">
        <div class="text-xs text-gray-500 font-mono mb-1.5">${esc(s.file_path||'')} &nbsp;[${(s.span||[]).join(' : ')}]</div>
        <pre class="text-gray-800 leading-relaxed">${esc(s.answer||'')}</pre>
      </div>`).join('')
    : '<p class="text-xs text-gray-400">(no benchmark answer found)</p>';

  const origRow = data.original_query && data.original_query !== data.query
    ? `<div class="mt-2 text-xs text-gray-500">
         <span class="font-medium">Original benchmark query:</span>
         <span class="italic">${esc(data.original_query)}</span>
       </div>`
    : '';

  const nHits = data.retrieved.filter(c => c.is_hit).length;
  const chunksHtml = data.retrieved.map(c => {
    const badge = c.is_hit
      ? '<span class="text-xs font-semibold text-green-700 bg-green-100 px-1.5 py-0.5 rounded">HIT</span>'
      : '<span class="text-xs text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">miss</span>';
    const overlapInfo = (c.gt_overlaps||[]).map(o =>
      `<span class="text-xs text-green-700">⟵ ${o.overlap_chars} ch overlap (${o.overlap_pct_of_gt.toFixed(1)}% of GT · ${o.overlap_pct_of_chunk.toFixed(1)}% of chunk)</span>`
    ).join(' ');
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
      <pre class="text-gray-700 border-t pt-2 mt-1">${highlightText(c.text, c.gt_overlaps, c.char_start)}</pre>
    </div>`;
  }).join('');

  return `
  <div class="p-6 max-w-5xl mx-auto space-y-6 pb-16">

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

    <section>
      <h2 class="font-semibold text-gray-700 mb-2">
        Ground Truth
        <span class="font-normal text-xs text-gray-400 ml-1">${data.total_gt_chars} chars total</span>
      </h2>
      <div class="space-y-2">${gtHtml}</div>
    </section>

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
