# LegalRAG

Retrieval over Canadian legal documents, backed by OpenSearch. The repo contains:

1. **LegalRAG Explore**, the main application. An exploration-first legal assistant over the
   [CanLegalRAGBench](https://huggingface.co/datasets/UBC-VL/CanLegalRAGBench) corpus. It
   doesn't write an answer. It retrieves sources and turns them into interactive,
   source-cited views: case landscape, decision timeline, "what if" fact toggles, and
   LLM-chosen diagrams such as legal tests, concept maps and comparisons. It's personalised
   by a user profile (province → jurisdiction filter, familiarity, goal).
2. **Evaluation** on LegalBench-RAG (chunk-level precision/recall), plus a static eval viewer.

The original CanLII answer-generation pipeline is archived in
[archive/canlii_pipeline/](archive/canlii_pipeline/README.md).

```
web/ (React + Vite)  ──POST /api/explore (NDJSON stream)──▶  legalrag/api/app.py (FastAPI)
                                                                │
                                         legalrag/explore/engine.py  ExplorationEngine
  1. formulate      LLM rewrites the question for search                    (Qwen3 via OpenAI-compatible API)
  2. retrieve       kNN in OpenSearch, top 60,                              (Qwen3-Embedding-0.6B)
                    filtered to {user's province, federal, unknown}
  3. rerank         keep the best 20 passages → evidence e1..e20            (Qwen3-Reranker-0.6B)
  4. built-in views case landscape + case timeline, from metadata           (no LLM)
  5. LLM views      fact toggles  ‖  planner → 0–3 extra views it judges useful
  6. grounding      every item must cite e# (or q = the question); others are hidden
                                                                │
                              every step is traced → "Under the hood" panel + logs/explore_traces/
```

| Read next | |
|---|---|
| Architecture diagram + a full example request, step by step | [legalrag/README.md](legalrag/README.md) |
| Frontend structure and conventions | [web/README.md](web/README.md) |
| Building the index on the lab GPU, copying it to a laptop | [docs/lab_ingestion.md](docs/lab_ingestion.md) |
| Open decisions, prompts and templates to review | [docs/explore_app.md](docs/explore_app.md) |

---

## Run the website on localhost

Tested on macOS (Apple M2, 32 GB) with the versions below. Linux works the same way.

### 0. Prerequisites

| Tool | Version tested | Install |
|---|---|---|
| Git | any | |
| Conda (Miniconda) or Python 3.11 | Python 3.11.14 | https://docs.conda.io/en/latest/miniconda.html |
| Docker Desktop (for OpenSearch) | 28.0 | https://www.docker.com/products/docker-desktop |
| Node.js ≥ 18 + npm | Node 24.13, npm 11.6 | https://nodejs.org or `brew install node` |
| Ollama | 0.40.3 | https://ollama.com/download or `brew install ollama` |

Disk: about 12 GB (Ollama `qwen3:8b` 5.2 GB, Qwen3 embedding and reranker 2.2 GB, PyTorch,
the 0.3 GB index). Memory: 16 GB minimum, 32 GB recommended (the 8B LLM, the two
Qwen3 models and OpenSearch all run at once).

### 1. Get the code and the Python environment

```bash
git clone <this repo> LegalRAG && cd LegalRAG

conda env create -f environment.yml        # Python 3.11 + pinned requirements.txt
conda activate legalrag
# Without conda: python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Check:** `python -c "import legalrag, fastapi, torch; print('ok')"` prints `ok`.

### 2. Configure

```bash
cp .env.example .env
```

The defaults point at the **lab vLLM** (through an SSH tunnel, step 5) and the Docker
OpenSearch. vLLM serves the app's parallel LLM calls (views, chat) at the same time; Ollama
answers them one by one. To run fully offline instead, comment out the vLLM block in `.env`
and uncomment the Ollama one. Each block sets `LLM_THINKING_OFF`, which must match the
server (it switches Qwen3's slow "thinking" off).

### 3. Start OpenSearch 3.4.0

```bash
docker compose up -d opensearch
```

**Check:** `curl -s localhost:9200/_cluster/health` shows `"status":"green"` or `"yellow"`
(the first start takes about 30 s).

### 4. Load the search index (`canlegal-qwen3`)

Pick **one** option.

**A. Restore a snapshot (recommended, about 1 minute).** Ask a teammate for the
`opensearch-snapshots/` folder (about 330 MB; built on the lab GPU, see
[docs/lab_ingestion.md](docs/lab_ingestion.md)). Put it at the repo root, then:

```bash
curl -X PUT localhost:9200/_snapshot/canlegal -H 'Content-Type: application/json' \
     -d '{"type":"fs","settings":{"location":"/usr/share/opensearch/snapshots"}}'
curl -X POST "localhost:9200/_snapshot/canlegal/canlegal-qwen3-v1/_restore?wait_for_completion=true" \
     -H 'Content-Type: application/json' -d '{"indices":"canlegal-qwen3"}'
```

**B. Build it yourself.** This downloads the corpus from Hugging Face and embeds about 63K
chunks. It takes minutes on a CUDA GPU and **5–9 hours on an Apple M2**.

```bash
python scripts/ingest_canlegal.py --index-name canlegal-qwen3 \
    --embedding-provider sentence_transformers --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --chunk-size 1024 --chunk-overlap 128
```

**Check:** `curl -s localhost:9200/canlegal-qwen3/_count` reports `"count":62942`.

### 5. Start the LLM

**vLLM on this Mac (Apple Silicon).** Install once with Homebrew, then serve Qwen3-8B on port 8001 (leave it running; stop Ollama first):

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
bash scripts/serve_qwen3_vllm_metal.sh
```

On an M2 this ran a full exploration in ~6.5 min against ~11 min on Ollama, because the view calls run concurrently. It needs ~15 GB while serving.

**Lab vLLM (fastest, needs the SFU network / VPN).** One-time install and run steps are at the top of
[scripts/serve_qwen3_vllm.sh](scripts/serve_qwen3_vllm.sh) (venv + model cache on `/local-scratch`). On the lab:

```bash
source /local-scratch/$USER/vllm-env/bin/activate && HF_HOME=/local-scratch/$USER/hf bash scripts/serve_qwen3_vllm.sh
```

On the laptop, on the SFU network or VPN (leave it running):

```bash
ssh -N -o ServerAliveInterval=15 -L 8001:127.0.0.1:8001 secb1010-d11.csil.sfu.ca
```

**Check:** `curl -s localhost:8001/v1/models` lists `qwen3`.

**Ollama (offline fallback)** — use the Ollama block in `.env`, then:

In a new terminal (leave it running):

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve    # the prompts need more than the 4K default context
```

Then, once:

```bash
ollama pull qwen3:8b
```

If the Ollama desktop app is already running, quit it first, so the server picks up
`OLLAMA_CONTEXT_LENGTH`.

**Check:** `curl -s localhost:11434/api/tags` lists `qwen3:8b`.

### 6. Start the API

In a new terminal, from the repo root (leave it running):

```bash
conda activate legalrag
uvicorn legalrag.api.app:app --port 8000
```

The first start downloads Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B from Hugging Face
(about 2.2 GB). After that it starts in a few seconds.

**Check:** `curl -s localhost:8000/api/health` returns
`{"status":"ok","index":"canlegal-qwen3","chunks":62942}`. Interactive API docs:
http://localhost:8000/docs.

### 7. Start the website

In a new terminal (leave it running):

```bash
cd web
npm ci            # exact versions from package-lock.json
npm run dev
```

Open **http://localhost:5173**. Vite forwards `/api` to the API on port 8000.

### 8. Use it

1. **Log in** with any name. This is a dev-only login: no password, and profiles are JSON
   files in `data/app/profiles/`.
2. **Build your profile:** province (sets the jurisdiction filter), familiarity with the
   law, why you're here, topics.
3. **Ask a question**, or click one of the four benchmark questions (two from your
   province, two on federal law). Those open instantly: they are precomputed demos
   (`data/demos/`, built by `scripts/build_demos.py`).
4. **Explore the tabs:** Overview, What-if, Sources, Timeline, Relations. While a live run
   builds, the Overview shows its progress and tabs unlock as they become ready. Click any
   source tag to read the passage in context.
5. **Ask** (top right) opens a chat beside every tab. It answers from this exploration's
   passages only and cites them.
6. Click **under the hood** to see every step: the rewrite, retrieval candidates and
   scores, LLM prompts and responses, and timings.

How long one exploration takes (measured 2026-10-18, embedder + reranker on an M2):

| LLM | whole run | LLM steps |
|---|---|---|
| Lab vLLM, Qwen3-30B-A3B (4090) | ~1 min (median 58 s) | facts ~8 s, summary ~3 s, each map ~8 s; the M2 reranker (~45 s) dominates |
| vllm-metal, Qwen3-8B (M2) | ~6.5 min | calls run concurrently |
| Ollama, qwen3:8b (M2) | ~11 min | calls queue one after another |

With Ollama `qwen3:8b` on a laptop:
- Sources and the built-in views (Sources, Timeline) appear after about 40 s.
- The fact toggles (What-if) take about 2–5 minutes and the planner (Overview) about 4.
- Extra views (Relations) often **time out**. Ollama answers one request at a time, so they
  wait behind the fact toggles and hit the 3-minute limit (`planner.extra_view_timeout_s`).

To get the extra views locally, raise that limit or start Ollama with
`OLLAMA_NUM_PARALLEL=2` (more memory). For full-speed runs, use the lab vLLM (see
`scripts/serve_qwen3_vllm.sh` and `.env.example`).

### Stopping

Press Ctrl-C in the API, website and Ollama terminals, then run `docker compose stop opensearch`
(the index is kept in a Docker volume).

### Troubleshooting

| Symptom | Fix |
|---|---|
| UI: "Cannot reach the API" | The API isn't running on :8000 (step 6). |
| `402 … no remaining credits` | `.env` still points at the Hugging Face router. Use the vLLM or Ollama block from `.env.example` and restart the API. |
| `404 model "…" not found` | `LLM_MODEL` must match the server: `qwen3` for vLLM, `qwen3:8b` for Ollama (after `ollama pull qwen3:8b`). |
| Every LLM step is very slow, with long "thinking" in the trace | `LLM_THINKING_OFF` in `.env` must match the server (`vllm` or `ollama`). |
| `Connection error` on every LLM step | The vLLM tunnel isn't up (step 5): `curl localhost:8001/v1/models`. Or switch `.env` to the Ollama block. |
| `Query vector has invalid dimension` | The index wasn't built with Qwen3-Embedding-0.6B. Check `EXPLORE_EMBEDDING_MODEL` and the index (step 4). |
| Extra views show "timed out after 180s" | This is expected with Ollama on a laptop. Raise `planner.extra_view_timeout_s`, set `OLLAMA_NUM_PARALLEL=2`, or use the lab vLLM. |
| `/api/health` reports 0 chunks or an error | The index is missing (step 4), or OpenSearch isn't running (step 3). |
| Port already in use | Another API or Vite instance is still running. Stop it, or use `--port`. |

---

## Seeing what happens under the hood

Every exploration is traced. Each trace records:
- step timings;
- the query rewrite;
- all retrieval candidates, with retrieval and rerank scores and which passages were kept;
- every LLM call: system and user prompts, raw response, thinking (if enabled), token usage, validation retries, errors.

Ways to view it:
- **UI:** the **under the hood** button on a run (it updates live while the run is in progress).
- **Server log:** the uvicorn terminal and `logs/explore_api.log`, one INFO line per step and LLM call, prefixed `[trace_id]`.
- **Files:** `logs/explore_traces/<date>/<time>_<trace_id>.json`.
- **API:** `GET /api/traces`, `GET /api/traces/{trace_id}`.
- **Terminal:** `python scripts/explore_cli.py --query-id 792` prints the event stream and a step and LLM timing summary.

---

## Evaluation (LegalBench-RAG)

Chunk-level Precision@K / Recall@K over the LegalBench-RAG benchmarks. Commands and baseline
results: [CLAUDE.md](CLAUDE.md#legalbench-rag-evaluation-commands) and
[docs/evaluation.md](docs/evaluation.md). Code: `evaluation/LegalBenchRAG/` (Gemini query
rewrite needs `pip install -e ".[eval]"`). Static results viewer: `eval-viewer/` (built by
`scripts/build_static.py`, deployed with `vercel.json`).

---

## Key design decisions

| Decision | Choice | Notes |
|---|---|---|
| Vector store | OpenSearch 3.4.0 | kNN (HNSW, Lucene) + BM25; native hybrid query with RRF pipeline |
| Explore corpus | CanLegalRAGBench (1,117 docs) | Jurisdiction tagged at ingestion (`canlegal_courts.yaml`) |
| Explore chunking | Flat recursive splitter, 1024 chars / 128 overlap | Same as UBC's reference config; LangChain-style overlap |
| Explore embedder | Qwen3-Embedding-0.6B (1024-dim) | `EXPLORE_EMBEDDING_MODEL`; must match the index |
| Explore retrieval | Dense kNN, top 60 | `retrieval.mode` in `explore_policy.yaml` (`hybrid` / `lexical` available) |
| Explore reranker | Qwen3-Reranker-0.6B | `EXPLORE_RERANKER_MODEL`; instruction in `explore_policy.yaml` |
| LLM | Qwen3 via any OpenAI-compatible server | Local Ollama `qwen3:8b` or lab vLLM; `LLM_*` / `EXPLORE_LLM_MODEL` |
| Structured LLM output | pydantic-ai, native JSON-schema mode | `legalrag/explore/schemas.py`; `llm.output_mode` |
| Prompts and policies | YAML, not Python | `legalrag/prompts/*.yaml` |
| Output style | Views, not an answer | Every item must cite retrieved evidence |

## Project structure

```
LegalRAG/
├── legalrag/                 Python backend package → see legalrag/README.md
│   ├── api/                  FastAPI app (exploration API)
│   ├── explore/              exploration engine, views, schemas, profiles, tracing
│   ├── core/                 settings, shared models, interfaces
│   ├── ingestion/            CanLegalRAGBench loader, chunkers, embedders, indexer
│   ├── query/                retriever, rerankers
│   ├── opensearch/           OpenSearch client
│   ├── prompts/              YAML prompts + policies
│   └── utils/                LLM client factory, logging
├── web/                      React + Vite exploration UI → see web/README.md
├── scripts/                  CLIs: ingest, explore, chunking audit, vLLM launcher, eval viewer build
├── evaluation/               LegalBench-RAG evaluation
├── tests/                    pytest suite (no OpenSearch needed)
├── docs/                     explore_app, lab_ingestion, evaluation
├── archive/                  archived CanLII answer pipeline (not maintained)
├── Demo/                     static visualization demo (Reyhaneh); renderers reused in web/
├── eval-viewer/              static evaluation results site
├── notebooks/                analysis notebooks
├── requirements.txt          pinned Python dependencies (+ environment.yml for conda)
└── docker-compose.yml        OpenSearch 3.4.0 (+ Dashboards), snapshot repo mount
```

## Tests

```bash
pytest tests/ -v                     # unit tests
python scripts/check_chunking.py --chunker recursive --chunk-size 1024 --chunk-overlap 128
cd web && npm run smoke              # browser end-to-end (API + dev server running)
```
