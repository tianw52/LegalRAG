# Coding Session Log

## 2026-10-12 — Exploration UI (CanLegalRAGBench) + local dev environment

**Goal:** an exploration-first legal chatbot. Ingest corpus → user asks → retrieve →
generate interactive views (pre-defined + agent-chosen), personalised by a user profile.
There is no plain-text answer step. Full guide and open decisions: `docs/explore_app.md`.

### Environment
- **Laptop setup mirrors the lab.** OpenSearch 3.4.0 runs through `docker compose up -d opensearch`, and the `legalrag` conda env is used throughout.
- **New `app` extra in `pyproject.toml`:** `fastapi<0.130`, `uvicorn`. FastAPI 0.130+ requires `opentelemetry-api>=1.44`, which conflicts with logfire's pin.
- **Ollama + `qwen3:8b`** installed locally as the LLM fallback. `scripts/serve_qwen3_vllm.sh` serves Qwen3 on the lab 4090 through vLLM.

### Data / ingestion
- **New `legalrag/ingestion/canlegal_loader.py` + `canlegal_courts.yaml`.** Court and jurisdiction mapping for the dataset's court codes, plus citation/name regex rules for UNKNOWN and STATUTE rows. Unmatched rows get `"unknown"`. YAML jurisdiction codes must be quoted, because bare `ON` parses as `true`.
- **New `scripts/ingest_canlegal.py`.** Qwen3-Embedding-0.6B with flat RCTS 1024/128, matching UBC's reference config (NLP-UBC/CanLegalRAGBench `configs/base.yaml`). Index name: `canlegal-qwen3`.
- **Index mapping gains** `court_abbrev`, `jurisdiction`, `doc_type`, `year`, `case_name`, `url`. The indexer and retriever populate them.

### Shared-code fixes (these affect LegalBench-RAG experiments; re-ingest to pick them up)
- **`HierarchicalChunker`: parent IDs are now `stable_id(doc_id, "parent", s, e)`.** Before, a parent no longer than one child got the child's ID and was overwritten on upsert. That hit 27% of documents at parent 4096.
- **`RecursiveCharacterTextSplitter` overlap now follows LangChain's `_merge_splits`.** It carries over whole trailing pieces ≤ `chunk_overlap`. Before, overlap was character-based: 61% of chunks started mid-word and windows could exceed `chunk_size` (up to 1152 at 1024). On the corpus: 0% mid-word starts, max 1024, median overlap 3, 41% of pairs with no overlap (same as LangChain/UBC). `test_recursive_overlap_creates_shared_context` was updated to the new contract, and regression tests were added.
- **`OpenSearchClient`:** list filters become `terms` queries (`_filter_clauses`); added `index_exists()` and `get_chunks_by_doc_id()`.
- **`StructuredQuery.jurisdiction_filter`**, set from the profile (not the LLM).
- **Pre-existing failure, unchanged:** `test_metadata_propagated` builds `HierarchicalChunker()` without `parent_size`.

### Exploration engine + API
- **`legalrag/explore/`** (engine, views, schemas, profile).
- **`legalrag/api/app.py`:** FastAPI with an NDJSON streaming `/api/explore`.
- **Templates:** `legalrag/prompts/explore_policy.yaml` and `explore_views.yaml`.
- **Views:**
  - case landscape and case timeline (from metadata, no LLM);
  - fact toggles (LLM);
  - a planner agent that picks 1–2 ad-hoc views (concept map, comparison, legal test, rule hierarchy, process flow, scenario timeline), rendered with Reyhaneh's Demo renderers.
- **Grounding:** every item cites `e1..eN` or `q` (the question). Invalid IDs are stripped and uncited items hidden.
- **`ExploreSettings` no longer sets `populate_by_name`.** It made pydantic-settings match `.env`'s `EMBEDDING_MODEL` / `LLM_MODEL` by field name, so the API loaded legal-bert against the 1024-dim Qwen index.

### Frontend (`web/`, React 18 + Vite 5, JavaScript)
- **Flow:** login (dev-only, name), then onboarding (province, familiarity, goal, interests), then the workspace.
- **Workspace:** exploration trail, question box with benchmark examples, streamed views, evidence panel with cross-highlighting, document reader with passage highlights, follow-up chips, and a scope toggle (province vs all of Canada).

### Known issues
- The HF router key has no credits (402). The LLM must come from lab vLLM or local Ollama.
- The full ingest on the M2 is slow (hours); the lab 4090 is much faster.

### Lab ingestion workflow
- **Ingestion moves to the lab 4090; the index reaches the laptop via an OpenSearch filesystem snapshot.** Step-by-step guide for humans and Claude on either machine: `docs/lab_ingestion.md`.
- **Laptop `docker-compose.yml`** now sets `path.repo=/usr/share/opensearch/snapshots` and bind-mounts `./opensearch-snapshots` (gitignored).
- **`CLAUDE.md`** has a short "Exploration App" section pointing to these docs.

## 2026-10-12 (cont.) — LLM views verified, reusable structure

- **Index built on the lab, restored on the laptop:** 62,942 chunks, an exact match.
- **End-to-end exploration verified with local Ollama `qwen3:8b`** (`scripts/explore_cli.py --query-id 792`):
  - fact toggles: 7 facts, all backed;
  - the planner chose `legal_test` and `scenario_timeline`;
  - every view item cites retrieved evidence;
  - ~15 min end to end on the M2, against a ~minute expected on the lab vLLM.
- **Bugs fixed:**
  - **Orphaned LLM calls.** View tasks kept running after the client disconnected, saturating Ollama. They're now cancelled in `ExplorationEngine.explore()`'s `finally`.
  - **LLM client and timeouts.** The client now comes from the shared `get_async_client().with_options(timeout, max_retries)` factory. `llm.timeout: 600` and `llm.http_retries: 1` live in `explore_policy.yaml`.
- **Prompt fixes:**
  - follow-ups are phrased as questions the user would ask, with no inline citations;
  - every decision-tree edge must be labelled, and sibling labels must differ.
- **Reusable structure:**
  - `OpenSearchClient.from_settings(embedding_dim, index_name=…)` replaces the copy-pasted per-index settings blocks (engine, ingest script).
  - `scripts/check_chunking.py`: chunker invariant and quality audit.
  - `scripts/explore_cli.py`: terminal exploration runner.
  - Frontend shared code: `web/src/lib/` (exploration reducer, selection model, jurisdiction helpers, storage) and `web/src/components/ViewHead.jsx` / `Placeholder.jsx`.
  - `web/scripts/smoke.mjs` (`npm run smoke`): headless-Chrome end-to-end test.
  - `web/README.md` documents the structure and conventions.

## 2026-10-12 (cont.) — Qwen3 reranker, tracing, optional extra views, READMEs

- **Reranker.**
  - `legalrag/query/reranker.py` gains `Qwen3Reranker`, which implements the official Qwen3-Reranker scoring (P("yes") with an instruction), and a `build_reranker(name)` factory.
  - The explore app uses `EXPLORE_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B`, with its instruction in `explore_policy.yaml → reranker`.
  - On 8 benchmark queries it kept 28 of 35 retrievable ground-truth documents in the top 20 (ms-marco: 25).
  - It's slower on the M2: ~28 s against ~1 s per 60 candidates.
  - The CanLII pipeline keeps ms-marco via `RERANKER_MODEL`.
- **Tracing.** `legalrag/explore/trace.py` adds `Trace` and `TraceStore`.
  - **The engine records:** step timings, the rewrite, all retrieval candidates with hybrid and rerank scores, and every LLM call. `views._run` captures, via pydantic-ai `capture_run_messages`, the prompts, raw responses, thinking, usage, validation retries and errors.
  - **Endpoints:** `GET /api/traces`, `GET /api/traces/{id}` (live while running).
  - **Files:** `logs/explore_traces/`.
  - **UI:** "Under the hood" drawer (`TracePanel.jsx`).
  - **CLI:** `explore_cli.py` prints a trace summary.
- **Server log.** The API now writes `logs/explore_api.log`. Previously `configure_logging(log_file=None)` silently fell back to `logs/queries.log`; the `None` semantics in `utils/logging.py` don't match its docstring.
- **Extra views are optional.** The planner judges usefulness and chooses 0–3 (`planner.max_extra_views`), with `why_useful` per view and `extra_views_reason`, shown in the UI.
- **Docs.** Root `README.md` rewritten for the current system; new `legalrag/README.md` (backend); `.env.example` gains the `EXPLORE_*` settings.
- **Memory.** Running Ollama 8B, the API (embedder + reranker), Docker and a reranker benchmark at once pushed the 24 GB M2 to ~40 GB of swap. Unload Ollama (`ollama stop qwen3:8b`) when it isn't needed.

## 2026-10-12 (cont.) — dense retrieval default, view timeout, UI redesign

- **Retrieval.** Dense kNN is now the default (`explore_policy.yaml → retrieval.mode: semantic`; `hybrid` and `lexical` are still available). Trace fields were renamed `retrieval_rank` / `retrieval_score`, and the `query` event carries a `retrieved` count.
- **Extra views time out.** Each extra view is limited to `planner.extra_view_timeout_s: 180`, using `asyncio.wait_for`, which also cancels the LLM request. The UI shows "timed out" in the Maps tab, and cancelled LLM calls are still recorded in the trace.
- **UI redesign** following the "LegalRAG Explorer Redesign" canvas:
  - **Look:** paper/ink/accent, Archivo condensed caps, IBM Plex Mono labels, IBM Plex Serif reader.
  - **Pages:** Login, a one-step-at-a-time Onboarding with a province tile map, and Home (ask, benchmark questions, explorations).
  - **Exploration tabs:** Overview / What-if / Cases / Timeline / Maps / Sources (with an in-place reader).
  - **New reusable modules:** `lib/useExplorations.js`, `lib/stats.js`, `components/{AppHeader,ProvinceTiles,primitives,HighlightedText,DemoViz}.jsx`.
  - **Removed:** the old Workspace, EvidencePanel and DocumentReader components, the old views, and `styles.css`.
  - **Run state:** `viewErrors` holds per-view failures.
- **Smoke test.** `web/scripts/smoke.mjs` was rewritten for the new UI. It runs a quick mode by default (no LLM wait); `FULL=1` waits for the LLM views. Text waits are case-insensitive because headings are uppercased by CSS. The quick run passes.
- **Ops note.** An `ollama stop` can leave the runner stuck in "Stopping…", so that requests hang. Restart `ollama serve` if `ollama ps` shows that state.

## 2026-10-12 (cont.) — location options, INFO logging, thinking actually off

- **Ontario bug.** Unquoted `ON` in `explore_policy.yaml` parsed as boolean `true`, so the UI showed "All of Canada" and the profile couldn't be saved. All codes are now quoted, and `tests/explore/test_profile.py` guards against this.
- **Location choices.** "Explore all of Canada", "Outside Canada" and "Prefer not to say" are separate options (`all`, `outside`, `undisclosed`; marked `all_of_canada: true`). All three apply no filter. `profile.filter_province()` is the single place that decides; the UI shows "All of Canada" for them.
- **Under-the-hood logging at INFO** (stdout + `logs/explore_api.log`, prefixed `[trace_id]`), via `Trace`:
  - run start (scope and profile);
  - each step with duration and key fields;
  - the top-10 retrieval candidates and the kept passages after reranking (scores and ranks);
  - each LLM call (model, time, tokens, retries, errors, thinking);
  - timeouts and errors as WARNING.

  Full prompts and raw responses are logged at DEBUG.
- **Thinking is actually off now.** `/no_think` was being ignored by Ollama, so Qwen3 kept thinking. The new setting `llm.thinking_off` is either `ollama` (sends `reasoning_effort: "none"`) or `vllm` (sends `enable_thinking: false`); the `/no_think` suffix was removed. Query rewrite went from 32 s to 6.4 s.
- **Local LLM limits.** On local Ollama 8B the parallel view calls queue, so 3-minute extra-view timeouts fire. That's expected locally; use the lab vLLM for full runs.
- **Structured output is now `native`** (`llm.output_mode`). With thinking off, `prompted` mode made `qwen3:8b` echo the JSON-schema wrapper (`{"properties": …, "title": …}`), and since temperature is 0 every retry was identical, so the fact toggles failed. With `native` (server-side `response_format` JSON schema) it succeeded first try: 7/7 facts backed, 0 retries.

### .env.example: Ollama as the default LLM (2026-10-12)
- LLM section now defaults to local Ollama (`qwen3:8b`), with lab vLLM and the paid HF router as
  commented alternatives, each noting the matching `explore_policy.yaml → llm.thinking_off` value.
  Prompted by a 402 "no remaining credits" from the HF router when `.env` still pointed there.

### Cleanup, archive, reproducible env, docs (2026-10-12)
- **Archived the original CanLII answer pipeline** (unused by both the exploration app and the
  LegalBench-RAG evaluation) to `archive/canlii_pipeline/`, keeping original relative paths:
  `query/{pipeline,router,generator,formulator}.py`, `ingestion/{pipeline,loader,metadata_extractor}.py`,
  `prompts/generator.yaml`, `scripts/{query,ingest}.py`, their tests, `docs/pipeline_breakdown.md`,
  `assets/vector_rag_architecture.png`. Archive README lists what's needed to restore.
- Removed helpers only the archived code used: `BaseQueryFormulator`/`BaseRouter`/`BaseGenerator`,
  `RAGResponse`, `get_sync_client`, `OpenSearchClient.get_parent/get_by_chunk_id/delete_index`,
  `RetrievalSettings.rerank_top_k` (`RERANK_TOP_K`).
- The query-rewrite schema moved from `query/formulator.py` to `explore/schemas.py` as
  `QueryRewriteOutput` (same fields; prompt still `prompts/formulator.yaml`).
- LLM defaults in `core/config.py` now match `.env.example` (local Ollama, `qwen3:8b`).
- **Reproducible env:** `requirements.txt` (pinned direct deps from the working env + `-e .`),
  `environment.yml` (conda, Python 3.11). `pyproject.toml`: dropped unused `tiktoken`/`rich`,
  added directly-imported `transformers`/`torch`/`huggingface_hub`; `eval` extra now
  `google-genai` (what evaluation actually imports; `pytrec_eval`/`datasets` were unused);
  ruff excludes `archive/`. `web/package.json` gets `engines.node >=18`; `package-lock.json`
  re-synced (`npm ci` failed before: missing `@types/node`).
- **Docs:** root README rewritten with step-by-step localhost setup on Ollama (prereqs, snapshot
  restore or ingest, checks, troubleshooting); `legalrag/README.md` has Mermaid architecture +
  sequence diagrams and a step-by-step walk-through of a real run (trace `5ce32f64fd43`);
  CLAUDE.md, docs/explore_app.md, docs/evaluation.md, web/README.md, opensearch/use_guide.md
  updated for the removed pipeline.
- Known: `notebooks/inspect_chunk_spans.ipynb` imports the archived `ingestion/loader.py`;
  `tests/ingestion/test_chunker.py::test_metadata_propagated` still fails (pre-existing).

### Fix: fact toggles flipped together (2026-10-12)
- Toggling one fact flipped every fact with the same status. Since `output_mode: native`,
  qwen3:8b filled `Fact.id` with the status (`stated`, `unknown`, …), and the What-if tab keys
  toggles by id. `Fact` no longer has an LLM-filled `id`; `views.fact_toggle` assigns `f1..fN`.
  Regression test: `tests/explore/test_views.py`.

### Minimal .env (2026-10-12)
- `.env` and `.env.example` now hold only the LLM server choice (`LLM_*`, Ollama active; lab
  vLLM and HF router commented with their `thinking_off` value). Everything else uses the
  defaults in `core/config.py`, which already match the running system: the exploration app
  takes its index/models from `EXPLORE_*` defaults, sizes from `explore_policy.yaml`, and
  embedding dims from the embedder. Dropped `EMBEDDING_*`, `OPENSEARCH_*`, `RERANKER_MODEL`,
  `RETRIEVAL_TOP_K`, `RERANK_TOP_K`, `CHUNK_*`. Verified: settings resolve and OpenSearch pings.
