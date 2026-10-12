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
