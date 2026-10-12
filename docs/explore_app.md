# LegalRAG Explore — exploration-first UI

An exploration tool over the CanLegalRAGBench corpus (UBC-VL). The user asks a question.
The system retrieves passages and turns them into interactive views, each citing its
sources. There is **no free-text answer step**; the visual structures are the output.

```
React + Vite (web/)  ──NDJSON──▶  FastAPI (legalrag/api/app.py)
                                    └─ ExplorationEngine (legalrag/explore/engine.py)
                                         formulate (LLM) → hybrid retrieve (OpenSearch, profile jurisdiction filter)
                                         → cross-encoder rerank → evidence e1..eN
                                         → built-in views: case landscape, case timeline        (no LLM)
                                         → fact toggles (LLM)  ‖  planner agent → 1–2 ad-hoc views (LLM)
```

## Run locally (laptop setup mirroring the lab machine)

```bash
# 1. OpenSearch 3.4.0 (same version as the lab)
docker compose up -d opensearch
curl localhost:9200/_cluster/health

# 2. Python env
conda activate legalrag
pip install -e ".[dev,app]"

# 3. Ingest CanLegalRAGBench (downloads from HF on first run)
#    Qwen3-Embedding-0.6B + flat RCTS 1024/128 (same as UBC's base config).
#    On the M2 this takes several hours; on the lab 4090 it takes minutes.
python scripts/ingest_canlegal.py --index-name canlegal-qwen3 \
    --embedding-model Qwen/Qwen3-Embedding-0.6B --chunk-size 1024 --chunk-overlap 128

# 4. LLM (Qwen3). Pick one and set it in .env:
#    a) lab vLLM: see scripts/serve_qwen3_vllm.sh, then
#       ssh -N -L 8001:127.0.0.1:8001 secb1010-d11.csil.sfu.ca
#       LLM_BASE_URL=http://localhost:8001/v1  LLM_API_KEY=EMPTY  LLM_MODEL=qwen3
#    b) local fallback: brew install ollama && ollama serve && ollama pull qwen3:8b
#       LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen3:8b

# 5. API (http://localhost:8000/docs)
uvicorn legalrag.api.app:app --port 8000

# 6. Web UI (http://localhost:5173; proxies /api to :8000)
cd web && npm install && npm run dev
```

### Build the index on the lab GPU instead

Ingesting on the M2 takes hours. Build `canlegal-qwen3` on the lab 4090 and restore it on
the laptop via an OpenSearch snapshot; see **[lab_ingestion.md](lab_ingestion.md)**.

Explore-specific settings (`EXPLORE_*` in `.env`, see `legalrag/core/config.py`):
`EXPLORE_INDEX_NAME` (default `canlegal-qwen3`), `EXPLORE_EMBEDDING_MODEL`
(must match the index), `EXPLORE_LLM_MODEL` (overrides `LLM_MODEL` for views),
`EXPLORE_PROFILES_DIR`, `EXPLORE_CORS_ORIGINS`.

## Files

| File | Role |
|---|---|
| `legalrag/ingestion/canlegal_loader.py` | Loads `documents.json`; sets court/jurisdiction from `canlegal_courts.yaml` |
| `scripts/ingest_canlegal.py` | Download + chunk + embed + index |
| `legalrag/explore/engine.py` | Orchestrates one exploration; streams events; document reader |
| `legalrag/explore/views.py` | Built-in views, LLM agents (pydantic-ai), grounding check |
| `legalrag/explore/schemas.py` | Evidence/View models + LLM output schemas (match Demo renderers) |
| `legalrag/explore/profile.py` | User profile + JSON-file store (dev only) |
| `legalrag/prompts/explore_policy.yaml` | Profile options, retrieval, jurisdiction policy, LLM settings |
| `legalrag/prompts/explore_views.yaml` | All view-generation prompts + ad-hoc view catalog |
| `legalrag/api/app.py` | FastAPI endpoints |
| `web/src/…` | React UI; `web/src/vendor/` = Reyhaneh's Demo renderers (copied) |
| `scripts/serve_qwen3_vllm.sh` | vLLM launcher for the lab 4090 |

### Event stream (`POST /api/explore`, one JSON object per line)
`status` → `query` (rewrite, keywords, jurisdictions, relaxed) → `evidence` (q + e1..eN)
→ `view` × N (`case_landscape`, `case_timeline`, `fact_toggle`, `adhoc_0..1`) → `plan`
(topic, orientation, follow-ups) → `view_error` (non-fatal) → `done` | `error`.

## Decisions & templates to fill in

Items marked ✅ were decided in the 2026-10-12 session; the others are open.

1. **LLM.** ✅ Qwen3, on lab vLLM with an Ollama fallback. Open:
   - Which model on the 4090: `Qwen3-30B-A3B-GPTQ-Int4` (default) or `Qwen3-14B-AWQ`.
   - Whether to set `EXPLORE_LLM_MODEL` separately from `LLM_MODEL`.

   The HF router key has no credits left (HTTP 402).
2. **Prompts** (`legalrag/prompts/explore_views.yaml`): all are first drafts. Review:
   - `preamble`: the grounding rules and the no-advice rule.
   - `fact_toggle`: 4–8 facts, statuses stated / assumed / unknown.
   - `planner`: the orientation is ≤2 neutral sentences. This is the only prose shown; confirm it's acceptable for "no answer generation".
   - Each catalog builder: node and item counts, and vocabularies.
3. **View catalog** (`catalog:` in the same file, plus `ADHOC_OUTPUTS` in `schemas.py`). Which ad-hoc kinds to offer: concept_map, comparison, legal_test, rule_hierarchy, process_flow, scenario_timeline. Add new ones here, plus a renderer.
4. **Pre-defined views.** Currently: case landscape, case timeline, fact toggles. Decide whether to add others (e.g. a statute/provision map, or "how courts decided" outcome bars from metadata).
5. **Schemas** (`legalrag/explore/schemas.py`). Enumerations to confirm:
   - matrix values: Yes/No/Maybe/Likely/Fails/Missing/N/A;
   - decision outcomes: allowed/prohibited/conditional;
   - rule categories;
   - timeline categories (event/legal);
   - network entity types.
6. **Profile options** (`explore_policy.yaml → profile`). Confirm:
   - the familiarity levels and their `audience` text;
   - the goals and their `planner_hint`;
   - the interest chips.

   Currently interests only feed the planner prompt. Decide whether they should also drive example questions or retrieval boosts.
7. **Jurisdiction policy** (`explore_policy.yaml → jurisdiction_policy`). ✅ Filter by profile; no UBC-style `"I am in {province}."` prefix. Open:
   - `always_include: [federal, unknown]`;
   - `min_results: 8` before relaxing to all of Canada.
8. **Court mapping** (`legalrag/ingestion/canlegal_courts.yaml`). Open:
   - Verify the `CM` code.
   - 73 documents (6.5%) still resolve to `unknown`, mostly `CANLII`-cited ones. Map them manually, infer from the text, or leave as is.
9. **Retrieval sizes** (`explore_policy.yaml → retrieval`). Open:
   - `top_k` 60, `evidence_k` 20, `llm_evidence_k` 15;
   - `use_formulator` (adds one LLM call).
10. **Embedding.** ✅ Qwen3-Embedding-0.6B with no query instruction prompt (same as UBC). Open: whether to try `prompt_name="query"` for queries.
11. **Chunking.** ✅ Flat RCTS, 1024 characters, 128 overlap, LangChain-style whole-piece overlap.
12. **Grounding.** Unknown evidence IDs are stripped, and items left with no citation are hidden by the renderers. Open: the check only validates that an ID *exists*, not that the passage supports the item. Consider an LLM-judge or entailment pass.
13. **Auth.** Name-only login and JSON-file profiles are **dev-only**. Choose real auth (e.g. SFU CAS, OAuth, email links) and a database before any public deployment.
14. **Hosting.** Open:
    - The frontend is static and can go anywhere (Netlify, or the same VM).
    - The backend needs a persistent machine running OpenSearch, the embedder, the reranker and an LLM endpoint; Netlify can't host it.
    - Candidates: an Alliance Cloud persistent VM (needs a PI request through RAS), an SFU CS server-room host (request via CSTS), or the lab machine for dev only (firewalled, reachable by SSH tunnel).
15. **Ingest speed.** ✅ Build on the lab 4090 and restore by snapshot ([lab_ingestion.md](lab_ingestion.md)).
16. **Demo renderers.** `web/src/vendor/` is a copy of `Demo/js/viz-renderers.js` and its CSS. Agree with Reyhaneh on a single source of truth.
