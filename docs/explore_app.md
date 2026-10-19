# LegalRAG Explore — exploration-first UI

An exploration tool over the CanLegalRAGBench corpus (UBC-VL). The user asks a question.
The system retrieves passages and turns them into interactive views, each citing its
sources. There is **no free-text answer step**; the visual structures are the output.

```
React + Vite (web/)  ──NDJSON──▶  FastAPI (legalrag/api/app.py)
                                    └─ ExplorationEngine (legalrag/explore/engine.py)
                                         formulate (LLM) → dense kNN retrieve (OpenSearch, profile jurisdiction filter)
                                         → Qwen3-Reranker → evidence e1..eN
                                         → built-in views: case landscape, case timeline        (no LLM)
                                         → fact toggles (LLM)  ‖  planner agent → 0–3 extra views it judges useful (LLM)
                                         every step traced → "Under the hood" panel, logs/explore_traces/
```

Backend details (modules, API, configuration, tracing, extending): [legalrag/README.md](../legalrag/README.md).

## Run locally

Step-by-step setup with Ollama (prerequisites, index snapshot or ingest, checks,
troubleshooting): **[README.md → Run the website on localhost](../README.md#run-the-website-on-localhost)**.
An annotated example request: [legalrag/README.md](../legalrag/README.md#how-a-request-runs-end-to-end).

To use the lab GPU for the LLM instead of Ollama, run `scripts/serve_qwen3_vllm.sh` on the
lab machine and tunnel it (`ssh -N -L 8001:127.0.0.1:8001 secb1010-d11.csil.sfu.ca`); the
vLLM block is the `.env.example` default and sets `LLM_THINKING_OFF=vllm`.

### Build the index on the lab GPU instead

Ingesting on the M2 takes hours. Build `canlegal-qwen3` on the lab 4090 and restore it on
the laptop via an OpenSearch snapshot; see **[lab_ingestion.md](lab_ingestion.md)**.

Explore-specific settings (`EXPLORE_*` in `.env`, see `legalrag/core/config.py`):
`EXPLORE_INDEX_NAME` (default `canlegal-qwen3`), `EXPLORE_EMBEDDING_MODEL`
(must match the index), `EXPLORE_LLM_MODEL` (overrides `LLM_MODEL` for views),
`EXPLORE_PROFILES_DIR`, `EXPLORE_CORS_ORIGINS`.

## Files

**Backend** (`legalrag/`)

| File | Role |
|---|---|
| `ingestion/canlegal_loader.py` | Loads `documents.json`; court/jurisdiction from `canlegal_courts.yaml`; `download_dataset()` |
| `explore/engine.py` | `ExplorationEngine`: one exploration → event stream; document reader |
| `explore/views.py` | Built-in views, LLM agents (`_agent`, `_run`, `formulate`), `ground()` evidence check |
| `explore/schemas.py` | `Evidence` / `View` envelopes + LLM output schemas (field names match the Demo renderers) |
| `explore/profile.py` | `UserProfile`, `ProfileStore` (JSON files, dev only), onboarding options |
| `api/app.py` | FastAPI endpoints |
| `opensearch/client.py` | Shared client; `OpenSearchClient.from_settings(embedding_dim, index_name=…)` for any index |
| `prompts/explore_policy.yaml` | Profile options, retrieval sizes, jurisdiction policy, LLM settings |
| `prompts/explore_views.yaml` | All view-generation prompts + ad-hoc view catalog |

**Scripts** (`scripts/`)

| Script | Role |
|---|---|
| `ingest_canlegal.py` | Download + chunk + embed + index (see `docs/lab_ingestion.md`) |
| `check_chunking.py` | Audit chunker invariants and quality on the corpus before re-ingesting |
| `explore_cli.py` | Run one exploration through the API from the terminal; save the event stream |
| `serve_qwen3_vllm.sh` | vLLM launcher for the lab 4090 |

**Frontend** (`web/`): structure and conventions in [`web/README.md`](../web/README.md).
`web/scripts/smoke.mjs` (`npm run smoke`) is a headless-Chrome end-to-end test.

### Event stream (`POST /api/explore`, one JSON object per line)
`trace` (trace_id) → `status` → `query` (rewrite, keywords, jurisdictions, relaxed) →
`evidence` (q + e1..eN) → `view` × N (`case_landscape`, `case_timeline`, `fact_toggle`,
`adhoc_0..2`) → `plan` (topic, orientation, extra_views_reason, follow-ups) → `view_error`
(non-fatal) → `done` | `error`.

### Under the hood (tracing)
Each run is recorded by `legalrag/explore/trace.py`. The record has:
- step timings;
- the query rewrite;
- all 60 retrieval candidates, with retrieval rank and score, rerank score and whether each was kept;
- every LLM call: system and user prompts, raw response, thinking (when enabled), token usage, validation retries, errors.

View it in the UI ("Under the hood" button, live while running), in
`logs/explore_traces/<date>/<time>_<trace_id>.json`, or at `GET /api/traces/{id}`.
`scripts/explore_cli.py` prints a step and LLM timing summary at the end.

## Decisions & templates to fill in

Items marked ✅ were decided in the 2026-10-12 session; the others are open.

1. **LLM.** ✅ Qwen3, on lab vLLM (default in `.env`), with vllm-metal or Ollama on the laptop as fallbacks.
   ✅ `Qwen3-30B-A3B-GPTQ-Int4` on the 4090 runs a whole exploration in ~1 min (2026-10-18); the
   laptop reranker is now the slowest step (~45 s). Open:
   - Whether to set `EXPLORE_LLM_MODEL` separately from `LLM_MODEL`.

   The HF router key has no credits left (HTTP 402); `.env.example` defaults to vLLM on :8001.
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
17. **Reranker.** ✅ Qwen3-Reranker-0.6B (`EXPLORE_RERANKER_MODEL`). On 8 benchmark queries it kept 28 of the 35 ground-truth documents found by retrieval in its top 20; ms-marco kept 25. Open:
    - The task instruction (`explore_policy.yaml → reranker.instruction`).
    - Latency: about 28 s per 60 candidates on the M2, against about 1 s for ms-marco. Use 4B/8B on the lab GPU, or lower `retrieval.top_k` on the laptop.
    - The rerank query is the LLM rewrite, not the raw question. Decide whether to keep that.
18. **Extra views.** ✅ The planner may choose 0 to `planner.max_extra_views` (3) and must justify each choice (`why_useful`, `extra_views_reason`).
19. **LLM thinking.** Off for speed via `LLM_THINKING_OFF` (in `.env`, per server block), which must match the server (`ollama` or `vllm`). The `/no_think` suffix is ignored by Ollama: a test call took 17 s with thinking against 0.7 s with `reasoning_effort: none`. Use `none` to see the reasoning in traces, at the cost of latency.
