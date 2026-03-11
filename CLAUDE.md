# CLAUDE.md — LegalRAG Project Context

Modular Retrieval-Augmented Generation system for 3000 CanLII Canadian legal documents,
backed by OpenSearch. All components are swappable via abstract interfaces.

---

## Environment Setup

```bash
# Create and activate conda env
conda create -n legalrag python=3.11 -y
conda activate legalrag
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env: set LLM_API_KEY=hf_... (HuggingFace token)
# All other defaults work for local Docker setup
```

---

## Common Commands

```bash
# Start OpenSearch + Dashboards
docker compose up -d
curl http://localhost:9200/_cluster/health          # verify healthy

# Ingest documents
python scripts/ingest.py data/subset/               # small subset (~20 docs)
python scripts/ingest.py data/extracted_court_documents/  # full 3000 docs

# Query
python scripts/query.py "What is the standard of review for s.7 Charter claims?"
python scripts/query.py "..." --log-level DEBUG     # verbose with HTTP requests

# Delete index and re-ingest (required after changing EMBEDDING_MODEL/EMBEDDING_DIM)
curl -X DELETE http://localhost:9200/legalrag
python scripts/ingest.py data/subset/

# Run tests
pytest tests/ -v
pytest tests/ingestion/ -v                          # ingestion tests only

# Check indexed document count
curl http://localhost:9200/legalrag/_count
```

---

## LegalBench-RAG Evaluation Commands

Metrics: **chunk-level Precision@K and Recall@K** (binary hit/miss per GT snippet at rank
cutoffs K). A single retrieval pass fetches `max(ks)` chunks; metrics are computed at each K
by slicing the ranked list. Default K values: 20, 40, 60.

```bash
# ── Step 1: Ingest corpus ──────────────────────────────────────────────────────

# Ingest only corpus files referenced by benchmark tests (recommended)
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG

# Ingest only files for specific sub-benchmarks
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad maud

# Fast smoke test: ingest files for first 10 test cases per benchmark
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --limit 10

# Ingest entire corpus (all *.txt files, ignores benchmark filter)
python -m evaluation.LegalBenchRAG.ingest \
    --data-dir data/LegalBenchRAG \
    --all

# Delete index and re-ingest (required after changing EMBEDDING_MODEL/EMBEDDING_DIM)
curl -X DELETE http://localhost:9200/legalrag-legalbenchrag
python -m evaluation.LegalBenchRAG.ingest --data-dir data/LegalBenchRAG


# ── Step 2: Sample a subset of queries (optional) ─────────────────────────────

# The benchmarks_subset/ dir holds a fixed 50-query sample (seed=42, ~12-13 per benchmark)
# To recreate it:
python3 -c "
import json, random
random.seed(42)
import pathlib
out = pathlib.Path('data/LegalBenchRAG/benchmarks_subset')
out.mkdir(exist_ok=True)
for name, n in [('contractnli',13),('cuad',13),('maud',12),('privacy_qa',12)]:
    tests = json.load(open(f'data/LegalBenchRAG/benchmarks/{name}.json'))['tests']
    json.dump({'tests': random.sample(tests, n)}, open(out/f'{name}.json','w'), indent=2)
"


# ── Step 3: Evaluate ──────────────────────────────────────────────────────────

# Full evaluation on all 4 benchmarks at K=20,40,60
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --ks 20 40 60

# Evaluate on the 50-query subset (fast iteration)
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks-dir data/LegalBenchRAG/benchmarks_subset \
    --ks 20 40 60

# Custom K values and specific sub-benchmarks
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --benchmarks cuad maud \
    --ks 10 20 50 100

# Quick smoke test (10 queries per benchmark, verbose)
python -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir data/LegalBenchRAG \
    --limit 10 \
    --ks 20 40 60 \
    --log-level INFO

# Check index doc count
curl http://localhost:9200/legalrag-legalbenchrag/_count

# Check index embedding dimension (must match EMBEDDING_DIM in .env)
curl -s http://localhost:9200/legalrag-legalbenchrag/_mapping \
    | python3 -c "import json,sys; m=json.load(sys.stdin); \
      print(list(m.values())[0]['mappings']['properties']['embedding']['dimension'])"
```

### Baseline results (50-query subset, nlpaueb/legal-bert-base-uncased, seed=42)

| | R@20 | R@40 | R@60 | P@20 | P@40 | P@60 |
|---|---|---|---|---|---|---|
| contractnli | 0.231 | 0.308 | 0.462 | 0.0154 | 0.0096 | 0.0090 |
| cuad | 0.154 | 0.333 | 0.641 | 0.0077 | 0.0135 | 0.0141 |
| maud | 0.083 | 0.125 | 0.167 | 0.0042 | 0.0063 | 0.0056 |
| privacy_qa | 0.313 | 0.563 | 0.646 | 0.0333 | 0.0229 | 0.0167 |
| **OVERALL** | **0.195** | **0.332** | **0.482** | **0.0150** | **0.0130** | **0.0113** |

---

## Architecture

### Ingestion Pipeline

```
TxtFileLoader
  → CanLIIMetadataExtractor    (parses 5-line CanLII header)
  → doc_id_from_citation()     (content-based ID: MD5 of citation)
  → clean_document_text()      (strip header block + page markers)
  → HierarchicalChunker        (parent ~1500 chars, child ~512 chars, 64 overlap)
  → SentenceTransformerEmbedder (all-MiniLM-L6-v2, 384-dim, child chunks only)
  → OpenSearchIndexer           (bulk upsert into single index "legalrag")
```

### Query Pipeline

```
LLMQueryFormulator    (pydantic-ai Agent → StructuredQuery with filters)
  → OpenSearchRetriever (hybrid: kNN + BM25 via RRF, top-80)
  → ThresholdRouter     (rerank all 80, route on confidence)
    → CrossEncoderReranker  (ms-marco-MiniLM-L-6-v2, top-20)
  → LLMGenerator        (parent-expand top-5 chunks → Qwen3.5 answer)
```

---

## Key Files

| File | Responsibility |
|---|---|
| `legalrag/core/config.py` | All settings via Pydantic BaseSettings; reads from `.env` |
| `legalrag/core/models.py` | Shared Pydantic schemas + `stable_id()` + `doc_id_from_citation()` |
| `legalrag/core/interfaces.py` | Abstract base classes for every swappable component |
| `legalrag/ingestion/loader.py` | `TxtFileLoader` + `clean_document_text()` |
| `legalrag/ingestion/metadata_extractor.py` | `CanLIIMetadataExtractor` — parses CanLII headers |
| `legalrag/ingestion/chunker.py` | `HierarchicalChunker` — parent + child chunks |
| `legalrag/ingestion/embedder.py` | `SentenceTransformerEmbedder` / `OpenAIEmbedder` |
| `legalrag/ingestion/indexer.py` | `OpenSearchIndexer` — bulk upsert |
| `legalrag/ingestion/pipeline.py` | `IngestionPipeline` — orchestrates all ingestion steps |
| `legalrag/prompts/formulator.yaml` | System prompt + field docs for query formulator |
| `legalrag/prompts/generator.yaml` | System prompt + context templates for answer generator |
| `legalrag/prompts/loader.py` | Cached YAML loader (`lru_cache`) |
| `legalrag/query/formulator.py` | `LLMQueryFormulator` (pydantic-ai Agent) |
| `legalrag/query/retriever.py` | `OpenSearchRetriever` — hybrid kNN + BM25 |
| `legalrag/query/reranker.py` | `CrossEncoderReranker` |
| `legalrag/query/router.py` | `ThresholdRouter` — fast/deep-search path |
| `legalrag/query/generator.py` | `LLMGenerator` — parent expansion + LLM call |
| `legalrag/query/pipeline.py` | `QueryPipeline` — orchestrates all query steps; logs full run trace |
| `legalrag/opensearch/client.py` | `OpenSearchClient` — kNN, BM25, hybrid, bulk, mapping |
| `legalrag/utils/llm_client.py` | Factory for sync/async OpenAI-compatible clients |
| `legalrag/utils/logging.py` | structlog config; stdout + `logs/queries.log` (append) |
| `docs/pipeline_breakdown.md` | Full technical walkthrough of ingestion + query pipeline |
| `docs/evaluation.md` | CLERC + LegalBench-RAG evaluation guides with runnable code |
| `cursor.md` | Session log of all major development changes |

---

## Design Decisions

**Document IDs**: `doc_id = MD5("citation|2015 ONSC 7241")` — content-based, survives file
renames. Falls back to `MD5("stem|<filename stem>")` for ~0.7% of docs without a citation.
`chunk_id = MD5(doc_id|char_start|char_end)` — re-ingestion is idempotent (upsert by `_id`).

**Hierarchical chunking**: Child chunks (512 chars) are embedded and retrieved. Parent chunks
(1500 chars) are stored without embeddings and fetched at generation time for richer context
(small-to-big retrieval pattern). Splitting is done **purely by character position** using
`_split_positions()` — never by reconstructing text from tokens (which loses whitespace and
causes `str.find()` to fail on legal documents with multi-space / newline runs, leaving large
unindexed gaps).

**Hybrid search**: kNN and BM25 results are merged via Reciprocal Rank Fusion (RRF):
`score = Σ 1/(60 + rank)`. No native hybrid query needed — works on OpenSearch 2.x.

**Prompts as YAML**: System prompts and message templates live in `legalrag/prompts/*.yaml`.
Edit there to change LLM behaviour without touching Python.

**LLM client**: Uses the `openai` Python library as a generic HTTP client. Swap backends
by changing `LLM_BASE_URL` + `LLM_MODEL` in `.env` — works with HF Router, vLLM, Ollama,
DashScope, OpenAI.

---

## OpenSearch Index Schema

Single index `legalrag`. Both parent and child chunks stored together, distinguished by `is_parent`.

| Field | Type | Notes |
|---|---|---|
| `text` | `text` | `legal_analyzer`: standard + lowercase + stop words |
| `embedding` | `knn_vector` (384-dim) | HNSW, cosine, nmslib; `ef_construction=128`, `m=24` |
| `court`, `citation`, `source_path` | `keyword` | Exact-match filterable |
| `decision_date` | `date` | ISO `yyyy-MM-dd`; range-filterable |
| `is_parent` | `boolean` | Filters child-only for kNN/BM25 searches |
| `chunk_id`, `doc_id`, `parent_chunk_id` | `keyword` | Linkage fields |

**Not stored** (extracted but dropped): `case_name`, `court_abbrev`, `year`, `pages`, `url`, `jurisdiction`

---

## Metadata Extracted per Document

From the CanLII 5-line header:
- `citation` — e.g. `2015 ONSC 7241`
- `case_name` — e.g. `Parkinson v The Corporation of The City of Brampton`
- `court_abbrev` — e.g. `ONSC`
- `court` — e.g. `Ontario Superior Court of Justice` (from `COURT_ABBREV_MAP`)
- `jurisdiction` — e.g. `ON` (from `COURT_JURISDICTION_MAP`)
- `year` — e.g. `2015`
- `decision_date` — defaults to `YYYY-01-01`, refined to specific date from body text
- `pages` — e.g. `12`
- `url` — CanLII source URL

---

## Gotchas

- **Embedding dim mismatch**: If you change `EMBEDDING_MODEL` or `EMBEDDING_DIM` in `.env`,
  you must delete and recreate the index: `curl -X DELETE http://localhost:9200/legalrag`
  then re-run ingestion. The index mapping is fixed at creation time.

- **Default dim is 1024**: `config.py` defaults to 1024 (for `bge-large-en-v1.5`). Current
  `.env` sets `EMBEDDING_DIM=384` for `all-MiniLM-L6-v2`. Make sure `.env` exists before
  the index is created.

- **Docker port binding**: Ports are bound to `127.0.0.1` (localhost only). Other machines
  on the same network cannot connect.

- **Query log**: Every run is appended to `logs/queries.log` (gitignored). Each run logs:
  reformulated query, all 80 retrieved chunks with scores, all reranked results, and the
  final answer. Third-party HTTP logs (`httpx`, `opensearch`, `sentence_transformers`) are
  suppressed to WARNING. Pass `--log-level DEBUG` to see them. To disable file logging,
  pass `log_file=None` to `configure_logging()`.

- **Deep Search is a placeholder**: The router falls back to top-N reranked candidates when
  routed to deep_search. The iterative multi-query expansion loop is TODO.
  See `legalrag/query/router.py` and arxiv.org/abs/2512.24601.

- **Date range filtering is a placeholder**: `date_from`/`date_to` are extracted by the
  formulator but not yet applied as OpenSearch range filters in the retriever.

- **court_filter exact match**: The retriever applies `court_filter` as an exact `term`
  query on the full court name. If the LLM extracts a partial name it won't match.

---

## Testing

```bash
pytest tests/ -v
pytest tests/ingestion/test_metadata_extractor.py -v   # metadata extraction
pytest tests/ingestion/test_chunker.py -v              # chunking + ID stability
pytest tests/ingestion/test_loader.py -v               # loading + text cleaning
```

Tests use real CanLII header samples. No OpenSearch required for unit tests.
