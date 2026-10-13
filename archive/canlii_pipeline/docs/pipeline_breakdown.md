# LegalRAG — End-to-End Pipeline Breakdown

---

## INGESTION PIPELINE

### Step 1 — Load (`TxtFileLoader`)

`legalrag/ingestion/loader.py`

- Reads each `.txt` file from disk as raw bytes decoded as UTF-8.
- The **raw text including the CanLII header** is preserved unchanged so the
  metadata extractor can read it.
- A `RawDocument` object is created with `source_path` as the only metadata
  field at this point.
- Supports loading a single file or an entire directory (recursively finds `*.txt`).

---

### Step 2 — Metadata Extraction (`CanLIIMetadataExtractor`)

`legalrag/ingestion/metadata_extractor.py`

Parses the structured 5-line CanLII header found in the first ~600 characters:

```
CASE: 2015 ONSC 7241 Parkinson v The Corporation of The City of Brampton
YEAR: 2015
COURT: ONSC
PAGES: 12
URL: https://www.canlii.org/...
================================================================================
```

**Fields extracted:**

| Field | Source | Example |
|---|---|---|
| `citation` | CASE line prefix (regex `^\d{4} [A-Z]+ \d+`) | `2015 ONSC 7241` |
| `case_name` | CASE line remainder after citation | `Parkinson v The Corporation...` |
| `court_abbrev` | COURT line directly | `ONSC` |
| `court` | Looked up from `COURT_ABBREV_MAP` (28 courts) | `Ontario Superior Court of Justice` |
| `jurisdiction` | Looked up from `COURT_JURISDICTION_MAP` | `ON` |
| `year` | YEAR line | `2015` |
| `decision_date` | YEAR line → `date(year, 1, 1)`, then refined to specific date from body if found | `2015-03-15` |
| `pages` | PAGES line | `12` |
| `url` | URL line (optional — absent in older files) | `https://www.canlii.org/...` |
| `doc_type` | Hardcoded `"case"` for all CanLII documents | `case` |

**Edge cases handled:**
- Older files where `COURT` is a numeric CanLII ID (e.g. `2170`): court abbreviation
  extracted from parentheses in the CASE line, e.g. `(FCA)`.
- Files without a URL line: regex makes the URL group optional.
- Fallback body scan: if header parse fails, regexes scan the first 3000 chars
  of body text for citation, court name, and date.

---

### Step 2b — Finalise `doc_id`

`legalrag/core/models.py` → `doc_id_from_citation()`

After extraction, a stable document ID is computed:

- **Primary** (≈99.3% of files): `MD5("citation|2015 ONSC 7241")`
  Content-based — survives file renames, directory moves, re-downloads.
- **Fallback** (≈0.7% of files with no parseable citation): `MD5("stem|<filename stem>")`
  Survives directory renames but not file renames.

All chunk IDs subsequently derive from this `doc_id`, so re-ingestion is idempotent.

---

### Step 2c — Clean Text (`clean_document_text`)

`legalrag/ingestion/loader.py`

Before chunking, the text is cleaned:
- Header block stripped (everything up to and including the `===` separator line).
- Page marker lines removed: `--- PAGE N ---`
- Consecutive blank lines collapsed to at most 2.
- Leading/trailing whitespace stripped.

---

### Step 3 — Hierarchical Chunking (`HierarchicalChunker`)

`legalrag/ingestion/chunker.py`

**Two-level hierarchy ("small-to-big retrieval"):**

**Parent chunks** (~1500 chars):
- Text split at sentence boundaries: regex `(?<=[.!?])\s+`
- Sentences greedily merged up to 1500 characters per parent.
- Stored in OpenSearch **without an embedding** — used only for context
  expansion at generation time.
- `parent_chunk_id = None`

**Child chunks** (~512 chars, 64-char overlap):
- Sliding window over each parent's text.
- Step size = `512 − 64 = 448` characters.
- Each child carries `parent_chunk_id` pointing back to its parent.

**Deterministic IDs** (MD5 hashes — makes re-ingestion idempotent):
```
parent_id = MD5(doc_id | char_start | char_end)
child_id  = MD5(doc_id | char_start | char_end)
```

Typical document yield: ~5–15 parent chunks, ~20–60 child chunks (varies by document length).

---

### Step 4 — Embedding (`SentenceTransformerEmbedder`)

`legalrag/ingestion/embedder.py`

Only **child chunks** are embedded. Parents are stored as plain text.

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Hardware**: CPU (MPS on Apple Silicon if available)
- **Batch size**: 64 texts per forward pass
- **Normalisation**: L2-normalised (`normalize_embeddings=True`) — enables
  dot-product = cosine similarity without extra normalisation at query time.
- **Alternative**: `OpenAIEmbedder` (swap via `EMBEDDING_PROVIDER=openai` in `.env`)

---

### Step 5 — Index (`OpenSearchIndexer`)

`legalrag/ingestion/indexer.py` + `legalrag/opensearch/client.py`

**Single index `legalrag`** — parents and children in the same index, distinguished
by the `is_parent` boolean field.

Upserted via `opensearch-py` bulk API. `_id` = `chunk_id`, so re-indexing the same
document overwrites existing chunks rather than creating duplicates.

**Index mapping:**

| Field | OpenSearch type | Details |
|---|---|---|
| `text` | `text` | Custom `legal_analyzer`: standard tokenizer + lowercase + stop-words filter |
| `embedding` | `knn_vector` (384-dim) | HNSW, cosine similarity, nmslib engine |
| `court` | `keyword` | Exact-match filterable |
| `citation` | `keyword` | Exact-match filterable |
| `source_path` | `keyword` | Exact-match filterable |
| `decision_date` | `date` (`yyyy-MM-dd`) | Range-filterable |
| `is_parent` | `boolean` | Filters child-only for kNN/BM25 |
| `chunk_id`, `doc_id`, `parent_chunk_id` | `keyword` | Linkage |
| `char_start`, `char_end` | `integer` | Byte offsets in original document |

**HNSW parameters:**
- Engine: `nmslib`
- Space: `cosinesimil`
- `ef_construction`: 128 (build quality — higher = better graph, slower build)
- `m`: 24 (bidirectional links — higher = better recall, more memory)
- `ef_search`: 100 (query-time beam width — higher = better recall, slower query)

**Not stored** (extracted but dropped at index time):
`case_name`, `court_abbrev`, `year`, `pages`, `url`, `jurisdiction`

---

## QUERY PIPELINE

### Step 1 — Query Formulation (`LLMQueryFormulator`)

`legalrag/query/formulator.py` + `legalrag/prompts/formulator.yaml`

A **pydantic-ai Agent** backed by `Qwen/Qwen3.5-397B-A17B:novita` via the
HuggingFace Inference Router receives the raw user question.

The system prompt is loaded from `legalrag/prompts/formulator.yaml` (edit there
to change LLM behaviour without touching Python).

**Output schema** (`_FormulatorOutput` Pydantic model):

| Field | Purpose |
|---|---|
| `reformulated_query` | Semantically optimised rewrite for dense retrieval |
| `lexical_keywords` | 3–8 BM25 keywords (legal terms, party names, statutes) |
| `court_filter` | Exact court name if user specifies one; `null` otherwise |
| `citation_filter` | Neutral citation if user specifies one; `null` otherwise |
| `date_from` / `date_to` | ISO date range if mentioned; `null` otherwise |

On LLM failure, falls back to using the raw query unchanged (`PassthroughQueryFormulator` behaviour).

---

### Step 2 — Hybrid Retrieval (`OpenSearchRetriever`)

`legalrag/query/retriever.py` + `legalrag/opensearch/client.py`

The reformulated query is embedded with the **same** `all-MiniLM-L6-v2` model
(dimension must match the indexed vectors — mismatch causes a 400 error).

Two searches run separately, then merged:

**kNN search** (semantic):
- OpenSearch `knn` query on the `embedding` field.
- `is_parent: false` filter ensures only child chunks are retrieved.
- Any `court_filter` / `citation_filter` added as additional `term` filters.
- Returns top-K=80 by vector cosine similarity.

**BM25 search** (lexical):
- OpenSearch `match` query on the `text` field (analysed with `legal_analyzer`).
- Same `is_parent: false` + metadata filters.
- Returns top-K=80 by BM25 score.

**Reciprocal Rank Fusion (RRF)** merges the two lists:
```
score(doc) = Σ  1 / (60 + rank_in_list)
```
Top-80 merged results returned. No native hybrid query needed — compatible with OpenSearch 2.x.

**Date range filtering**: `date_from`/`date_to` are extracted by the formulator
but **not yet applied** as range filters (placeholder — marked TODO in `retriever.py`).

**court_filter caveat**: Applied as an exact `term` match on the full court name
(e.g. `"Ontario Superior Court of Justice"`). Partial or abbreviated names won't match.

---

### Step 3 — Routing + Reranking (`ThresholdRouter` + `CrossEncoderReranker`)

`legalrag/query/router.py` + `legalrag/query/reranker.py`

**Cross-encoder reranking** — all 80 candidates scored:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally, no API key)
- Jointly encodes `(query, chunk_text)` pairs — captures interaction between
  query and passage, unlike bi-encoders which score independently.
- Output: raw logits (unbounded real numbers, ~−10 to +10). Higher = more relevant.
  Negative scores are normal; what matters is relative ranking.
- Model loaded lazily on first use (`@cached_property`).

**Routing decision:**

| Condition | Path | Result |
|---|---|---|
| ≥ 3 candidates with `rerank_score ≥ 0.6` | `reranker` (fast) | Top-20 reranked chunks |
| Fewer than 3 confident candidates | `deep_search` (slow) | **Placeholder** — also returns top-20 reranked chunks |

Deep Search (RLM-inspired iterative multi-query expansion) is **TODO**.
See `router.py` docstring and [arxiv:2512.24601](https://arxiv.org/abs/2512.24601).

---

### Step 4 — Generation (`LLMGenerator`)

`legalrag/query/generator.py` + `legalrag/prompts/generator.yaml`

**Parent expansion (small-to-big)**:
- For each of the top-20 reranked chunks, the generator fetches its **parent chunk**
  from OpenSearch by `parent_chunk_id`.
- Parent text (~1500 chars) replaces the child snippet (~512 chars), giving the LLM
  richer context around the matched passage.
- At most 5 parent-expanded excerpts are included in the prompt (configurable via
  `max_context_chunks`).

**Context assembly** (template from `generator.yaml`):
```
[Excerpt 1 | Court: Ontario Superior Court of Justice | Citation: 2015 ONSC 7241]
<parent text ~1500 chars>

[Excerpt 2 | Court: ... | Citation: ...]
<parent text>
...
```

**LLM call**:
- Model: `Qwen/Qwen3.5-397B-A17B:novita` via HuggingFace Inference Router
- Temperature: 0.1 (near-deterministic)
- System prompt loaded from `legalrag/prompts/generator.yaml`

**Output**: `RAGResponse` with:
- `answer` — generated answer text
- `retrieved_chunks` — list of `RetrievedChunk` objects with scores (provenance)
- `router_path` — `"reranker"` or `"deep_search"`

---

## LOGGING

Every query run is logged to both stdout and `logs/queries.log`:

```
────────────────────────────────────────────────────────────────────────
RAW QUERY   : <original user question>
REFORMULATED: <LLM-rewritten query>
KEYWORDS    : <comma-separated BM25 keywords>
RETRIEVED   : 80 candidates
  [01] 2015 ONSC 7241      | Ontario Superior Court of Justice  | sem=0.7194 lex=0.7194
  [02] ...
ROUTER PATH : reranker → 20 chunks kept
RERANKED RESULTS:
  [01] 2015 ONSC 7241      | Ontario Superior Court of Justice  | rerank=3.2100
  [02] ...
ANSWER      :
<generated answer text>
────────────────────────────────────────────────────────────────────────
```

Third-party library HTTP logs (`httpx`, `opensearch`, `sentence_transformers`) are
suppressed to WARNING level to keep output clean. Pass `--log-level DEBUG` to see them.
