# LegalRAG

A modular Retrieval-Augmented Generation system for legal documents, backed by OpenSearch.

---

## Architecture

![architecture- vector RAG](assets/vector_rag_architecture.png)

### Offline Ingestion Pipeline

```
Input (.txt files)
      │
      ▼
 TxtFileLoader          (legalrag/ingestion/loader.py)
      │  RawDocument
      ▼
 RegexMetadataExtractor (legalrag/ingestion/metadata_extractor.py)
      │  RawDocument + metadata
      ▼
 HierarchicalChunker    (legalrag/ingestion/chunker.py)
      │  [parent chunks + child chunks]
      ▼
 Embedder               (legalrag/ingestion/embedder.py)
      │  child chunks with dense vectors
      ▼
 OpenSearchIndexer      (legalrag/ingestion/indexer.py)
      │
      ▼
 OpenSearch Index       (kNN vector + BM25 full-text)
```

### Online Query Pipeline

```
Raw user query
      │
      ▼
 LLMQueryFormulator     (legalrag/query/formulator.py)
      │  StructuredQuery (reformulated text + metadata filters)
      ▼
 OpenSearchRetriever    (legalrag/query/retriever.py)
      │  top-K RetrievedChunks (hybrid kNN + BM25)
      ▼
 ThresholdRouter        (legalrag/query/router.py)
      │
      ├── [high confidence] → CrossEncoderReranker (legalrag/query/reranker.py)
      │                              │ top-N chunks
      │
      └── [low confidence] → Deep Search [TODO – see below]
                                     │ top-N chunks
      ▼
 LLMGenerator           (legalrag/query/generator.py)
      │  parent-expanded context → LLM prompt
      ▼
 RAGResponse (answer + provenance)
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Document format | Plain text (.txt) | Initial scope; interfaces support adding PDF/DOCX loaders |
| Chunking strategy | Hierarchical (parent + child) | Small-to-big retrieval: child for precise vector match, parent for rich context |
| Embedding model | Swappable (default: sentence-transformers) | `EMBEDDING_PROVIDER` env var; `SentenceTransformerEmbedder` or `OpenAIEmbedder` |
| LLM backend | Qwen via OpenAI-compatible API | `LLM_BASE_URL` env var; swap to any OpenAI-compatible endpoint |
| Vector store | OpenSearch (local Docker) | kNN (HNSW / nmslib) + BM25 in one system; hybrid search via RRF |
| Retrieval mode | Hybrid (kNN + BM25) via RRF | Combines semantic precision with lexical recall; tunable weight |
| Metadata fields | court, citation, decision_date | Chosen for legal domain relevance; easy to extend |
| Reranker | Cross-encoder (ms-marco-MiniLM) | Standard cross-encoder gives best quality/latency trade-off |
| Router paths | Reranker (fast) / Deep Search (slow) | Confidence-score threshold; deep search triggers on complex queries |
| Deep Search | RLM-inspired (placeholder) | See arxiv 2512.24601 – recursive LLM over document snippets |

---

## Deep Search – Design Reference

The **Deep Search** path is designed around the *Recursive Language Models* paper
(Zhang et al., 2025, [arxiv:2512.24601](https://arxiv.org/abs/2512.24601)).

The core idea: treat the document corpus as an **external environment**.
When the initial retrieval is not confident, instead of giving up, the LLM:
1. Analyses the initial candidate set.
2. Generates targeted follow-up queries.
3. Retrieves again and merges results via RRF.
4. Repeats until a confidence threshold or max-iteration budget is reached.

**Current status:** The interface (`BaseRouter`, `ThresholdRouter._deep_search`) is in
place. The iterative retrieval loop is `TODO` in `legalrag/query/router.py`.

---

## Project Structure

```
LegalRAG/
├── docker-compose.yml          # OpenSearch + Dashboards (local dev)
├── pyproject.toml              # dependencies & tooling config
├── .env.example                # environment variable template
├── scripts/
│   ├── ingest.py               # CLI: ingest documents
│   └── query.py                # CLI: run a query
├── legalrag/
│   ├── core/
│   │   ├── config.py           # Pydantic settings (env-driven)
│   │   ├── models.py           # Shared domain models 
│   │   └── interfaces.py       # Abstract base classes for all components
│   ├── ingestion/
│   │   ├── loader.py           # TxtFileLoader + clean_document_text()
│   │   ├── metadata_extractor.py  # CanLIIMetadataExtractor (header + body fallback)
│   │   ├── chunker.py          # HierarchicalChunker (deterministic chunk IDs)
│   │   ├── embedder.py         # SentenceTransformerEmbedder / OpenAIEmbedder
│   │   ├── indexer.py          # OpenSearchIndexer
│   │   └── pipeline.py         # IngestionPipeline orchestrator
│   ├── prompts/                # ← prompt configs (edit here, not in Python)
│   │   ├── loader.py           # YAML loader
│   │   ├── formulator.yaml     # system prompt + field docs + model params
│   │   └── generator.yaml      # system prompt + context templates + model params
│   ├── query/
│   │   ├── formulator.py       # LLMQueryFormulator (pydantic-ai Agent)
│   │   ├── retriever.py        # OpenSearchRetriever (hybrid kNN + BM25)
│   │   ├── reranker.py         # CrossEncoderReranker
│   │   ├── router.py           # ThresholdRouter (fast/slow path)
│   │   ├── generator.py        # LLMGenerator (parent-expansion + streaming)
│   │   └── pipeline.py         # QueryPipeline orchestrator
│   ├── opensearch/
│   │   └── client.py           # OpenSearchClient (kNN, BM25, hybrid, bulk)
│   └── utils/
│       ├── llm_client.py       # OpenAI-compatible client factory (sync + async)
│       └── logging.py          # structlog configuration
└── tests/
    ├── ingestion/
    ├── query/
    └── opensearch/
```

---

## Quickstart

### 1. Create and activate the conda environment

```bash
conda create -n legalrag python=3.11 -y
conda activate legalrag
pip install -e ".[dev]"
```


### 2. Configure environment variables

```bash
cp .env.example .env
```

The default LLM is **Qwen3.5-397B-A17B** via the HuggingFace Inference Router. Get a free token at https://huggingface.co/settings/tokens, then:

```bash
export HF_TOKEN=hf_...      # picked up automatically, or set LLM_API_KEY in .env
```

To run fully locally with a smaller model instead:
```bash
# Ollama (easiest)
ollama pull qwen2.5:7b
# then in .env:
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_API_KEY=ollama
# LLM_MODEL=qwen2.5:7b
```

### 3. Start OpenSearch (Docker)

```bash
docker compose up -d
# Wait ~30 s then verify:
curl http://localhost:9200/_cluster/health
```

OpenSearch Dashboards will be available at http://localhost:5601.

### 4. Ingest documents

```bash
python scripts/ingest.py data/extracted_court_documents/
```

### 5. Query

```bash
python scripts/query.py "What is the standard of review for s.7 Charter claims?"
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Next Steps

### Short-term (Phase 1)
- [ ] **Unit tests** for chunker, metadata extractor, and retriever (mock OpenSearch)
- [ ] **Evaluation harness**: build a small gold-standard Q&A set to measure retrieval recall@K and answer correctness
- [ ] **Date range filtering**: add OpenSearch `range` query for `decision_date` in retriever
- [ ] **LLM metadata extractor**: replace regex heuristics with an LLM call for better coverage

### Medium-term (Phase 2)
- [ ] **Deep Search implementation**: iterative multi-query expansion loop in `ThresholdRouter._deep_search`, following the RLM paper (arxiv:2512.24601)
- [ ] **Parent-chunk summarisation**: store LLM-generated summaries as parent chunks instead of raw text for better context quality

### Long-term (Phase 3)
- [ ] **Citation graph**: link cases by citation relationships and use graph traversal in Deep Search
- [ ] **Feedback loop**: collect user relevance signals to improve routing thresholds
- [ ] **Multi-index support**: separate indices per jurisdiction or doc_type with a meta-router

---


