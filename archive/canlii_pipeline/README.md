# Archived: original CanLII answer pipeline

The first LegalRAG system: question → LLM query rewrite → hybrid retrieval → router →
cross-encoder rerank → **generated answer**, over ~3,000 CanLII `.txt` decisions.

Archived on 2026-10-12. Neither the exploration app (`legalrag/explore`, `legalrag/api`)
nor the LegalBench-RAG evaluation (`evaluation/`) used it. Files keep their original
repo-relative paths under this folder.

```
Ingestion:  TxtFileLoader → CanLIIMetadataExtractor → HierarchicalChunker → Embedder → OpenSearchIndexer
Query:      LLMQueryFormulator → OpenSearchRetriever (hybrid) → ThresholdRouter
              → CrossEncoderReranker (fast path) | Deep Search (TODO) → LLMGenerator (answer)
```

Full walkthrough: [docs/pipeline_breakdown.md](docs/pipeline_breakdown.md).

## Not runnable as-is

This folder is not an installed package. Its code also depends on a few helpers that were
removed from the live package at the same time:

- `BaseQueryFormulator`, `BaseRouter` and `BaseGenerator` in `legalrag/core/interfaces.py`;
- `RAGResponse` in `legalrag/core/models.py`;
- `get_sync_client` in `legalrag/utils/llm_client.py`;
- `OpenSearchClient.get_parent` / `get_by_chunk_id` / `delete_index`;
- `RetrievalSettings.rerank_top_k` (`RERANK_TOP_K`).

`notebooks/inspect_chunk_spans.ipynb` imports `legalrag.ingestion.loader`, so it no longer
runs either.

## Restoring

Everything is in git history. To check out the last full working tree, use the commit
before the one that added this folder:

```bash
git log --oneline -- archive/canlii_pipeline/README.md     # find the archiving commit
git checkout <archiving-commit>~1                           # full pre-archive tree
```

To bring a single file back, `git mv` it from here to its original path, then restore any
helpers it needs from the list above.
