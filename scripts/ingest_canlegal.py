#!/usr/bin/env python3
"""
Ingest the UBC-VL/CanLegalRAGBench corpus into an OpenSearch index.

Downloads documents.json / queries.json from the HF Hub on first run.

Chunking mirrors UBC's reference setup (NLP-UBC/CanLegalRAGBench,
configs/base.yaml): flat recursive character splitting, 1024 chars / 128
overlap.  Every chunk is embedded and retrievable (no parent chunks).

Usage
-----
    conda activate legalrag
    python scripts/ingest_canlegal.py \
        --index-name canlegal-qwen3 \
        --embedding-provider sentence_transformers \
        --embedding-model Qwen/Qwen3-Embedding-0.6B \
        --chunk-size 1024 --chunk-overlap 128

    # quick smoke test on the first 20 documents
    python scripts/ingest_canlegal.py --index-name canlegal-dev --limit 20

The index name / embedding model must match EXPLORE_INDEX_NAME /
EXPLORE_EMBEDDING_MODEL used by the API (see legalrag/core/config.py).
"""

from __future__ import annotations

import argparse
import logging
import sys

from tqdm import tqdm

from legalrag.core.config import OpenSearchSettings, settings
from legalrag.ingestion.canlegal_loader import CanLegalRAGBenchLoader, download_dataset
from legalrag.ingestion.chunker import RecursiveCharacterTextSplitter
from legalrag.ingestion.embedder import build_embedder
from legalrag.ingestion.indexer import OpenSearchIndexer
from legalrag.opensearch.client import OpenSearchClient
from legalrag.utils.logging import configure_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/CanLegalRAGBench")
    p.add_argument("--index-name", required=True)
    p.add_argument("--embedding-provider", default="sentence_transformers")
    p.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--chunk-overlap", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="Only ingest the first N documents")
    p.add_argument("--log-level", default="WARNING")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    data_dir = download_dataset(args.data_dir)
    docs = CanLegalRAGBenchLoader().load(str(data_dir))
    if args.limit:
        docs = docs[: args.limit]

    cfg = settings.opensearch
    os_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": args.index_name,
        }
    )
    embedder = build_embedder(model_name=args.embedding_model, provider=args.embedding_provider)
    os_client = OpenSearchClient(cfg=os_cfg, embedding_dim=embedder.dim)
    os_client.ensure_index()
    indexer = OpenSearchIndexer(os_client)
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    n_chunks = 0
    for doc in tqdm(docs, desc="Ingesting", unit="doc"):
        chunks = chunker.chunk(doc)
        for chunk, emb in zip(chunks, embedder.embed([c.text for c in chunks])):
            chunk.embedding = emb
        indexer.index(chunks)
        n_chunks += len(chunks)

    logger.warning("Done: %d documents, %d chunks → index '%s'", len(docs), n_chunks, args.index_name)


if __name__ == "__main__":
    main(sys.argv[1:])
