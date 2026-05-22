#!/usr/bin/env python3
"""Fast full-corpus ingest: larger SentenceTransformer batch and accumulated child chunks.

Drop-in replacement for the standard ingestion pipeline with no changes to
``evaluation/LegalBenchRAG/pipeline.py``.  Adds the repo root to sys.path automatically.

Usage::

    cd /path/to/LegalRAG
    python tools/full_corpus_fast/fast_ingest.py \\
      --parquet /scratch/.../passages.parquet \\
      --index-name legalrag-reglab-barexam-full \\
      --st-encode-batch 256 --child-chunk-batch 1024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# repo root = .../LegalRAG
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tqdm import tqdm  # noqa: E402

from evaluation.LegalBenchRAG.loader import LegalBenchRAGCorpusLoader  # noqa: E402
from evaluation.LegalBenchRAG.pipeline import _build_chunker  # noqa: E402
from legalrag.core.config import settings  # noqa: E402
from legalrag.core.models import Chunk  # noqa: E402
from legalrag.ingestion.embedder import (  # noqa: E402
    HuggingFaceEmbedder,
    OpenAIEmbedder,
    build_embedder,
)
from legalrag.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from legalrag.ingestion.indexer import OpenSearchIndexer  # noqa: E402
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings  # noqa: E402

# Same directory as this script (no need to install the tools.* package)
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from parquet_corpus_loader import ParquetCorpusLoader  # noqa: E402

logger = logging.getLogger(__name__)


def build_fast_embedder(
    model_name: str | None,
    provider: str | None,
    *,
    st_encode_batch: int,
    hf_encode_batch: int,
):
    resolved = provider or settings.embedding.provider
    if resolved == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name, batch_size=st_encode_batch)
    if resolved == "huggingface":
        return HuggingFaceEmbedder(model_name=model_name, batch_size=hf_encode_batch)
    if resolved == "openai":
        # OpenAIEmbedder default batch already 512
        return OpenAIEmbedder(model=model_name, batch_size=max(512, hf_encode_batch))
    return build_embedder(model_name=model_name, provider=provider)


def run_ingest(
    loader,
    *,
    chunker: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    parent_size: int | None,
    embedding_model: str | None,
    embedding_provider: str | None,
    index_name: str,
    child_chunk_batch: int,
    st_encode_batch: int,
    hf_encode_batch: int,
    log_level: str,
) -> None:
    configure_logging = __import__(
        "legalrag.utils.logging", fromlist=["configure_logging"]
    ).configure_logging
    configure_logging(level=log_level)

    chunker_obj = _build_chunker(
        chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        parent_size=parent_size,
    )
    embedder = build_fast_embedder(
        embedding_model,
        embedding_provider,
        st_encode_batch=st_encode_batch,
        hf_encode_batch=hf_encode_batch,
    )

    cfg = settings.opensearch
    lb_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": index_name,
        }
    )
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    os_client.ensure_index()
    indexer = OpenSearchIndexer(os_client)

    batch_chunks: list[Chunk] = []
    total_docs = 0
    total_child = 0
    hierarchical = chunker_obj.is_hierarchical

    desc = "Ingesting corpus"
    iterator = loader.iter()

    for doc in tqdm(iterator, desc=desc, unit="doc"):
        chunks = chunker_obj.chunk(doc)
        for chunk in chunks:
            if chunk.metadata is None:
                chunk.metadata = doc.metadata

        if hierarchical:
            child_chunks = [c for c in chunks if not c.is_parent]
            parent_chunks = [c for c in chunks if c.is_parent]
            if parent_chunks:
                indexer.index(parent_chunks)
        else:
            child_chunks = chunks
            parent_chunks = []

        batch_chunks.extend(child_chunks)
        if len(batch_chunks) >= child_chunk_batch:
            _embed_and_index(embedder, indexer, batch_chunks)
            total_child += len(batch_chunks)
            batch_chunks = []

        total_docs += 1

    if batch_chunks:
        _embed_and_index(embedder, indexer, batch_chunks)
        total_child += len(batch_chunks)

    logger.info(
        "Fast ingest complete — docs=%d child_chunks=%d (child_chunk_batch=%d, st_encode_batch=%d)",
        total_docs,
        total_child,
        child_chunk_batch,
        st_encode_batch,
    )


def _embed_and_index(embedder, indexer, chunks: list[Chunk]) -> None:
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
    indexer.index(chunks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--corpus-dir",
        type=Path,
        help="e.g. .../barexam_qa/corpus — all *.txt files underneath (same as LegalBenchRAGCorpusLoader)",
    )
    src.add_argument(
        "--parquet",
        type=Path,
        help=".parquet file or directory containing passages_part_*.parquet shards (build_passages_parquet.py)",
    )
    p.add_argument("--index-name", required=True)
    p.add_argument("--chunker", default="hierarchical", choices=("hierarchical", "recursive"))
    p.add_argument("--parent-size", type=int, default=2048)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-overlap", type=int, default=None)
    p.add_argument("--embedding-provider", default=None, choices=["sentence_transformers", "huggingface", "openai"])
    p.add_argument("--embedding-model", default=None)
    p.add_argument(
        "--child-chunk-batch",
        type=int,
        default=1024,
        help="Number of child chunks to accumulate before a single embed + bulk call (default 1024, vs 512 in the main pipeline)",
    )
    p.add_argument(
        "--st-encode-batch",
        type=int,
        default=256,
        help="batch_size passed to SentenceTransformer.encode (default 64 in build_embedder)",
    )
    p.add_argument(
        "--hf-encode-batch",
        type=int,
        default=128,
        help="batch_size for the HuggingFace embedder",
    )
    p.add_argument(
        "--parquet-read-batch",
        type=int,
        default=4096,
        help="Parquet rows per iter_batches call",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    if args.parquet:
        loader = ParquetCorpusLoader(args.parquet, read_batch_rows=args.parquet_read_batch)
    else:
        loader = LegalBenchRAGCorpusLoader(args.corpus_dir.resolve(), file_paths=None)

    run_ingest(
        loader,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_size=args.parent_size,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        index_name=args.index_name,
        child_chunk_batch=args.child_chunk_batch,
        st_encode_batch=args.st_encode_batch,
        hf_encode_batch=args.hf_encode_batch,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
