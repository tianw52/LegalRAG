"""LegalBench-RAG ingestion pipeline.

Strategy
--------
Each corpus document is a legal contract (plain text). They are indexed into a
dedicated OpenSearch index (``legalrag-legalbenchrag``) using the same
``HierarchicalChunker`` + ``SentenceTransformerEmbedder`` stack used by the
main LegalRAG pipeline.

The *relative file path* (relative to ``corpus/``) is stored as
``metadata.citation`` on every chunk. The evaluator uses this to match
retrieved chunks back to ground-truth snippets.

Usage
-----
    from evaluation.LegalBenchRAG.pipeline import LegalBenchRAGIngestionPipeline

    pipeline = LegalBenchRAGIngestionPipeline.build(corpus_dir="data/LegalBenchRAG/corpus")
    pipeline.run(file_paths=["cuad/contract_001.txt", ...])   # or None to ingest all

Index
-----
``legalrag-legalbenchrag`` — separate from the CanLII ``legalrag`` index so
that the two corpora never interfere.  Delete and re-create when re-ingesting::

    curl -X DELETE http://localhost:9200/legalrag-legalbenchrag
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from legalrag.core.config import settings
from legalrag.core.models import Chunk
from legalrag.ingestion.chunker import HierarchicalChunker
from legalrag.ingestion.embedder import build_embedder
from legalrag.ingestion.indexer import OpenSearchIndexer
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings

from evaluation.LegalBenchRAG.loader import LegalBenchRAGCorpusLoader

logger = logging.getLogger(__name__)

INDEX_NAME = "legalrag-legalbenchrag"

_BATCH_SIZE = 512   # embed + index this many child chunks at once


class LegalBenchRAGIngestionPipeline:
    """Ingests LegalBench-RAG corpus documents into ``legalrag-legalbenchrag``."""

    def __init__(
        self,
        loader: LegalBenchRAGCorpusLoader,
        chunker: HierarchicalChunker,
        embedder,
        indexer: OpenSearchIndexer,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._indexer = indexer

    @classmethod
    def build(
        cls,
        corpus_dir: str,
        file_paths: list[str] | None = None,
    ) -> "LegalBenchRAGIngestionPipeline":
        """Factory: build pipeline from global settings.

        Parameters
        ----------
        corpus_dir:
            Path to the ``corpus/`` folder inside the downloaded data dir.
        file_paths:
            Explicit list of relative file paths to load.  Pass ``None`` to
            discover and ingest all ``*.txt`` files under ``corpus_dir``.
        """
        cfg = settings.opensearch
        lb_cfg = OpenSearchSettings(
            **{
                "OPENSEARCH_HOST": cfg.host,
                "OPENSEARCH_PORT": cfg.port,
                "OPENSEARCH_USER": cfg.user,
                "OPENSEARCH_PASSWORD": cfg.password,
                "OPENSEARCH_USE_SSL": cfg.use_ssl,
                "OPENSEARCH_INDEX_NAME": INDEX_NAME,
            }
        )
        embedder = build_embedder()
        os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
        os_client.ensure_index()

        return cls(
            loader=LegalBenchRAGCorpusLoader(corpus_dir, file_paths=file_paths),
            chunker=HierarchicalChunker(),
            embedder=embedder,
            indexer=OpenSearchIndexer(os_client),
        )

    def run(self, file_paths: list[str] | None = None) -> None:
        """Ingest documents.

        Parameters
        ----------
        file_paths:
            Override the file paths set at construction time.  Useful when
            you want to ingest only the files referenced by a specific
            benchmark subset.  ``None`` uses the paths configured in the
            loader (or discovers all ``*.txt`` files).
        """
        if file_paths is not None:
            self._loader._file_paths = file_paths

        batch_chunks: list[Chunk] = []
        total_docs = 0
        total_chunks = 0

        for doc in tqdm(self._loader.iter(), desc="Ingesting corpus", unit="doc"):
            chunks = self._chunker.chunk(doc)

            # Propagate the corpus-relative file path from doc metadata to each chunk
            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = doc.metadata

            child_chunks = [c for c in chunks if c.parent_chunk_id is not None]
            parent_chunks = [c for c in chunks if c.parent_chunk_id is None]

            # Index parent chunks immediately (no embeddings needed)
            if parent_chunks:
                self._indexer.index(parent_chunks)

            # Buffer child chunks for batch embedding
            batch_chunks.extend(child_chunks)
            if len(batch_chunks) >= _BATCH_SIZE:
                self._embed_and_index(batch_chunks)
                total_chunks += len(batch_chunks)
                batch_chunks = []

            total_docs += 1

        # Flush remaining child chunks
        if batch_chunks:
            self._embed_and_index(batch_chunks)
            total_chunks += len(batch_chunks)

        logger.info(
            "LegalBenchRAG ingestion complete — docs=%d child_chunks=%d",
            total_docs,
            total_chunks,
        )

    def _embed_and_index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self._indexer.index(chunks)

