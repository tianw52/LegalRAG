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
``legalrag-lbr-hierarchical`` — separate from the CanLII ``legalrag`` index so
that the two corpora never interfere.  Delete and re-create when re-ingesting::

``legalrag-lbr-recursive`` — separate from the CanLII ``legalrag`` index so
that the two corpora never interfere.  Delete and re-create when re-ingesting::

To delete:
    curl -X DELETE http://localhost:9200/legalrag-lbr-recursive

"""

from __future__ import annotations

import logging

from tqdm import tqdm

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseChunker
from legalrag.core.models import Chunk
from legalrag.ingestion.chunker import HierarchicalChunker, RecursiveCharacterTextSplitter
from legalrag.ingestion.embedder import build_embedder
from legalrag.ingestion.indexer import OpenSearchIndexer
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings

from evaluation.LegalBenchRAG.loader import LegalBenchRAGCorpusLoader

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "legalrag-legalbenchrag"

_BATCH_SIZE = 512   # embed + index this many child chunks at once


class LegalBenchRAGIngestionPipeline:
    """Ingests LegalBench-RAG corpus documents into a dedicated OpenSearch index."""

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
        chunker: str = "hierarchical",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_size: int | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> "LegalBenchRAGIngestionPipeline":
        """Factory: build pipeline from global settings with optional overrides.

        Parameters
        ----------
        corpus_dir:
            Path to the ``corpus/`` folder inside the downloaded data dir.
        file_paths:
            Explicit list of relative file paths to load.  Pass ``None`` to
            discover and ingest all ``*.txt`` files under ``corpus_dir``.
        chunker:
            Chunker strategy: ``"hierarchical"`` (default) or ``"recursive"``.
        chunk_size:
            Child chunk size in characters (overrides config / env default).
        chunk_overlap:
            Overlap between consecutive child chunks in characters.
        parent_size:
            Parent chunk size in characters (hierarchical chunker only).
        embedding_model:
            HuggingFace model name for the sentence-transformers embedder,
            e.g. ``"BAAI/bge-large-en-v1.5"``.  Overrides config / env default.
        index_name:
            OpenSearch index to ingest into (default: ``legalrag-legalbenchrag``).
            Use a different name to keep multiple experiments isolated.
        """
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

        chunker_obj: BaseChunker = _build_chunker(
            chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_size=parent_size,
        )

        embedder = build_embedder(model_name=embedding_model, provider=embedding_provider)

        os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
        os_client.ensure_index()

        return cls(
            loader=LegalBenchRAGCorpusLoader(corpus_dir, file_paths=file_paths),
            chunker=chunker_obj,
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

        hierarchical = self._chunker.is_hierarchical

        for doc in tqdm(self._loader.iter(), desc="Ingesting corpus", unit="doc"):
            chunks = self._chunker.chunk(doc)

            # Propagate the corpus-relative file path from doc metadata to each chunk
            for chunk in chunks:
                if chunk.metadata is None:
                    chunk.metadata = doc.metadata

            if hierarchical:
                # Parent chunks: store text-only for context expansion
                # Child chunks: embed and retrieve; we search for child chunks only
                child_chunks = [c for c in chunks if not c.is_parent]
                parent_chunks = [c for c in chunks if c.is_parent]
                if parent_chunks:
                    self._indexer.index(parent_chunks)
            else:
                # Flat chunker: every chunk is a retrievable leaf — embed them all
                child_chunks = chunks
                parent_chunks = []

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_chunker(
    name: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
    parent_size: int | None,
) -> BaseChunker:
    """Instantiate a chunker by name with optional parameter overrides."""
    if name == "hierarchical":
        kwargs: dict = {}
        if parent_size is not None:
            kwargs["parent_size"] = parent_size
        if chunk_size is not None:
            kwargs["child_size"] = chunk_size
        if chunk_overlap is not None:
            kwargs["child_overlap"] = chunk_overlap
        return HierarchicalChunker(**kwargs)
    if name == "recursive":
        kwargs = {}
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            kwargs["chunk_overlap"] = chunk_overlap
        return RecursiveCharacterTextSplitter(**kwargs)
    raise ValueError(f"Unknown chunker: {name!r}. Choose 'hierarchical' or 'recursive'.")

