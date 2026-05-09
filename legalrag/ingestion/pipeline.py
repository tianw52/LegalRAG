"""
Offline ingestion pipeline orchestrator.

Wires together: Loader → MetadataExtractor → Chunker → Embedder → Indexer.

Usage
-----
    pipeline = IngestionPipeline.default()
    pipeline.run("/path/to/legal_docs/")
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from legalrag.core.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseIndexer,
    BaseLoader,
    BaseMetadataExtractor,
)
from legalrag.core.models import doc_id_from_citation
from legalrag.ingestion.chunker import HierarchicalChunker, RecursiveCharacterTextSplitter
from legalrag.ingestion.embedder import SentenceTransformerEmbedder, build_embedder
from legalrag.ingestion.indexer import OpenSearchIndexer
from legalrag.ingestion.loader import TxtFileLoader, clean_document_text
from legalrag.ingestion.metadata_extractor import CanLIIMetadataExtractor
from legalrag.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full offline ingestion flow."""

    def __init__(
        self,
        loader: BaseLoader,
        extractor: BaseMetadataExtractor,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        indexer: BaseIndexer,
    ) -> None:
        self.loader = loader
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer

    @classmethod
    def default(
        cls,
        chunker: str = "hierarchical",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        parent_size: int | None = None,
        embedding_model: str | None = None,
    ) -> "IngestionPipeline":
        """Construct the pipeline from default config-driven components.

        Parameters
        ----------
        chunker:
            ``"hierarchical"`` (default) or ``"recursive"``.
        chunk_size:
            Child chunk size in characters (overrides config default).
        chunk_overlap:
            Overlap between consecutive chunks in characters.
        parent_size:
            Parent chunk size in characters — hierarchical only (default 1500).
        embedding_model:
            HuggingFace sentence-transformers model name.  Overrides config.
        """
        embedder = (
            SentenceTransformerEmbedder(model_name=embedding_model)
            if embedding_model
            else build_embedder()
        )
        os_client = OpenSearchClient.from_settings(embedding_dim=embedder.dim)
        os_client.ensure_index()

        chunker_obj = _build_chunker(
            chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            parent_size=parent_size,
        )

        return cls(
            loader=TxtFileLoader(),
            extractor=CanLIIMetadataExtractor(),
            chunker=chunker_obj,
            embedder=embedder,
            indexer=OpenSearchIndexer(os_client),
        )

    def run(self, source: str) -> None:
        """Run the full pipeline for *source* (file or directory)."""
        logger.info("Starting ingestion for: %s", source)

        documents = self.loader.load(source)
        logger.info("Loaded %d document(s)", len(documents))

        for doc in tqdm(documents, desc="Ingesting", unit="doc"):
            # Step 1 – metadata extraction (reads raw text incl. CanLII header)
            doc = self.extractor.extract(doc)

            # Step 2 – finalise doc_id now that citation is known.
            # Uses citation if available (content-based, path-independent),
            # falls back to filename stem for docs without a parseable citation.
            doc.metadata.doc_id = doc_id_from_citation(
                doc.metadata.citation, doc.metadata.source_path
            )

            # Step 3 – clean body text (strip header block + page markers)
            doc.text = clean_document_text(doc.text)

            # Step 4 – chunking
            chunks = self.chunker.chunk(doc)

            # Step 5 – embed retrievable chunks
            # Hierarchical: only child chunks are embedded; parents stored text-only.
            # Flat (recursive): every chunk is retrievable, embed them all.
            if self.chunker.is_hierarchical:
                child_chunks = [c for c in chunks if c.parent_chunk_id is not None]
            else:
                child_chunks = chunks
            if child_chunks:
                texts = [c.text for c in child_chunks]
                embeddings = self.embedder.embed(texts)
                for chunk, emb in zip(child_chunks, embeddings):
                    chunk.embedding = emb

            # Step 6 – index everything
            self.indexer.index(chunks)

        logger.info("Ingestion complete.")


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
