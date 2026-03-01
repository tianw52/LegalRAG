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
from legalrag.ingestion.chunker import HierarchicalChunker
from legalrag.ingestion.embedder import build_embedder
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
    def default(cls) -> "IngestionPipeline":
        """Construct the pipeline from default config-driven components."""
        os_client = OpenSearchClient.from_settings()
        os_client.ensure_index()
        embedder = build_embedder()
        return cls(
            loader=TxtFileLoader(),
            extractor=CanLIIMetadataExtractor(),
            chunker=HierarchicalChunker(),
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

            # Step 2 – clean body text (strip header block + page markers)
            doc.text = clean_document_text(doc.text)

            # Step 3 – chunking
            chunks = self.chunker.chunk(doc)

            # Step 4 – embed child chunks only (parents are stored as-is)
            child_chunks = [c for c in chunks if c.parent_chunk_id is not None]
            if child_chunks:
                texts = [c.text for c in child_chunks]
                embeddings = self.embedder.embed(texts)
                for chunk, emb in zip(child_chunks, embeddings):
                    chunk.embedding = emb

            # Step 5 – index everything
            self.indexer.index(chunks)

        logger.info("Ingestion complete.")
