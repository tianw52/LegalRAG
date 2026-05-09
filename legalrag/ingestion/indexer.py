"""Indexer: writes embedded chunks into OpenSearch.

Handles both parent chunks (stored for context expansion, no vector) and
child chunks (embedded, used for kNN retrieval).
"""

from __future__ import annotations

import logging

from legalrag.core.interfaces import BaseIndexer
from legalrag.core.models import Chunk
from legalrag.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

# Parent chunks have no embedding; they are stored text-only for context fetch.
_PARENT_CHUNK_INDEX_SUFFIX = "_parents"


class OpenSearchIndexer(BaseIndexer):
    """Upserts Chunk objects into OpenSearch using bulk API."""

    def __init__(self, client: OpenSearchClient) -> None:
        self._client = client

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        parents = [c for c in chunks if c.is_parent]
        children = [c for c in chunks if not c.is_parent]

        if parents:
            self._bulk_upsert(parents, vector=False)
        if children:
            self._bulk_upsert(children, vector=True)

        logger.info(
            "Indexed %d parent chunks and %d child chunks", len(parents), len(children)
        )

    def delete(self, doc_id: str) -> None:
        self._client.delete_by_doc_id(doc_id)
        logger.info("Deleted all chunks for doc_id=%s", doc_id)

    def _bulk_upsert(self, chunks: list[Chunk], *, vector: bool) -> None:
        actions = []
        for chunk in chunks:
            doc: dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "parent_chunk_id": chunk.parent_chunk_id,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "is_parent": chunk.is_parent,
                # Metadata fields (flattened for easy filtering)
                "source_path": chunk.metadata.source_path,
                "court": chunk.metadata.court,
                "citation": chunk.metadata.citation,
                "decision_date": (
                    chunk.metadata.decision_date.isoformat()
                    if chunk.metadata.decision_date
                    else None
                ),
            }
            if vector and chunk.embedding is not None:
                doc["embedding"] = chunk.embedding

            actions.append(
                {
                    "_index": self._client.index_name,
                    "_id": chunk.chunk_id,
                    "_source": doc,
                }
            )

        self._client.bulk(actions)
