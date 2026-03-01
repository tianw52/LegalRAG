"""Tests for HierarchicalChunker."""

import pytest

from legalrag.core.models import LegalDocumentMetadata, RawDocument
from legalrag.ingestion.chunker import HierarchicalChunker


@pytest.fixture()
def sample_doc() -> RawDocument:
    text = (
        "The Supreme Court of Canada held in R v. Smith that the standard of review "
        "for s. 7 Charter claims is correctness. The court reasoned that life, liberty, "
        "and security of the person are fundamental rights that cannot be compromised by "
        "administrative deference. Furthermore, the principles of fundamental justice "
        "demand a de novo assessment on questions of law. The accused in this case was "
        "acquitted on all counts."
    )
    metadata = LegalDocumentMetadata(source_path="/tmp/test.txt")
    return RawDocument(metadata=metadata, text=text)


def test_produces_parent_and_child_chunks(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    chunks = chunker.chunk(sample_doc)

    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]

    assert len(parents) >= 1, "Expected at least one parent chunk"
    assert len(children) >= 1, "Expected at least one child chunk"


def test_child_chunks_reference_valid_parent(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    chunks = chunker.chunk(sample_doc)

    parent_ids = {c.chunk_id for c in chunks if c.parent_chunk_id is None}
    for child in [c for c in chunks if c.parent_chunk_id is not None]:
        assert child.parent_chunk_id in parent_ids


def test_metadata_propagated(sample_doc: RawDocument) -> None:
    chunker = HierarchicalChunker()
    chunks = chunker.chunk(sample_doc)
    for chunk in chunks:
        if chunk.metadata is not None:
            assert chunk.metadata.source_path == "/tmp/test.txt"
