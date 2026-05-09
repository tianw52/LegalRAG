"""Tests for HierarchicalChunker and RecursiveCharacterTextSplitter."""

import pytest

from legalrag.core.models import LegalDocumentMetadata, RawDocument
from legalrag.ingestion.chunker import HierarchicalChunker, RecursiveCharacterTextSplitter


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


def test_chunk_ids_are_stable_across_runs(sample_doc: RawDocument) -> None:
    """Re-chunking the same document must produce identical chunk_ids."""
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    run1 = {c.chunk_id for c in chunker.chunk(sample_doc)}
    run2 = {c.chunk_id for c in chunker.chunk(sample_doc)}
    assert run1 == run2, "chunk_ids changed between runs – duplicates would be created"


def test_doc_id_is_stable_for_same_source_path() -> None:
    """Two metadata objects with the same source_path must have the same doc_id."""
    from legalrag.core.models import LegalDocumentMetadata
    m1 = LegalDocumentMetadata(source_path="/data/case.txt")
    m2 = LegalDocumentMetadata(source_path="/data/case.txt")
    assert m1.doc_id == m2.doc_id


def test_doc_id_differs_for_different_source_paths() -> None:
    from legalrag.core.models import LegalDocumentMetadata
    m1 = LegalDocumentMetadata(source_path="/data/case_a.txt")
    m2 = LegalDocumentMetadata(source_path="/data/case_b.txt")
    assert m1.doc_id != m2.doc_id


def test_doc_id_from_citation_is_path_independent() -> None:
    """doc_id_from_citation must return the same ID regardless of file location."""
    from legalrag.core.models import doc_id_from_citation
    id1 = doc_id_from_citation("2010 BCCA 220", "/old/path/file.txt")
    id2 = doc_id_from_citation("2010 BCCA 220", "/new/renamed/folder/file.txt")
    assert id1 == id2, "doc_id must not change when the file is moved or renamed"


def test_doc_id_from_citation_falls_back_to_stem() -> None:
    """Without a citation the ID must be derived from the filename stem, not full path."""
    from legalrag.core.models import doc_id_from_citation
    id1 = doc_id_from_citation(None, "/old/dir/2010 BCCA 220 Smith v Jones.txt")
    id2 = doc_id_from_citation(None, "/new/dir/2010 BCCA 220 Smith v Jones.txt")
    assert id1 == id2, "stem-based fallback ID must be stable across directory renames"


def test_doc_id_citation_differs_from_stem_fallback() -> None:
    """Citation-based and stem-based IDs for the 'same' document must be distinct."""
    from legalrag.core.models import doc_id_from_citation
    citation_id = doc_id_from_citation("2010 BCCA 220", "/data/case.txt")
    stem_id = doc_id_from_citation(None, "/data/case.txt")
    assert citation_id != stem_id


# ── HierarchicalChunker: char span correctness ────────────────────────────────


def test_hierarchical_chunk_text_matches_offsets(sample_doc: RawDocument) -> None:
    """chunk.text must equal document.text[char_start:char_end] for every chunk."""
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    for chunk in chunker.chunk(sample_doc):
        expected = sample_doc.text[chunk.char_start:chunk.char_end]
        assert chunk.text == expected, (
            f"Offset mismatch for chunk_id={chunk.chunk_id}: "
            f"text[{chunk.char_start}:{chunk.char_end}]={expected!r} "
            f"!= chunk.text={chunk.text!r}"
        )


def test_hierarchical_parents_cover_full_document(sample_doc: RawDocument) -> None:
    """Parent chunks must tile the document with no gaps or overlaps."""
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    parents = sorted(
        [c for c in chunker.chunk(sample_doc) if c.is_parent],
        key=lambda c: c.char_start,
    )
    assert parents, "Expected at least one parent chunk"
    assert parents[0].char_start == 0, "First parent must start at 0"
    assert parents[-1].char_end == len(sample_doc.text), "Last parent must end at len(text)"
    for a, b in zip(parents, parents[1:]):
        assert a.char_end == b.char_start, (
            f"Gap or overlap between parents: [{a.char_start}:{a.char_end}] -> [{b.char_start}:{b.char_end}]"
        )


def test_hierarchical_child_offsets_within_parent(sample_doc: RawDocument) -> None:
    """Every child's char span must fall within its parent's char span."""
    chunker = HierarchicalChunker(parent_size=300, child_size=100, child_overlap=20)
    chunks = chunker.chunk(sample_doc)
    parent_map = {c.chunk_id: c for c in chunks if c.is_parent}
    for child in [c for c in chunks if not c.is_parent]:
        parent = parent_map[child.parent_chunk_id]
        assert child.char_start >= parent.char_start and child.char_end <= parent.char_end, (
            f"Child [{child.char_start}:{child.char_end}] outside "
            f"parent [{parent.char_start}:{parent.char_end}]"
        )


# ── RecursiveCharacterTextSplitter ────────────────────────────────────────────


@pytest.fixture()
def long_doc() -> RawDocument:
    """A document long enough to produce multiple chunks at small chunk_size."""
    text = (
        "The accused was charged under s. 267 of the Criminal Code.\n\n"
        "The trial judge found that the Crown had not proven intent beyond a reasonable doubt.\n\n"
        "On appeal, the majority held that the standard of review for findings of credibility "
        "is palpable and overriding error. The dissent would have ordered a new trial.\n\n"
        "The Supreme Court dismissed the appeal in a unanimous decision. "
        "Costs were awarded to the respondent."
    )
    metadata = LegalDocumentMetadata(source_path="/tmp/recursive_test.txt")
    return RawDocument(metadata=metadata, text=text)


def test_recursive_produces_flat_chunks(long_doc: RawDocument) -> None:
    """All chunks must be flat (no parent/child hierarchy)."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(long_doc)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.parent_chunk_id is None, "RecursiveCharacterTextSplitter must produce flat chunks"


def test_recursive_chunks_respect_size_limit(long_doc: RawDocument) -> None:
    """Each chunk must not exceed chunk_size characters."""
    chunk_size = 100
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=10)
    chunks = chunker.chunk(long_doc)

    for chunk in chunks:
        assert len(chunk.text) <= chunk_size, (
            f"Chunk exceeds size limit: {len(chunk.text)} > {chunk_size}"
        )


def test_recursive_covers_full_text(long_doc: RawDocument) -> None:
    """The union of all chunk spans must cover the full document text."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(long_doc)
    text = long_doc.text

    covered = bytearray(len(text))
    for chunk in chunks:
        for i in range(chunk.char_start, chunk.char_end):
            covered[i] = 1

    assert all(covered), "Some characters are not covered by any chunk"


def test_recursive_chunk_text_matches_offsets(long_doc: RawDocument) -> None:
    """chunk.text must equal document.text[char_start:char_end]."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    for chunk in chunker.chunk(long_doc):
        expected = long_doc.text[chunk.char_start:chunk.char_end]
        assert chunk.text == expected, "Chunk text does not match source offsets"


def test_recursive_chunk_ids_are_stable(long_doc: RawDocument) -> None:
    """Re-chunking the same document must produce identical chunk_ids."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    ids1 = {c.chunk_id for c in chunker.chunk(long_doc)}
    ids2 = {c.chunk_id for c in chunker.chunk(long_doc)}
    assert ids1 == ids2


def test_recursive_metadata_propagated(long_doc: RawDocument) -> None:
    chunker = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=15)
    for chunk in chunker.chunk(long_doc):
        assert chunk.metadata.source_path == "/tmp/recursive_test.txt"


def test_recursive_overlap_creates_shared_context(long_doc: RawDocument) -> None:
    """Consecutive chunks must share at least some characters (overlap > 0)."""
    chunker = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk(long_doc)

    if len(chunks) < 2:
        pytest.skip("Document too short to test overlap")

    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end, (
            f"Expected overlap between consecutive chunks: "
            f"chunk[i] ends at {a.char_end}, chunk[i+1] starts at {b.char_start}"
        )


def test_recursive_short_doc_yields_single_chunk() -> None:
    """A document shorter than chunk_size must produce exactly one chunk."""
    text = "Short legal note."
    metadata = LegalDocumentMetadata(source_path="/tmp/short.txt")
    doc = RawDocument(metadata=metadata, text=text)

    chunker = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(doc)

    assert len(chunks) == 1
    assert chunks[0].text == text
