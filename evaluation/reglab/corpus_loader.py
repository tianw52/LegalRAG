"""Corpus loader that sets ``metadata.court`` from Housing statute paths.

Paths produced by :mod:`evaluation.reglab.prepare` look like
``statutes/Alabama/<sha256>.txt``.  The CS&Law ''25 retrieval setup restricts
Housing Statute QA to the passage pool for the query jurisdiction; OpenSearch
filters use the indexed ``court`` field (see :class:`legalrag.query.retriever.OpenSearchRetriever`).
"""

from __future__ import annotations

from evaluation.LegalBenchRAG.loader import LegalBenchRAGCorpusLoader


class RegLabCorpusLoader(LegalBenchRAGCorpusLoader):
    """Same as :class:`LegalBenchRAGCorpusLoader` but tags U.S. state on statute docs."""

    def iter(self):
        for doc in super().iter():
            rel = doc.metadata.citation or ""
            parts = rel.split("/")
            if len(parts) >= 3 and parts[0] == "statutes":
                doc.metadata.court = parts[1]
            yield doc
