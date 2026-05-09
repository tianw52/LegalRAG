"""
Online query pipeline orchestrator.

Wires together:
  QueryFormulator → Retriever → Router → [Reranker | DeepSearch] → Generator

Usage
-----
    pipeline = QueryPipeline.default()
    response = pipeline.run("What is the standard of review for Charter s.7 claims?")
    print(response.answer)
"""

from __future__ import annotations

import logging

from legalrag.core.interfaces import (
    BaseGenerator,
    BaseQueryFormulator,
    BaseRetriever,
    BaseRouter,
)
from legalrag.core.models import RAGResponse
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient
from legalrag.query.formulator import LLMQueryFormulator
from legalrag.query.generator import LLMGenerator
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.query.router import ThresholdRouter

logger = logging.getLogger(__name__)

_DIVIDER = "─" * 72


class QueryPipeline:
    """Orchestrates the full online query-and-answer flow."""

    def __init__(
        self,
        formulator: BaseQueryFormulator,
        retriever: BaseRetriever,
        router: BaseRouter,
        generator: BaseGenerator,
    ) -> None:
        self.formulator = formulator
        self.retriever = retriever
        self.router = router
        self.generator = generator

    @classmethod
    def default(cls) -> "QueryPipeline":
        """Construct the pipeline from default config-driven components."""
        os_client = OpenSearchClient.from_settings()
        embedder = build_embedder()
        generator = LLMGenerator(os_client=os_client, expand_to_parent=True)
        return cls(
            formulator=LLMQueryFormulator(),
            retriever=OpenSearchRetriever(os_client, embedder, mode="hybrid"),
            router=ThresholdRouter(),
            generator=generator,
        )

    def run(self, raw_query: str) -> RAGResponse:
        """Execute the full pipeline for *raw_query* and return a RAGResponse."""
        logger.info(_DIVIDER)
        logger.info("RAW QUERY   : %s", raw_query)

        # Step 1 – query formulation
        structured = self.formulator.formulate(raw_query)
        logger.info("REFORMULATED: %s", structured.reformulated_query)
        if structured.lexical_keywords:
            logger.info("KEYWORDS    : %s", ", ".join(structured.lexical_keywords))
        if structured.court_filter:
            logger.info("FILTER court: %s", structured.court_filter)
        if structured.citation_filter:
            logger.info("FILTER cite : %s", structured.citation_filter)
        if structured.date_from or structured.date_to:
            logger.info("FILTER dates: %s → %s", structured.date_from, structured.date_to)

        # Step 2 – retrieval
        candidates = self.retriever.retrieve(structured)
        logger.info("RETRIEVED   : %d candidates", len(candidates))
        for i, rc in enumerate(candidates, 1):
            m = rc.chunk.metadata
            logger.info(
                "  [%02d] %-30s | %-45s | sem=%.4f lex=%.4f",
                i,
                (m.citation or "no-citation")[:30],
                (m.court or "unknown")[:45],
                rc.semantic_score or 0.0,
                rc.lexical_score or 0.0,
            )

        # Step 3 – routing (includes reranking on fast path)
        path, final_chunks = self.router.route(structured, candidates)
        logger.info("ROUTER PATH : %s → %d chunks kept", path, len(final_chunks))
        logger.info("RERANKED RESULTS:")
        for i, rc in enumerate(final_chunks, 1):
            m = rc.chunk.metadata
            logger.info(
                "  [%02d] %-30s | %-45s | rerank=%.4f",
                i,
                (m.citation or "no-citation")[:30],
                (m.court or "unknown")[:45],
                rc.rerank_score or 0.0,
            )

        # Step 4 – generation
        response = self.generator.generate(structured.raw_query, final_chunks)
        response.router_path = path
        logger.info("ANSWER      :\n%s", response.answer)
        logger.info(_DIVIDER)
        return response
