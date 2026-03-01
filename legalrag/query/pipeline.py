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
        logger.info("Query: %r", raw_query)

        # Step 1 – query formulation
        structured = self.formulator.formulate(raw_query)
        logger.debug("Structured query: %s", structured.model_dump())

        # Step 2 – retrieval
        candidates = self.retriever.retrieve(structured)
        logger.info("Retrieved %d candidates.", len(candidates))

        # Step 3 – routing (includes reranking on fast path)
        path, final_chunks = self.router.route(structured, candidates)
        logger.info("Router path: %s → %d chunks", path, len(final_chunks))

        # Step 4 – generation
        response = self.generator.generate(structured.raw_query, final_chunks)
        response.router_path = path
        return response
