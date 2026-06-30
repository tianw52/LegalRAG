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
from legalrag.query.formulator import LLMQueryFormulator, PassthroughQueryFormulator
from legalrag.query.generator import LLMGenerator, LocalHFGenerator
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

    @classmethod
    def local(
        cls,
        *,
        index_name: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        retrieval_mode: str = "hybrid",
        local_model_id: str = LocalHFGenerator.DEFAULT_MODEL,
        max_context_chunks: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        expand_to_parent: bool = True,
        hf_offline: bool = False,
    ) -> "QueryPipeline":
        """Construct the pipeline using a local HuggingFace LLM for generation.

        This factory is designed for cluster use (no API server needed).
        The embedding model is configured via the usual .env / env vars and
        can be overridden via the keyword arguments below.

        Parameters
        ----------
        index_name:
            OpenSearch index to query.  If ``None``, uses the value from
            ``OPENSEARCH_INDEX_NAME`` in the environment / .env file.
        embedding_provider:
            ``"sentence_transformers"`` or ``"huggingface"``.  Defaults to
            ``EMBEDDING_PROVIDER`` env var.
        embedding_model:
            HF model ID for the retrieval embedder.  Defaults to
            ``EMBEDDING_MODEL`` env var.
        retrieval_mode:
            ``"hybrid"`` (default), ``"semantic"``, or ``"lexical"``.
        local_model_id:
            HuggingFace model ID for the local generator.  Defaults to
            ``Qwen/Qwen2.5-7B-Instruct``.
        max_context_chunks:
            Number of retrieved chunks to include in the generation prompt.
        max_new_tokens:
            Token budget for the generated answer.
        temperature:
            Sampling temperature (0.0 = greedy).
        expand_to_parent:
            Expand child chunks to parent for richer context.
        hf_offline:
            Load both embedder and generator from local HF cache only.
        """
        import os

        if hf_offline:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from legalrag.core.config import settings as _settings

        if index_name:
            os.environ["OPENSEARCH_INDEX_NAME"] = index_name
        if embedding_provider:
            os.environ["EMBEDDING_PROVIDER"] = embedding_provider
        if embedding_model:
            os.environ["EMBEDDING_MODEL"] = embedding_model

        os_client = OpenSearchClient.from_settings()
        embedder = build_embedder()
        generator = LocalHFGenerator(
            model_id=local_model_id,
            os_client=os_client if expand_to_parent else None,
            expand_to_parent=expand_to_parent,
            max_context_chunks=max_context_chunks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            hf_offline=hf_offline,
        )
        return cls(
            formulator=PassthroughQueryFormulator(),
            retriever=OpenSearchRetriever(os_client, embedder, mode=retrieval_mode),
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
