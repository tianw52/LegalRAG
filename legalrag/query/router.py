"""
Query router – decides between the fast path (Reranker) and the slow path
(Deep Search / RLM-style recursive retrieval).

Routing logic
-------------
The router scores candidate quality from retrieval.  If the top candidates
are confident enough (scores above a threshold), we take the fast path.
Otherwise, or if the query contains signals of complexity, we fall back to
Deep Search.

Deep Search – placeholder (RLM-inspired, arxiv 2512.24601)
------------------------------------------------------------
The paper "Recursive Language Models" (Zhang et al., 2025) proposes treating
the full document corpus as an external environment and letting the LLM
programmatically decompose, peek, and recursively call itself over document
snippets.  The deep_search module will implement a simplified version of this:

  1. Iterative multi-query expansion: the LLM generates follow-up queries
     based on the current candidate set.
  2. Each follow-up query is independently retrieved and merged via RRF.
  3. The process repeats until a confidence threshold or a max-iteration limit.

This is marked TODO – the interface is established so integration is a
drop-in once the module is built.

Reference: https://arxiv.org/html/2512.24601v2
"""

from __future__ import annotations

import logging

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseReranker, BaseRetriever, BaseRouter
from legalrag.core.models import RetrievedChunk, StructuredQuery
from legalrag.query.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Routing thresholds (tune empirically)
_CONFIDENCE_THRESHOLD = 0.6   # min rerank score to take fast path
_FAST_PATH_CANDIDATES = 3     # min number of confident candidates for fast path


class ThresholdRouter(BaseRouter):
    """
    Routes to 'reranker' (fast) or 'deep_search' (slow) based on retrieval
    confidence scores.
    """

    def __init__(
        self,
        reranker: BaseReranker | None = None,
        deep_search_retriever: BaseRetriever | None = None,
        rerank_top_n: int | None = None,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        fast_path_min_candidates: int = _FAST_PATH_CANDIDATES,
    ) -> None:
        self._reranker: BaseReranker = reranker or CrossEncoderReranker()
        self._deep_search_retriever = deep_search_retriever  # None = not yet implemented
        self._rerank_top_n = rerank_top_n or settings.retrieval.rerank_top_k
        self._confidence_threshold = confidence_threshold
        self._fast_path_min_candidates = fast_path_min_candidates

    def route(
        self, query: StructuredQuery, candidates: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk]]:
        # Always rerank first – we need scores to decide
        reranked = self._reranker.rerank(
            query.reformulated_query or query.raw_query,
            candidates,
            top_n=len(candidates),
        )

        confident = [
            c for c in reranked if (c.rerank_score or 0.0) >= self._confidence_threshold
        ]

        if len(confident) >= self._fast_path_min_candidates:
            logger.info("Router → fast path (reranker). Confident candidates: %d", len(confident))
            return "reranker", reranked[: self._rerank_top_n]

        logger.info(
            "Router → deep_search path (low confidence: %d/%d candidates above threshold).",
            len(confident),
            len(reranked),
        )
        return "deep_search", self._deep_search(query, reranked)

    def _deep_search(
        self, query: StructuredQuery, initial_candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Placeholder for RLM-inspired deep search.

        TODO: implement iterative multi-query expansion + recursive retrieval.
        See module docstring and arxiv.org/abs/2512.24601 for design reference.

        Current behaviour: falls back to top-N of the initial reranked candidates.
        """
        logger.warning(
            "Deep search is not yet implemented – falling back to reranked candidates."
        )
        return initial_candidates[: self._rerank_top_n]
