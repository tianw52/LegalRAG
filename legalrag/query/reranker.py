"""Cross-encoder reranker.

Takes the top-K candidates from retrieval and re-scores them with a
cross-encoder model (query × passage) to produce a refined top-N list.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
Swap via RERANKER_MODEL env var or constructor argument.
"""

from __future__ import annotations

import logging
from functools import cached_property

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseReranker
from legalrag.core.models import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.retrieval.reranker_model

    @cached_property
    def _model(self):  # type: ignore[return]
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: %s", self._model_name)
        return CrossEncoder(self._model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.chunk.text) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score

        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        logger.debug(
            "Reranked %d → %d candidates (top score=%.4f)",
            len(candidates),
            top_n,
            ranked[0].rerank_score if ranked else 0,
        )
        return ranked[:top_n]
