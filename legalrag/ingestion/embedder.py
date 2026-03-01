"""Embedding providers.

All embedders implement BaseEmbedder so they can be swapped via config.

Available implementations
--------------------------
SentenceTransformerEmbedder  – local HuggingFace model (default)
OpenAIEmbedder               – OpenAI / compatible API (e.g. text-embedding-3-*)

Add new providers by subclassing BaseEmbedder.
"""

from __future__ import annotations

import logging
from functools import cached_property

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding via sentence-transformers (runs on CPU or GPU)."""

    def __init__(self, model_name: str | None = None, batch_size: int = 64) -> None:
        self._model_name = model_name or settings.embedding.model
        self._batch_size = batch_size

    @cached_property
    def _model(self):  # type: ignore[return]
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self._model_name)
        return SentenceTransformer(self._model_name)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()  # type: ignore[return-value]


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI-compatible embedding API (works with text-embedding-3-* or vLLM)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 512,
    ) -> None:
        self._model = model or settings.embedding.model
        self._api_key = api_key or settings.llm.api_key
        self._base_url = base_url or settings.llm.base_url
        self._batch_size = batch_size
        self._dim: int | None = None

    @cached_property
    def _client(self):  # type: ignore[return]
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe with a single token
            probe = self.embed(["probe"])
            self._dim = len(probe[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            results.extend([item.embedding for item in response.data])
        return results


def build_embedder() -> BaseEmbedder:
    """Factory: instantiate the correct embedder from settings."""
    provider = settings.embedding.provider
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder()
    if provider == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider!r}")
