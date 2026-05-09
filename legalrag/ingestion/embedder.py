"""Embedding providers.

All embedders implement BaseEmbedder so they can be swapped via config.

Available implementations
--------------------------
SentenceTransformerEmbedder  – local HuggingFace model via sentence-transformers (default)
HuggingFaceEmbedder          – local HuggingFace model via AutoTokenizer + AutoModel
                               (for models not packaged as sentence-transformers,
                               e.g. jhu-clsp/BERT-DPR-CLERC-ft)
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


class HuggingFaceEmbedder(BaseEmbedder):
    """Local embedding via AutoTokenizer + AutoModel with mean pooling.

    Use this for models that are not packaged as sentence-transformers,
    e.g. ``jhu-clsp/BERT-DPR-CLERC-ft``.

    Token embeddings from the last hidden state are mean-pooled over
    non-padding positions and then L2-normalised.
    """

    def __init__(self, model_name: str | None = None, batch_size: int = 64) -> None:
        self._model_name = model_name or settings.embedding.model
        self._batch_size = batch_size

    @cached_property
    def _tokenizer(self):  # type: ignore[return]
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer: %s", self._model_name)
        return AutoTokenizer.from_pretrained(self._model_name)

    @cached_property
    def _model(self):  # type: ignore[return]
        import torch
        from transformers import AutoModel

        logger.info("Loading model: %s", self._model_name)
        model = AutoModel.from_pretrained(self._model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    @property
    def dim(self) -> int:
        return self._model.config.hidden_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch
        import torch.nn.functional as F

        device = next(self._model.parameters()).device
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self._model(**encoded)

            # Mean pool over non-padding tokens
            token_embeddings = output.last_hidden_state  # (B, T, D)
            attention_mask = encoded["attention_mask"]   # (B, T)
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts  # (B, D)

            normalised = F.normalize(pooled, p=2, dim=1)
            all_embeddings.extend(normalised.cpu().tolist())

        return all_embeddings


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


def build_embedder(
    model_name: str | None = None,
    provider: str | None = None,
) -> BaseEmbedder:
    """Factory: instantiate the correct embedder from settings.

    Parameters
    ----------
    model_name:
        Override the model name from ``EMBEDDING_MODEL`` in ``.env``.
    provider:
        Override the provider from ``EMBEDDING_PROVIDER`` in ``.env``.
        Choices: ``sentence_transformers``, ``huggingface``, ``openai``.
    """
    resolved_provider = provider or settings.embedding.provider
    if resolved_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name)
    if resolved_provider == "huggingface":
        return HuggingFaceEmbedder(model_name=model_name)
    if resolved_provider == "openai":
        return OpenAIEmbedder(model=model_name)
    raise ValueError(f"Unknown embedding provider: {resolved_provider!r}")
