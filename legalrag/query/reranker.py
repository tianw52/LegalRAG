"""Rerankers: re-score retrieval candidates (query × passage) and keep the top-N.

Implementations
---------------
CrossEncoderReranker
    sentence-transformers CrossEncoder, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
    (default for the CanLII pipeline via RERANKER_MODEL).  Scores are raw logits.
Qwen3Reranker
    Qwen/Qwen3-Reranker-{0.6B,4B,8B}: a causal LM asked whether the document
    meets the query; score = P("yes") in [0, 1].  Instruction-aware.

Use ``build_reranker(model_name)`` to get the right class for a model name.
"""

from __future__ import annotations

import logging
from functools import cached_property

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseReranker
from legalrag.core.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _apply_scores(
    candidates: list[RetrievedChunk], scores: list[float], top_n: int
) -> list[RetrievedChunk]:
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = float(score)
    ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
    logger.debug(
        "Reranked %d → %d candidates (top score=%.4f)",
        len(candidates),
        top_n,
        ranked[0].rerank_score if ranked else 0,
    )
    return ranked[:top_n]


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
        return _apply_scores(candidates, self._model.predict(pairs).tolist(), top_n)


# Prompt format from the Qwen3-Reranker model card (do not change the wrapper:
# the model was trained on it; customise behaviour through `instruction`).
_QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3Reranker(BaseReranker):
    """Qwen3-Reranker via transformers; score = P(yes | instruction, query, document)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> None:
        self._model_name = model_name
        self._instruction = instruction or _QWEN3_DEFAULT_INSTRUCTION
        self._batch_size = batch_size
        self._max_length = max_length

    @cached_property
    def _bundle(self):  # (tokenizer, model, device, yes_id, no_id, prefix_ids, suffix_ids)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32
        logger.info("Loading reranker model: %s on %s", self._model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(self._model_name, torch_dtype=dtype).to(device).eval()
        return (
            tokenizer,
            model,
            device,
            tokenizer.convert_tokens_to_ids("yes"),
            tokenizer.convert_tokens_to_ids("no"),
            tokenizer.encode(_QWEN3_PREFIX, add_special_tokens=False),
            tokenizer.encode(_QWEN3_SUFFIX, add_special_tokens=False),
        )

    def _score_batch(self, texts: list[str]) -> list[float]:
        import torch

        tokenizer, model, device, yes_id, no_id, prefix, suffix = self._bundle
        enc = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._max_length - len(prefix) - len(suffix),
        )
        enc["input_ids"] = [prefix + ids + suffix for ids in enc["input_ids"]]
        batch = tokenizer.pad(enc, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            last = model(**batch).logits[:, -1, :]
        pair = torch.stack([last[:, no_id], last[:, yes_id]], dim=1).float()
        return torch.nn.functional.log_softmax(pair, dim=1)[:, 1].exp().tolist()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        texts = [
            f"<Instruct>: {self._instruction}\n<Query>: {query}\n<Document>: {c.chunk.text}"
            for c in candidates
        ]
        scores: list[float] = []
        for i in range(0, len(texts), self._batch_size):
            scores += self._score_batch(texts[i : i + self._batch_size])
        return _apply_scores(candidates, scores, top_n)


def build_reranker(model_name: str | None = None, **kwargs) -> BaseReranker:
    """Return the reranker implementation that matches *model_name*.

    ``Qwen/Qwen3-Reranker-*`` → Qwen3Reranker (kwargs: instruction, batch_size,
    max_length); anything else → CrossEncoderReranker.
    """
    name = model_name or settings.retrieval.reranker_model
    if "qwen3-reranker" in name.lower():
        return Qwen3Reranker(name, **kwargs)
    return CrossEncoderReranker(name)
