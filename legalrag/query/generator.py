"""Answer generator: prompts the LLM with retrieved context to produce a final answer.

System prompt and message templates live in:

    legalrag/prompts/generator.yaml

Edit that file to tune tone, citation style, or context formatting without
touching Python.

Context assembly
----------------
For each retrieved chunk we optionally expand to its parent chunk for richer
context (small-to-big retrieval pattern).  Parent expansion is done lazily
via the OpenSearch client to avoid loading all parents upfront.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseGenerator
from legalrag.core.models import RAGResponse, RetrievedChunk
from legalrag.opensearch.client import OpenSearchClient
from legalrag.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class LLMGenerator(BaseGenerator):
    """Generates answers using an LLM conditioned on retrieved context.

    Prompt configuration (system prompt, context template, user turn template)
    is loaded from ``legalrag/prompts/generator.yaml`` at construction time.
    """

    def __init__(
        self,
        os_client: OpenSearchClient | None = None,
        expand_to_parent: bool = True,
        max_context_chunks: int = 5,
        model: str | None = None,
    ) -> None:
        from legalrag.utils.llm_client import get_async_client, get_sync_client

        self._sync_client = get_sync_client()
        self._async_client = get_async_client()
        self._model = model or settings.llm.model
        self._os_client = os_client
        self._expand_to_parent = expand_to_parent
        self._max_context_chunks = max_context_chunks

        # Load prompt config once; cached by loader for the process lifetime
        self._prompt_cfg = load_prompt("generator")

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> RAGResponse:
        messages = self._build_messages(query, context_chunks)
        cfg = self._prompt_cfg.get("model_params", {})
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=cfg.get("temperature", 0.1),
        )
        answer = response.choices[0].message.content or ""
        logger.debug("Generated answer (%d chars)", len(answer))
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=context_chunks,
        )

    async def stream(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        messages = self._build_messages(query, context_chunks)
        cfg = self._prompt_cfg.get("model_params", {})
        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=cfg.get("temperature", 0.1),
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
        self, query: str, context_chunks: list[RetrievedChunk]
    ) -> list[dict]:
        system_prompt: str = self._prompt_cfg["system"]
        context_str = self._build_context(context_chunks)
        user_turn: str = self._prompt_cfg["user_turn_template"].format(
            question=query,
            context=context_str,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_turn},
        ]

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        selected = chunks[: self._max_context_chunks]
        item_tpl: str = self._prompt_cfg["context_item_template"]
        separator: str = self._prompt_cfg.get("context_separator", "\n\n")
        parts: list[str] = []
        for i, rc in enumerate(selected, start=1):
            text = self._get_text(rc)
            parts.append(
                item_tpl.format(
                    i=i,
                    court=rc.chunk.metadata.court if rc.chunk.metadata else "unknown",
                    citation=rc.chunk.metadata.citation if rc.chunk.metadata else "unknown",
                    text=text,
                )
            )
        return separator.join(parts)

    def _get_text(self, rc: RetrievedChunk) -> str:
        """Return parent text if expand_to_parent is enabled, else child text."""
        if (
            self._expand_to_parent
            and self._os_client is not None
            and rc.chunk.parent_chunk_id
        ):
            parent = self._os_client.get_parent(rc.chunk.parent_chunk_id)
            if parent:
                return parent.get("text", rc.chunk.text)
        return rc.chunk.text
