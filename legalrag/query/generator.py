"""Answer generator: prompts the LLM with retrieved context to produce a final answer.

Uses the OpenAI-compatible API so it works with Qwen (via vLLM / DashScope),
GPT-4o, Claude (via proxy), or any compatible endpoint.

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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert legal research assistant.

Answer the user's question using ONLY the provided legal document excerpts.
If the answer cannot be determined from the provided context, say so clearly.
Cite the relevant document (court and citation) when making specific legal claims.
Be precise, professional, and concise.
"""

_CONTEXT_TEMPLATE = """\
[Excerpt {i} | Court: {court} | Citation: {citation}]
{text}
"""


class LLMGenerator(BaseGenerator):
    """Generates answers using an LLM conditioned on retrieved context."""

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

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> RAGResponse:
        context_str = self._build_context(context_chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_str}",
            },
        ]
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
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
        context_str = self._build_context(context_chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_str}",
            },
        ]
        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        selected = chunks[: self._max_context_chunks]
        parts: list[str] = []
        for i, rc in enumerate(selected, start=1):
            text = self._get_text(rc)
            parts.append(
                _CONTEXT_TEMPLATE.format(
                    i=i,
                    court=rc.chunk.metadata.court or "unknown",
                    citation=rc.chunk.metadata.citation or "unknown",
                    text=text,
                )
            )
        return "\n\n".join(parts)

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
