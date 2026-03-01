"""Query formulation: transform a raw user question into a StructuredQuery.

Uses an LLM (Qwen via OpenAI-compatible API) with a Pydantic output schema
to extract:
  - a cleaned, reformulated query for dense retrieval
  - lexical keywords for BM25
  - optional metadata filters (court, citation, date range)

This is the first stage of the online pipeline and has a significant effect
on retrieval quality.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import TYPE_CHECKING

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseQueryFormulator
from legalrag.core.models import StructuredQuery

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a legal research assistant. Your task is to analyse a user's legal question
and return a JSON object with the following fields:

{
  "reformulated_query": "<concise, precise version of the question optimised for semantic search>",
  "lexical_keywords": ["<keyword1>", "<keyword2>", ...],
  "court_filter": "<exact court name or null>",
  "citation_filter": "<exact citation string or null>",
  "date_from": "<YYYY-MM-DD or null>",
  "date_to": "<YYYY-MM-DD or null>"
}

Return ONLY the JSON object, no explanation.
"""


class LLMQueryFormulator(BaseQueryFormulator):
    """Formulates a StructuredQuery using an LLM call."""

    def __init__(self, client: "OpenAI | None" = None, model: str | None = None) -> None:
        from legalrag.utils.llm_client import get_sync_client

        self._client = client or get_sync_client()
        self._model = model or settings.llm.model

    def formulate(self, raw_query: str) -> StructuredQuery:
        logger.debug("Formulating query: %r", raw_query)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_json = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_json)
        except Exception as exc:
            logger.warning("Query formulation LLM call failed: %s – using fallback.", exc)
            parsed = {}

        return StructuredQuery(
            raw_query=raw_query,
            reformulated_query=parsed.get("reformulated_query", raw_query),
            lexical_keywords=parsed.get("lexical_keywords", []),
            court_filter=parsed.get("court_filter"),
            citation_filter=parsed.get("citation_filter"),
            date_from=_parse_date(parsed.get("date_from")),
            date_to=_parse_date(parsed.get("date_to")),
        )


class PassthroughQueryFormulator(BaseQueryFormulator):
    """No-op formulator: returns the raw query unchanged (useful for testing)."""

    def formulate(self, raw_query: str) -> StructuredQuery:
        return StructuredQuery(raw_query=raw_query, reformulated_query=raw_query)


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except ValueError:
        return None
