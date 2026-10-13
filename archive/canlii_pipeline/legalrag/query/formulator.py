"""Query formulation: transform a raw user question into a StructuredQuery.

Uses a pydantic-ai Agent backed by the configured OpenAI-compatible LLM.
The system prompt and output-field descriptions live in:

    legalrag/prompts/formulator.yaml

Edit that file to tune the LLM's behaviour without touching Python.
"""

from __future__ import annotations

import logging
from datetime import date

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from legalrag.core.config import settings
from legalrag.core.interfaces import BaseQueryFormulator
from legalrag.core.models import StructuredQuery
from legalrag.prompts.loader import load_prompt

logger = logging.getLogger(__name__)


# ── Pydantic output schema for the agent ─────────────────────────────────────

class _FormulatorOutput(BaseModel):
    """Structured output the LLM must return."""

    reformulated_query: str = Field(
        description="Concise, precise version of the question optimised for semantic search."
    )
    lexical_keywords: list[str] = Field(
        default_factory=list,
        description="Legal terms and keywords for BM25 retrieval.",
    )
    court_filter: str | None = Field(
        default=None,
        description="Exact court name filter; null when not specified.",
    )
    citation_filter: str | None = Field(
        default=None,
        description="Neutral citation filter (e.g. '2010 BCCA 220'); null when not specified.",
    )
    date_from: str | None = Field(
        default=None,
        description="Start of date range in YYYY-MM-DD format; null when not specified.",
    )
    date_to: str | None = Field(
        default=None,
        description="End of date range in YYYY-MM-DD format; null when not specified.",
    )


# ── Agent factory (lazy, module-level singleton) ──────────────────────────────

def _build_agent() -> Agent[None, _FormulatorOutput]:
    """Construct the pydantic-ai Agent from settings + YAML prompt config."""
    cfg = load_prompt("formulator")
    api_key = settings.llm.api_key or _resolve_hf_token()
    provider = OpenAIProvider(
        base_url=str(settings.llm.base_url),
        api_key=api_key,
    )
    model = OpenAIModel(settings.llm.model, provider=provider)
    return Agent(
        model=model,
        output_type=_FormulatorOutput,
        system_prompt=cfg["system"],
    )


def _resolve_hf_token() -> str:
    """Fall back to HF_TOKEN env var when LLM_API_KEY is not set."""
    import os
    return os.environ.get("HF_TOKEN", "none")


# Singleton – created once on first use, not at import time.
_agent: Agent[None, _FormulatorOutput] | None = None


def _get_agent() -> Agent[None, _FormulatorOutput]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


# ── Formulator implementations ────────────────────────────────────────────────

class LLMQueryFormulator(BaseQueryFormulator):
    """Formulates a StructuredQuery using a pydantic-ai Agent.

    The agent prompt is loaded from ``legalrag/prompts/formulator.yaml``.
    Model and API connection settings come from ``legalrag/core/config.py``
    (driven by the ``.env`` file).
    """

    def formulate(self, raw_query: str) -> StructuredQuery:
        logger.debug("Formulating query: %r", raw_query)
        try:
            result = _get_agent().run_sync(raw_query)
            out: _FormulatorOutput = result.output
        except Exception as exc:
            logger.warning("Query formulation LLM call failed: %s – using fallback.", exc)
            return StructuredQuery(raw_query=raw_query, reformulated_query=raw_query)

        return StructuredQuery(
            raw_query=raw_query,
            reformulated_query=out.reformulated_query,
            lexical_keywords=out.lexical_keywords,
            court_filter=out.court_filter,
            citation_filter=out.citation_filter,
            date_from=_parse_date(out.date_from),
            date_to=_parse_date(out.date_to),
        )


class PassthroughQueryFormulator(BaseQueryFormulator):
    """No-op formulator: returns the raw query unchanged (useful for testing)."""

    def formulate(self, raw_query: str) -> StructuredQuery:
        return StructuredQuery(raw_query=raw_query, reformulated_query=raw_query)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return date.fromisoformat(val)
    except ValueError:
        return None
