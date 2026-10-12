"""Exploration engine: question + profile → stream of evidence and views.

Flow (events are yielded as soon as each step finishes):

    formulate (LLM, optional) → hybrid retrieve (jurisdiction from profile)
      → rerank → evidence + built-in views (case landscape, case timeline)
      → in parallel: fact toggles (LLM)  |  planner (agent) → ad-hoc views

There is deliberately no free-text answer step: the outputs are structures
the user explores, each item citing retrieved evidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from legalrag.core.config import OpenSearchSettings, settings
from legalrag.core.models import RetrievedChunk, StructuredQuery
from legalrag.explore import views
from legalrag.explore.profile import UserProfile
from legalrag.explore.schemas import Evidence, ExplorationPlan, View
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient
from legalrag.prompts.loader import load_prompt
from legalrag.query.reranker import CrossEncoderReranker
from legalrag.query.retriever import OpenSearchRetriever

logger = logging.getLogger(__name__)

Scope = Literal["profile", "all"]


def build_os_client(embedding_dim: int) -> OpenSearchClient:
    cfg = settings.opensearch
    explore_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": settings.explore.index_name,
        }
    )
    return OpenSearchClient(cfg=explore_cfg, embedding_dim=embedding_dim)


class ExplorationEngine:
    def __init__(self) -> None:
        policy = load_prompt("explore_policy")
        self._retrieval_cfg = policy["retrieval"]
        self._juris_cfg = policy["jurisdiction_policy"]
        self._question_id = policy["grounding"]["question_evidence_id"]

        embedder = build_embedder(
            model_name=settings.explore.embedding_model,
            provider=settings.explore.embedding_provider,
        )
        self.os_client = build_os_client(embedder.dim)
        if not self.os_client.index_exists():
            raise RuntimeError(
                f"OpenSearch index '{self.os_client.index_name}' not found — "
                "run scripts/ingest_canlegal.py first (see EXPLORE_INDEX_NAME)."
            )
        self.os_client.ensure_index()  # index exists → only ensures the hybrid pipeline
        self._retriever = OpenSearchRetriever(
            self.os_client, embedder, mode="hybrid", top_k=self._retrieval_cfg["top_k"]
        )
        self._reranker = CrossEncoderReranker()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _jurisdictions(self, profile: UserProfile | None, scope: Scope) -> list[str] | None:
        if scope == "all" or self._juris_cfg["mode"] != "filter":
            return None
        if not profile or not profile.province:
            return None
        return [profile.province, *self._juris_cfg["always_include"]]

    async def _formulate(self, question: str) -> tuple[StructuredQuery, str | None]:
        """Return (query, warning).  Falls back to the raw question if the LLM fails."""
        if not self._retrieval_cfg["use_formulator"]:
            return StructuredQuery(raw_query=question, reformulated_query=question), None
        try:
            out = await views.formulate(question)
        except Exception as exc:
            logger.warning("Query formulation failed: %s — using the raw question", exc)
            return (
                StructuredQuery(raw_query=question, reformulated_query=question),
                f"Query rewrite unavailable ({exc.__class__.__name__}); searched with your exact wording.",
            )
        # Only the rewrite + keywords are used: the formulator's court/citation
        # filters are exact-match and easily wrong; jurisdiction comes from the profile.
        return (
            StructuredQuery(
                raw_query=question,
                reformulated_query=out.reformulated_query or question,
                lexical_keywords=out.lexical_keywords,
            ),
            None,
        )

    def _retrieve(
        self, sq: StructuredQuery, jurisdictions: list[str] | None
    ) -> tuple[list[RetrievedChunk], bool]:
        sq.jurisdiction_filter = jurisdictions
        candidates = self._retriever.retrieve(sq)
        relaxed = False
        if jurisdictions and len(candidates) < self._juris_cfg["min_results"]:
            sq.jurisdiction_filter = None
            candidates = self._retriever.retrieve(sq)
            relaxed = True
        reranked = self._reranker.rerank(
            sq.reformulated_query or sq.raw_query,
            candidates,
            top_n=self._retrieval_cfg["evidence_k"],
        )
        return reranked, relaxed

    def _evidence(self, question: str, chunks: list[RetrievedChunk]) -> list[Evidence]:
        items = [Evidence(evidence_id=self._question_id, kind="question", text=question)]
        for i, rc in enumerate(chunks, start=1):
            c, m = rc.chunk, rc.chunk.metadata
            items.append(
                Evidence(
                    evidence_id=f"e{i}",
                    text=c.text,
                    rank=i,
                    rerank_score=rc.rerank_score,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    citation=m.citation if m else None,
                    case_name=m.case_name if m else None,
                    court=m.court if m else None,
                    court_abbrev=m.court_abbrev if m else None,
                    jurisdiction=m.jurisdiction if m else None,
                    year=m.year if m else None,
                    doc_type=m.doc_type if m else None,
                    url=m.url if m else None,
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
            )
        return items

    # ── Main entry point ──────────────────────────────────────────────────────

    async def explore(
        self, question: str, profile: UserProfile | None, scope: Scope = "profile"
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def mark(name: str) -> None:
            timings[name] = round(time.perf_counter() - t0, 2)

        yield {"event": "status", "stage": "formulating", "message": "Understanding your question…"}
        sq, warning = await self._formulate(question)
        mark("formulated")
        if warning:
            yield {"event": "view_error", "view_id": "formulator", "message": warning}

        jurisdictions = self._jurisdictions(profile, scope)
        yield {"event": "status", "stage": "retrieving", "message": "Searching Canadian legal sources…"}
        chunks, relaxed = await asyncio.to_thread(self._retrieve, sq, jurisdictions)
        mark("retrieved")
        yield {
            "event": "query",
            "reformulated": sq.reformulated_query,
            "keywords": sq.lexical_keywords,
            "jurisdictions": None if relaxed else jurisdictions,
            "relaxed": relaxed,
            "scope": scope,
        }

        evidence = self._evidence(question, chunks)
        yield {"event": "evidence", "items": [e.model_dump() for e in evidence]}
        for view in (views.case_landscape(evidence), views.case_timeline(evidence)):
            yield {"event": "view", "view": view.model_dump()}

        if not chunks:
            yield {"event": "done", "timings": timings}
            return

        valid_ids = {e.evidence_id for e in evidence}
        ctx = views.prompt_context(
            question,
            profile,
            views.format_evidence(evidence[1:], self._retrieval_cfg["llm_evidence_k"]),
        )

        yield {"event": "status", "stage": "building", "message": "Building views…"}
        pending: dict[asyncio.Task, str] = {
            asyncio.create_task(views.fact_toggle(ctx, valid_ids)): "fact_toggle",
            asyncio.create_task(views.plan(ctx)): "plan",
        }
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                label = pending.pop(task)
                if task.exception() is not None:
                    logger.exception("View '%s' failed", label, exc_info=task.exception())
                    yield {"event": "view_error", "view_id": label, "message": str(task.exception())}
                    continue
                result = task.result()
                mark(label)
                if isinstance(result, ExplorationPlan):
                    yield {"event": "plan", **result.model_dump()}
                    for i, pv in enumerate(result.views):
                        t = asyncio.create_task(views.build_adhoc(pv, i, ctx, valid_ids))
                        pending[t] = f"adhoc_{i}"
                elif isinstance(result, View):
                    yield {"event": "view", "view": result.model_dump()}

        yield {"event": "done", "timings": timings}

    # ── Document reader ───────────────────────────────────────────────────────

    def document(self, doc_id: str) -> dict[str, Any] | None:
        chunks = self.os_client.get_chunks_by_doc_id(doc_id, parents=False)
        if not chunks:
            return None
        text = ""
        for c in chunks:  # stitch flat RCTS chunks by char offset, dropping the overlap
            start = c.get("char_start") or 0
            text += c["text"][len(text) - start:] if start < len(text) else c["text"]
        first = chunks[0]
        meta_keys = ("doc_id", "citation", "case_name", "court", "court_abbrev",
                     "jurisdiction", "year", "doc_type", "url")
        return {**{k: first.get(k) for k in meta_keys}, "text": text}
