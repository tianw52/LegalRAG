"""Exploration engine: question + profile → stream of evidence and views.

Flow (events are yielded as soon as each step finishes):

    formulate (LLM, optional) → retrieve (dense kNN by default; jurisdiction from profile)
      → rerank → evidence + built-in views (case landscape, case timeline)
      → in parallel: fact toggles (LLM)  |  planner (agent) → ad-hoc views

There is deliberately no free-text answer step: the outputs are structures
the user explores, each item citing retrieved evidence.

Every run is traced (legalrag/explore/trace.py): step timings, the query
rewrite, retrieval candidates and rerank scores, and every LLM call (prompts,
raw responses, thinking, tokens).  Traces are served by GET /api/traces/{id}
and written to logs/explore_traces/.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Literal

from legalrag.core.config import settings
from legalrag.core.models import RetrievedChunk, StructuredQuery
from legalrag.explore import views
from legalrag.explore.profile import UserProfile, filter_province
from legalrag.explore.schemas import Evidence, ExplorationPlan, View
from legalrag.explore.trace import Trace, TraceStore
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient
from legalrag.prompts.loader import load_prompt
from legalrag.query.reranker import build_reranker
from legalrag.query.retriever import OpenSearchRetriever

logger = logging.getLogger(__name__)

Scope = Literal["profile", "all"]
_LOG_TOP = 10  # retrieval candidates listed in the INFO log


class ExplorationEngine:
    def __init__(self) -> None:
        policy = load_prompt("explore_policy")
        self._retrieval_cfg = policy["retrieval"]
        self._juris_cfg = policy["jurisdiction_policy"]
        self._question_id = policy["grounding"]["question_evidence_id"]
        self._extra_view_timeout = policy["planner"]["extra_view_timeout_s"]

        embedder = build_embedder(
            model_name=settings.explore.embedding_model,
            provider=settings.explore.embedding_provider,
        )
        self.os_client = OpenSearchClient.from_settings(
            embedding_dim=embedder.dim, index_name=settings.explore.index_name
        )
        if not self.os_client.index_exists():
            raise RuntimeError(
                f"OpenSearch index '{self.os_client.index_name}' not found — "
                "run scripts/ingest_canlegal.py first (see EXPLORE_INDEX_NAME)."
            )
        self.os_client.ensure_index()  # index exists → only ensures the hybrid pipeline
        self._retriever = OpenSearchRetriever(
            self.os_client,
            embedder,
            mode=self._retrieval_cfg["mode"],
            top_k=self._retrieval_cfg["top_k"],
        )
        rr_cfg = policy["reranker"]
        self._reranker = build_reranker(
            settings.explore.reranker_model,
            instruction=rr_cfg["instruction"],
            batch_size=rr_cfg["batch_size"],
        )
        self.traces = TraceStore()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _jurisdictions(self, profile: UserProfile | None, scope: Scope) -> list[str] | None:
        if scope == "all" or self._juris_cfg["mode"] != "filter":
            return None
        province = filter_province(profile.province) if profile else None
        if not province:
            return None
        return [province, *self._juris_cfg["always_include"]]

    async def _formulate(self, question: str, trace: Trace) -> tuple[StructuredQuery, str | None]:
        """Return (query, warning).  Falls back to the raw question if the LLM fails."""
        with trace.step("formulate", enabled=self._retrieval_cfg["use_formulator"]) as rec:
            if not self._retrieval_cfg["use_formulator"]:
                return StructuredQuery(raw_query=question, reformulated_query=question), None
            try:
                out = await views.formulate(question, trace)
            except Exception as exc:
                logger.warning("Query formulation failed: %s — using the raw question", exc)
                rec["fallback"] = f"{exc.__class__.__name__}: {exc}"
                return (
                    StructuredQuery(raw_query=question, reformulated_query=question),
                    f"Query rewrite unavailable ({exc.__class__.__name__}); searched with your exact wording.",
                )
            # Only the rewrite + keywords are used: the formulator's court/citation
            # filters are exact-match and easily wrong; jurisdiction comes from the profile.
            rec.update(reformulated=out.reformulated_query, keywords=out.lexical_keywords)
            return (
                StructuredQuery(
                    raw_query=question,
                    reformulated_query=out.reformulated_query or question,
                    lexical_keywords=out.lexical_keywords,
                ),
                None,
            )

    def _search(
        self, sq: StructuredQuery, jurisdictions: list[str] | None, trace: Trace
    ) -> tuple[list[RetrievedChunk], bool]:
        """Search (mode from explore_policy.yaml); relax the jurisdiction filter if too few hits."""
        with trace.step(
            "retrieve",
            mode=self._retrieval_cfg["mode"],
            query_text=sq.reformulated_query or sq.raw_query,
            top_k=self._retrieval_cfg["top_k"],
            jurisdictions=jurisdictions,
        ) as rec:
            sq.jurisdiction_filter = jurisdictions
            candidates = self._retriever.retrieve(sq)
            relaxed = False
            if jurisdictions and len(candidates) < self._juris_cfg["min_results"]:
                rec["filtered_hits"] = len(candidates)
                sq.jurisdiction_filter = None
                candidates = self._retriever.retrieve(sq)
                relaxed = True
            rec["relaxed"] = relaxed
            trace.log_rows(
                f"retrieved {len(candidates)} ({self._retrieval_cfg['mode']}), top {_LOG_TOP}:",
                [
                    f"#{i:<3} {rc.semantic_score if rc.semantic_score is not None else rc.lexical_score or 0:.4f}  "
                    f"{(rc.chunk.metadata.citation or '')[:28]:<28} {rc.chunk.metadata.jurisdiction or '':<8} "
                    f"{' '.join(rc.chunk.text.split())[:90]}"
                    for i, rc in enumerate(candidates[:_LOG_TOP], start=1)
                ],
            )
            rec["candidates"] = [
                {
                    "retrieval_rank": i,
                    "retrieval_score": rc.semantic_score if rc.semantic_score is not None else rc.lexical_score,
                    "chunk_id": rc.chunk.chunk_id,
                    "citation": rc.chunk.metadata.citation,
                    "jurisdiction": rc.chunk.metadata.jurisdiction,
                    "char_start": rc.chunk.char_start,
                    "text_preview": rc.chunk.text[:240],
                }
                for i, rc in enumerate(candidates, start=1)
            ]
            return candidates, relaxed

    def _rerank(
        self, sq: StructuredQuery, candidates: list[RetrievedChunk], trace: Trace
    ) -> list[RetrievedChunk]:
        """Score every candidate; keep the top evidence_k."""
        query = sq.reformulated_query or sq.raw_query
        keep = self._retrieval_cfg["evidence_k"]
        with trace.step(
            "rerank", model=settings.explore.reranker_model, query=query, keep=keep
        ) as rec:
            retrieval_rank = {rc.chunk.chunk_id: i for i, rc in enumerate(candidates, start=1)}
            ranked = self._reranker.rerank(query, candidates, top_n=len(candidates))
            rec["scores"] = [
                {
                    "rerank_rank": i,
                    "rerank_score": rc.rerank_score,
                    "retrieval_rank": retrieval_rank[rc.chunk.chunk_id],
                    "citation": rc.chunk.metadata.citation,
                    "kept": i <= keep,
                }
                for i, rc in enumerate(ranked, start=1)
            ]
            trace.log_rows(
                f"kept {min(keep, len(ranked))} after rerank (rerank score | retrieval rank):",
                [
                    f"e{i:<3} {rc.rerank_score or 0:.4f} | #{retrieval_rank[rc.chunk.chunk_id]:<3} "
                    f"{(rc.chunk.metadata.citation or '')[:28]:<28} {rc.chunk.metadata.jurisdiction or ''}"
                    for i, rc in enumerate(ranked[:keep], start=1)
                ],
            )
            return ranked[:keep]

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
        trace = Trace(question, profile.model_dump() if profile else None, scope)
        self.traces.add(trace)
        status = "cancelled"  # if the client disconnects before we finish
        try:
            yield {"event": "trace", "trace_id": trace.trace_id}
            async for event in self._explore(question, profile, scope, trace):
                yield event
            status = "done"
        except Exception:
            status = "error"
            raise
        finally:
            trace.finish(status)

    async def _explore(
        self, question: str, profile: UserProfile | None, scope: Scope, trace: Trace
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"event": "status", "stage": "formulating", "message": "Understanding your question…"}
        sq, warning = await self._formulate(question, trace)
        if warning:
            yield {"event": "view_error", "view_id": "formulator", "message": warning}

        jurisdictions = self._jurisdictions(profile, scope)
        yield {"event": "status", "stage": "retrieving", "message": "Searching Canadian legal sources…"}
        candidates, relaxed = await asyncio.to_thread(self._search, sq, jurisdictions, trace)
        chunks = await asyncio.to_thread(self._rerank, sq, candidates, trace)
        yield {
            "event": "query",
            "reformulated": sq.reformulated_query,
            "keywords": sq.lexical_keywords,
            "jurisdictions": None if relaxed else jurisdictions,
            "relaxed": relaxed,
            "scope": scope,
            "retrieved": len(candidates),
        }

        evidence = self._evidence(question, chunks)
        yield {"event": "evidence", "items": [e.model_dump() for e in evidence]}
        with trace.step("builtin_views"):
            builtins = [views.case_landscape(evidence), views.case_timeline(evidence)]
        for view in builtins:
            yield {"event": "view", "view": view.model_dump()}

        if not chunks:
            yield {"event": "done", "trace_id": trace.trace_id}
            return

        valid_ids = {e.evidence_id for e in evidence}
        ctx = views.prompt_context(
            question,
            profile,
            views.format_evidence(evidence[1:], self._retrieval_cfg["llm_evidence_k"]),
        )

        yield {"event": "status", "stage": "building", "message": "Building views…"}
        pending: dict[asyncio.Task, str] = {
            self._task(trace, "fact_toggle", views.fact_toggle(ctx, valid_ids, trace)): "fact_toggle",
            self._task(trace, "plan", views.plan(ctx, trace)): "plan",
        }
        try:
            while pending:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    label = pending.pop(task)
                    exc = task.exception()
                    if exc is not None:
                        logger.error("View '%s' failed: %r", label, exc)
                        message = (
                            f"timed out after {self._extra_view_timeout}s"
                            if isinstance(exc, TimeoutError)
                            else str(exc)
                        )
                        yield {"event": "view_error", "view_id": label, "message": message}
                        continue
                    result = task.result()
                    if isinstance(result, ExplorationPlan):
                        yield {"event": "plan", **result.model_dump()}
                        for i, pv in enumerate(result.views):
                            coro = asyncio.wait_for(
                                views.build_adhoc(pv, i, ctx, valid_ids, trace),
                                timeout=self._extra_view_timeout,
                            )
                            pending[self._task(trace, f"adhoc_{i}:{pv.kind}", coro)] = f"adhoc_{i}"
                    elif isinstance(result, View):
                        yield {"event": "view", "view": result.model_dump()}
        finally:
            # Client disconnected (or a new exploration replaced this one):
            # don't leave LLM calls running for a response nobody will read.
            for task in pending:
                task.cancel()

        yield {"event": "done", "trace_id": trace.trace_id}

    @staticmethod
    def _task(trace: Trace, name: str, coro: Awaitable[Any]) -> asyncio.Task:
        """Run a view coroutine as a task inside its own trace step."""

        async def run() -> Any:
            with trace.step(name) as rec:
                result = await coro
                if isinstance(result, View):
                    rec["audit"] = result.audit
                elif isinstance(result, ExplorationPlan):
                    rec["topic"] = result.topic
                    rec["chosen_views"] = [f"{v.kind} ({v.title})" for v in result.views]
                    rec["extra_views_reason"] = result.extra_views_reason
                return result

        return asyncio.create_task(run())

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
