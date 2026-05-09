"""Passage-level metrics from Zheng et al., CS&Law 2025 (Tables 21–25).

The paper reports retrieval over **passage** units.  A LegalRAG index stores
**chunks**; we convert a ranked chunk list to a ranked **unique passage** list
(using chunk ``metadata.citation`` = corpus-relative path) in first-hit order,
then compute Recall@K and MRR@10 as in standard ODQA / BEIR-style reporting.

* **Bar Exam QA**: one gold passage per question — binary Recall@K, MRR@10.
* **Housing Statute QA (upper, main paper)**: success if **at least one** gold
  statute passage appears in the top-K passages.
* **Housing (lower, Appendix H)**: success if **all** gold passages appear in
  the top-K passage list (meaningful for K ≥ number of gold labels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalrag.core.models import RetrievedChunk


def ranked_passage_citations(results: list["RetrievedChunk"]) -> list[str]:
    """Dedupe ranked chunk hits into a passage ordering (citation = corpus path)."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        m = r.chunk.metadata
        cite = m.citation if m else None
        if not cite:
            continue
        if cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


def recall_at_k_upper(passage_rank: list[str], gold_paths: set[str], k: int) -> float:
    """Recall: at least one relevant passage in the top-K passages (indicator)."""
    if k <= 0:
        return 0.0
    topk = set(passage_rank[:k])
    return 1.0 if topk & gold_paths else 0.0


def recall_at_k_lower(passage_rank: list[str], gold_paths: set[str], k: int) -> float:
    """Appendix H lower bound — all gold passages must appear in the top-K passages."""
    if k <= 0 or not gold_paths:
        return 0.0
    topk = set(passage_rank[:k])
    return 1.0 if gold_paths <= topk else 0.0


def mrr_at_cutoff(passage_rank: list[str], gold_paths: set[str], cutoff: int = 10) -> float:
    """MRR@cutoff: reciprocal rank of the first retrieved passage that matches any gold."""
    for i, p in enumerate(passage_rank[:cutoff], start=1):
        if p in gold_paths:
            return 1.0 / i
    return 0.0