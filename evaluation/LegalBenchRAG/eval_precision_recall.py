"""LegalBench-RAG evaluation — character-level AND chunk-level Precision@K & Recall@K.

Evaluation methodology
----------------------
Two complementary metrics are computed simultaneously.

Character-level (fine-grained):
    For each query, the top-K retrieved chunks are unioned into a single set of
    character spans per file.  Metrics are computed against GT snippet spans:

        CharRecall@K    = |intersection(GT chars, retrieved chars)| / |GT chars|
        CharPrecision@K = |intersection(GT chars, retrieved chars)| / |retrieved chars|

    Intersection is computed span-by-span within the same file.

Chunk-level (binary hit/miss):
    Each GT snippet is "hit" if at least one top-K chunk from the same file has
    a character span that overlaps it (non-empty intersection).  Each GT snippet
    counts once regardless of how many chunks cover it.  A retrieved chunk is a
    "hit" if it overlaps at least one GT snippet.

        ChunkRecall@K    = # GT snippets hit by ≥1 top-K chunk / # GT snippets
        ChunkPrecision@K = # top-K chunks that hit ≥1 GT snippet / K

    For every hit pair (chunk, GT snippet), the overlap span and its percentage
    relative to both the chunk and the GT snippet are recorded in the trace.

Mapping retrieved chunks back to character positions
-----------------------------------------------------
Each indexed chunk has:
  - ``metadata.citation``  → the relative corpus file path (e.g. ``cuad/001.txt``)
  - ``char_start`` / ``char_end``  → character offsets within the original document

Usage
-----
# Default: evaluate at K=1,5,10,20 (all 4 sub-benchmarks)
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG

# Evaluate on the 50-query subset at K=5,10,20
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG \\
    --benchmarks-dir data/LegalBenchRAG/benchmarks_subset \\
    --ks 5 10 20

# Evaluate only cuad + maud, retrieve up to top-50
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG \\
    --benchmarks cuad maud \\
    --ks 5 10 20 50

# Quick smoke test (10 cases per benchmark, verbose)
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG \\
    --limit 10 \\
    --log-level INFO

Requirements
------------
No extra packages needed beyond the main legalrag dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from legalrag.core.config import settings
from legalrag.core.models import StructuredQuery
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.loader import (
    BenchmarkTestCase,
    load_benchmark,
)
from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME

logger = logging.getLogger(__name__)

# ── Retrieval ─────────────────────────────────────────────────────────────────


def build_retriever(
    top_k: int,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    embedding_cache: str | None = None,
    retrieval_mode: str = "hybrid",
) -> OpenSearchRetriever:
    cfg = settings.opensearch
    lb_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": index_name,
        }
    )
    embedder = build_embedder(model_name=embedding_model, provider=embedding_provider, cache_path=embedding_cache)
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    if retrieval_mode == "hybrid":
        os_client._ensure_hybrid_pipeline()
    return OpenSearchRetriever(os_client, embedder, mode=retrieval_mode, top_k=top_k)


# ── Per-query scoring ─────────────────────────────────────────────────────────


class QueryScore(NamedTuple):
    # Dicts keyed by K value, e.g. {1: 0.5, 5: 0.8, 10: 1.0}
    # char_recall_at_k:    intersection(GT, retrieved) / GT chars
    # char_precision_at_k: intersection(GT, retrieved) / retrieved chars
    char_recall_at_k: dict[int, float]
    char_precision_at_k: dict[int, float]
    # chunk_recall_at_k:    # GT snippets hit / # GT snippets
    # chunk_precision_at_k: # retrieved chunks that hit any GT / K
    chunk_recall_at_k: dict[int, float]
    chunk_precision_at_k: dict[int, float]
    tags: list[str]


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if two half-open [start, end) spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge a list of half-open [start, end) intervals into disjoint sorted spans."""
    if not spans:
        return []
    sorted_spans = sorted(spans)
    merged: list[tuple[int, int]] = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def span_intersection_chars(
    spans_a: list[tuple[int, int]],
    spans_b: list[tuple[int, int]],
) -> int:
    """Return the total number of characters in the intersection of two span lists.

    Each list is a set of half-open [start, end) intervals (within the same
    file).  The intersection is computed pairwise; overlapping pairs within
    the same list are handled by merging each list before intersecting.
    """
    ma = _merge_spans(spans_a)
    mb = _merge_spans(spans_b)

    total = 0
    j = 0
    for a_start, a_end in ma:
        while j < len(mb) and mb[j][1] <= a_start:
            j += 1
        i = j
        while i < len(mb) and mb[i][0] < a_end:
            overlap_start = max(a_start, mb[i][0])
            overlap_end = min(a_end, mb[i][1])
            total += max(0, overlap_end - overlap_start)
            i += 1
    return total


def span_total_chars(spans: list[tuple[int, int]]) -> int:
    """Return the total number of unique characters covered by a list of spans."""
    return sum(e - s for s, e in _merge_spans(spans))


def _chunk_level_score(
    gt_snippets: list[tuple[str, int, int]],
    top_k_chunks: list[tuple[str, int, int]],
    top_k_meta: list[dict],
    k: int,
) -> tuple[float, float, list[dict], list[dict]]:
    """Compute chunk-level recall and precision for one K cutoff.

    Parameters
    ----------
    gt_snippets:
        List of (file_path, char_start, char_end) for every GT snippet.
    top_k_chunks:
        Ranked list of (file_path, char_start, char_end) for retrieved chunks
        (already sliced to K).
    top_k_meta:
        Parallel list of metadata dicts for top_k_chunks (rank, score, etc.).
    k:
        The cutoff — used as the denominator for precision.

    Returns
    -------
    recall:    # unique GT snippets hit / # GT snippets
    precision: # retrieved chunks with ≥1 GT overlap / k
    chunk_hit_details:
        One dict per retrieved chunk with hit status and per-GT overlap info.
    gt_hit_details:
        One dict per GT snippet noting whether it was hit and by which chunks.
    """
    n_gt = len(gt_snippets)
    gt_hit_by: list[list[int]] = [[] for _ in range(n_gt)]  # gt_idx → [chunk ranks]

    chunk_hit_details: list[dict] = []
    for c_idx, (c_file, c_start, c_end) in enumerate(top_k_chunks):
        c_len = max(1, c_end - c_start)
        overlaps: list[dict] = []
        for g_idx, (g_file, g_start, g_end) in enumerate(gt_snippets):
            if c_file != g_file:
                continue
            if c_start >= g_end or g_start >= c_end:
                continue
            ov_start = max(c_start, g_start)
            ov_end = min(c_end, g_end)
            ov_chars = ov_end - ov_start
            g_len = max(1, g_end - g_start)
            overlaps.append({
                "gt_idx": g_idx,
                "overlap_span": [ov_start, ov_end],
                "overlap_chars": ov_chars,
                "overlap_pct_of_chunk": round(ov_chars / c_len * 100, 2),
                "overlap_pct_of_gt": round(ov_chars / g_len * 100, 2),
            })
            gt_hit_by[g_idx].append(top_k_meta[c_idx]["rank"])

        chunk_hit_details.append({
            **top_k_meta[c_idx],
            "is_chunk_hit": len(overlaps) > 0,
            "gt_overlaps": overlaps,
        })

    gt_hit_details: list[dict] = []
    n_gt_hit = 0
    for g_idx, (g_file, g_start, g_end) in enumerate(gt_snippets):
        hit = len(gt_hit_by[g_idx]) > 0
        if hit:
            n_gt_hit += 1
        gt_hit_details.append({
            "gt_idx": g_idx,
            "file": g_file,
            "span": [g_start, g_end],
            "is_hit": hit,
            "hit_by_chunk_ranks": gt_hit_by[g_idx],
        })

    n_chunk_hits = sum(1 for c in chunk_hit_details if c["is_chunk_hit"])
    recall = n_gt_hit / n_gt if n_gt > 0 else 0.0
    precision = n_chunk_hits / k if k > 0 else 0.0
    return recall, precision, chunk_hit_details, gt_hit_details


def score_query(
    test: BenchmarkTestCase,
    retriever: OpenSearchRetriever,
    ks: list[int],
    trace_fh=None,
    query_idx: int = 0,
) -> QueryScore:
    """Run retrieval for one test case and return both character- and chunk-level metrics.

    A single retrieval pass fetches ``max(ks)`` chunks.  Metrics are computed
    for every K by slicing the ranked list at position K.

    Character-level:
        CharRecall@K    = intersection(GT chars, retrieved chars) / GT chars
        CharPrecision@K = intersection(GT chars, retrieved chars) / retrieved chars

    Chunk-level:
        ChunkRecall@K    = # GT snippets hit by ≥1 top-K chunk / # GT snippets
        ChunkPrecision@K = # top-K chunks that overlap ≥1 GT snippet / K

    If ``trace_fh`` is provided, one JSON line is appended per query with the
    full retrieval trace: query text, GT snippets, all retrieved chunks (ranked),
    and per-K metrics for both methodologies.  Chunk-level hits include the
    overlap span and percentage relative to both the chunk and the GT snippet.
    """
    sq = StructuredQuery(
        raw_query=test.query,
        reformulated_query=test.query,
    )
    results = retriever.retrieve(sq)

    # Collect retrieved (file_path, char_start, char_end) in rank order
    retrieved: list[tuple[str, int, int]] = []
    retrieved_meta: list[dict] = []
    for rank, r in enumerate(results, start=1):
        chunk = r.chunk
        if chunk.char_start is None or chunk.char_end is None:
            continue
        file_path = chunk.metadata.citation if chunk.metadata else None
        if not file_path:
            continue
        retrieved.append((file_path, chunk.char_start, chunk.char_end))
        retrieved_meta.append({
            "rank": rank,
            "file": file_path,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "char_len": chunk.char_end - chunk.char_start,
            "score": r.semantic_score,
            "chunk_id": chunk.chunk_id,
        })

    # GT spans grouped by file (for char-level)
    gt_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for snippet in test.snippets:
        gt_by_file[snippet.file_path].append(snippet.span)
    total_gt_chars = sum(span_total_chars(spans) for spans in gt_by_file.values())

    # GT as a flat list of (file, start, end) for chunk-level
    gt_flat: list[tuple[str, int, int]] = [
        (s.file_path, s.span[0], s.span[1]) for s in test.snippets
    ]

    char_recall_at_k: dict[int, float] = {}
    char_precision_at_k: dict[int, float] = {}
    chunk_recall_at_k: dict[int, float] = {}
    chunk_precision_at_k: dict[int, float] = {}

    k_details: list[dict] = []

    for k in ks:
        top_k = retrieved[:k]
        top_k_meta = retrieved_meta[:k]

        # ── Character-level ────────────────────────────────────────────────────
        ret_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fp, cs, ce in top_k:
            ret_by_file[fp].append((cs, ce))
        total_ret_chars = sum(span_total_chars(spans) for spans in ret_by_file.values())

        intersection_chars = sum(
            span_intersection_chars(gt_by_file[fp], ret_by_file[fp])
            for fp in set(gt_by_file) & set(ret_by_file)
        )

        c_recall = intersection_chars / total_gt_chars if total_gt_chars > 0 else 0.0
        c_precision = intersection_chars / total_ret_chars if total_ret_chars > 0 else 0.0
        char_recall_at_k[k] = c_recall
        char_precision_at_k[k] = c_precision

        # ── Chunk-level ────────────────────────────────────────────────────────
        ck_recall, ck_precision, chunk_hits, gt_hits = _chunk_level_score(
            gt_flat, top_k, top_k_meta, k
        )
        chunk_recall_at_k[k] = ck_recall
        chunk_precision_at_k[k] = ck_precision

        if trace_fh is not None:
            k_details.append({
                "k": k,
                # character-level
                "char_recall": round(c_recall, 6),
                "char_precision": round(c_precision, 6),
                "intersection_chars": intersection_chars,
                "gt_chars": total_gt_chars,
                "retrieved_chars": total_ret_chars,
                # chunk-level
                "chunk_recall": round(ck_recall, 6),
                "chunk_precision": round(ck_precision, 6),
                "n_gt_snippets": len(gt_flat),
                "n_gt_hit": sum(1 for g in gt_hits if g["is_hit"]),
                "n_chunk_hits": sum(1 for c in chunk_hits if c["is_chunk_hit"]),
                # detailed hit info
                "chunk_hits": chunk_hits,
                "gt_snippet_hits": gt_hits,
            })

    if trace_fh is not None:
        record = {
            "query_idx": query_idx,
            "query": test.query,
            "tags": test.tags,
            "ground_truth": [
                {"file": s.file_path, "span": list(s.span)}
                for s in test.snippets
            ],
            "total_gt_chars": total_gt_chars,
            "n_retrieved": len(retrieved),
            "retrieved_all": retrieved_meta,
            "metrics_by_k": k_details,
        }
        trace_fh.write(json.dumps(record) + "\n")
        trace_fh.flush()

    return QueryScore(
        char_recall_at_k=char_recall_at_k,
        char_precision_at_k=char_precision_at_k,
        chunk_recall_at_k=chunk_recall_at_k,
        chunk_precision_at_k=chunk_precision_at_k,
        tags=test.tags,
    )


# ── Aggregate results ─────────────────────────────────────────────────────────


def aggregate(
    scores: list[QueryScore],
    benchmark_names: list[str],
    ks: list[int],
    index_name: str = DEFAULT_INDEX_NAME,
) -> None:
    """Print summary tables for both character-level and chunk-level metrics."""
    per_bm: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        for tag in score.tags:
            if tag in benchmark_names:
                per_bm[tag].append(score)

    def avg_at_k(score_list: list[QueryScore], metric: str, k: int) -> float:
        if not score_list:
            return 0.0
        vals = [getattr(s, metric)[k] for s in score_list]
        return sum(vals) / len(vals)

    col_w = 7  # width per K column

    def fmt_row(label: str, score_list: list[QueryScore], metric: str) -> str:
        vals = "  ".join(f"{avg_at_k(score_list, metric, k):>{col_w}.4f}" for k in ks)
        return f"  {label:<18}  {vals}  ({len(score_list)})"

    def fmt_overall_row(label: str, bm_avgs: list[list[float]]) -> str:
        """Average of per-benchmark averages (equal weight per benchmark)."""
        n = len(bm_avgs)
        vals = "  ".join(
            f"{(sum(col[i] for col in bm_avgs) / n):>{col_w}.4f}"
            for i in range(len(ks))
        )
        total_queries = sum(len(per_bm.get(name, [])) for name in benchmark_names)
        return f"  {label:<18}  {vals}  ({total_queries})"

    k_labels = "  ".join(f"{'K='+str(k):>{col_w}}" for k in ks)
    width = 22 + (col_w + 2) * len(ks) + 6

    def _print_section(title: str, legend_lines: list[str], metrics: list[tuple[str, str]]) -> None:
        print(f"\n{'─' * width}")
        print(f"  {title}")
        for line in legend_lines:
            print(f"  {line}")
        print(f"  OVERALL = macro-average of per-benchmark averages (equal benchmark weight)")
        print(f"{'─' * width}")
        for metric, label in metrics:
            print(f"\n  {label}")
            print(f"  {'Benchmark':<18}  {k_labels}   N")
            print(f"  {'─'*18}  {'  '.join(['─'*col_w]*len(ks))}  {'─'*5}")
            bm_avgs: list[list[float]] = []
            for name in benchmark_names:
                bm_scores = per_bm.get(name, [])
                if not bm_scores:
                    continue
                print(fmt_row(name, bm_scores, metric))
                bm_avgs.append([avg_at_k(bm_scores, metric, k) for k in ks])
            print(f"  {'─'*18}  {'  '.join(['─'*col_w]*len(ks))}  {'─'*5}")
            if bm_avgs:
                print(fmt_overall_row("OVERALL", bm_avgs))

    _print_section(
        "LegalBench-RAG — character-level @K",
        [
            "CharRecall@K    = intersection(GT, retrieved) / GT chars",
            "CharPrecision@K = intersection(GT, retrieved) / retrieved chars",
        ],
        [
            ("char_recall_at_k", "CharRecall"),
            ("char_precision_at_k", "CharPrecision"),
        ],
    )

    _print_section(
        "LegalBench-RAG — chunk-level @K",
        [
            "ChunkRecall@K    = # GT snippets hit by ≥1 top-K chunk / # GT snippets",
            "ChunkPrecision@K = # top-K chunks overlapping ≥1 GT snippet / K",
            "  A chunk 'hits' a GT snippet if their character spans overlap (same file).",
            "  Each GT snippet counts once regardless of how many chunks cover it.",
        ],
        [
            ("chunk_recall_at_k", "ChunkRecall"),
            ("chunk_precision_at_k", "ChunkPrecision"),
        ],
    )

    print(f"\n{'─' * width}")
    print(f"  Index : {index_name}  |  K values: {ks}")
    print()


# ── Structured results dict (for JSON output / plotting) ─────────────────────


def compute_aggregate_dict(
    scores: list[QueryScore],
    benchmark_names: list[str],
    ks: list[int],
    index_name: str,
    label: str,
    embedding_model: str | None = None,
) -> dict:
    """Return a JSON-serialisable dict of all per-benchmark and overall averages.

    Used by ``main()`` to write the results file consumed by ``plot_results.py``.

    Schema::

        {
          "run_info": {
            "label": str,          # display name (for plot legends)
            "embedding_model": str,
            "index_name": str,
            "ks": [int, ...],
            "timestamp": "ISO-8601"
          },
          "benchmarks": {
            "<name>": {
              "n_queries": int,
              "char_recall_at_k":    {"2": 0.12, "20": 0.31, ...},
              "char_precision_at_k": {...},
              "chunk_recall_at_k":   {...},
              "chunk_precision_at_k": {...}
            },
            ...
          },
          "overall": {
            "char_recall_at_k":    {...},
            "char_precision_at_k": {...},
            "chunk_recall_at_k":   {...},
            "chunk_precision_at_k": {...}
          }
        }
    """
    per_bm: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        for tag in score.tags:
            if tag in benchmark_names:
                per_bm[tag].append(score)

    def avg(score_list: list[QueryScore], metric: str, k: int) -> float:
        if not score_list:
            return 0.0
        return sum(getattr(s, metric)[k] for s in score_list) / len(score_list)

    metrics = [
        "char_recall_at_k",
        "char_precision_at_k",
        "chunk_recall_at_k",
        "chunk_precision_at_k",
    ]

    benchmarks_out: dict = {}
    for name in benchmark_names:
        bm_scores = per_bm.get(name, [])
        if not bm_scores:
            continue
        benchmarks_out[name] = {
            "n_queries": len(bm_scores),
            **{
                m: {str(k): round(avg(bm_scores, m, k), 6) for k in ks}
                for m in metrics
            },
        }

    bm_vals = list(benchmarks_out.values())
    n_bm = len(bm_vals)
    overall: dict = {}
    if n_bm > 0:
        for m in metrics:
            overall[m] = {
                str(k): round(sum(b[m][str(k)] for b in bm_vals) / n_bm, 6)
                for k in ks
            }

    return {
        "run_info": {
            "label": label,
            "embedding_model": embedding_model or "",
            "index_name": index_name,
            "ks": ks,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "benchmarks": benchmarks_out,
        "overall": overall,
    }


# ── Tee stdout → trace file ───────────────────────────────────────────────────


class _Tee:
    """Write to both the real stdout and a secondary file handle."""

    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        return self._secondary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def fileno(self) -> int:
        return self._primary.fileno()

    # Delegate everything else (isatty, etc.) to the primary stream
    def __getattr__(self, name):
        return getattr(self._primary, name)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.LegalBenchRAG.eval_precision_recall",
        description=(
            "Evaluate LegalRAG retrieval on LegalBench-RAG using "
            "chunk-level Precision@K & Recall@K."
        ),
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        metavar="PATH",
        help=(
            "Root of the downloaded LegalBench-RAG data directory. "
            "Must contain corpus/ and benchmarks/ sub-folders."
        ),
    )
    parser.add_argument(
        "--benchmarks-dir",
        default=None,
        metavar="PATH",
        help=(
            "Override the benchmarks directory (default: <data-dir>/benchmarks). "
            "Useful for evaluating a subset, e.g. data/LegalBenchRAG/benchmarks_subset."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Sub-benchmarks to evaluate. "
            "Choices: contractnli cuad maud privacy_qa. "
            "Defaults to all four."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of test cases loaded per benchmark. "
            "Useful for fast iteration."
        ),
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        metavar="K",
        help="Rank cutoffs to evaluate (default: 1 5 10 20). Retrieves max(ks) chunks.",
    )
    parser.add_argument(
        "--index-name",
        required=True,
        metavar="NAME",
        help="OpenSearch index to query. Must match the index used during ingestion.",
    )
    parser.add_argument(
        "--embedding-provider",
        required=True,
        choices=["sentence_transformers", "huggingface", "openai"],
        metavar="PROVIDER",
        help=(
            "Embedding provider: sentence_transformers, huggingface, or openai. "
            "Must match the provider used during ingestion."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        required=True,
        metavar="MODEL",
        help="Embedding model name. Must match the model used during ingestion.",
    )
    parser.add_argument(
        "--trace-file",
        required=True,
        metavar="PATH",
        help=(
            "Write a per-query retrieval trace to this file in JSONL format. "
            "Each line is a JSON object with: query, ground_truth spans, all "
            "retrieved chunks (ranked with scores), and per-K metrics with "
            "gt_overlap flags. Useful for debugging retrieval failures."
        ),
    )
    parser.add_argument(
        "--embedding-cache",
        default="data/cache/embeddings",
        metavar="PATH",
        help=(
            "Path to a diskcache directory for caching query embeddings across runs "
            "(default: data/cache/embeddings). Only active when --embedding-provider openai "
            "is used. Pass --no-embedding-cache to disable."
        ),
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Disable the embedding cache even for API-based providers.",
    )
    parser.add_argument(
        "--label",
        default=None,
        metavar="TEXT",
        help=(
            "Display label for this run in plot legends (default: embedding model name). "
            "Example: 'SBERT all-mpnet-base-v2'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — keeps output clean).",
    )
    parser.add_argument(
        "--retrieval-mode",
        default="hybrid",
        choices=["hybrid", "semantic", "lexical"],
        help="Retrieval mode: hybrid (KNN+BM25 via RRF), semantic (KNN only), or lexical (BM25 only). Default: hybrid.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    ks = sorted(set(args.ks))
    top_k = max(ks)

    data_dir = args.data_dir.rstrip("/")
    benchmarks_dir = args.benchmarks_dir.rstrip("/") if args.benchmarks_dir else f"{data_dir}/benchmarks"
    benchmark_names = args.benchmarks or ["contractnli", "cuad", "maud", "privacy_qa"]

    tests = load_benchmark(
        benchmarks_dir,
        names=benchmark_names,
        limit_per_benchmark=args.limit,
    )
    if not tests:
        print("No test cases found. Check --data-dir and --benchmarks.", file=sys.stderr)
        sys.exit(1)

    use_cache = (
        not args.no_embedding_cache
        and args.embedding_provider == "openai"
    )
    retriever = build_retriever(
        top_k=top_k,
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        embedding_cache=args.embedding_cache if use_cache else None,
        retrieval_mode=args.retrieval_mode,
    )

    trace_path = Path(args.trace_file)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    scores: list[QueryScore] = []
    trace_fh = open(trace_path, "w", encoding="utf-8")
    _real_stdout = sys.stdout
    sys.stdout = _Tee(_real_stdout, trace_fh)
    try:
        print(
            f"\nRunning evaluation: {len(tests)} queries, "
            f"K={ks}, top_k={top_k}, index={args.index_name} …"
        )
        for i, test in enumerate(tests, 1):
            score = score_query(test, retriever, ks=ks, trace_fh=trace_fh, query_idx=i)
            scores.append(score)
            if i % 50 == 0:
                print(f"  {i}/{len(tests)} queries done …")
        aggregate(scores, benchmark_names, ks=ks, index_name=args.index_name)

        embedding_model_name = args.embedding_model or settings.embedding.model
        summary = compute_aggregate_dict(
            scores,
            benchmark_names,
            ks=ks,
            index_name=args.index_name,
            label=args.label or embedding_model_name,
            embedding_model=embedding_model_name,
        )
        summary["_type"] = "run_summary"
        trace_fh.write(json.dumps(summary) + "\n")
        trace_fh.flush()
    finally:
        sys.stdout = _real_stdout
        trace_fh.close()
        print(f"  Trace written → {trace_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
