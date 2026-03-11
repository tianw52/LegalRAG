"""LegalBench-RAG evaluation — chunk-level Precision@K & Recall@K.

Evaluation methodology
----------------------
Scoring is at the *chunk level* (binary hit/miss), evaluated at multiple
rank cutoffs K simultaneously:

    Recall@K    = fraction of GT snippets covered by ≥1 of the top-K chunks
    Precision@K = fraction of the top-K chunks that overlap ≥1 GT snippet

A GT snippet is "covered" if any of the top-K retrieved chunks from the same
file has a character span that overlaps the snippet span (non-empty intersection).
A retrieved chunk "hits" if it overlaps at least one GT snippet.

This is insensitive to exact chunk boundary positions — only overlap matters.

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
import logging
import sys
from collections import defaultdict
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
from evaluation.LegalBenchRAG.pipeline import INDEX_NAME

logger = logging.getLogger(__name__)

# ── Retrieval ─────────────────────────────────────────────────────────────────


def build_retriever(top_k: int) -> OpenSearchRetriever:
    cfg = settings.opensearch
    lb_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": INDEX_NAME,
        }
    )
    embedder = build_embedder()
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    return OpenSearchRetriever(os_client, embedder, mode="hybrid", top_k=top_k)


# ── Per-query scoring ─────────────────────────────────────────────────────────


class QueryScore(NamedTuple):
    # Dicts keyed by K value, e.g. {1: 0.5, 5: 0.8, 10: 1.0}
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    tags: list[str]


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if two half-open [start, end) spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


def score_query(
    test: BenchmarkTestCase,
    retriever: OpenSearchRetriever,
    ks: list[int],
) -> QueryScore:
    """Run retrieval for one test case and return Precision@K / Recall@K scores.

    A single retrieval pass fetches ``max(ks)`` chunks.  Metrics are then
    computed for every K by slicing the ranked list at position K.

    Recall@K    = fraction of GT snippets covered by ≥1 of the top-K chunks.
    Precision@K = fraction of the top-K chunks that overlap ≥1 GT snippet.
    """
    sq = StructuredQuery(
        raw_query=test.query,
        reformulated_query=test.query,
    )
    results = retriever.retrieve(sq)

    # Collect retrieved (file_path, char_start, char_end) in rank order
    retrieved: list[tuple[str, int, int]] = []
    for r in results:
        chunk = r.chunk
        if chunk.char_start is None or chunk.char_end is None:
            continue
        file_path = chunk.metadata.citation if chunk.metadata else None
        if not file_path:
            continue
        retrieved.append((file_path, chunk.char_start, chunk.char_end))

    n_gt = len(test.snippets)
    recall_at_k: dict[int, float] = {}
    precision_at_k: dict[int, float] = {}

    for k in ks:
        top_k = retrieved[:k]

        n_gt_covered = sum(
            1
            for snippet in test.snippets
            if any(
                fp == snippet.file_path and spans_overlap((cs, ce), snippet.span)
                for fp, cs, ce in top_k
            )
        )
        n_retrieved_relevant = sum(
            1
            for fp, cs, ce in top_k
            if any(
                fp == snippet.file_path and spans_overlap((cs, ce), snippet.span)
                for snippet in test.snippets
            )
        )

        recall_at_k[k] = n_gt_covered / n_gt if n_gt > 0 else 0.0
        precision_at_k[k] = n_retrieved_relevant / len(top_k) if top_k else 0.0

    return QueryScore(recall_at_k=recall_at_k, precision_at_k=precision_at_k, tags=test.tags)


# ── Aggregate results ─────────────────────────────────────────────────────────


def aggregate(
    scores: list[QueryScore],
    benchmark_names: list[str],
    ks: list[int],
) -> None:
    """Print a summary table: Recall@K and Precision@K per benchmark + overall."""
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

    k_header = "  ".join(f"@{k:>2}" for k in ks)
    col_w = 7  # width per K column

    def fmt_row(label: str, score_list: list[QueryScore], metric: str) -> str:
        vals = "  ".join(f"{avg_at_k(score_list, metric, k):>{col_w}.4f}" for k in ks)
        return f"  {label:<18}  {vals}  ({len(score_list)})"

    k_labels = "  ".join(f"{'K='+str(k):>{col_w}}" for k in ks)
    width = 22 + (col_w + 2) * len(ks) + 6

    print(f"\n{'─' * width}")
    print(f"  LegalBench-RAG Evaluation — chunk-level @K")
    print(f"{'─' * width}")

    for metric, label in [("recall_at_k", "Recall"), ("precision_at_k", "Precision")]:
        print(f"\n  {label}")
        print(f"  {'Benchmark':<18}  {k_labels}   N")
        print(f"  {'─'*18}  {'  '.join(['─'*col_w]*len(ks))}  {'─'*5}")
        for name in benchmark_names:
            bm_scores = per_bm.get(name, [])
            if not bm_scores:
                continue
            print(fmt_row(name, bm_scores, metric))
        print(f"  {'─'*18}  {'  '.join(['─'*col_w]*len(ks))}  {'─'*5}")
        print(fmt_row("OVERALL", scores, metric))

    print(f"\n{'─' * width}")
    print(f"  Index : {INDEX_NAME}  |  K values: {ks}")
    print()


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
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — keeps output clean).",
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

    retriever = build_retriever(top_k=top_k)

    print(
        f"\nRunning evaluation: {len(tests)} queries, "
        f"K={ks}, top_k={top_k}, index={INDEX_NAME} …"
    )

    scores: list[QueryScore] = []
    for i, test in enumerate(tests, 1):
        score = score_query(test, retriever, ks=ks)
        scores.append(score)
        if i % 50 == 0:
            print(f"  {i}/{len(tests)} queries done …")

    aggregate(scores, benchmark_names, ks=ks)


if __name__ == "__main__":
    main(sys.argv[1:])
