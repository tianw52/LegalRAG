"""LegalBench-RAG evaluation — chunk-level Precision & Recall.

Evaluation methodology
----------------------
Scoring is at the *chunk level* (binary hit/miss per snippet), which avoids
penalising systems for the exact byte position of chunk boundaries:

    Recall    = fraction of GT snippets covered by ≥1 retrieved chunk
    Precision = fraction of retrieved chunks that overlap ≥1 GT snippet

A GT snippet is "covered" if any retrieved chunk from the same file has a
character span that overlaps the snippet's span (i.e. the intersection is
non-empty).  A retrieved chunk "hits" if it overlaps at least one GT snippet.

Mapping retrieved chunks back to character positions
-----------------------------------------------------
Each indexed chunk has:
  - ``metadata.citation``  → the relative corpus file path (e.g. ``cuad/001.txt``)
  - ``char_start`` / ``char_end``  → character offsets within the original document

Usage
-----
# Evaluate with default settings (all 4 sub-benchmarks, top_k=20)
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG

# Evaluate only cuad + maud, top_k=50
python -m evaluation.LegalBenchRAG.eval_precision_recall \\
    --data-dir data/LegalBenchRAG \\
    --benchmarks cuad maud \\
    --top-k 50

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
    recall: float
    precision: float
    tags: list[str]


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if two half-open [start, end) spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


def score_query(
    test: BenchmarkTestCase,
    retriever: OpenSearchRetriever,
) -> QueryScore:
    """Run retrieval for one test case and return chunk-level recall/precision.

    Recall    = fraction of GT snippets covered by ≥1 retrieved chunk.
    Precision = fraction of retrieved chunks that overlap ≥1 GT snippet.

    A GT snippet is "covered" when any retrieved chunk from the same file has a
    character span that overlaps the snippet span.  This is insensitive to the
    exact position of chunk boundaries.
    """
    sq = StructuredQuery(
        raw_query=test.query,
        reformulated_query=test.query,
    )
    results = retriever.retrieve(sq)

    # Collect retrieved (file_path, char_start, char_end) triples
    retrieved: list[tuple[str, int, int]] = []
    for r in results:
        chunk = r.chunk
        if chunk.char_start is None or chunk.char_end is None:
            continue
        file_path = chunk.metadata.citation if chunk.metadata else None
        if not file_path:
            continue
        retrieved.append((file_path, chunk.char_start, chunk.char_end))

    # Recall: for each GT snippet, check if any retrieved chunk overlaps it
    n_gt = len(test.snippets)
    n_gt_covered = sum(
        1
        for snippet in test.snippets
        if any(
            fp == snippet.file_path and spans_overlap((cs, ce), snippet.span)
            for fp, cs, ce in retrieved
        )
    )

    # Precision: for each retrieved chunk, check if it overlaps any GT snippet
    n_retrieved = len(retrieved)
    n_retrieved_relevant = sum(
        1
        for fp, cs, ce in retrieved
        if any(
            fp == snippet.file_path and spans_overlap((cs, ce), snippet.span)
            for snippet in test.snippets
        )
    )

    recall = n_gt_covered / n_gt if n_gt > 0 else 0.0
    precision = n_retrieved_relevant / n_retrieved if n_retrieved > 0 else 0.0

    return QueryScore(recall=recall, precision=precision, tags=test.tags)


# ── Aggregate results ─────────────────────────────────────────────────────────


def aggregate(
    scores: list[QueryScore],
    benchmark_names: list[str],
) -> None:
    """Print a summary table: per-benchmark + overall average."""
    overall_recall = sum(s.recall for s in scores) / len(scores) if scores else 0.0
    overall_precision = sum(s.precision for s in scores) / len(scores) if scores else 0.0

    per_bm: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        for tag in score.tags:
            if tag in benchmark_names:
                per_bm[tag].append(score)

    width = 44
    print(f"\n{'─' * width}")
    print(f"  LegalBench-RAG Evaluation — chunk-level")
    print(f"{'─' * width}")
    print(f"  {'Benchmark':<18} {'Recall':>8}  {'Precision':>9}  {'N':>5}")
    print(f"  {'─'*18} {'─'*8}  {'─'*9}  {'─'*5}")
    for name in benchmark_names:
        bm_scores = per_bm.get(name, [])
        if not bm_scores:
            continue
        bm_recall = sum(s.recall for s in bm_scores) / len(bm_scores)
        bm_prec = sum(s.precision for s in bm_scores) / len(bm_scores)
        print(
            f"  {name:<18} {bm_recall:>8.4f}  {bm_prec:>9.4f}  {len(bm_scores):>5}"
        )
    print(f"  {'─'*18} {'─'*8}  {'─'*9}  {'─'*5}")
    print(
        f"  {'OVERALL':<18} {overall_recall:>8.4f}  {overall_precision:>9.4f}  {len(scores):>5}"
    )
    print(f"{'─' * width}")
    print(f"  Index : {INDEX_NAME}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.LegalBenchRAG.eval_precision_recall",
        description=(
            "Evaluate LegalRAG retrieval on LegalBench-RAG using "
            "chunk-level Precision & Recall."
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
        "--top-k",
        type=int,
        default=20,
        metavar="K",
        help="Number of chunks to retrieve per query (default: 20).",
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

    retriever = build_retriever(top_k=args.top_k)

    print(
        f"\nRunning evaluation: {len(tests)} queries, "
        f"top_k={args.top_k}, index={INDEX_NAME} …"
    )

    scores: list[QueryScore] = []
    for i, test in enumerate(tests, 1):
        score = score_query(test, retriever)
        scores.append(score)
        if i % 50 == 0:
            print(f"  {i}/{len(tests)} queries done …")

    aggregate(scores, benchmark_names)


if __name__ == "__main__":
    main(sys.argv[1:])
