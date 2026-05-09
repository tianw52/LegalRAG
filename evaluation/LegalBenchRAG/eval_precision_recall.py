"""LegalBench-RAG evaluation — character-level Precision@K & Recall@K.

Evaluation methodology
----------------------
Scoring is at the *character level*, evaluated at multiple rank cutoffs K
simultaneously.  For each query, the top-K retrieved chunks are unioned into
a single set of character spans per file.  Metrics are then computed against
the ground-truth (GT) snippet spans:

    CharRecall@K    = |intersection(GT chars, retrieved chars)| / |GT chars|
    CharPrecision@K = |intersection(GT chars, retrieved chars)| / |retrieved chars|

"Intersection" is computed span-by-span within the same file: only characters
that appear in both the GT spans and at least one retrieved chunk span are
counted.  Characters from different files never mix.

# ── Commented-out chunk-level methodology ────────────────────────────────────
# (kept for reference; replaced by character-level metrics above)
#
# Scoring was at the *chunk level* (binary hit/miss), evaluated at multiple
# rank cutoffs K simultaneously:
#
#     Recall@K    = fraction of GT snippets covered by ≥1 of the top-K chunks
#     Precision@K = fraction of the top-K chunks that overlap ≥1 GT snippet
#
# A GT snippet was "covered" if any top-K chunk from the same file had a
# character span that overlapped the snippet span (non-empty intersection).
# A retrieved chunk "hit" if it overlapped at least one GT snippet.
# This was insensitive to exact chunk boundary positions — only overlap mattered.
# ─────────────────────────────────────────────────────────────────────────────

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
    embedder = build_embedder(model_name=embedding_model, provider=embedding_provider)
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    # Ensure the hybrid search pipeline exists (idempotent — safe to call every time)
    os_client._ensure_hybrid_pipeline()
    return OpenSearchRetriever(os_client, embedder, mode="hybrid", top_k=top_k)


# ── Per-query scoring ─────────────────────────────────────────────────────────


class QueryScore(NamedTuple):
    # Dicts keyed by K value, e.g. {1: 0.5, 5: 0.8, 10: 1.0}
    # char_recall_at_k:    intersection(GT, retrieved) / GT chars
    # char_precision_at_k: intersection(GT, retrieved) / retrieved chars
    char_recall_at_k: dict[int, float]
    char_precision_at_k: dict[int, float]
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


def score_query(
    test: BenchmarkTestCase,
    retriever: OpenSearchRetriever,
    ks: list[int],
    trace_fh=None,
    query_idx: int = 0,
) -> QueryScore:
    """Run retrieval for one test case and return character-level Precision@K / Recall@K.

    A single retrieval pass fetches ``max(ks)`` chunks.  Metrics are then
    computed for every K by slicing the ranked list at position K.

    CharRecall@K    = intersection(GT chars, retrieved chars) / GT chars
    CharPrecision@K = intersection(GT chars, retrieved chars) / retrieved chars

    Intersection is computed per-file: only characters in both GT spans and
    retrieved spans from the same file are counted.

    If ``trace_fh`` is provided, one JSON line is appended per query with the
    full retrieval trace: query text, GT snippets, all retrieved chunks (ranked),
    per-K metrics, and which retrieved chunks hit/missed at each K.
    """
    sq = StructuredQuery(
        raw_query=test.query,
        reformulated_query=test.query,
    )
    results = retriever.retrieve(sq)

    # Collect retrieved (file_path, char_start, char_end, score) in rank order
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

    # GT spans grouped by file
    gt_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for snippet in test.snippets:
        gt_by_file[snippet.file_path].append(snippet.span)
    total_gt_chars = sum(span_total_chars(spans) for spans in gt_by_file.values())

    char_recall_at_k: dict[int, float] = {}
    char_precision_at_k: dict[int, float] = {}

    # Per-K trace data
    k_details: list[dict] = []

    for k in ks:
        top_k = retrieved[:k]
        top_k_meta = retrieved_meta[:k]

        # Retrieved spans grouped by file
        ret_by_file: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for fp, cs, ce in top_k:
            ret_by_file[fp].append((cs, ce))
        total_ret_chars = sum(span_total_chars(spans) for spans in ret_by_file.values())

        # Intersection: per-file overlap between GT and retrieved spans
        intersection_chars = sum(
            span_intersection_chars(gt_by_file[fp], ret_by_file[fp])
            for fp in set(gt_by_file) & set(ret_by_file)
        )

        recall = intersection_chars / total_gt_chars if total_gt_chars > 0 else 0.0
        precision = intersection_chars / total_ret_chars if total_ret_chars > 0 else 0.0
        char_recall_at_k[k] = recall
        char_precision_at_k[k] = precision

        if trace_fh is not None:
            # Mark which chunks at this K overlap any GT snippet
            hits = []
            for m in top_k_meta:
                fp = m["file"]
                span = (m["char_start"], m["char_end"])
                gt_spans = gt_by_file.get(fp, [])
                overlaps = any(spans_overlap(span, g) for g in gt_spans)
                hits.append({**m, "gt_overlap": overlaps})
            k_details.append({
                "k": k,
                "char_recall": round(recall, 6),
                "char_precision": round(precision, 6),
                "intersection_chars": intersection_chars,
                "gt_chars": total_gt_chars,
                "retrieved_chars": total_ret_chars,
                "top_k_chunks": hits,
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
        tags=test.tags,
    )


def compute_overall_macro(
    scores: list[QueryScore],
    benchmark_names: list[str],
    ks: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Macro-average (equal weight per benchmark) of mean char recall / precision at each K."""
    per_bm: dict[str, list[QueryScore]] = defaultdict(list)
    for score in scores:
        for tag in score.tags:
            if tag in benchmark_names:
                per_bm[tag].append(score)

    char_recall: dict[int, float] = {}
    char_precision: dict[int, float] = {}
    for k in ks:
        bm_avg_r: list[float] = []
        bm_avg_p: list[float] = []
        for name in benchmark_names:
            sl = per_bm.get(name, [])
            if not sl:
                continue
            bm_avg_r.append(sum(s.char_recall_at_k[k] for s in sl) / len(sl))
            bm_avg_p.append(sum(s.char_precision_at_k[k] for s in sl) / len(sl))
        char_recall[k] = sum(bm_avg_r) / len(bm_avg_r) if bm_avg_r else 0.0
        char_precision[k] = sum(bm_avg_p) / len(bm_avg_p) if bm_avg_p else 0.0
    return char_recall, char_precision


# ── Aggregate results ─────────────────────────────────────────────────────────


def aggregate(
    scores: list[QueryScore],
    benchmark_names: list[str],
    ks: list[int],
    index_name: str = DEFAULT_INDEX_NAME,
) -> None:
    """Print a summary table: character-level Recall@K and Precision@K per benchmark + overall.

    # ── Commented-out chunk-level table header ────────────────────────────────
    # Previously printed:
    #   LegalBench-RAG Evaluation — chunk-level @K
    # with metrics "recall_at_k" and "precision_at_k" (binary hit/miss per chunk).
    # Replaced by character-level metrics below.
    # ─────────────────────────────────────────────────────────────────────────
    """
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

    print(f"\n{'─' * width}")
    print(f"  LegalBench-RAG Evaluation — character-level @K")
    print(f"  CharRecall@K    = intersection(GT, retrieved) / GT chars")
    print(f"  CharPrecision@K = intersection(GT, retrieved) / retrieved chars")
    print(f"  OVERALL = macro-average of per-benchmark averages (equal benchmark weight)")
    print(f"{'─' * width}")

    for metric, label in [
        ("char_recall_at_k", "CharRecall"),
        ("char_precision_at_k", "CharPrecision"),
    ]:
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

    print(f"\n{'─' * width}")
    print(f"  Index : {index_name}  |  K values: {ks}")
    print()


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
        default=DEFAULT_INDEX_NAME,
        metavar="NAME",
        help=(
            f"OpenSearch index to query (default: {DEFAULT_INDEX_NAME}). "
            "Must match the index used during ingestion."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
        metavar="PROVIDER",
        help=(
            "Embedding provider: sentence_transformers, huggingface, or openai. "
            "Overrides EMBEDDING_PROVIDER in .env. "
            "Must match the provider used during ingestion."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help=(
            "Override the query embedding model (default: EMBEDDING_MODEL in .env). "
            "Must match the model used during ingestion. "
            "Provider is taken from --embedding-provider or EMBEDDING_PROVIDER in .env."
        ),
    )
    parser.add_argument(
        "--metrics-json-out",
        default=None,
        metavar="PATH",
        help=(
            "Write one JSON file with macro OVERALL char_recall_overall / "
            "char_precision_overall series (for sweep aggregation/plots)."
        ),
    )
    parser.add_argument(
        "--metrics-model-label",
        default="",
        help="Short label stored in metrics JSON (e.g. SBERT (all-mpnet-base-v2)).",
    )
    parser.add_argument(
        "--metrics-chunker-tag",
        default="",
        help="Chunker tag in metrics JSON: hierarchical or recursive.",
    )
    parser.add_argument(
        "--trace-file",
        default=None,
        metavar="PATH",
        help=(
            "Write a per-query retrieval trace to this file in JSONL format. "
            "Each line is a JSON object with: query, ground_truth spans, all "
            "retrieved chunks (ranked with scores), and per-K metrics with "
            "gt_overlap flags. Useful for debugging retrieval failures. "
            "Default: no trace file written."
        ),
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

    retriever = build_retriever(
        top_k=top_k,
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
    )

    trace_path = Path(args.trace_file) if args.trace_file else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    scores: list[QueryScore] = []
    trace_fh = open(trace_path, "w", encoding="utf-8") if trace_path else None
    _real_stdout = sys.stdout
    if trace_fh:
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
    finally:
        sys.stdout = _real_stdout
        if trace_fh:
            trace_fh.close()
            print(f"  Trace written → {trace_path}")

    if args.metrics_json_out:
        r_o, p_o = compute_overall_macro(scores, benchmark_names, ks)
        payload = {
            "index_name": args.index_name,
            "embedding_model": args.embedding_model or "",
            "chunker_tag": args.metrics_chunker_tag,
            "model_label": args.metrics_model_label,
            "ks": ks,
            "char_recall_overall": r_o,
            "char_precision_overall": p_o,
            "n_queries": len(scores),
        }
        out_p = Path(args.metrics_json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  Metrics JSON → {out_p}")


if __name__ == "__main__":
    main(sys.argv[1:])
