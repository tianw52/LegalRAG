"""Character-level CharRecall@K / CharPrecision@K for RegLab loaded from Hugging Face.

Does not read ``benchmarks/*.json`` or ``corpus/*.txt``.

Smoke test::

    python -m evaluation.reglab.eval_char_hf \\
        --benchmark barexam_qa \\
        --index-name legalrag-reglab-barexam-hf-smoke \\
        --limit-queries 20 \\
        --limit-corpus-maps 5000 \\
        --ks 20 40 \\
        --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.eval_precision_recall import (
    QueryScore,
    aggregate,
    build_retriever,
    compute_overall_macro,
    score_query,
)
from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME
from evaluation.reglab.hf_reglab import (
    HFLoadStats,
    load_barexam_hf_benchmark_tests,
    load_housing_hf_benchmark_tests,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        "--reglab-source",
        dest="source",
        choices=("hf",),
        default="hf",
        help="Only Hugging Face loading is implemented in this module.",
    )
    p.add_argument(
        "--benchmark",
        required=True,
        choices=("barexam_qa", "housing_qa"),
        help="Which RegLab benchmark to evaluate.",
    )
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument(
        "--limit-corpus-maps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap rows when scanning passages/statutes for gold span lengths. "
            "Smaller = faster smoke; may drop QA rows whose gold is past the cap."
        ),
    )
    p.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
        metavar="K",
    )
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    p.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--metrics-json-out", default=None, metavar="PATH")
    p.add_argument("--trace-file", default=None, metavar="PATH")
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def _log_stats(benchmark: str, stats: HFLoadStats, limit_maps: int | None) -> None:
    logger.info(
        "RegLab HF char eval: benchmark=%s source=%s queries=%d gold_snippets=%d "
        "dropped_queries=%d corpus_map_rows=%d limit_corpus_maps=%s",
        benchmark,
        stats.source_mode,
        stats.n_queries_loaded,
        stats.n_gold_snippets,
        stats.n_queries_missing_any_gold,
        stats.n_corpus_rows_scanned_for_maps,
        limit_maps,
    )
    if stats.n_queries_missing_any_gold:
        logger.warning(
            "Some queries were dropped (missing gold in passage/statute maps). "
            "If this is unexpected, increase --limit-corpus-maps or omit it.",
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    if args.benchmark == "barexam_qa":
        tests, st = load_barexam_hf_benchmark_tests(
            limit_queries=args.limit_queries,
            passage_map_limit=args.limit_corpus_maps,
        )
    else:
        tests, st = load_housing_hf_benchmark_tests(
            limit_queries=args.limit_queries,
            statute_map_limit=args.limit_corpus_maps,
        )
    _log_stats(args.benchmark, st, args.limit_corpus_maps)

    if not tests:
        print("No test cases loaded.", file=sys.stderr)
        sys.exit(1)

    ks = sorted(set(args.ks))
    top_k = max(ks)
    retriever = build_retriever(
        top_k=top_k,
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
    )

    benchmark_names = [args.benchmark]
    trace_path = Path(args.trace_file) if args.trace_file else None
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    scores: list[QueryScore] = []
    trace_fh = open(trace_path, "w", encoding="utf-8") if trace_path else None
    print(
        f"\nRegLab HF char eval: n={len(tests)}, K={ks}, index={args.index_name}, "
        f"source={args.source}\n",
    )
    try:
        for i, test in enumerate(tests, 1):
            score = score_query(test, retriever, ks=ks, trace_fh=trace_fh, query_idx=i)
            scores.append(score)
            if i % 50 == 0:
                print(f"  {i}/{len(tests)} queries …")
        aggregate(scores, benchmark_names, ks=ks, index_name=args.index_name)
    finally:
        if trace_fh:
            trace_fh.close()
            print(f"  Trace written → {trace_path}")

    if args.metrics_json_out:
        r_o, p_o = compute_overall_macro(scores, benchmark_names, ks)
        payload = {
            "reglab_source": args.source,
            "benchmark": args.benchmark,
            "index_name": args.index_name,
            "embedding_model": args.embedding_model or "",
            "ks": ks,
            "char_recall_overall": r_o,
            "char_precision_overall": p_o,
            "n_queries": len(scores),
            "hf_stats": {
                "n_queries_loaded": st.n_queries_loaded,
                "n_gold_snippets": st.n_gold_snippets,
                "n_queries_missing_any_gold": st.n_queries_missing_any_gold,
                "n_corpus_rows_scanned_for_maps": st.n_corpus_rows_scanned_for_maps,
            },
        }
        out_p = Path(args.metrics_json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  Metrics JSON → {out_p}")


if __name__ == "__main__":
    main(sys.argv[1:])
