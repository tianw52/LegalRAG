"""CLI entrypoint for LegalBench-RAG corpus ingestion.

Ingests corpus documents into the ``legalrag-legalbenchrag`` OpenSearch index.
By default only documents referenced by the benchmark test cases are ingested
(much smaller than the full corpus).  Pass ``--all`` to ingest every ``*.txt``
file under ``--corpus-dir``.

Usage
-----
# Ingest only files referenced by benchmark tests (recommended first run)
python -m evaluation.LegalBenchRAG.ingest \\
    --data-dir data/LegalBenchRAG

# Ingest all files referenced by a single sub-benchmark
python -m evaluation.LegalBenchRAG.ingest \\
    --data-dir data/LegalBenchRAG \\
    --benchmarks cuad maud

# Ingest at most 50 test cases per benchmark (fast smoke test)
python -m evaluation.LegalBenchRAG.ingest \\
    --data-dir data/LegalBenchRAG \\
    --limit 50

# Ingest the entire corpus (all *.txt files, no benchmark filter)
python -m evaluation.LegalBenchRAG.ingest \\
    --data-dir data/LegalBenchRAG \\
    --all
"""

from __future__ import annotations

import argparse
import logging
import sys

from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.loader import (
    corpus_file_paths_for_tests,
    load_benchmark,
)
from evaluation.LegalBenchRAG.pipeline import LegalBenchRAGIngestionPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.LegalBenchRAG.ingest",
        description="Ingest LegalBench-RAG corpus into OpenSearch for evaluation.",
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
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Benchmark(s) to load for determining which corpus files to ingest. "
            "Choices: contractnli cuad maud privacy_qa. "
            "Defaults to all four when --all is not set."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of test cases loaded per benchmark when determining "
            "which corpus files to ingest.  Useful for fast smoke tests."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="ingest_all",
        help=(
            "Ingest every *.txt file under corpus/ regardless of benchmark coverage. "
            "Overrides --benchmarks and --limit."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    data_dir = args.data_dir.rstrip("/")
    corpus_dir = f"{data_dir}/corpus"
    benchmarks_dir = f"{data_dir}/benchmarks"

    if args.ingest_all:
        logger.info("Ingesting all corpus files under %s", corpus_dir)
        file_paths = None
    else:
        logger.info(
            "Loading benchmark tests to determine corpus files "
            "(benchmarks=%s, limit=%s)",
            args.benchmarks or "all",
            args.limit if args.limit is not None else "all",
        )
        tests = load_benchmark(
            benchmarks_dir,
            names=args.benchmarks,
            limit_per_benchmark=args.limit,
        )
        file_paths = corpus_file_paths_for_tests(tests)
        logger.info(
            "Will ingest %d unique corpus files referenced by %d test cases",
            len(file_paths),
            len(tests),
        )

    pipeline = LegalBenchRAGIngestionPipeline.build(
        corpus_dir=corpus_dir,
        file_paths=file_paths,
    )
    pipeline.run(file_paths=file_paths)
    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
