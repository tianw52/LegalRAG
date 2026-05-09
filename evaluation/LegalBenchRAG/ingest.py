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
from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME, LegalBenchRAGIngestionPipeline


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
        "--corpus-dir",
        default=None,
        metavar="PATH",
        help=(
            "Override the corpus directory (default: <data-dir>/corpus). "
            "Use this to point at a custom corpus subset, e.g. data/LegalBenchRAG/corpus_50."
        ),
    )
    # ── Chunker options ───────────────────────────────────────────────────────
    parser.add_argument(
        "--chunker",
        default="hierarchical",
        choices=["hierarchical", "recursive"],
        help=(
            "Chunking strategy. 'hierarchical' stores parent (~1500 chars) + child "
            "chunks (default); 'recursive' produces flat chunks only."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Child chunk size in characters "
            "(default: CHUNK_SIZE from config, usually 512)."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Character overlap between consecutive child chunks "
            "(default: CHUNK_OVERLAP from config, usually 64)."
        ),
    )
    parser.add_argument(
        "--parent-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parent chunk size in characters — hierarchical chunker only "
            "(default: 1500)."
        ),
    )
    # ── Embedding options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
        metavar="PROVIDER",
        help=(
            "Embedding provider: sentence_transformers, huggingface, or openai. "
            "Overrides EMBEDDING_PROVIDER in .env. "
            "Use 'huggingface' for models not packaged as sentence-transformers "
            "(e.g. jhu-clsp/BERT-DPR-CLERC-ft)."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help=(
            "Embedding model name. Overrides EMBEDDING_MODEL in .env. "
            "Provider is taken from --embedding-provider or EMBEDDING_PROVIDER in .env."
        ),
    )
    # ── Index ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--index-name",
        default=DEFAULT_INDEX_NAME,
        metavar="NAME",
        help=(
            f"OpenSearch index to ingest into (default: {DEFAULT_INDEX_NAME}). "
            "Use a different name to keep experiments with different chunkers or "
            "embedding models isolated without deleting the default index."
        ),
    )
    # ── Logging ───────────────────────────────────────────────────────────────
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
    corpus_dir = args.corpus_dir.rstrip("/") if args.corpus_dir else f"{data_dir}/corpus"
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
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_size=args.parent_size,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        index_name=args.index_name,
    )
    pipeline.run(file_paths=file_paths)
    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
