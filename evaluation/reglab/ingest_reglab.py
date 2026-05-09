"""Ingest RegLab exports with jurisdiction metadata for Housing (``court`` field).

Same CLI as :mod:`evaluation.LegalBenchRAG.ingest` but swaps in
:class:`~evaluation.reglab.corpus_loader.RegLabCorpusLoader` so each statute chunk
is filterable by U.S. state / territory (Section 5.2).

Usage::

    python -m evaluation.reglab.ingest_reglab \\
        --data-dir data/reglab_eval/housing_qa \\
        --index-name legalrag-reglab-housing-mpnet \\
        --chunker hierarchical \\
        --parent-size 2048 \\
        --chunk-size 512 \\
        --chunk-overlap 64 \\
        --embedding-provider sentence_transformers \\
        --embedding-model sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import logging
import sys

from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.ingest import parse_args
from evaluation.LegalBenchRAG.loader import corpus_file_paths_for_tests, load_benchmark
from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME, LegalBenchRAGIngestionPipeline
from evaluation.reglab.corpus_loader import RegLabCorpusLoader

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    data_dir = args.data_dir.rstrip("/")
    corpus_dir = args.corpus_dir.rstrip("/") if args.corpus_dir else f"{data_dir}/corpus"
    benchmarks_dir = f"{data_dir}/benchmarks"

    if args.ingest_all:
        file_paths = None
    else:
        tests = load_benchmark(
            benchmarks_dir,
            names=args.benchmarks,
            limit_per_benchmark=args.limit,
        )
        file_paths = corpus_file_paths_for_tests(tests)

    pipe = LegalBenchRAGIngestionPipeline.build(
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
    pipe._loader = RegLabCorpusLoader(corpus_dir, file_paths)  # type: ignore[method-assign]
    pipe.run(file_paths=file_paths)
    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
