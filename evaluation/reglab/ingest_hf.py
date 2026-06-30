"""Ingest RegLab corpus from Hugging Face (no ``corpus/*.txt`` export).

Uses the same chunking / embedding / indexing path as :mod:`evaluation.reglab.ingest_reglab`
but streams :class:`~legalrag.core.models.RawDocument` from HF via
:class:`~evaluation.reglab.hf_reglab.RegLabHFCorpusLoader`.

Smoke test (fast; uses cached HF data when available)::

    export HF_HUB_OFFLINE=1   # optional if reglab/* is already cached
    python -m evaluation.reglab.ingest_hf \\
        --dataset barexam_qa \\
        --limit-queries 20 \\
        --limit-corpus 1000 \\
        --index-name legalrag-reglab-barexam-hf-smoke

Full ingest (benchmark-filtered corpus only; default)::

    python -m evaluation.reglab.ingest_hf \\
        --dataset housing_qa \\
        --index-name legalrag-reglab-housing-hf

Ingest every statute/passage row (heavy)::

    python -m evaluation.reglab.ingest_hf --dataset housing_qa --ingest-all
"""

from __future__ import annotations

import argparse
import logging
import sys

from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME, LegalBenchRAGIngestionPipeline
from evaluation.reglab.hf_reglab import (
    RegLabHFCorpusLoader,
    citations_for_benchmark_tests,
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
        help="Only 'hf' is supported here; disk ingest uses evaluation.reglab.ingest_reglab.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=("barexam_qa", "housing_qa"),
        help="Which reglab Hugging Face dataset to ingest.",
    )
    p.add_argument(
        "--ingest-all",
        action="store_true",
        help="Ingest the full passages/statutes split (ignore benchmark filter).",
    )
    p.add_argument(
        "--limit-corpus",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N corpus rows (debug / smoke).",
    )
    p.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When not using --ingest-all, load at most N QA rows to decide which "
            "corpus citations to ingest."
        ),
    )
    p.add_argument(
        "--chunker",
        default="hierarchical",
        choices=["hierarchical", "recursive"],
        help="Chunking strategy (default: hierarchical).",
    )
    p.add_argument("--chunk-size", type=int, default=None, metavar="N")
    p.add_argument("--chunk-overlap", type=int, default=None, metavar="N")
    p.add_argument("--parent-size", type=int, default=None, metavar="N")
    p.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME, metavar="NAME")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    ds = args.dataset
    citation_filter: set[str] | None = None
    n_queries_for_log: int | None = None

    if not args.ingest_all:
        if ds == "barexam_qa":
            tests, st = load_barexam_hf_benchmark_tests(
                limit_queries=args.limit_queries,
                passage_map_limit=None,
            )
        else:
            tests, st = load_housing_hf_benchmark_tests(
                limit_queries=args.limit_queries,
                statute_map_limit=None,
            )
        citation_filter = citations_for_benchmark_tests(tests)
        n_queries_for_log = len(tests)
        logger.info(
            "RegLab HF ingest: source=%s dataset=%s mode=benchmark-filtered "
            "queries_loaded=%d unique_citations=%d missing_queries=%d corpus_map_rows=%d",
            args.source,
            ds,
            st.n_queries_loaded,
            len(citation_filter),
            st.n_queries_missing_any_gold,
            st.n_corpus_rows_scanned_for_maps,
        )
    else:
        logger.info(
            "RegLab HF ingest: source=%s dataset=%s mode=ingest-all cap=%s",
            args.source,
            ds,
            args.limit_corpus,
        )

    loader = RegLabHFCorpusLoader(
        ds,
        citation_filter=citation_filter,
        limit_corpus=args.limit_corpus,
    )

    pipe = LegalBenchRAGIngestionPipeline.build_with_loader(
        loader,
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_size=args.parent_size,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        index_name=args.index_name,
    )
    logger.info(
        "Starting ingest into index=%s (limit_corpus=%s, limit_queries=%s, ingest_all=%s)",
        args.index_name,
        args.limit_corpus,
        args.limit_queries,
        args.ingest_all,
    )
    pipe.run(file_paths=None)
    logger.info(
        "RegLab HF ingest done — index=%s dataset=%s queries_used_for_filter=%s",
        args.index_name,
        ds,
        n_queries_for_log,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
