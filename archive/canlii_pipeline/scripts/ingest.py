#!/usr/bin/env python
"""CLI script: ingest a directory or single file into the LegalRAG index.

Usage:
    python scripts/ingest.py /path/to/legal_docs/
    python scripts/ingest.py /path/to/case.txt

    # Use the recursive chunker with custom chunk size
    python scripts/ingest.py data/subset/ --chunker recursive --chunk-size 768

    # Override the embedding model
    python scripts/ingest.py data/subset/ --embedding-model BAAI/bge-large-en-v1.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legalrag.ingestion.pipeline import IngestionPipeline
from legalrag.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal documents into LegalRAG.")
    parser.add_argument("source", help="Path to a .txt file or directory of .txt files.")

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
        help="Child chunk size in characters (default: CHUNK_SIZE from config, usually 512).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="N",
        help="Character overlap between consecutive chunks (default: CHUNK_OVERLAP from config).",
    )
    parser.add_argument(
        "--parent-size",
        type=int,
        default=None,
        metavar="N",
        help="Parent chunk size in characters — hierarchical chunker only (default: 1500).",
    )

    # ── Embedding options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--embedding-model",
        default=None,
        metavar="MODEL",
        help=(
            "HuggingFace sentence-transformers model name, e.g. 'BAAI/bge-large-en-v1.5'. "
            "Overrides EMBEDDING_MODEL in .env."
        ),
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)

    pipeline = IngestionPipeline.default(
        chunker=args.chunker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parent_size=args.parent_size,
        embedding_model=args.embedding_model,
    )
    pipeline.run(args.source)


if __name__ == "__main__":
    main()
