#!/usr/bin/env python
"""CLI script: ingest a directory or single file into the LegalRAG index.

Usage:
    python scripts/ingest.py /path/to/legal_docs/
    python scripts/ingest.py /path/to/case.txt
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legalrag.ingestion.pipeline import IngestionPipeline
from legalrag.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal documents into LegalRAG.")
    parser.add_argument("source", help="Path to a .txt file or directory of .txt files")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    configure_logging(args.log_level)

    pipeline = IngestionPipeline.default()
    pipeline.run(args.source)


if __name__ == "__main__":
    main()
