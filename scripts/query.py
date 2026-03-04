#!/usr/bin/env python
"""CLI script: run a query against the LegalRAG system.

Usage:
    python scripts/query.py "What is the standard of review for s.7 Charter claims?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel

from legalrag.query.pipeline import QueryPipeline
from legalrag.utils.logging import configure_logging

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LegalRAG system.")
    parser.add_argument("query", help="Legal question to answer")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    configure_logging(args.log_level)

    pipeline = QueryPipeline.default()
    response = pipeline.run(args.query)

    console.print(Panel(response.answer, title="[bold green]Answer[/bold green]", expand=False))
    console.print(f"\n[dim]Router path: {response.router_path}[/dim]")
    console.print(f"[dim]Sources used: {len(response.retrieved_chunks)}[/dim]")

    for i, rc in enumerate(response.retrieved_chunks, start=1):
        m = rc.chunk.metadata
        console.print(
            f"  [{i}] {m.court or 'unknown court'} | {m.citation or 'no citation'}"
            f" | score={rc.rerank_score:.4f}"
        )


if __name__ == "__main__":
    main()
