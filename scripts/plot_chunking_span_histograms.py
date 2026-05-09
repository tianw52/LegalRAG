#!/usr/bin/env python3
"""Histogram of how many fixed-size chunks each ground-truth span touches.

Splits each corpus document conceptually into contiguous, non-overlapping chunks of
``chunk_size`` characters from the start of the file ([0, chunk_size),
[chunk_size, 2*chunk_size), ...). For each GT snippet span ``[start, end)``, counts how
many of those chunks have non-empty overlap with the span — i.e. how the evidence span
lies across chunk boundaries under a fixed tokenizer-free chunk grid.

Produces five PNGs: one per LegalBench-RAG benchmark (contractnli, cuad, maud,
privacy_qa) and one for all snippets combined.

Usage (from LegalRAG repo root)::

    python scripts/plot_chunking_span_histograms.py \\
        --benchmarks-dir data/LegalBenchRAG/benchmarks \\
        --out-dir results/gt_chunk_histograms \\
        --chunk-size 512

Requires: matplotlib and numpy.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ("contractnli", "cuad", "maud", "privacy_qa")

TITLE_SUFFIX = (
    "(contiguous chunks from document start,\ncount chunk indices overlapping each GT span)"
)


def num_chunks_overlap_span(span_start: int, span_end: int, chunk_size: int) -> int:
    """Return how many contiguous ``chunk_size`` blocks from offset 0 hit ``[start, end)``.
    Uses half-open span convention matching the corpus (``[start, end)``).
    """
    if chunk_size <= 0 or span_end <= span_start:
        return 0
    lo = span_start // chunk_size
    hi = (span_end - 1) // chunk_size
    return hi - lo + 1


def chunk_counts_from_benchmark_file(json_path: Path, chunk_size: int) -> list[int]:
    """Read one benchmarks/*.json and return one overlap count per GT snippet."""
    counts: list[int] = []
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    for t in payload.get("tests", []):
        for s in t.get("snippets", []):
            span = s["span"]
            n = num_chunks_overlap_span(int(span[0]), int(span[1]), chunk_size)
            if n > 0:
                counts.append(n)
    return counts


def plot_one_histogram(values: list[int], title: str, out_path: Path) -> None:
    if not values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No snippets loaded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    arr = np.asarray(values, dtype=int)
    max_n = int(arr.max())
    min_n = int(arr.min())

    bins = np.arange(min_n - 0.5, max_n + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(arr, bins=bins, color="steelblue", edgecolor="black", linewidth=0.35)
    ax.set_xlabel("Number of contiguous chunks overlapping GT span (fixed chunk grid)")
    ax.set_ylabel("Count (snippet instances)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


_BENCH_LABEL = {
    "contractnli": "ContractNLI",
    "cuad": "CUAD",
    "maud": "MAUD",
    "privacy_qa": "Privacy QA",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("data/LegalBenchRAG/benchmarks"),
        help="Directory with contractnli.json, cuad.json, ...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/gt_chunk_histograms"),
        help="Where to write PNG files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters",
    )
    args = parser.parse_args()
    benchmarks_dir = args.benchmarks_dir.expanduser()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_counts: list[int] = []

    for ds in DATASETS:
        json_path = benchmarks_dir / f"{ds}.json"
        lbl = _BENCH_LABEL.get(ds, ds)
        title = f"{lbl} — GT spans vs chunk size {args.chunk_size}\n{TITLE_SUFFIX}"
        outfile = args.out_dir / f"gt_chunks_{ds}_chunksize_{args.chunk_size}.png"
        if not json_path.is_file():
            print(f"Skipping missing benchmark (empty plot): {json_path}", flush=True)
            plot_one_histogram([], title, outfile)
            continue
        vals = chunk_counts_from_benchmark_file(json_path, args.chunk_size)
        all_counts.extend(vals)

        plot_one_histogram(vals, title, outfile)

    title_all = f"ALL datasets — GT spans vs chunk size {args.chunk_size}\n{TITLE_SUFFIX}"
    plot_one_histogram(all_counts, title_all, args.out_dir / f"gt_chunks_all_chunksize_{args.chunk_size}.png")

    print(f"Wrote PNGs under {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
