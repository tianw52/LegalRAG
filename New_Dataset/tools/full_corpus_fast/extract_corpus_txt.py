"""Extract only the corpus .txt files that appear in a benchmark JSON.

Instead of writing all 1.8M statute files (which exhausts inode quota),
this script reads the benchmark JSON to find the ~990 unique file_paths
referenced by the ground-truth snippets, then scans the Parquet shards to
find matching rows and writes them as .txt files under corpus/.

This gives eval_precision_recall.py the corpus_dir it needs for gt_text
tracing, using only ~1 000 inodes instead of 1.8 M.

Usage::

    # housing_qa (recommended — run once, finishes in seconds)
    python tools/full_corpus_fast/extract_corpus_txt.py \\
        --benchmark  $SCRATCH/reglab_eval/housing_qa/benchmarks/housing_qa.json \\
        --parquet    $SCRATCH/reglab_eval/housing_qa/passages_meta \\
        --corpus-out $SCRATCH/reglab_eval/housing_qa/corpus

    # barexam_qa (if corpus/ was accidentally deleted)
    python tools/full_corpus_fast/extract_corpus_txt.py \\
        --benchmark  $SCRATCH/reglab_eval/barexam_qa/benchmarks/barexam_qa.json \\
        --parquet    $SCRATCH/reglab_eval/barexam_qa/passages_meta \\
        --corpus-out $SCRATCH/reglab_eval/barexam_qa/corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def extract(benchmark_path: Path, parquet_path: Path, corpus_out: Path) -> None:
    # ── 1. Collect needed citation paths from benchmark ──────────────────────
    bm = json.loads(benchmark_path.read_text(encoding="utf-8"))
    needed: set[str] = set()
    for test in bm.get("tests", []):
        for snippet in test.get("snippets", []):
            fp = snippet.get("file_path")
            if fp:
                needed.add(fp)

    logger.info("Benchmark references %d unique file paths", len(needed))
    if not needed:
        logger.warning("No file_paths found in benchmark — nothing to write.")
        return

    # ── 2. Scan Parquet shards for matching rows ─────────────────────────────
    import pyarrow.parquet as pq
    import glob

    if parquet_path.is_file():
        shard_files = [parquet_path]
    else:
        shard_files = sorted(parquet_path.glob("passages_part_*.parquet"))
        if not shard_files:
            raise FileNotFoundError(
                f"No passages_part_*.parquet under {parquet_path}"
            )

    found: dict[str, str] = {}   # citation → text
    remaining = set(needed)

    for shard in shard_files:
        if not remaining:
            break
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=50_000, columns=["citation", "text"]):
            citations = batch.column("citation").to_pylist()
            texts = batch.column("text").to_pylist()
            for c, t in zip(citations, texts):
                if c in remaining:
                    found[c] = t or ""
                    remaining.discard(c)
            if not remaining:
                break
        logger.info(
            "After shard %s: found %d/%d, remaining %d",
            shard.name, len(found), len(needed), len(remaining),
        )

    if remaining:
        logger.warning(
            "%d file_paths in benchmark were not found in parquet: %s …",
            len(remaining),
            sorted(remaining)[:5],
        )

    # ── 3. Write .txt files ───────────────────────────────────────────────────
    corpus_out.mkdir(parents=True, exist_ok=True)
    written = 0
    for citation, text in found.items():
        out_path = corpus_out / citation
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        written += 1

    logger.info(
        "Wrote %d .txt files under %s  (skipped/missing: %d)",
        written, corpus_out, len(needed) - written,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract only the benchmark-referenced .txt files from Parquet shards. "
            "Avoids writing millions of files while still supplying corpus_dir for tracing."
        )
    )
    p.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Path to the benchmark JSON (e.g. benchmarks/housing_qa.json).",
    )
    p.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Parquet file or directory of passages_part_*.parquet shards.",
    )
    p.add_argument(
        "--corpus-out",
        required=True,
        type=Path,
        help="Output corpus/ directory; .txt files are written under it mirroring citation paths.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    extract(
        benchmark_path=args.benchmark.resolve(),
        parquet_path=args.parquet.resolve(),
        corpus_out=args.corpus_out.resolve(),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
