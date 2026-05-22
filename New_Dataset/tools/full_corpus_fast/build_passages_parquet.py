#!/usr/bin/env python3
"""Pack all ``*.txt`` files under a corpus root into a compressed Parquet file.

Output can be a **single file** (``--out``) or **multiple shards** (``--output-dir`` +
``passages_part_000.parquet``, ...).  The shard mode is safer for long-running jobs:
if the job is interrupted, completed shards are preserved and ``--resume`` continues
from where it left off.

Usage::
    # single file
    python tools/full_corpus_fast/build_passages_parquet.py \\
        --corpus-root /path/to/barexam_qa/corpus \\
        --out /path/to/passages.parquet

    # sharded + resumable
    python tools/full_corpus_fast/build_passages_parquet.py \\
        --corpus-root /path/to/corpus \\
        --output-dir /path/to/passages_meta \\
        --shard-rows 50000 --resume
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PART_RE = re.compile(r"^passages_part_(\d+)\.parquet$")


def _sorted_txt_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.txt"))


def _scan_resume_shards(
    output_dir: Path,
    *,
    import_pq: object,
) -> tuple[int, int]:
    """Sum valid rows across existing shards; the last incomplete shard is removed.

    Returns:
        (resume_skip_rows, next_part_index)
    """
    pq = import_pq  # type: ignore[assignment]
    part_files: list[tuple[int, Path]] = []
    for p in output_dir.glob("passages_part_*.parquet"):
        m = _PART_RE.match(p.name)
        if m:
            part_files.append((int(m.group(1)), p))
    part_files.sort(key=lambda x: x[0])

    resume_skip_rows = 0
    next_part_index = 0
    for idx, path in part_files:
        try:
            pf = pq.ParquetFile(path)
            n = int(pf.metadata.num_rows)
        except Exception as exc:  # noqa: BLE001 — any parquet read error = corrupt shard
            logger.warning("Removing invalid shard %s (%s)", path, exc)
            try:
                path.unlink()
            except OSError as uexc:
                logger.warning("Could not delete %s: %s", path, uexc)
            next_part_index = idx
            break
        resume_skip_rows += n
        next_part_index = idx + 1

    if resume_skip_rows:
        logger.info(
            "Resume: %d complete rows in shards → skip first %d docs, next part index %d",
            resume_skip_rows,
            resume_skip_rows,
            next_part_index,
        )
    return resume_skip_rows, next_part_index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-root",
        type=Path,
        required=True,
        help="Directory like .../barexam_qa/corpus (contains passages/) or .../housing_qa/corpus (contains statutes/<State>/)",
    )
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", type=Path, help="Output path for single-file .parquet")
    out.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for passages_part_000.parquet, ...",
    )
    p.add_argument(
        "--shard-rows",
        type=int,
        default=50_000,
        help="Max document rows per shard file (only with --output-dir)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="With --output-dir: keep valid shards and continue from the last complete row count",
    )
    p.add_argument(
        "--flush-rows",
        type=int,
        default=50_000,
        help="Write to Arrow buffer every N rows (controls memory usage within each shard)",
    )
    p.add_argument(
        "--compression",
        default="zstd",
        choices=("zstd", "snappy", "gzip", "none"),
        help="Column compression codec",
    )
    args = p.parse_args()

    if args.output_dir and args.shard_rows < 1:
        raise SystemExit("--shard-rows must be >= 1")
    if args.out and args.resume:
        raise SystemExit("--resume is only valid with --output-dir")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        logger.error("pyarrow is required: pip install pyarrow (on Compute Canada: module load arrow)")
        raise SystemExit(1) from e

    root: Path = args.corpus_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"corpus-root not a directory: {root}")

    paths = _sorted_txt_paths(root)
    logger.info("Found %d .txt files under %s", len(paths), root)

    comp = None if args.compression == "none" else args.compression

    if args.out:
        _run_single_file(
            paths=paths,
            root=root,
            out=args.out.resolve(),
            flush_rows=args.flush_rows,
            compression=comp,
            pa=pa,
            pq=pq,
        )
        return

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_skip = 0
    part_idx = 0
    if args.resume:
        resume_skip, part_idx = _scan_resume_shards(output_dir, import_pq=pq)

    _run_sharded(
        paths=paths,
        root=root,
        output_dir=output_dir,
        resume_skip_rows=resume_skip,
        start_part_index=part_idx,
        shard_rows=args.shard_rows,
        flush_rows=args.flush_rows,
        compression=comp,
        pa=pa,
        pq=pq,
    )


def _run_single_file(
    *,
    paths: list[Path],
    root: Path,
    out: Path,
    flush_rows: int,
    compression: str | None,
    pa: object,
    pq: object,
) -> None:
    citations: list[str] = []
    texts: list[str] = []
    total = 0
    writer: object | None = None
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        for path in paths:
            if not path.is_file():
                continue
            rel = str(path.relative_to(root))
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("skip %s: %s", path, exc)
                continue
            citations.append(rel)
            texts.append(txt)
            total += 1

            if len(citations) >= flush_rows:
                table = pa.table({"citation": citations, "text": texts})  # type: ignore[attr-defined]
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema, compression=compression)  # type: ignore[attr-defined]
                writer.write_table(table)
                citations.clear()
                texts.clear()
                logger.info("Wrote batch, total rows so far: %d", total)

        if citations:
            table = pa.table({"citation": citations, "text": texts})  # type: ignore[attr-defined]
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression=compression)  # type: ignore[attr-defined]
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    logger.info("Done: %d documents → %s", total, out)


def _run_sharded(
    *,
    paths: list[Path],
    root: Path,
    output_dir: Path,
    resume_skip_rows: int,
    start_part_index: int,
    shard_rows: int,
    flush_rows: int,
    compression: str | None,
    pa: object,
    pq: object,
) -> None:
    citations: list[str] = []
    texts: list[str] = []
    total_docs = 0
    successful_seen = 0  # count of successfully readable rows (same as Parquet row count)
    rows_in_part = 0
    part_index = start_part_index
    writer: object | None = None
    eff_flush = max(1, min(flush_rows, shard_rows))

    def _part_path(idx: int) -> Path:
        return output_dir / f"passages_part_{idx:03d}.parquet"

    def _close_writer() -> None:
        nonlocal writer
        if writer is not None:
            writer.close()
            writer = None

    def _open_writer_for_schema(schema: object) -> None:
        nonlocal writer
        _close_writer()
        path = _part_path(part_index)
        writer = pq.ParquetWriter(path, schema, compression=compression)  # type: ignore[attr-defined]
        logger.info("Opened shard %s", path.name)

    def _flush_buffer() -> None:
        """Flush the in-memory buffer; each shard holds at most ``shard_rows`` rows."""
        nonlocal rows_in_part, writer, part_index
        while citations:
            if rows_in_part >= shard_rows:
                _close_writer()
                part_index += 1
                rows_in_part = 0
            cap = shard_rows - rows_in_part
            take = min(len(citations), cap)
            if take <= 0:
                break
            table = pa.table(  # type: ignore[attr-defined]
                {"citation": citations[:take], "text": texts[:take]}
            )
            if writer is None:
                _open_writer_for_schema(table.schema)
            assert writer is not None
            writer.write_table(table)
            del citations[:take]
            del texts[:take]
            rows_in_part += take
            if rows_in_part >= shard_rows:
                _close_writer()
                part_index += 1
                rows_in_part = 0

    try:
        for path in paths:
            if not path.is_file():
                continue

            rel = str(path.relative_to(root))
            try:
                txt = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("skip %s: %s", path, exc)
                continue

            if successful_seen < resume_skip_rows:
                successful_seen += 1
                continue

            citations.append(rel)
            texts.append(txt)
            total_docs += 1
            successful_seen += 1

            if len(citations) >= eff_flush:
                _flush_buffer()
                logger.info(
                    "Wrote batch (successful docs seen=%d), shard %03d rows_in_shard=%d",
                    successful_seen,
                    part_index,
                    rows_in_part,
                )

        _flush_buffer()
    finally:
        _close_writer()

    logger.info(
        "Done: %d new documents appended (after resume skip), shards under %s",
        total_docs,
        output_dir,
    )


if __name__ == "__main__":
    main()
