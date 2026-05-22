"""Build housing_qa Parquet shards + benchmarks JSON **without** intermediate .txt files.

This bypasses the two-step prepare.py → build_passages_parquet.py pipeline to avoid
creating ~1.7M individual statute files that would exhaust the inode quota.

Output
------
<out_dir>/
    passages_part_000.parquet
    passages_part_001.parquet
    ...
<benchmarks_dir>/
    housing_qa.json

Each Parquet shard has columns:
    citation (str)  – relative path ``statutes/<state>/<sha256>.txt``
    text     (str)  – full statute text

``citation`` is the same key used by ``ParquetCorpusLoader`` and the benchmark JSON
``file_path`` field so retrieval and evaluation are consistent.

Usage::

    python tools/full_corpus_fast/build_housing_direct_parquet.py \\
        --parquet-dir   $SCRATCH/reglab_eval/housing_qa_parquet \\
        --benchmarks-dir $SCRATCH/reglab_eval/housing_qa/benchmarks \\
        --rows-per-shard 200000

    # smoke test
    python tools/full_corpus_fast/build_housing_direct_parquet.py \\
        --parquet-dir   data/smoke/housing_parquet \\
        --benchmarks-dir data/smoke/housing_benchmarks \\
        --max-corpus-docs 5000

The script is **resumable**: if shard N already exists on disk it is skipped and
the statute idx→citation map is re-read from that shard so benchmarks remain
consistent.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility (mirrors evaluation.reglab.util to avoid import path issues)
# ---------------------------------------------------------------------------

import hashlib


def _statute_relpath(state: str, idx) -> str:
    h = hashlib.sha256(str(idx).encode()).hexdigest()
    st = (state or "unknown").strip().replace("/", "_")
    return f"statutes/{st}/{h}.txt"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _write_shard(rows: list[tuple[str, str]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    citations = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    tbl = pa.table({"citation": citations, "text": texts})
    pq.write_table(tbl, path, compression="snappy")
    logger.info("Wrote shard %s  (%d rows)", path.name, len(rows))


def build(
    parquet_dir: Path,
    benchmarks_dir: Path,
    rows_per_shard: int = 200_000,
    max_corpus_docs: int | None = None,
) -> None:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "Install the HuggingFace datasets package: pip install -e '.[eval]'"
        ) from exc

    parquet_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1 – statutes → sharded Parquet
    # -----------------------------------------------------------------------
    logger.info("Loading reglab/housing_qa statutes …")
    stat_ds = load_dataset("reglab/housing_qa", "statutes", trust_remote_code=True)
    split = next(s for s in ("corpus", "train", "validation", "test") if s in stat_ds)
    rows_iter = iter(stat_ds[split])

    idx_to_citation: dict[str, str] = {}
    idx_to_textlen: dict[str, int] = {}
    shard_idx = 0
    buffer: list[tuple[str, str]] = []
    n_written = 0

    def _flush(buf: list[tuple[str, str]], shard: int) -> None:
        _write_shard(buf, parquet_dir / f"passages_part_{shard:03d}.parquet")

    for row in rows_iter:
        if max_corpus_docs is not None and n_written >= max_corpus_docs:
            break

        idx = row["idx"]
        idx_str = str(idx)
        state = (row.get("state") or "").strip()
        text = row.get("text") or ""
        citation = _statute_relpath(state, idx)

        idx_to_citation[idx_str] = citation
        idx_to_textlen[idx_str] = len(text)

        buffer.append((citation, text))
        n_written += 1

        if len(buffer) >= rows_per_shard:
            shard_path = parquet_dir / f"passages_part_{shard_idx:03d}.parquet"
            if not shard_path.exists():
                _flush(buffer, shard_idx)
            else:
                logger.info("Shard %s already exists — skipping write", shard_path.name)
            shard_idx += 1
            buffer = []

    if buffer:
        shard_path = parquet_dir / f"passages_part_{shard_idx:03d}.parquet"
        if not shard_path.exists():
            _flush(buffer, shard_idx)
        else:
            logger.info("Final shard %s already exists — skipping write", shard_path.name)

    logger.info("Total statutes written: %d across %d shards", n_written, shard_idx + 1)

    # -----------------------------------------------------------------------
    # Step 2 – questions → benchmarks JSON
    # -----------------------------------------------------------------------
    logger.info("Loading reglab/housing_qa questions …")
    q_ds = load_dataset("reglab/housing_qa", "questions", trust_remote_code=True)
    q_split = next(s for s in ("test", "train", "validation") if s in q_ds)
    q_rows = q_ds[q_split]

    tests = []
    skipped = 0
    for row in q_rows:
        state = (row.get("state") or "").strip()
        question = (row.get("question") or "").strip()
        qtext = (
            f"Consider statutory law for {state} in the year 2021.\n"
            f"{question}\n"
            'Answer "Yes" or "No".'
        )

        statutes = row.get("statutes") or []
        snippets = []
        ok = True
        for st_ref in statutes:
            sid = st_ref.get("statute_idx")
            if sid is None:
                ok = False
                break
            sid_str = str(sid)
            citation = idx_to_citation.get(sid_str)
            if not citation:
                ok = False
                break
            n_chars = idx_to_textlen.get(sid_str, 0)
            if n_chars == 0:
                ok = False
                break
            snippets.append({"file_path": citation, "span": [0, n_chars]})

        if not ok or not snippets:
            skipped += 1
            continue

        tests.append({
            "query": qtext,
            "snippets": snippets,
            "tags": ["housing_qa"],
            "jurisdiction": state,
        })

    bm_path = benchmarks_dir / "housing_qa.json"
    bm_path.write_text(json.dumps({"tests": tests}, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d benchmark tests → %s  (skipped %d)",
        len(tests), bm_path, skipped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build housing_qa Parquet shards + benchmarks JSON from HuggingFace.",
    )
    p.add_argument(
        "--parquet-dir",
        required=True,
        type=Path,
        help="Output directory for passages_part_*.parquet shards.",
    )
    p.add_argument(
        "--benchmarks-dir",
        required=True,
        type=Path,
        help="Output directory for housing_qa.json benchmark.",
    )
    p.add_argument(
        "--rows-per-shard",
        type=int,
        default=200_000,
        metavar="N",
        help="Statutes per Parquet shard (default: 200000).",
    )
    p.add_argument(
        "--max-corpus-docs",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N statutes (for smoke tests).",
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
    build(
        parquet_dir=args.parquet_dir.resolve(),
        benchmarks_dir=args.benchmarks_dir.resolve(),
        rows_per_shard=args.rows_per_shard,
        max_corpus_docs=args.max_corpus_docs,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
