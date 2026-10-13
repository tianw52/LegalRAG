#!/usr/bin/env python3
"""
Audit a chunker configuration against the CanLegalRAGBench corpus.

Checks the invariants the rest of the system relies on, and reports quality
statistics, before you spend an ingest run on a new chunking setup:

  * every chunk's text equals document.text[char_start:char_end]
  * chunks cover the whole document (no gaps) and are ordered
  * chunk_ids are unique (no parent/child collisions)
  * chunks never exceed the size limit
  * hierarchical: parents tile the document; children stay inside their parent
  * quality: chunks starting/ending mid-word, overlap between consecutive chunks

Usage
-----
    python scripts/check_chunking.py --chunker recursive --chunk-size 1024 --chunk-overlap 128
    python scripts/check_chunking.py --chunker hierarchical --parent-size 4096 --chunk-size 1024
    python scripts/check_chunking.py --chunker recursive --chunk-size 1024 --limit 100

Exit code 1 if any invariant is violated.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict

from legalrag.core.models import Chunk, RawDocument
from legalrag.ingestion.canlegal_loader import CanLegalRAGBenchLoader
from legalrag.ingestion.chunker import HierarchicalChunker, RecursiveCharacterTextSplitter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/CanLegalRAGBench")
    p.add_argument("--chunker", choices=["recursive", "hierarchical"], required=True)
    p.add_argument("--chunk-size", type=int, default=1024, help="Chunk (or child) size in chars")
    p.add_argument("--chunk-overlap", type=int, default=128)
    p.add_argument("--parent-size", type=int, default=None, help="Hierarchical only")
    p.add_argument("--limit", type=int, default=None, help="Only audit the first N documents")
    return p.parse_args(argv)


def mid_word(text: str, pos: int) -> bool:
    """True if *pos* splits a word (alphanumeric on both sides)."""
    return 0 < pos < len(text) and text[pos - 1].isalnum() and text[pos].isalnum()


def audit_document(doc: RawDocument, chunks: list[Chunk], size_limit: int) -> tuple[Counter, dict]:
    text = doc.text
    errors: Counter = Counter()
    stats: dict[str, list] = defaultdict(list)

    ids = [c.chunk_id for c in chunks]
    errors["duplicate_chunk_id"] += len(ids) - len(set(ids))
    for c in chunks:
        if text[c.char_start:c.char_end] != c.text:
            errors["text_offset_mismatch"] += 1

    parents = [c for c in chunks if c.is_parent]
    retrievable = [c for c in chunks if not c.is_parent]

    # Coverage + ordering of the retrievable (embedded) chunks
    covered = bytearray(len(text))
    for c in retrievable:
        covered[c.char_start:c.char_end] = b"\x01" * (c.char_end - c.char_start)
    if not all(covered):
        errors["coverage_gap"] += 1
    ordered = sorted(retrievable, key=lambda c: (c.char_start, c.char_end))
    for a, b in zip(ordered, ordered[1:], strict=False):
        stats["overlap"].append(max(0, a.char_end - b.char_start))

    for c in retrievable:
        stats["length"].append(len(c.text))
        if len(c.text) > size_limit:
            errors["over_size_limit"] += 1
        stats["mid_word_start"].append(mid_word(text, c.char_start))
        stats["mid_word_end"].append(mid_word(text, c.char_end))

    if parents:
        ps = sorted(parents, key=lambda c: c.char_start)
        if ps[0].char_start != 0 or ps[-1].char_end != len(text):
            errors["parents_do_not_cover_document"] += 1
        errors["parent_gap_or_overlap"] += sum(a.char_end != b.char_start for a, b in zip(ps, ps[1:], strict=False))
        by_id = {p.chunk_id: p for p in parents}
        for c in retrievable:
            p = by_id.get(c.parent_chunk_id)
            if p is None:
                errors["orphan_child"] += 1
            elif not (p.char_start <= c.char_start and c.char_end <= p.char_end):
                errors["child_outside_parent"] += 1
        stats["parent_length"] += [p.char_end - p.char_start for p in parents]
    return errors, stats


def summary(values: list) -> str:
    if not values:
        return "n/a"
    return f"min {min(values)} / median {int(statistics.median(values))} / max {max(values)}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    docs = CanLegalRAGBenchLoader().load(args.data_dir)[: args.limit]
    if args.chunker == "hierarchical":
        if not args.parent_size:
            raise SystemExit("--parent-size is required for the hierarchical chunker")
        chunker = HierarchicalChunker(
            parent_size=args.parent_size, child_size=args.chunk_size, child_overlap=args.chunk_overlap
        )
    else:
        chunker = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    errors: Counter = Counter()
    stats: dict[str, list] = defaultdict(list)
    for doc in docs:
        e, s = audit_document(doc, chunker.chunk(doc), args.chunk_size)
        errors.update(e)
        for k, v in s.items():
            stats[k] += v

    n = len(stats["length"])
    pct = lambda flags: f"{sum(flags) / max(1, len(flags)):.1%}"  # noqa: E731
    print(f"{args.chunker} chunk_size={args.chunk_size} overlap={args.chunk_overlap}"
          + (f" parent_size={args.parent_size}" if args.parent_size else "")
          + f" — {len(docs)} documents, {n} retrievable chunks")
    print(f"  chunk length        : {summary(stats['length'])}")
    print(f"  overlap (chars)     : {summary(stats['overlap'])}, none for {pct([o == 0 for o in stats['overlap']])} of pairs")
    print(f"  starts mid-word     : {pct(stats['mid_word_start'])}")
    print(f"  ends mid-word       : {pct(stats['mid_word_end'])}")
    if stats["parent_length"]:
        print(f"  parent length       : {summary(stats['parent_length'])} ({len(stats['parent_length'])} parents)")
    violations = {k: v for k, v in errors.items() if v}
    print(f"  invariant violations: {violations or 'none'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
