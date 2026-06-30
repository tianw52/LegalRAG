#!/usr/bin/env python3
"""Summarize paper-aligned flat Housing Statute QA results vs Zheng et al. Table 3."""

from __future__ import annotations

import json
from pathlib import Path

PAPER = {
    "flat-bm25-baseline": ("BM25 upper R@10", 40.8),
    "flat-e5-baseline": ("E5-large-v2 upper R@10", 50.6),
    "flat-bm25-qexp": ("BM25 + structured reasoning R@10", 51.1),
    "flat-e5-qexp": ("E5 + structured reasoning R@10", 52.8),
}

PRIOR_CHILD_CHUNK = {
    "hier-bm25": 22.11,
    "hier-e5-dense": None,  # filled from diag if available
}


def _load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("paper_housing_*.json")):
        d = json.loads(f.read_text())
        label = d.get("diag_label") or f.stem.replace("paper_housing_upper_", "").rsplit("_", 2)[0]
        r = d.get("recall_passage_percent") or {}
        rows.append(
            {
                "file": f.name,
                "status": d.get("status", "unknown"),
                "label": label,
                "chunker": d.get("chunker", "?"),
                "query_expansion": d.get("query_expansion", False),
                "retrieval_mode": d.get("retrieval_mode", "?"),
                "model": d.get("embedding_model", ""),
                "index_document_count": d.get("index_document_count"),
                "R@1": r.get("1"),
                "R@10": r.get("10"),
                "R@100": r.get("100"),
                "R@1000": r.get("1000"),
                "MRR@10": d.get("mrr_10_percent"),
            }
        )
    return rows


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/scratch/ram112/reglab_eval/results_housing_paper_flat"),
    )
    p.add_argument(
        "--prior-diag-dir",
        type=Path,
        default=Path("/scratch/ram112/reglab_eval/results_housing_diag"),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = _load_rows(args.results_dir)
    valid = [r for r in rows if r["status"] == "valid"]
    invalid = [r for r in rows if r["status"] != "valid"]

    print("=== Paper-aligned flat Housing results ===")
    if not rows:
        print(f"No results under {args.results_dir}")
        return

    print(
        f"{'Label':<24} {'Mode':<8} {'QExp':>4} {'R@1':>6} {'R@10':>6} "
        f"{'R@100':>7} {'R@1000':>7} {'MRR@10':>7} {'IndexDocs':>10}"
    )
    print("-" * 92)
    for r in sorted(valid, key=lambda x: float(x["R@10"] or 0), reverse=True):
        print(
            f"{r['label']:<24} {r['retrieval_mode']:<8} "
            f"{'yes' if r['query_expansion'] else 'no':>4} "
            f"{float(r['R@1'] or 0):6.2f} {float(r['R@10'] or 0):6.2f} "
            f"{float(r['R@100'] or 0):7.2f} {float(r['R@1000'] or 0):7.2f} "
            f"{float(r['MRR@10'] or 0):7.2f} {r['index_document_count'] or '?':>10}"
        )

    if invalid:
        print("\n=== Invalid runs ===")
        for r in invalid:
            print(f"  {r['label']}: status={r['status']} index_docs={r.get('index_document_count')}")

    print("\n=== Direct paper comparison (R@10, percentage points) ===")
    print(f"{'Config':<28} {'Ours':>7} {'Paper':>7} {'Gap':>8}")
    print("-" * 54)
    by_label = {r["label"]: r for r in valid}
    for label, (ref_name, ref_val) in PAPER.items():
        row = by_label.get(label)
        if not row:
            print(f"{label:<28} {'—':>7} {ref_val:7.2f} {'pending':>8}")
            continue
        ours = float(row["R@10"] or 0)
        gap = ours - ref_val
        print(f"{label:<28} {ours:7.2f} {ref_val:7.2f} {gap:+8.2f}pp  ({ref_name})")

    # Prior child-chunk BM25 from diagnostic
    prior_bm25 = PRIOR_CHILD_CHUNK["hier-bm25"]
    if args.prior_diag_dir.is_dir():
        for f in args.prior_diag_dir.glob("paper_housing_*hier-bm25*.json"):
            d = json.loads(f.read_text())
            if d.get("status") == "valid":
                prior_bm25 = float((d.get("recall_passage_percent") or {}).get("10", prior_bm25))
                break

    flat_bm25 = by_label.get("flat-bm25-baseline")
    if flat_bm25 and prior_bm25 is not None:
        flat_r10 = float(flat_bm25["R@10"] or 0)
        print("\n=== Child-chunk vs flat BM25 (chunking hypothesis) ===")
        print(f"  Prior hier/rec child-chunk BM25 R@10: {prior_bm25:.2f}%")
        print(f"  Flat statute-level BM25 R@10:         {flat_r10:.2f}%")
        print(f"  Lift from flat indexing:              {flat_r10 - prior_bm25:+.2f}pp")
        print(f"  Remaining gap to paper (40.8%):      {flat_r10 - 40.8:+.2f}pp")

    out = args.out or (args.results_dir / "summary_housing_paper_flat.json")
    out.write_text(
        json.dumps({"valid": valid, "invalid": invalid, "paper_refs": PAPER}, indent=2) + "\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
