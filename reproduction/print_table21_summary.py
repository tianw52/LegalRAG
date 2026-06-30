#!/usr/bin/env python3
"""Print final Table 22 reproduction once all SLURM evaluation jobs have finished.

Loads per-model metric JSON files written by reproduce_table21_baseline.py,
then prints a three-section comparison table:
  1. Our reproduced numbers
  2. Paper Table 22 reference values
  3. Absolute difference (ours − paper) for every metric

Usage:
    python reproduction/print_table21_summary.py
    python reproduction/print_table21_summary.py --results-dir /path/to/table21_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

RECALL_CUTOFFS = (1, 10, 100, 1000)
MRR_CUTOFF = 10

MODEL_ORDER = ["bm25", "e5-small", "e5-base", "e5-large", "e5-mistral"]
MODEL_DISPLAY = {
    "bm25":       "BM25",
    "e5-small":   "E5-small-v2",
    "e5-base":    "E5-base-v2",
    "e5-large":   "E5-large-v2",
    "e5-mistral": "E5-mistral-7b",
}

# Paper Table 22 (Historical MBE subset) — baseline rows only
# Columns: (R@1, R@10, MRR@10, R@100, R@1000)
PAPER_TABLE22: dict[str, tuple[float, float, float, float, float]] = {
    "bm25":       (0.25, 0.75,  0.37,  2.26,  8.79),
    "e5-small":   (0.08, 0.59,  0.18,  2.68,  9.29),
    "e5-base":    (0.25, 0.84,  0.39,  3.51, 11.21),
    "e5-large":   (0.17, 0.92,  0.34,  4.27, 12.30),
    "e5-mistral": (0.84, 3.26,  1.45,  9.71, 26.36),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

METRIC_KEYS = ["recall@1", "recall@10", "mrr@10", "recall@100", "recall@1000"]
COL_HDR = (
    f"  {'Method':<20}  "
    f"{'R@1':>6}  {'R@10':>6}  {'MRR@10':>7}  {'R@100':>7}  {'R@1000':>8}"
)
SEP = "  " + "─" * 70


def _val_row(display: str, vals: tuple[float, ...]) -> str:
    return (
        f"  {display:<20}  "
        f"{vals[0]:6.2f}  {vals[1]:6.2f}  {vals[2]:7.2f}  {vals[3]:7.2f}  {vals[4]:8.2f}"
    )


def _diff_row(display: str, ours: tuple[float, ...], ref: tuple[float, ...]) -> str:
    diffs = tuple(o - r for o, r in zip(ours, ref))

    def _fmt(d: float) -> str:
        return f"{d:+.2f}"

    return (
        f"  {display:<20}  "
        f"{_fmt(diffs[0]):>6}  {_fmt(diffs[1]):>6}  {_fmt(diffs[2]):>7}  "
        f"{_fmt(diffs[3]):>7}  {_fmt(diffs[4]):>8}"
    )


def load_metrics(results_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in MODEL_ORDER:
        fname = "bm25_metrics.json" if key == "bm25" else f"{key}_metrics.json"
        f = results_dir / fname
        if f.exists():
            out[key] = json.loads(f.read_text())
    return out


def extract_vals(m: dict) -> tuple[float, ...]:
    return (
        m.get("recall@1",    0.0),
        m.get("recall@10",   0.0),
        m.get("mrr@10",      0.0),
        m.get("recall@100",  0.0),
        m.get("recall@1000", 0.0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "table21_results",
        help="Directory containing *_metrics.json files (default: reproduction/table21_results/).",
    )
    args = p.parse_args()

    metrics = load_metrics(args.results_dir)
    done = [k for k in MODEL_ORDER if k in metrics]
    missing = [k for k in MODEL_ORDER if k not in metrics]

    W = 74
    print()
    print("=" * W)
    print("  Table 22 Reproduction — Bar Exam QA, Historical MBE Subset")
    print("  Baseline Retrievers Only (no query expansion)")
    print("  Zheng et al., CS&Law 2025  (arXiv:2505.03970)")
    print("=" * W)

    if not done:
        print(f"\n  No results found in {args.results_dir}")
        print("  Jobs are still running — check with:  squeue -u $USER")
        print()
        return

    # ── Section 1: Our reproduced numbers ─────────────────────────────────────
    print(f"\n  ① OUR RESULTS  (reglab/barexam_qa · Historical MBE · no query expansion)")
    print(COL_HDR)
    print(SEP)
    our: dict[str, tuple[float, ...]] = {}
    for key in done:
        v = extract_vals(metrics[key])
        our[key] = v
        print(_val_row(MODEL_DISPLAY[key], v))
    if missing:
        for key in missing:
            print(f"  {MODEL_DISPLAY[key]:<20}  {'(job pending or failed)':>52}")

    # ── Section 2: Paper Table 22 reference ───────────────────────────────────
    print(f"\n  ② PAPER Table 22  (Zheng et al. 2025 — Historical MBE, baseline rows)")
    print(COL_HDR)
    print(SEP)
    for key in MODEL_ORDER:
        print(_val_row(MODEL_DISPLAY[key], PAPER_TABLE22[key]))

    # ── Section 3: Absolute differences ───────────────────────────────────────
    print(f"\n  ③ ABSOLUTE DIFFERENCE  (ours − paper, positive = above paper)")
    print(COL_HDR)
    print(SEP)
    for key in done:
        print(_diff_row(MODEL_DISPLAY[key], our[key], PAPER_TABLE22[key]))
    if missing:
        for key in missing:
            print(f"  {MODEL_DISPLAY[key]:<20}  {'(not yet available)':>52}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    if missing:
        print(f"\n  Still waiting for: {[MODEL_DISPLAY[k] for k in missing]}")
        print(f"  Check job status:  squeue -u $USER")
        print(f"  Re-run this script when all jobs complete.")
    else:
        print("\n  All 5 models complete.")
    print()


if __name__ == "__main__":
    main()
