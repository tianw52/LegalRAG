#!/usr/bin/env python3
"""
Plot evaluation results for the BarExam full-corpus retrieval experiment.

Generates a 2×2 figure (Character-Level Precision@K, Character-Level Recall@K,
Chunk-Level Precision@K, Chunk-Level Recall@K) matching Tian's original figure
style and color palette exactly, for each chunking strategy separately.

Colors were extracted via pixel analysis of Tian's original figures:
  - all-mpnet-base-v2:            #0000ff  (pure blue)
  - BERT-DPR-CLERC-ft:            #aaaaaa  (light gray)
  - legal-bert-base-uncased:      #555555  (dark gray)
  - Legal-Embed-bge-base-en-v1.5: #007800  (dark green)
  - Octen-Embedding-0.6B:         #ff69b4  (hot pink)
  - Qwen3-Embedding-0.6B:         #ffbec8  (light pink)

Usage (from LegalRAG root):
  module load scipy-stack
  python scripts/plot_barexam_eval_results.py
  python scripts/plot_barexam_eval_results.py \\
      --results-dir /scratch/ram112/reglab_eval/results \\
      --out-dir results/barexam_eval_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Model display names, order, colors, markers, line styles
# Matched to Tian's original plots (pixel-analysed from Tian's figures)
# ---------------------------------------------------------------------------

# Legend order matches Tian's reference screenshot exactly
MODEL_ORDER = [
    "all-mpnet-base-v2",
    "BERT-DPR-CLERC-ft",
    "legal-bert-base-uncased",
    "Legal-Embed-bge-base-en-v1.5",
    "Octen-Embedding-0.6B",
    "Qwen3-Embedding-0.6B",
]

# Colors pixel-extracted from Tian's original figures
MODEL_COLORS = {
    "all-mpnet-base-v2":             "#0000ff",   # pure blue
    "BERT-DPR-CLERC-ft":             "#aaaaaa",   # light gray
    "legal-bert-base-uncased":       "#555555",   # dark gray
    "Legal-Embed-bge-base-en-v1.5":  "#007800",   # dark green
    "Octen-Embedding-0.6B":          "#ff69b4",   # hot pink
    "Qwen3-Embedding-0.6B":          "#ffbec8",   # light pink
}

# Markers matching Tian's notebook (index 0-5 → o, s, ^, D, v, *)
MARKERS = {
    "all-mpnet-base-v2":             "o",   # circle
    "BERT-DPR-CLERC-ft":             "s",   # square
    "legal-bert-base-uncased":       "^",   # triangle-up
    "Legal-Embed-bge-base-en-v1.5":  "D",   # diamond
    "Octen-Embedding-0.6B":          "v",   # triangle-down
    "Qwen3-Embedding-0.6B":          "*",   # star
}

# Linestyles matching Tian's notebook (index 0-5 → -, --, -., :, -, --)
LINESTYLES = {
    "all-mpnet-base-v2":             "-",
    "BERT-DPR-CLERC-ft":             "--",
    "legal-bert-base-uncased":       "-.",
    "Legal-Embed-bge-base-en-v1.5":  ":",
    "Octen-Embedding-0.6B":          "-",
    "Qwen3-Embedding-0.6B":          "--",
}

# Mapping from results-dir model folder name → display label
MODEL_DIR_TO_LABEL = {
    "mpnet":     "all-mpnet-base-v2",
    "clerc":     "BERT-DPR-CLERC-ft",
    "legalbert": "legal-bert-base-uncased",
    "legal-bge": "Legal-Embed-bge-base-en-v1.5",
    "octen":     "Octen-Embedding-0.6B",
    "qwen3":     "Qwen3-Embedding-0.6B",
}

KS = [2, 4, 6, 10, 15, 20, 40, 60]


# ---------------------------------------------------------------------------
# Metric extraction from trace JSONL files
# ---------------------------------------------------------------------------

def parse_trace(fpath: Path) -> dict[int, dict[str, float]]:
    """
    Parse a char-eval .jsonl trace file and return per-K overall metrics:
      { k: { char_recall, char_precision, chunk_recall, chunk_precision } }
    Each metric is the macro-average (mean over queries).
    """
    accum: dict[int, dict[str, float]] = defaultdict(
        lambda: {"char_recall": 0.0, "char_precision": 0.0,
                 "chunk_recall": 0.0, "chunk_precision": 0.0, "n": 0}
    )

    with open(fpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "metrics_by_k" not in rec:
                continue

            gt_snippets = rec.get("ground_truth") or []
            n_gt = len(gt_snippets) if gt_snippets else 1

            for entry in rec["metrics_by_k"]:
                k = entry["k"]
                a = accum[k]
                a["char_recall"]    += entry.get("char_recall", 0.0)
                a["char_precision"] += entry.get("char_precision", 0.0)
                a["n"]              += 1

                chunks = entry.get("top_k_chunks") or []
                if chunks:
                    # ChunkPrecision@K = hits / K
                    n_hits = sum(1 for c in chunks if c.get("gt_overlap"))
                    a["chunk_precision"] += n_hits / len(chunks)

                    # ChunkRecall@K = covered GT snippets / n_gt
                    covered_files = {c["file"] for c in chunks if c.get("gt_overlap")}
                    if gt_snippets:
                        covered = sum(1 for g in gt_snippets
                                      if g["file"] in covered_files)
                        a["chunk_recall"] += covered / n_gt
                    else:
                        a["chunk_recall"] += float(n_hits > 0)

    result: dict[int, dict[str, float]] = {}
    for k, v in sorted(accum.items()):
        n = v["n"]
        if n == 0:
            continue
        result[k] = {
            "char_recall":     v["char_recall"]     / n,
            "char_precision":  v["char_precision"]  / n,
            "chunk_recall":    v["chunk_recall"]    / n,
            "chunk_precision": v["chunk_precision"] / n,
        }
    return result


def load_all_results(
    results_dir: Path,
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """
    Returns { label: { 'hier': {k: metrics}, 'rec': {k: metrics} } }
    """
    data: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    for dir_name, label in MODEL_DIR_TO_LABEL.items():
        mdir = results_dir / dir_name
        if not mdir.is_dir():
            print(f"WARN: missing results dir {mdir}", file=sys.stderr)
            continue

        for tag, pattern in [("hier", "char_barexam_hier_*.jsonl"),
                              ("rec",  "char_barexam_rec_*.jsonl")]:
            files = sorted(mdir.glob(pattern))
            if not files:
                print(f"WARN: no {pattern} in {mdir}", file=sys.stderr)
                continue
            fpath = files[-1]  # pick latest job ID
            print(f"  Loading {label:40s} / {tag}: {fpath.name}")
            metrics = parse_trace(fpath)
            data.setdefault(label, {})[tag] = metrics

    return data


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _get_series(
    data: dict[str, dict[str, dict[int, dict[str, float]]]],
    chunker: str,
    metric: str,
) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}
    for label in MODEL_ORDER:
        entry = data.get(label, {}).get(chunker, {})
        if not entry:
            continue
        ks_vals = [(k, entry[k][metric]) for k in KS if k in entry]
        if ks_vals:
            ks, vals = zip(*ks_vals)
            series[label] = (list(ks), list(vals))
    return series


def _draw_panel(
    ax: "plt.Axes",
    data: dict[str, dict[str, dict[int, dict[str, float]]]],
    chunker: str,
    metric_key: str,
    ylabel: str,
    title: str,
    legend_loc: str = "best",
) -> None:
    series = _get_series(data, chunker, metric_key)

    for label in MODEL_ORDER:
        if label not in series:
            continue
        ks, vals = series[label]
        ax.plot(
            ks, vals,
            label=label,
            color=MODEL_COLORS[label],
            marker=MARKERS[label],
            linestyle=LINESTYLES[label],
            linewidth=2.5,
            markersize=8,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("K (Retrieval Depth)", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc=legend_loc, frameon=True, shadow=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)


def plot_2x2(
    data: dict[str, dict[str, dict[int, dict[str, float]]]],
    chunker: str,
    out_png: Path,
    dpi: int = 150,
) -> None:
    """2×2 figure matching Tian's layout: CharPrec|CharRecall / ChunkPrec|ChunkRecall."""
    chunker_label = "Hierarchical" if chunker == "hier" else "Recursive"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panel_defs = [
        (axes[0, 0], "char_precision",  "CharPrecision@K",
         "Character-Level Precision@K — Overall", "upper right"),
        (axes[0, 1], "char_recall",     "CharRecall@K",
         "Character-Level Recall@K — Overall", "upper left"),
        (axes[1, 0], "chunk_precision", "ChunkPrecision@K",
         "Chunk-Level Precision@K — Overall", "upper right"),
        (axes[1, 1], "chunk_recall",    "ChunkRecall@K",
         "Chunk-Level Recall@K — Overall", "upper left"),
    ]

    for ax, metric_key, ylabel, title, legend_loc in panel_defs:
        _draw_panel(ax, data, chunker, metric_key, ylabel, title, legend_loc)

    fig.suptitle(
        f"BarExam Full-Corpus Evaluation — {chunker_label} Chunking",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="/scratch/ram112/reglab_eval/results",
        help="Directory containing per-model result subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        default="results/barexam_eval_figures",
        help="Output directory for PNG figures",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    print("Loading results...")
    data = load_all_results(results_dir)

    # Two separate figures: hierarchical and recursive
    for chunker in ("hier", "rec"):
        label = "hierarchical" if chunker == "hier" else "recursive"
        plot_2x2(data, chunker, out_dir / f"barexam_{label}_2x2.png", dpi=args.dpi)

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
