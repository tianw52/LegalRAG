#!/usr/bin/env python3
"""Aggregate metrics JSONs from a 6-embedding × 2-chunker sweep and plot CharRecall@K / CharPrecision@K vs K.

Reads ``metrics/metrics_*.json`` (each file: one chunker × one embedding). Writes:

  - ``<out-prefix>_overall.csv`` — same schema as before
  - ``<out-prefix>_{recursive|hierarchical}_{recall|precision}_synced.png``

Expected embedding set (HF ids in ``run_b50_4embed_sweep_one_variant.sh``):

  - ``nlpaueb/legal-bert-base-uncased``, ``sentence-transformers/all-mpnet-base-v2``,
    ``axondendriteplus/Legal-Embed-bge-base-en-v1.5``, ``jhu-clsp/BERT-DPR-CLERC-ft``,
    ``Qwen/Qwen3-Embedding-0.6B``, ``Octen/Octen-Embedding-0.6B``

Use ``model_label`` matching ``MODEL_ORDER`` below (or ``embedding_model`` → label via ``EMBEDDING_MODEL_TO_LABEL``).

Usage::

  python3 scripts/aggregate_plot_mistral_4embed_sweep.py \\
    --metrics-dir results/mistral_4embed_sweep/metrics \\
    --out-prefix results/mistral_4embed_sweep/mistral_reform_4embed

  # Minimum y-axis maxima; expanded if data exceeds (avoids clipping):
  python3 scripts/aggregate_plot_mistral_4embed_sweep.py \\
    --metrics-dir results/qwen72b_4embed_sweep/metrics \\
    --out-prefix results/qwen72b_4embed_sweep/qwen72b_reform_4embed \\
    --ymax-precision 0.035 --ymax-recall 0.35

  # Serif / bold labels like paper figures; only hierarchical precision + recall PNGs:
  python3 scripts/aggregate_plot_mistral_4embed_sweep.py \\
    --metrics-dir results/qwen35_9b_4embed_sweep/metrics \\
    --out-prefix results/qwen35_9b_4embed_sweep/qwen35_9b_reform_4embed \\
    --ymax-precision 0.035 --ymax-recall 0.35 \\
    --publication-style \\
    --only-plots hierarchical_precision,hierarchical_recall

Requires: matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Must match plotting order and styling (same order as sweep steps in run_b50_4embed_sweep_one_variant.sh).
K_VALUES = [2, 4, 6, 10, 15, 20, 40, 60]
MODEL_ORDER = [
    "LegalBERT",
    "SBERT (all-mpnet-base-v2)",
    "Legal-Embed-bge-base",
    "BERT-DPR-CLERC-ft",
    "Qwen3",
    "Octen",
]
MODEL_COLORS = {
    "LegalBERT": "red",
    "SBERT (all-mpnet-base-v2)": "blue",
    "Legal-Embed-bge-base": "green",
    "BERT-DPR-CLERC-ft": "purple",
    "Qwen3": "pink",
    "Octen": "orange",
}
MARKERS = ["o", "s", "^", "D", "v", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]

# If eval writes HF id but short plot name is desired:
EMBEDDING_MODEL_TO_LABEL = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3",
    "Octen/Octen-Embedding-0.6B": "Octen",
}


def _canonical_label(obj: dict) -> str:
    lab = (obj.get("model_label") or "").strip()
    emb = (obj.get("embedding_model") or "").strip()
    if lab in MODEL_ORDER:
        return lab
    if emb in EMBEDDING_MODEL_TO_LABEL:
        return EMBEDDING_MODEL_TO_LABEL[emb]
    return lab or emb or "unknown"


def _chunker_tag(obj: dict) -> str:
    return (obj.get("chunker_tag") or "").strip()


def load_all_metrics(metrics_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(metrics_dir.glob("metrics_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skip {path}: {e}", file=sys.stderr)
            continue
        data["_path"] = str(path)
        out.append(data)
    return out


def write_overall_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = [
        "chunker_tag",
        "model_label",
        "embedding_model",
        "index_name",
        "k",
        "char_recall_overall",
        "char_precision_overall",
        "n_queries",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def metrics_to_csv_rows(records: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for obj in records:
        tag = _chunker_tag(obj)
        mlabel = _canonical_label(obj)
        emb = obj.get("embedding_model", "")
        idx = obj.get("index_name", "")
        ks = obj.get("ks") or []
        r = obj.get("char_recall_overall") or []
        p = obj.get("char_precision_overall") or []
        nq = obj.get("n_queries", "")
        if len(ks) != len(r) or len(ks) != len(p):
            print(
                f"WARN: length mismatch in {obj.get('_path')}: ks={len(ks)} r={len(r)} p={len(p)}",
                file=sys.stderr,
            )
            continue
        for k, rv, pv in zip(ks, r, p):
            flat.append(
                {
                    "chunker_tag": tag,
                    "model_label": mlabel,
                    "embedding_model": emb,
                    "index_name": idx,
                    "k": k,
                    "char_recall_overall": rv,
                    "char_precision_overall": pv,
                    "n_queries": nq,
                }
            )
    return flat


def _series_by_model(
    records: list[dict],
    chunker: str,
    metric: str,
) -> dict[str, tuple[list[int], list[float]]]:
    """chunker: 'recursive' | 'hierarchical'; metric: 'recall' | 'precision'."""
    key_arr = "char_recall_overall" if metric == "recall" else "char_precision_overall"
    series: dict[str, tuple[list[int], list[float]]] = {}
    for obj in records:
        if _chunker_tag(obj) != chunker:
            continue
        lab = _canonical_label(obj)
        ks = list(obj.get("ks") or [])
        vals = list(obj.get(key_arr) or [])
        if len(ks) != len(vals):
            continue
        series[lab] = (ks, vals)
    return series


def _publication_rc() -> dict:
    """Match reference figure: serif (Times-like), bold title & axis labels, readable ticks."""
    return {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Bitstream Vera Serif",
            "New Century Schoolbook",
            "Century Schoolbook L",
            "serif",
        ],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    }


def plot_one(
    records: list[dict],
    chunker: str,
    metric: str,
    out_png: Path,
    title_suffix: str,
    dpi: int,
    *,
    ymax_precision: float | None = None,
    ymax_recall: float | None = None,
    publication_style: bool = False,
) -> None:
    series = _series_by_model(records, chunker, metric)
    ordered = [m for m in MODEL_ORDER if m in series]
    if not ordered:
        print(f"WARN: no data for {chunker} {metric}; skip {out_png}", file=sys.stderr)
        return

    rc = _publication_rc() if publication_style else {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x_pos = np.arange(len(K_VALUES))
        k_to_pos = {k: i for i, k in enumerate(K_VALUES)}
        all_y_values: list[float] = []

        for model in ordered:
            ks, ys = series[model]
            xs: list[int] = []
            yv: list[float] = []
            for k, y in zip(ks, ys):
                if k in k_to_pos:
                    xs.append(k_to_pos[k])
                    yv.append(y)
                    all_y_values.append(float(y))
            if not xs:
                continue
            mi = MODEL_ORDER.index(model)
            ax.plot(
                xs,
                yv,
                label=model,
                color=MODEL_COLORS.get(model, "gray"),
                marker=MARKERS[mi % len(MARKERS)],
                linestyle=LINESTYLES[mi % len(LINESTYLES)],
                linewidth=2,
                markersize=7,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(k) for k in K_VALUES])
        if publication_style:
            ax.set_xlabel("K (Retrieval Depth)")
            ylab = (
                "Overall CharRecall@K (Macro-Avg)"
                if metric == "recall"
                else "Overall CharPrecision@K (Macro-Avg)"
            )
            ax.set_ylabel(ylab)
            base_title = (
                "Character-Level Recall@K Comparison"
                if metric == "recall"
                else "Character-Level Precision@K Comparison"
            )
            ttl = base_title
            if title_suffix:
                ttl = f"{base_title}\n{title_suffix}"
            ax.set_title(ttl)
        else:
            ax.set_xlabel("K")
            ylab = "CharRecall@K (OVERALL)" if metric == "recall" else "CharPrecision@K (OVERALL)"
            ax.set_ylabel(ylab)
            ch_label = "Recursive (RTCS)" if chunker == "recursive" else "Hierarchical"
            ttl = f"{ch_label} — {ylab}"
            if title_suffix:
                ttl = f"{ttl}\n{title_suffix}"
            ax.set_title(ttl, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
        # Upper bound = at least user's ymax (for comparable scale) but never clip peaks
        _ypad = 1.08
        if metric == "precision" and ymax_precision is not None:
            cap = float(ymax_precision)
            if all_y_values:
                cap = max(cap, max(all_y_values) * _ypad)
            ax.set_ylim(0.0, cap)
        elif metric == "recall" and ymax_recall is not None:
            cap = float(ymax_recall)
            if all_y_values:
                cap = max(cap, max(all_y_values) * _ypad)
            ax.set_ylim(0.0, cap)
        if publication_style:
            ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.35)
            ax.legend(
                loc="upper right",
                frameon=True,
                fancybox=True,
                shadow=True,
                framealpha=1.0,
                edgecolor="black",
            )
        else:
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(loc="best", fontsize=7, framealpha=0.92)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", type=Path, required=True, help="Directory containing metrics_*.json")
    p.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Output path prefix (no extension), e.g. results/mistral_4embed_sweep/mistral_reform_4embed",
    )
    p.add_argument("--plot-title-suffix", default="", help="Optional second line in plot titles")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument(
        "--ymax-precision",
        type=float,
        default=None,
        metavar="Y",
        help=(
            "Minimum y-axis max for precision plots (e.g. 0.035); min stays 0. "
            "If any series exceeds this, the axis extends automatically (no clipping)."
        ),
    )
    p.add_argument(
        "--ymax-recall",
        type=float,
        default=None,
        metavar="Y",
        help=(
            "Minimum y-axis max for recall plots (e.g. 0.35); min stays 0. "
            "If any series exceeds this, the axis extends automatically (no clipping)."
        ),
    )
    p.add_argument(
        "--publication-style",
        action="store_true",
        help=(
            "Serif font, bold title/axis labels, grid on both axes, legend with shadow "
            "(match paper-style figures)."
        ),
    )
    p.add_argument(
        "--only-plots",
        default="",
        metavar="LIST",
        help=(
            "Comma-separated subset to write, e.g. hierarchical_precision,hierarchical_recall. "
            "Each token is <chunker>_<metric> with chunker recursive|hierarchical and "
            "metric recall|precision. Empty = all four plots."
        ),
    )
    args = p.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    if not metrics_dir.is_dir():
        print(f"ERROR: not a directory: {metrics_dir}", file=sys.stderr)
        sys.exit(1)

    records = load_all_metrics(metrics_dir)
    if not records:
        print(f"ERROR: no metrics_*.json under {metrics_dir}", file=sys.stderr)
        sys.exit(1)

    rows = metrics_to_csv_rows(records)
    prefix = args.out_prefix
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    csv_path = Path(str(prefix) + "_overall.csv")
    write_overall_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    suffix = args.plot_title_suffix.strip()
    only_raw = (args.only_plots or "").strip()
    only_set: set[str] | None = None
    if only_raw:
        only_set = {t.strip().lower() for t in only_raw.split(",") if t.strip()}
        valid = {
            "recursive_recall",
            "recursive_precision",
            "hierarchical_recall",
            "hierarchical_precision",
        }
        bad = only_set - valid
        if bad:
            print(f"ERROR: unknown --only-plots token(s): {sorted(bad)}", file=sys.stderr)
            print(f"  expected one or more of: {sorted(valid)}", file=sys.stderr)
            sys.exit(1)

    for chunker, slug in (("recursive", "recursive"), ("hierarchical", "hierarchical")):
        for metric, mslug in (("recall", "recall"), ("precision", "precision")):
            key = f"{slug}_{mslug}"
            if only_set is not None and key not in only_set:
                continue
            out_png = Path(str(prefix) + f"_{slug}_{mslug}_synced.png")
            plot_one(
                records,
                chunker,
                metric,
                out_png,
                suffix,
                args.dpi,
                ymax_precision=args.ymax_precision,
                ymax_recall=args.ymax_recall,
                publication_style=args.publication_style,
            )


if __name__ == "__main__":
    main()
