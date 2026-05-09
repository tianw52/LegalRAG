#!/usr/bin/env python3
"""
Plot CharRecall@K / CharPrecision@K for LegalBench benchmark-50 reformulated runs.

Numeric tables are copied verbatim from the author's benchmark summary (same figures as numbers.pdf /
slide tables): Mistral + Qwen72b reformulations × Legal-Embed-bge-base vs BERT-DPR-CLERC-ft × RTCS vs
hierarchical. MAUD row for Qwen hierarchical recall uses **0.0019** at K=10 (PDF typo `0.0$019`).

Requires: pip install matplotlib

Usage (from repo root):
  python scripts/visualize_benchmark50_reformulated_results.py
  python scripts/visualize_benchmark50_reformulated_results.py --out-dir docs/experiments/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

KS = (1, 5, 10, 20)
KS_POS = np.arange(len(KS))
BENCHMARKS = ("contractnli", "cuad", "maud", "privacy_qa", "OVERALL")
BENCH_LABELS = {
    "contractnli": "ContractNLI",
    "cuad": "CUAD",
    "maud": "MAUD",
    "privacy_qa": "Privacy QA",
    "OVERALL": "OVERALL",
}

# fmt: off
# Keys: reformulator -> embedding_family -> chunking -> metric -> benchmark -> tuple(K=1,5,10,20)
# Cross-check OVERALL CharRecall@K=20: Mistral LE 0.1273 / 0.1260; Mistral DPR 0.0675 / 0.0475;
# Qwen LE 0.1221 / 0.1298; Qwen DPR 0.0715 / 0.0852 (RTCS / hierarchical).
RAW: dict[str, dict[str, dict[str, dict[str, dict[str, tuple[float, float, float, float]]]]]] = {
    "mistral": {
        "legal_embed": {
            "rtcs": {
                "recall": {
                    "contractnli": (0.0029, 0.0413, 0.0535, 0.0601),
                    "cuad": (0.0003, 0.0481, 0.0978, 0.1680),
                    "maud": (0.0000, 0.0199, 0.0199, 0.0199),
                    "privacy_qa": (0.0420, 0.1295, 0.1927, 0.2611),
                    "OVERALL": (0.0113, 0.0597, 0.0910, 0.1273),
                },
                "precision": {
                    "contractnli": (0.0030, 0.0097, 0.0071, 0.0040),
                    "cuad": (0.0047, 0.0135, 0.0123, 0.0119),
                    "maud": (0.0000, 0.0029, 0.0015, 0.0008),
                    "privacy_qa": (0.0640, 0.0502, 0.0364, 0.0244),
                    "OVERALL": (0.0179, 0.0191, 0.0143, 0.0103),
                },
            },
            "hierarchical": {
                "recall": {
                    "contractnli": (0.0106, 0.0788, 0.0908, 0.1098),
                    "cuad": (0.0003, 0.0538, 0.0932, 0.1103),
                    "maud": (0.0000, 0.0200, 0.0200, 0.0200),
                    "privacy_qa": (0.0296, 0.1516, 0.2085, 0.2642),
                    "OVERALL": (0.0101, 0.0760, 0.1031, 0.1260),
                },
                "precision": {
                    "contractnli": (0.0091, 0.0120, 0.0078, 0.0047),
                    "cuad": (0.0027, 0.0169, 0.0164, 0.0093),
                    "maud": (0.0000, 0.0024, 0.0015, 0.0007),
                    "privacy_qa": (0.0355, 0.0418, 0.0312, 0.0197),
                    "OVERALL": (0.0118, 0.0183, 0.0142, 0.0086),
                },
            },
        },
        "dpr_clerc": {
            "rtcs": {
                "recall": {
                    "contractnli": (0.0029, 0.0047, 0.0178, 0.0443),
                    "cuad": (0.0043, 0.0069, 0.0341, 0.0935),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0000),
                    "privacy_qa": (0.0269, 0.0529, 0.0994, 0.1322),
                    "OVERALL": (0.0085, 0.0161, 0.0378, 0.0675),
                },
                "precision": {
                    "contractnli": (0.0030, 0.0020, 0.0032, 0.0027),
                    "cuad": (0.0109, 0.0023, 0.0060, 0.0069),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0000),
                    "privacy_qa": (0.0377, 0.0191, 0.0208, 0.0134),
                    "OVERALL": (0.0129, 0.0059, 0.0075, 0.0058),
                },
            },
            "hierarchical": {
                "recall": {
                    "contractnli": (0.0095, 0.0118, 0.0158, 0.0215),
                    "cuad": (0.0000, 0.0048, 0.0193, 0.0696),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0000),
                    "privacy_qa": (0.0287, 0.0574, 0.0893, 0.0988),
                    "OVERALL": (0.0096, 0.0185, 0.0311, 0.0475),
                },
                "precision": {
                    "contractnli": (0.0080, 0.0033, 0.0023, 0.0014),
                    "cuad": (0.0000, 0.0023, 0.0029, 0.0074),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0000),
                    "privacy_qa": (0.0332, 0.0168, 0.0141, 0.0079),
                    "OVERALL": (0.0103, 0.0056, 0.0048, 0.0042),
                },
            },
        },
    },
    "qwen72b": {
        "legal_embed": {
            "rtcs": {
                "recall": {
                    "contractnli": (0.0050, 0.0279, 0.0401, 0.0651),
                    "cuad": (0.0249, 0.0585, 0.0801, 0.1723),
                    "maud": (0.0000, 0.0025, 0.0025, 0.0048),
                    "privacy_qa": (0.0532, 0.1657, 0.1931, 0.2460),
                    "OVERALL": (0.0208, 0.0637, 0.0789, 0.1221),
                },
                "precision": {
                    "contractnli": (0.0069, 0.0078, 0.0064, 0.0046),
                    "cuad": (0.0100, 0.0061, 0.0067, 0.0096),
                    "maud": (0.0000, 0.0026, 0.0013, 0.0007),
                    "privacy_qa": (0.1017, 0.0634, 0.0423, 0.0266),
                    "OVERALL": (0.0297, 0.0200, 0.0142, 0.0104),
                },
            },
            "hierarchical": {
                "recall": {
                    "contractnli": (0.0132, 0.0673, 0.0862, 0.1468),
                    "cuad": (0.0000, 0.0628, 0.0852, 0.1323),
                    "maud": (0.0000, 0.0000, 0.0019, 0.0085),
                    "privacy_qa": (0.0295, 0.1340, 0.2110, 0.2316),
                    "OVERALL": (0.0107, 0.0660, 0.0961, 0.1298),
                },
                "precision": {
                    "contractnli": (0.0143, 0.0175, 0.0107, 0.0081),
                    "cuad": (0.0000, 0.0118, 0.0098, 0.0095),
                    "maud": (0.0000, 0.0000, 0.0004, 0.0005),
                    "privacy_qa": (0.0400, 0.0452, 0.0334, 0.0212),
                    "OVERALL": (0.0136, 0.0186, 0.0136, 0.0098),
                },
            },
        },
        "dpr_clerc": {
            "rtcs": {
                "recall": {
                    "contractnli": (0.0000, 0.0086, 0.0309, 0.0651),
                    "cuad": (0.0000, 0.0038, 0.0123, 0.0710),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0023),
                    "privacy_qa": (0.0381, 0.0732, 0.0909, 0.1475),
                    "OVERALL": (0.0095, 0.0214, 0.0335, 0.0715),
                },
                "precision": {
                    "contractnli": (0.0000, 0.0017, 0.0046, 0.0044),
                    "cuad": (0.0000, 0.0007, 0.0018, 0.0041),
                    "maud": (0.0000, 0.0000, 0.0000, 0.0001),
                    "privacy_qa": (0.0526, 0.0263, 0.0177, 0.0158),
                    "OVERALL": (0.0132, 0.0072, 0.0060, 0.0061),
                },
            },
            "hierarchical": {
                "recall": {
                    "contractnli": (0.0000, 0.0537, 0.0687, 0.0971),
                    "cuad": (0.0000, 0.0000, 0.0243, 0.0977),
                    "maud": (0.0000, 0.0019, 0.0019, 0.0085),
                    "privacy_qa": (0.0109, 0.0845, 0.0988, 0.1374),
                    "OVERALL": (0.0027, 0.0350, 0.0484, 0.0852),
                },
                "precision": {
                    "contractnli": (0.0000, 0.0110, 0.0077, 0.0060),
                    "cuad": (0.0000, 0.0000, 0.0035, 0.0058),
                    "maud": (0.0000, 0.0009, 0.0004, 0.0005),
                    "privacy_qa": (0.0109, 0.0254, 0.0166, 0.0131),
                    "OVERALL": (0.0027, 0.0093, 0.0071, 0.0064),
                },
            },
        },
    },
}
# fmt: on

TITLE_EMBED = {
    "mistral": "Legal-Embed-bge-base\n(Reformulated queries by Mistral)",
    "qwen72b": "Legal-Embed-bge-base\n(Reformulated queries by Qwen72b)",
}
TITLE_DPR = {
    "mistral": "ft-BERT-DB ERPR (CLERC)\n(Reformulated queries by Mistral)",
    "qwen72b": "ft-BERT-DB ERPR (CLERC)\n(Reformulated queries by Qwen72b)",
}


def plot_panel_group(
    reformulator: str,
    embed_key: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """2×5 grid: rows = recall / precision; columns = benchmarks; 2 lines each (RTCS vs hierarchical)."""
    data = RAW[reformulator][embed_key]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.8), sharex=True, sharey=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for row, metric in enumerate(("recall", "precision")):
        for col, bench in enumerate(BENCHMARKS):
            ax = axes[row, col]
            y_rtcs = data["rtcs"][metric][bench]
            y_hier = data["hierarchical"][metric][bench]
            ax.plot(KS_POS, y_rtcs, marker="o", linestyle="-", color="#1f77b4", label="RTCS", linewidth=2)
            ax.plot(
                KS_POS,
                y_hier,
                marker="s",
                linestyle="--",
                color="#ff7f0e",
                label="Hierarchical",
                linewidth=2,
            )
            # Label K=20 values so RTCS vs Hier. are not confused (e.g. contractnli: ~0.06 vs ~0.11).
            kx = float(KS_POS[-1])
            ax.annotate(
                f"{y_rtcs[-1]:.3f}",
                xy=(kx, y_rtcs[-1]),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=6,
                color="#1f77b4",
            )
            ax.annotate(
                f"{y_hier[-1]:.3f}",
                xy=(kx, y_hier[-1]),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=6,
                color="#ff7f0e",
            )
            ax.set_xticks(KS_POS)
            ax.set_xticklabels([str(k) for k in KS], fontsize=8)
            ymax = max(max(y_rtcs), max(y_hier), 1e-9)
            ax.set_ylim(0, ymax * 1.18)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(loc="upper left", fontsize=6, framealpha=0.92)
            if row == 0:
                ax.set_title(BENCH_LABELS[bench], fontsize=9, fontweight="bold")
            if col == 0:
                ylab = "CharRecall@K" if metric == "recall" else "CharPrecision@K"
                ax.set_ylabel(ylab, fontsize=9)
            if row == 1:
                ax.set_xlabel("K", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overall_bars_four_panel(out_path: Path, dpi: int) -> None:
    """Single figure 2x2: each panel = embedding×reformulator; grouped bars OVERALL @ K for RTCS vs Hier."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("mistral", "legal_embed", TITLE_EMBED["mistral"]),
        ("mistral", "dpr_clerc", TITLE_DPR["mistral"]),
        ("qwen72b", "legal_embed", TITLE_EMBED["qwen72b"]),
        ("qwen72b", "dpr_clerc", TITLE_DPR["qwen72b"]),
    ]
    width = 0.35
    for ax, (ref, emb, ttl) in zip(axes.flat, panels):
        r_rtcs = RAW[ref][emb]["rtcs"]["recall"]["OVERALL"]
        r_hier = RAW[ref][emb]["hierarchical"]["recall"]["OVERALL"]
        p_rtcs = RAW[ref][emb]["rtcs"]["precision"]["OVERALL"]
        p_hier = RAW[ref][emb]["hierarchical"]["precision"]["OVERALL"]
        x = KS_POS
        ax.bar(x - width / 2, r_rtcs, width, label="Recall RTCS", color="#2ecc71", alpha=0.85)
        ax.bar(x + width / 2, r_hier, width, label="Recall Hier.", color="#27ae60", alpha=0.55)
        ax2 = ax.twinx()
        ax2.plot(x, p_rtcs, color="#e74c3c", marker="o", label="Prec. RTCS", linewidth=2)
        ax2.plot(x, p_hier, color="#c0392b", marker="s", linestyle="--", label="Prec. Hier.", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in KS])
        ax.set_ylabel("CharRecall")
        ax2.set_ylabel("CharPrecision")
        ax.set_ylim(0, max(max(r_rtcs), max(r_hier)) * 1.15 + 1e-6)
        ax2.set_ylim(0, max(max(p_rtcs), max(p_hier)) * 1.2 + 1e-6)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=7, ncol=2)

    fig.suptitle("OVERALL metrics: recall (bars) + precision (lines)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/experiments/figures"),
        help="Directory for PNG outputs",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for ref in ("mistral", "qwen72b"):
        for emb, ttl, slug in (
            ("legal_embed", TITLE_EMBED[ref], "legal_embed"),
            ("dpr_clerc", TITLE_DPR[ref], "dpr_clerc"),
        ):
            path = out_dir / f"benchmark50_{ref}_{slug}_recall_precision_by_k.png"
            plot_panel_group(ref, emb, ttl, path, args.dpi)
            print(f"Wrote {path}")

    summary = out_dir / "benchmark50_overall_recall_bars_precision_lines_2x2.png"
    plot_overall_bars_four_panel(summary, args.dpi)
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
