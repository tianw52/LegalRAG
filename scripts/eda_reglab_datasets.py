#!/usr/bin/env python3
"""Exploratory analysis for Stanford REGLab legal QA datasets (Hugging Face).

Loads data directly from the dataset repos (CSV/TSV/JSON) so you do not need the
``datasets`` package or PyArrow — useful on clusters where PyArrow wheels are
restricted.

Sources (CC BY-SA 4.0):
  - https://huggingface.co/datasets/reglab/barexam_qa
  - https://huggingface.co/datasets/reglab/housing_qa

Recommended on Alliance / Compute Canada login nodes::

    module load scipy-stack/2024b
    python scripts/eda_reglab_datasets.py --out-dir results/eda_reglab

Outputs PNG figures and ``eda_summary.txt`` under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAREXAM_QA_CSV = (
    "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main/data/qa/qa.csv"
)
BAREXAM_PASSAGES_TSV = (
    "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main/data/passages/passages.tsv"
)
HOUSING_QUESTIONS_ZIP = (
    "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/questions.json.zip"
)
HOUSING_QUESTIONS_AUX_ZIP = (
    "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/questions_aux.json.zip"
)
HOUSING_STATUTES_TSV = (
    "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/statutes.tsv"
)


def _len_str(x: object) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return len(str(x))


def _fetch_json_list_from_zip(url: str) -> list[dict]:
    raw = urllib.request.urlopen(url, timeout=300).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    inner = zf.namelist()[0]
    return json.loads(zf.read(inner).decode("utf-8"))


def plot_hist_log(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 60,
    color: str = "#2c5282",
) -> None:
    vals = values[values > 0]
    if vals.size == 0:
        ax.text(0.5, 0.5, "no positive values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(np.log10(vals.astype(float)), bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(f"log10({xlabel})")
    ax.set_ylabel("count")


def plot_bar_counts(
    ax: plt.Axes,
    labels: list[str],
    counts: list[int],
    title: str,
    xlabel: str,
    rotate: bool = False,
) -> None:
    ax.bar(range(len(labels)), counts, color="#2b6cb0", edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xlabel(xlabel)


def eda_barexam(out_dir: Path, passage_rows: int) -> None:
    print("Loading Bar Exam QA (qa.csv)…")
    qa = pd.read_csv(BAREXAM_QA_CSV)
    lines: list[str] = []
    lines.append("=== reglab/barexam_qa · qa ===")
    lines.append(f"rows: {len(qa):,}")
    lines.append(f"columns: {list(qa.columns)}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ans = qa["answer"].astype(str).str.upper().value_counts()
    order = ["A", "B", "C", "D"]
    ac = [int(ans.get(k, 0)) for k in order]
    plot_bar_counts(axes[0, 0], order, ac, "Correct answer label (MCQ)", "answer")

    subj = qa["subject"].fillna("(missing)").astype(str)
    top_subj = subj.value_counts().head(12)
    plot_bar_counts(
        axes[0, 1],
        list(top_subj.index),
        [int(x) for x in top_subj.values],
        "Subject (top 12; missing grouped)",
        "subject",
        rotate=True,
    )

    src = qa["source"].astype(str).value_counts().head(12)
    plot_bar_counts(
        axes[1, 0],
        list(src.index),
        [int(x) for x in src.values],
        "Dataset source (top 12)",
        "source",
        rotate=True,
    )

    gpl = qa["gold_passage"].map(_len_str).values
    plot_hist_log(axes[1, 1], np.asarray(gpl, dtype=int), "Gold passage character length", "chars")
    fig.suptitle("Bar Exam QA — task-level summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "barexam_qa_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Text lengths: question, prompt, choices total
    q_lens = qa["question"].map(_len_str).values
    p_lens = qa["prompt"].map(_len_str).values
    choice_sum = (
        qa["choice_a"].map(_len_str)
        + qa["choice_b"].map(_len_str)
        + qa["choice_c"].map(_len_str)
        + qa["choice_d"].map(_len_str)
    ).values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plot_hist_log(axes[0], np.asarray(p_lens, dtype=int), "Prompt length", "chars")
    plot_hist_log(axes[1], np.asarray(q_lens, dtype=int), "Question length", "chars")
    plot_hist_log(axes[2], np.asarray(choice_sum, dtype=int), "All four choices (sum) length", "chars")
    fig.suptitle("Bar Exam QA — text size (characters)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "barexam_qa_text_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines.append(
        f"gold_passage chars: min={int(np.min(gpl))} median={int(np.median(gpl))} max={int(np.max(gpl))}"
    )

    # Passages pool sample
    print(f"Sampling Bar Exam passages (first {passage_rows:,} rows of passages.tsv)…")
    psg = pd.read_csv(BAREXAM_PASSAGES_TSV, sep="\t", nrows=passage_rows)
    lines.append("")
    lines.append(f"=== reglab/barexam_qa · passages (first {len(psg):,} rows only) ===")
    lines.append(
        "(Passages are read from the start of passages.tsv; distribution should be stable "
        "at large N but compare with full ~900K if needed.)"
    )
    lines.append(f"Unique idx in sample: {psg['idx'].nunique():,}")
    src2 = psg["source"].astype(str).value_counts().head(15)
    lines.append("top sources in sample:\n" + src2.to_string())

    tl = psg["text"].map(_len_str).values
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_hist_log(ax, np.asarray(tl, dtype=int), "Passage text length (sample)", "chars")
    fig.suptitle(
        f"Bar Exam passages — text length (first {len(psg):,} / ~900K rows)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "barexam_passages_text_length_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lbl = list(src2.index)
    plot_bar_counts(ax, lbl, [int(x) for x in src2.values], "Passage source (sample, top 15)", "source", rotate=True)
    fig.tight_layout()
    fig.savefig(out_dir / "barexam_passages_source_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "eda_summary.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def eda_housing(out_dir: Path, statute_rows: int) -> None:
    lines: list[str] = []
    print("Loading HousingQA questions…")
    qs = _fetch_json_list_from_zip(HOUSING_QUESTIONS_ZIP)
    qdf = pd.DataFrame(qs)
    lines.append("=== reglab/housing_qa · questions (questions.json.zip; HF loader uses split=test) ===")
    lines.append(f"rows: {len(qdf):,}")

    print("Loading HousingQA questions_aux…")
    aux = _fetch_json_list_from_zip(HOUSING_QUESTIONS_AUX_ZIP)
    lines.append(f"questions_aux rows: {len(aux):,}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    vc = qdf["answer"].astype(str).value_counts()
    labels = list(vc.index)
    plot_bar_counts(axes[0, 0], labels, [int(vc[l]) for l in labels], "Answer (Yes / No)", "answer")

    st_counts = qdf["state"].astype(str).value_counts().sort_values(ascending=False)
    top_n = 20
    st_top = st_counts.head(top_n)
    plot_bar_counts(
        axes[0, 1],
        list(st_top.index),
        [int(x) for x in st_top.values],
        f"Top {top_n} states by number of questions",
        "state",
        rotate=True,
    )

    qg = qdf["question_group"].value_counts().sort_values(ascending=False).head(20)
    plot_bar_counts(
        axes[1, 0],
        [str(i) for i in qg.index],
        [int(x) for x in qg.values],
        "question_group (top 20)",
        "group id",
        rotate=True,
    )

    n_stat = []
    excerpt_lens: list[int] = []
    for row in qs:
        sts = row.get("statutes") or []
        n_stat.append(len(sts))
        for s in sts:
            excerpt_lens.append(_len_str(s.get("excerpt")))
    mx = max(n_stat) if n_stat else 0
    bins = np.arange(0, min(mx + 2, 40)) - 0.5
    axes[1, 1].hist(
        n_stat,
        bins=bins if mx < 35 else 30,
        color="#276749",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1, 1].set_title("Supporting statute excerpts per question")
    axes[1, 1].set_xlabel("# statute excerpts annotated")
    axes[1, 1].set_ylabel("count")

    fig.suptitle("HousingQA — questions subset", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "housing_qa_questions_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    el = np.asarray(excerpt_lens, dtype=int)
    plot_hist_log(ax, el, "Annotated statute excerpt length", "chars")
    fig.suptitle("HousingQA — gold excerpt sizes (in questions[].statutes)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "housing_qa_excerpt_lengths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines.append(
        f"statute excerpts per Q: min={min(n_stat)} max={max(n_stat)} "
        f"mean={sum(n_stat)/len(n_stat):.2f}"
    )
    if excerpt_lens:
        el_arr = np.asarray(excerpt_lens)
        lines.append(
            f"excerpt chars: median={int(np.median(el_arr))} max={int(np.max(el_arr))}"
        )

    # Statutes corpus sample
    print(f"Sampling housing statutes.tsv (first {statute_rows:,} rows)…")
    st = pd.read_csv(HOUSING_STATUTES_TSV, sep="\t", nrows=statute_rows)
    lines.append("")
    lines.append(f"=== reglab/housing_qa · statutes (first {len(st):,} rows; full ~1.7M) ===")
    lines.append(
        "(Sequential rows from the TSV can skew the state mix vs. the full corpus; "
        "increase --statute-sample-rows or shuffle offline for a fairer estimate.)"
    )
    st_state = st["state"].astype(str).value_counts().head(20)
    lines.append("top states in sample:\n" + st_state.to_string())

    tlen = st["text"].map(_len_str).values
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plot_hist_log(ax, np.asarray(tlen, dtype=int), "Statute text length (sample)", "chars")
    fig.suptitle(
        f"Housing statutes — text length (first {len(st):,} rows)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "housing_statutes_text_length_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_bar_counts(
        ax,
        list(st_state.index),
        [int(x) for x in st_state.values],
        "Statutes corpus — state (sample, top 20)",
        "state",
        rotate=True,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "housing_statutes_state_sample.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "eda_summary.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/eda_reglab"),
        help="Directory for PNGs and eda_summary.txt",
    )
    ap.add_argument(
        "--passage-sample-rows",
        type=int,
        default=200_000,
        help="Rows to read from barexam passages.tsv (full pool is ~900K).",
    )
    ap.add_argument(
        "--statute-sample-rows",
        type=int,
        default=250_000,
        help="Rows to read from housing statutes.tsv (full corpus ~1.7M).",
    )
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "eda_summary.txt"
    summary_path.write_text(
        "REGLab dataset EDA\n"
        "==================\n"
        "barexam_qa: https://huggingface.co/datasets/reglab/barexam_qa\n"
        "housing_qa: https://huggingface.co/datasets/reglab/housing_qa\n\n",
        encoding="utf-8",
    )
    eda_barexam(out, passage_rows=max(1, args.passage_sample_rows))
    eda_housing(out, statute_rows=max(1, args.statute_sample_rows))
    print(f"Done. Figures and summary in: {out}")


if __name__ == "__main__":
    main()
