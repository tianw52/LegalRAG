#!/usr/bin/env python3
"""Professional EDA report for REGLab barexam_qa + housing_qa (Hugging Face).

Loads CSV/TSV/zipped JSON over HTTPS (no ``datasets`` / PyArrow).

Outputs (default: ``LegalRAG/viz_results/``)::
  - Many high-DPI PNG figures (general EDA + **RAG chunk-budget** plots prefixed with ``rag_``)
  - ``summary_statistics.txt`` with distributional summaries and a **RAG CHUNK ANALYSIS** section

Chunk plots use a **fixed character grid** (default 512 chars), identical in spirit to
``scripts/plot_chunking_span_histograms.py`` for a span ``[0, L)``: number of chunks =
``ceil(L / chunk_size)`` when the passage starts at grid offset 0.

Example::

    module load scipy-stack/2024b
    cd LegalRAG && python scripts/reglab_dataset_visual_report.py \\
        --chunk-size 512 --chunk-hist-cap 80

CC BY-SA 4.0 dataset cards:
  https://huggingface.co/datasets/reglab/barexam_qa
  https://huggingface.co/datasets/reglab/housing_qa
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# -----------------------------------------------------------------------------
URLS = {
    "barexam_qa": "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main/data/qa/qa.csv",
    "barexam_passages": (
        "https://huggingface.co/datasets/reglab/barexam_qa/resolve/main/data/passages/passages.tsv"
    ),
    "housing_q": (
        "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/questions.json.zip"
    ),
    "housing_q_aux": (
        "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/questions_aux.json.zip"
    ),
    "housing_statutes": (
        "https://huggingface.co/datasets/reglab/housing_qa/resolve/main/data/statutes.tsv"
    ),
}

COLORS = {
    "primary": "#1a365d",
    "accent": "#2b6cb0",
    "ok": "#276749",
    "warn": "#c05621",
    "bad": "#9b2c2c",
    "muted": "#718096",
}


def _len_str(x: object) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return len(str(x))


def fetch_json_zip(url: str) -> list[dict]:
    raw = urllib.request.urlopen(url, timeout=300).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    return json.loads(zf.read(zf.namelist()[0]).decode("utf-8"))


def ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(a.astype(float))
    n = len(x)
    if n == 0:
        return x, x
    return x, np.arange(1, n + 1) / n


def describe_series(s: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return pd.Series({f"{name}_n": 0})
    return pd.Series(
        {
            f"{name}_n": len(s),
            f"{name}_mean": float(s.mean()),
            f"{name}_std": float(s.std()),
            f"{name}_min": float(s.min()),
            f"{name}_p05": float(s.quantile(0.05)),
            f"{name}_p25": float(s.quantile(0.25)),
            f"{name}_p50": float(s.quantile(0.50)),
            f"{name}_p75": float(s.quantile(0.75)),
            f"{name}_p95": float(s.quantile(0.95)),
            f"{name}_max": float(s.max()),
        }
    )


def save_fig(path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()


def cramers_v(conf: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(conf, correction=False)
    n = conf.values.sum()
    r, k = conf.shape
    if n == 0 or min(r, k) < 2:
        return float("nan")
    return float(math.sqrt(chi2 / (n * (min(k, r) - 1))))


def log_section(fh: TextIO, title: str) -> None:
    fh.write("\n")
    fh.write("=" * 72 + "\n")
    fh.write(f" {title}\n")
    fh.write("=" * 72 + "\n")


def chunks_needed_nonoverlap_chars(n_chars: np.ndarray | pd.Series, chunk_size: int) -> np.ndarray:
    """How many contiguous char buckets [0,cs), [cs,2cs), ... are touched by [0, n_chars).

    Same grid as ``plot_chunking_span_histograms.num_chunks_overlap_span(0, L, cs)``.
    Empty text → 0 buckets.
    """
    n = np.asarray(n_chars, dtype=np.int64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = (n + chunk_size - 1) // chunk_size
    return np.where(n <= 0, 0, out)


def _hist_chunks_bars(
    ax: plt.Axes,
    chunk_counts: np.ndarray,
    cap_display: int,
    title: str,
    xlabel: str,
    ylabel: str = "count",
    color: str = "#2b6cb0",
    annotate_max_if_merged: bool = True,
) -> tuple[int, int]:
    """Bar chart: x = #chunks, y = frequency. Last bar = ≥cap when tail is merged."""
    cc = np.asarray(chunk_counts, dtype=np.int64).ravel()
    cc = cc[cc >= 0]
    n_items = int(cc.size)
    if n_items == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return 0, 0
    true_max = int(cc.max())
    cap_d = max(1, int(cap_display))
    if true_max == 0:
        z = int(np.sum(cc == 0))
        ax.bar([0], [z], color=color, edgecolor="white", width=0.5)
        ax.set_xticks([0])
        ax.set_xticklabels(["0"], fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return 0, n_items
    if true_max <= cap_d:
        bc = np.bincount(cc, minlength=true_max + 1)
        heights = bc[1 : true_max + 1].astype(float)
        xs = np.arange(1, true_max + 1, dtype=float)
        tick_labels = [str(int(i)) for i in xs]
    else:
        capped = np.minimum(cc, cap_d)
        bc = np.bincount(capped, minlength=cap_d + 1)
        heights = bc[1 : cap_d + 1].astype(float)
        xs = np.arange(1, cap_d + 1, dtype=float)
        tick_labels = [str(i) for i in range(1, cap_d)] + [f"≥{cap_d}"]
        if annotate_max_if_merged:
            ax.text(
                0.99,
                0.98,
                f"max={true_max}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#4a5568",
            )
    ax.bar(xs, heights, color=color, edgecolor="white", linewidth=0.45)
    n_x = int(len(xs))
    if n_x <= 16:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=9)
    else:
        max_ticks = 16
        idx = np.linspace(0, n_x - 1, num=min(max_ticks, n_x), dtype=int)
        idx = np.unique(idx)
        ax.set_xticks(xs[idx])
        ax.set_xticklabels([tick_labels[i] for i in idx], fontsize=8, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=8)
    return true_max, n_items


def rag_chunk_and_retrieval_report(
    out: Path,
    txt_path: Path,
    bar: pd.DataFrame,
    psg: pd.DataFrame,
    st: pd.DataFrame,
    hq: pd.DataFrame,
    excerpt_lens_arr: np.ndarray,
    *,
    chunk_size: int,
    pool_hist_cap: int,
    dpi: int,
) -> None:
    """RAG chunk-budget figures: fixed-size character chunks from text start, ceil(L/k)."""
    g_chunks = chunks_needed_nonoverlap_chars(bar["gold_passage_nchars"].to_numpy(), chunk_size)
    p_chunks = chunks_needed_nonoverlap_chars(psg["text_nchars"].to_numpy(), chunk_size)
    s_chunks = chunks_needed_nonoverlap_chars(st["text_nchars"].to_numpy(), chunk_size)
    ex_chunks = chunks_needed_nonoverlap_chars(excerpt_lens_arr, chunk_size)

    max_per_q: list[int] = []
    concat_chunks_per_q: list[int] = []
    for sts in hq["statutes"]:
        if not isinstance(sts, list) or not sts:
            max_per_q.append(0)
            concat_chunks_per_q.append(0)
            continue
        lens = [_len_str(x.get("excerpt")) for x in sts]
        ch = chunks_needed_nonoverlap_chars(np.asarray(lens, dtype=np.int64), chunk_size)
        max_per_q.append(int(ch.max()))
        joined = "\n".join(str(x.get("excerpt") or "") for x in sts)
        concat_chunks_per_q.append(
            int(chunks_needed_nonoverlap_chars(np.asarray([len(joined)], dtype=np.int64), chunk_size)[0])
        )
    max_q = np.asarray(max_per_q, dtype=np.int64)
    hq_concat_q_chunks = np.asarray(concat_chunks_per_q, dtype=np.int64)

    # --- Gold passages (all labels, discrete bars) ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    gmx = max(1, int(g_chunks.max()))
    _hist_chunks_bars(
        ax,
        g_chunks,
        cap_display=max(gmx, 1),
        title=f"Bar Exam — chunks to cover each gold passage ({chunk_size} chars)",
        xlabel="Chunk count",
        ylabel="Gold passages",
        color="steelblue",
        annotate_max_if_merged=False,
    )
    plt.tight_layout()
    save_fig(out / f"rag_barexam_gold_passage_chunk_hist_{chunk_size}c.png", dpi)

    # --- Housing gold excerpts (same discrete-bin style as barexam gold passages) ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    emx = max(1, int(ex_chunks.max()))
    _hist_chunks_bars(
        ax,
        ex_chunks,
        cap_display=max(emx, 1),
        title=f"Housing — chunks to cover each gold excerpt ({chunk_size} chars)",
        xlabel="Chunk count",
        ylabel="Gold excerpts",
        color="darkorange",
        annotate_max_if_merged=False,
    )
    plt.tight_layout()
    save_fig(out / f"rag_housing_gold_excerpt_chunk_hist_{chunk_size}c.png", dpi)

    # --- Housing: excerpts newline-joined per question (one chunk count per row, like bar gold passage) ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    cmx = max(1, int(hq_concat_q_chunks.max()))
    _hist_chunks_bars(
        ax,
        hq_concat_q_chunks,
        cap_display=max(cmx, 1),
        title=(
            f"Housing — chunks to cover all gold excerpts per question, "
            f"concatenated with newlines ({chunk_size} chars)"
        ),
        xlabel="Chunk count",
        ylabel="Gold passages",
        color="darkorange",
        annotate_max_if_merged=False,
    )
    plt.tight_layout()
    save_fig(out / f"rag_housing_gold_concat_per_question_chunk_hist_{chunk_size}c.png", dpi)

    # --- Retrieval corpus (full row text): same discrete-bin style as gold hists, not gold labels ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    pmx_corpus = max(1, int(p_chunks.max()))
    _hist_chunks_bars(
        ax,
        p_chunks,
        cap_display=max(pmx_corpus, 1),
        title=(
            f"Bar Exam — full corpus passage text: chunks per row "
            f"({chunk_size} chars, n={len(p_chunks):,}, loaded prefix)"
        ),
        xlabel="Chunk count",
        ylabel="Passages",
        color="steelblue",
        annotate_max_if_merged=False,
    )
    plt.tight_layout()
    save_fig(out / f"rag_barexam_corpus_passage_chunk_hist_{chunk_size}c.png", dpi)

    fig, ax = plt.subplots(figsize=(8.8, 5))
    smx_corpus = max(1, int(s_chunks.max()))
    _hist_chunks_bars(
        ax,
        s_chunks,
        cap_display=max(smx_corpus, 1),
        title=(
            f"Housing — full corpus statute text: chunks per row "
            f"({chunk_size} chars, n={len(s_chunks):,}, loaded prefix)"
        ),
        xlabel="Chunk count",
        ylabel="Statute rows",
        color="darkorange",
        annotate_max_if_merged=False,
    )
    plt.tight_layout()
    save_fig(out / f"rag_housing_corpus_statute_chunk_hist_{chunk_size}c.png", dpi)

    # --- Passage pool (tail merged) ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    _hist_chunks_bars(
        ax,
        p_chunks,
        cap_display=pool_hist_cap,
        title=f"Bar Exam — passage corpus ({len(p_chunks):,} rows)",
        xlabel="Chunk count",
        ylabel="Passages",
        color=COLORS["accent"],
    )
    plt.tight_layout()
    save_fig(out / f"rag_barexam_passage_pool_chunk_hist_{chunk_size}c.png", dpi)

    # --- Housing: excerpts vs statutes ---
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ex_cap = min(40, max(15, pool_hist_cap))
    _hist_chunks_bars(
        axes[0],
        ex_chunks,
        cap_display=ex_cap,
        title=f"Housing — gold excerpts (n={len(ex_chunks):,})",
        xlabel="Chunk count",
        ylabel="Excerpts",
        color="#553c9a",
    )
    _hist_chunks_bars(
        axes[1],
        s_chunks,
        cap_display=pool_hist_cap,
        title=f"Housing — statute rows (n={len(s_chunks):,})",
        xlabel="Chunk count",
        ylabel="Rows",
        color="#285e61",
    )
    plt.tight_layout()
    save_fig(out / f"rag_housing_excerpt_and_statute_chunk_hist_{chunk_size}c.png", dpi)

    # --- Side-by-side share: gold vs pool (same chunk buckets) ---
    gmx_c = int(g_chunks.max())
    pmx_c = int(p_chunks.max())
    K = min(20, max(gmx_c, pmx_c, 6))
    K = max(K, 2)
    x = np.arange(1, K + 1, dtype=float)
    prop_g = np.zeros(K)
    prop_p = np.zeros(K)
    for k in range(1, K):
        prop_g[k - 1] = (g_chunks == k).mean()
        prop_p[k - 1] = (p_chunks == k).mean()
    prop_g[K - 1] = (g_chunks >= K).mean()
    prop_p[K - 1] = (p_chunks >= K).mean()
    fig, ax = plt.subplots(figsize=(9.2, 5))
    w = 0.38
    ax.bar(x - w / 2, prop_g * 100, width=w, label="Gold passages", color="#5b21b6", edgecolor="white", linewidth=0.35)
    ax.bar(x + w / 2, prop_p * 100, width=w, label="Passage sample", color=COLORS["accent"], edgecolor="white", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(1, K)] + [f"≥{K}"])
    ax.set_xlabel(f"Chunk count ({chunk_size} chars)")
    ax.set_ylabel("Percent of items")
    ax.set_title("Bar Exam — gold vs passage sample: share per chunk count")
    ax.legend(loc="upper right")
    _top = float(max(np.max(prop_g), np.max(prop_p)) * 100)
    ax.set_ylim(0, max(8.0, _top * 1.12))
    plt.tight_layout()
    save_fig(out / f"rag_barexam_gold_vs_pool_chunk_share_{chunk_size}c.png", dpi)

    # --- ECDF chunk counts ---
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    gx, gy = ecdf(g_chunks.astype(float))
    px, py = ecdf(p_chunks.astype(float))
    ax.step(gx, gy, where="post", lw=2.4, color="#5b21b6", label="Gold passages")
    ax.step(px, py, where="post", lw=2.4, color=COLORS["accent"], label=f"Passage sample (n={len(p_chunks):,})")
    ax.set_xlabel(f"Chunk count ({chunk_size} chars)")
    ax.set_ylabel("ECDF")
    ax.set_title("Bar Exam — chunk count distribution")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    save_fig(out / f"rag_barexam_chunk_count_ecdf_{chunk_size}c.png", dpi)

    # --- Housing: max chunks among excerpts per question ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    _hist_chunks_bars(
        ax,
        max_q,
        cap_display=min(28, max(8, int(max_q.max()) + 2)),
        title=f"Housing — max chunk count among excerpts per question ({chunk_size} chars)",
        xlabel="Chunk count",
        ylabel="Questions",
        color="#744210",
    )
    plt.tight_layout()
    save_fig(out / f"rag_housing_max_excerpt_chunks_per_question_{chunk_size}c.png", dpi)

    # --- Housing ECDF excerpts vs statutes ---
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ex_x, ex_y = ecdf(ex_chunks.astype(float))
    st_x, st_y = ecdf(s_chunks.astype(float))
    ax.step(ex_x, ex_y, where="post", lw=2.2, color="#553c9a", label="Gold excerpts")
    ax.step(st_x, st_y, where="post", lw=2.2, color="#285e61", label=f"Statute sample (n={len(s_chunks):,})")
    ax.set_xlabel(f"Chunk count ({chunk_size} chars)")
    ax.set_ylabel("ECDF")
    ax.set_title("Housing — excerpt vs full statute row lengths (chunks)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    save_fig(out / f"rag_housing_chunk_ecdf_{chunk_size}c.png", dpi)

    # --- Cumulative: gold ---
    if int(g_chunks.max()) > 0:
        ks = np.arange(1, int(g_chunks.max()) + 1)
        cum_pct = 100.0 * np.array([(g_chunks <= k).mean() for k in ks], dtype=float)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        ax.plot(ks, cum_pct, lw=2.8, color="#5b21b6", marker="o", ms=5)
        ax.axhline(90, ls=":", color="#718096")
        ax.set_xlabel(f"Chunk count (≤ k, {chunk_size} chars)")
        ax.set_ylabel("% of gold passages")
        ax.set_title("Bar Exam gold — cumulative coverage")
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.28)
        plt.tight_layout()
        save_fig(out / f"rag_barexam_gold_cumulative_chunks_{chunk_size}c.png", dpi)

    def _summ(name: str, arr: np.ndarray, fh: TextIO) -> None:
        arr = np.asarray(arr, dtype=np.int64)
        arr = arr[arr >= 0]
        if len(arr) == 0:
            fh.write(f"\n{name}: (empty)\n")
            return
        fh.write(f"\n{name}:\n")
        fh.write(f"  n={len(arr):,}  min={int(arr.min())}  median={float(np.median(arr)):.1f}  mean={float(arr.mean()):.2f}  p90={float(np.percentile(arr, 90)):.1f}  max={int(arr.max())}\n")
        fh.write(f"  fraction requiring >1 chunk: {float((arr > 1).mean()):.4f}\n")
        fh.write(f"  fraction requiring >2 chunks: {float((arr > 2).mean()):.4f}\n")

    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, f"RAG CHUNK ANALYSIS ({chunk_size} characters, non-overlapping from offset 0)")
        fh.write(
            "Each text is aligned to chunk grid [0,k), [k,2k), ... from the start of that string. "
            "Number of chunks touching [0, L) equals ceil(L/k) for L>0.\n"
            "This matches overlap counting for span [0,L) in plot_chunking_span_histograms.py.\n"
            "Housing “concat per question”: all `statutes[].excerpt` for a row are joined with '\\n' then treated as one span.\n\n"
        )
        _summ("Bar Exam — gold_passage chunks", g_chunks, fh)
        _summ("Bar Exam — passage pool text chunks (loaded prefix)", p_chunks, fh)
        _summ("Housing — flattened gold excerpt chunks", ex_chunks, fh)
        _summ("Housing — concat gold excerpts per question (chunks)", hq_concat_q_chunks, fh)
        _summ("Housing — statute sample full-text chunks", s_chunks, fh)
        _summ("Housing — max excerpt chunks per question", max_q, fh)


# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("viz_results"),
        help="Directory for PNGs + summary_statistics.txt (default: ./viz_results)",
    )
    ap.add_argument("--passage-rows", type=int, default=200_000)
    ap.add_argument("--statute-rows", type=int, default=250_000)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Char chunk size for RAG chunk-budget plots (fixed grid from text start).",
    )
    ap.add_argument(
        "--chunk-hist-cap",
        type=int,
        default=80,
        help="Merge chunk-count tail at this value for huge corpora (pool + statutes).",
    )
    args = ap.parse_args()
    out: Path = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dpi = max(72, args.dpi)

    txt_path = out / "summary_statistics.txt"

    # --- Load barexam QA ---
    print("Loading barexam_qa …", file=sys.stderr)
    bar = pd.read_csv(URLS["barexam_qa"])
    bar["answer"] = bar["answer"].astype(str).str.upper().str.strip()
    for c in ["prompt", "question", "choice_a", "choice_b", "choice_c", "choice_d", "gold_passage"]:
        bar[f"{c}_nchars"] = bar[c].map(_len_str)
    bar["choices_total_nchars"] = (
        bar["choice_a_nchars"]
        + bar["choice_b_nchars"]
        + bar["choice_c_nchars"]
        + bar["choice_d_nchars"]
    )

    with open(txt_path, "w", encoding="utf-8") as fh:
        def tw(x: str = "") -> None:
            fh.write(x + "\n")

        tw(f"REGLab dataset statistics — generated {datetime.now(timezone.utc).isoformat()} UTC")
        tw(f"Figures directory: {out}")
        tw(f"Samples: passages.tsv nrows={args.passage_rows:,} | statutes.tsv nrows={args.statute_rows:,}")
        tw("")
        log_section(fh, "BAR EXAM QA (qa.csv)")
        tw(f"Rows: {len(bar):,}")
        tw(f"Columns: {list(bar.columns)}")
        tw("\nMissingness (% of rows) — top 15:")
        miss = (bar.isna().mean() * 100).sort_values(ascending=False).head(15)
        tw(miss.to_string())
        tw("\nNumeric / engineered length summaries:")
        for col in [
            "prompt_nchars",
            "question_nchars",
            "choices_total_nchars",
            "gold_passage_nchars",
        ]:
            tw(describe_series(bar[col], col).to_string())

        counts = bar["answer"].value_counts().reindex(list("ABCD")).fillna(0).astype(int)
        chi2, p = stats.chisquare(counts.values, np.full(4, len(bar) / 4))
        tw(f"\nAnswer counts vs uniform: chi-square={chi2:.4f}, p={p:.4g}")
        tw(counts.to_string())

        qpp = bar.groupby("prompt_id", observed=True).size()
        tw("\nQuestions per prompt_id:")
        tw(qpp.describe().to_string())
        tw(f"Fraction of prompt_id with >1 question: {(qpp > 1).mean():.4f}")

        tsrc = bar["source"].astype(str).value_counts().head(6).index
        sub6 = bar.loc[bar["source"].astype(str).isin(tsrc), ["source", "answer"]]
        cta = pd.crosstab(sub6["source"].astype(str), sub6["answer"].astype(str))
        tw(f"\nCramér V (top-6 source × answer): {cramers_v(cta):.4f}")
        fh.flush()

    # Figures: Bar exam
    mpl.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

    fig, ax = plt.subplots(figsize=(6.5, 6))
    cc = counts.reindex(list("ABCD")).fillna(0).astype(int)
    _cols = [COLORS["accent"], COLORS["ok"], COLORS["warn"], "#805ad5"]
    ax.pie(
        cc.values,
        labels=list(cc.index),
        autopct=lambda p: f"{p:.1f}%",
        startangle=45,
        colors=_cols,
    )
    ax.set_title(f"Bar Exam QA — correct MCQ label (n={len(bar):,})")
    save_fig(out / "barexam_qa_01_answer_distribution.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    mv = (bar.isna().mean() * 100).sort_values(ascending=False).head(12)
    ax.barh(mv.index[::-1], mv.values[::-1], color=COLORS["muted"])
    ax.set_xlabel("Missing %")
    ax.set_title("Bar Exam QA — columns with highest missing rates")
    save_fig(out / "barexam_qa_02_missingness.png", dpi)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, col, color, title in zip(
        axes.flat,
        ["prompt_nchars", "question_nchars", "choices_total_nchars", "gold_passage_nchars"],
        ["#2b6cb0", "#276749", "#c05621", "#9f7aea"],
        ["Prompt (chars)", "Question (chars)", "Sum of 4 choices (chars)", "Gold passage (chars)"],
    ):
        ax.hist(bar[col], bins=50, color=color, edgecolor="white", alpha=0.92)
        ax.set_title(title)
        ax.set_ylabel("Frequency")
    fig.suptitle("Bar Exam QA — text length histograms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(out / "barexam_qa_03_length_histograms.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = {
        "prompt_nchars": "prompt",
        "question_nchars": "question",
        "choices_total_nchars": "Σ choices",
        "gold_passage_nchars": "gold passage",
    }
    colors = ["#2b6cb0", "#276749", "#c05621", "#805ad5"]
    for (col, lab), c in zip(labels.items(), colors):
        x, y = ecdf(bar[col].values)
        ax.plot(x, y, lw=2.2, label=lab, color=c)
    ax.set_xlabel("Characters")
    ax.set_ylabel("ECDF")
    ax.set_title("Bar Exam QA — empirical CDFs of field lengths")
    ax.legend(loc="lower right")
    save_fig(out / "barexam_qa_04_length_ecdf.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = bar["question_nchars"].values
    y = bar["gold_passage_nchars"].values
    hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1, linewidths=0)
    plt.colorbar(hb, ax=ax, label="count / bin")
    ax.set_xlabel("Question length (chars)")
    ax.set_ylabel("Gold passage length (chars)")
    ax.set_title("Bar Exam QA — joint density (hexbin): stem vs evidence size")
    save_fig(out / "barexam_qa_05_hexbin_question_vs_gold.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    order = list("ABCD")
    data = [bar.loc[bar["answer"] == k, "gold_passage_nchars"] for k in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, 4))):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xlabel("Correct answer")
    ax.set_ylabel("Gold passage length (chars)")
    ax.set_title("Bar Exam QA — gold passage length by correct label (no outliers)")
    save_fig(out / "barexam_qa_06_boxplot_gold_by_answer.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(qpp.values, bins=np.arange(0.5, qpp.max() + 1.5), color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("# questions sharing same prompt_id")
    ax.set_ylabel("Count")
    ax.set_title("Bar Exam QA — how often the same stem (prompt_id) spans multiple items")
    save_fig(out / "barexam_qa_07_prompt_id_cluster_sizes.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sv = bar["source"].astype(str).value_counts().head(18)
    ax.barh(sv.index[::-1], sv.values[::-1], color=COLORS["accent"])
    ax.set_xlabel("Count")
    ax.set_title("Bar Exam QA — top 18 `source` values")
    save_fig(out / "barexam_qa_08_source_top18.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sj = bar["subject"].fillna("(missing)").astype(str).value_counts().head(18)
    ax.barh(sj.index[::-1], sj.values[::-1], color=COLORS["warn"])
    ax.set_xlabel("Count")
    ax.set_title("Bar Exam QA — top 18 `subject` (missing grouped)")
    save_fig(out / "barexam_qa_09_subject_top18.png", dpi)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cols = list(labels.keys())
    rho = bar[cols].corr(method="spearman")
    if HAS_SNS:
        sns.heatmap(rho, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    else:
        im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        short = list(labels.values())
        ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_yticklabels(short)
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Bar Exam QA — Spearman correlation of length features")
    plt.tight_layout()
    save_fig(out / "barexam_qa_10_spearman_corr_heatmap.png", dpi)

    _ns, _nj = 10, 10
    _src_top = bar["source"].astype(str).value_counts().head(_ns).index
    _subj = bar["subject"].fillna("(missing)").astype(str)
    _sub_top = _subj.value_counts().head(_nj).index
    _mask = bar["source"].astype(str).isin(_src_top) & _subj.isin(_sub_top)
    _cthm = pd.crosstab(bar.loc[_mask, "source"].astype(str), _subj.loc[_mask])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    if HAS_SNS:
        sns.heatmap(_cthm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "count"})
    else:
        im = ax.imshow(_cthm.values, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xticks(np.arange(_cthm.shape[1]))
        ax.set_yticks(np.arange(_cthm.shape[0]))
        ax.set_xticklabels(_cthm.columns, rotation=45, ha="right")
        ax.set_yticklabels(_cthm.index)
        plt.colorbar(im, ax=ax)
    ax.set_title(f"Bar Exam QA — heatmap: top-{_ns} sources × top-{_nj} subjects")
    plt.tight_layout()
    save_fig(out / "barexam_qa_11_source_subject_heatmap.png", dpi)

    # Passages
    print("Loading barexam passages …", file=sys.stderr)
    psg = pd.read_csv(URLS["barexam_passages"], sep="\t", nrows=args.passage_rows)
    psg["text_nchars"] = psg["text"].map(_len_str)

    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, f"BAR EXAM PASSAGES (first {len(psg):,} rows of passages.tsv)")
        tw = lambda s="": fh.write(s + "\n")  # noqa: E731
        tw(psg["source"].astype(str).value_counts().to_string())
        tw("\n" + describe_series(psg["text_nchars"], "passage_text_nchars").to_string())

    fig, ax = plt.subplots(figsize=(6.5, 6))
    _psrc = psg["source"].astype(str).str.strip().str.lower()
    _mw = psg[_psrc.isin({"mbe", "wex"})]
    mx = _mw["source"].astype(str).str.strip().str.lower().value_counts()
    _ord = [k for k in ("mbe", "wex") if k in mx.index]
    mx = mx.reindex(_ord)
    _lbl = [s.upper() for s in mx.index]
    ax.pie(mx.values, labels=_lbl, autopct=lambda p: f"{p:.1f}%", startangle=45)
    ax.set_title(f"Bar Exam passages — MBE vs WEX (n={len(_mw):,} of {len(psg):,} prefix sample)")
    save_fig(out / "barexam_passages_01_source_mix_pie.png", dpi)

    tn = psg["text_nchars"].to_numpy()
    tn = tn[tn > 0]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.hist(tn, bins=72, color=COLORS["accent"], edgecolor="white")
    ax.set_xlabel("Passage length (characters)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bar Exam passages — length (n={len(psg):,} rows loaded)")
    save_fig(out / "barexam_passages_02_length_hist.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    gx, gy = ecdf(bar["gold_passage_nchars"].values)
    cx, cy = ecdf(psg["text_nchars"].values)
    ax.plot(gx, gy, lw=2.4, color=COLORS["bad"], label="gold passages (labels)")
    ax.plot(cx, cy, lw=2.4, color=COLORS["accent"], label=f"corpus prefix (n={len(psg):,})")
    ax.set_xscale("log")
    ax.set_xlabel("Characters (log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title("Bar Exam — gold passage lengths vs passage-pool prefix")
    ax.legend()
    save_fig(out / "barexam_passages_03_gold_vs_corpus_ecdf.png", dpi)

    hit = bar["gold_idx"].astype(str).isin(psg["idx"].astype(str))
    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, "BAR EXAM gold_idx overlap with passages prefix")
        fh.write(f"Questions with gold_idx in prefix: {hit.sum()} / {len(bar)} ({100*hit.mean():.2f}%)\n")

    # Housing
    print("Loading housing_qa …", file=sys.stderr)
    hq_list = fetch_json_zip(URLS["housing_q"])
    hq_aux = fetch_json_zip(URLS["housing_q_aux"])
    hq = pd.DataFrame(hq_list)
    hq["answer_norm"] = hq["answer"].astype(str).str.strip().str.title()
    hq["answer_bin"] = hq["answer"].astype(str).str.lower().str.strip().eq("yes").astype(int)
    hq["n_statute_excerpts"] = hq["statutes"].map(lambda s: len(s) if isinstance(s, list) else 0)
    hq["excerpt_chars_total"] = hq["statutes"].map(
        lambda sts: sum(_len_str(x.get("excerpt")) for x in sts) if isinstance(sts, list) else 0
    )

    excerpt_lens: list[int] = []
    idx_in_labels: set[int] = set()
    for sts in hq["statutes"]:
        if not isinstance(sts, list):
            continue
        for x in sts:
            excerpt_lens.append(_len_str(x.get("excerpt")))
            if isinstance(x, dict) and x.get("statute_idx") is not None:
                try:
                    idx_in_labels.add(int(x["statute_idx"]))
                except (TypeError, ValueError):
                    pass
    excerpt_lens_arr = np.asarray(excerpt_lens, dtype=int)

    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, "HOUSING QA (questions.json.zip)")
        fh.write(f"questions rows: {len(hq):,}\n")
        fh.write(f"questions_aux rows: {len(hq_aux):,}\n")
        fh.write(f"Unique states: {hq['state'].nunique()}\n")
        fh.write(f"Unique question_group: {hq['question_group'].nunique()}\n")
        fh.write(f"Overall Yes rate: {100*hq['answer_bin'].mean():.2f}%\n")
        fh.write("\nanswer_norm counts:\n")
        fh.write(hq["answer_norm"].value_counts().to_string())
        fh.write("\n\nn_statute_excerpts:\n")
        fh.write(hq["n_statute_excerpts"].describe().to_string())
        fh.write("\n\nexcerpt field lengths (flattened):\n")
        fh.write(describe_series(pd.Series(excerpt_lens_arr), "excerpt_chars").to_string())
        fh.write("\n\ntop states by n:\n")
        fh.write(hq.groupby("state", observed=True).size().sort_values(ascending=False).head(20).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 6))
    _ne, _na = len(hq), len(hq_aux)
    ax.pie(
        [_ne, _na],
        labels=["questions.json", "questions_aux.json"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=45,
        colors=[COLORS["accent"], COLORS["muted"]],
    )
    ax.set_title(f"Housing QA — dataset files by row count (n={_ne + _na:,})")
    save_fig(out / "housing_q_00_dataset_files_pie.png", dpi)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    vc = hq["answer_norm"].value_counts()
    _ord = [x for x in ("No", "Yes") if x in vc.index]
    vc = vc.reindex(_ord)
    ax.pie(
        vc.values,
        labels=list(vc.index),
        autopct=lambda p: f"{p:.1f}%",
        startangle=45,
        colors=[COLORS["bad"], COLORS["ok"]][: len(vc)],
    )
    ax.set_title(f"Housing QA — Yes vs No (n={len(hq):,}; 2021 law slice)")
    save_fig(out / "housing_q_01_yes_no_counts.png", dpi)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    stc = hq.groupby("state", observed=True).size().sort_values(ascending=False).head(22)
    ax.barh(stc.index[::-1].astype(str), stc.values[::-1], color="#744210")
    ax.set_xlabel("# questions")
    ax.set_title("Housing QA — top 22 jurisdictions by volume")
    save_fig(out / "housing_q_02_states_by_volume.png", dpi)

    MIN_N = 40
    marg = (
        hq.groupby("state", observed=True)
        .agg(n=("answer_bin", "size"), y=("answer_bin", "mean"))
        .query("n >= @MIN_N")
        .assign(y=lambda d: d["y"] * 100)
        .sort_values("y")
    )
    _nh = max(5.0, min(22.0, 0.28 * len(marg)))
    fig, ax = plt.subplots(figsize=(8, _nh))
    colors = [COLORS["ok"] if v > 50 else COLORS["bad"] for v in marg["y"]]
    ax.barh(marg.index.astype(str), marg["y"], color=colors)
    ax.axvline(50, color="black", ls=":", lw=1)
    ax.set_xlabel(f"Yes rate % (states with n ≥ {MIN_N})")
    ax.set_title("Housing QA — Yes rate by state (coverage filter)")
    save_fig(out / "housing_q_03_yes_rate_by_state.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ne = hq["n_statute_excerpts"]
    ax.hist(ne, bins=np.arange(0, ne.max() + 2) - 0.5, color=COLORS["primary"], edgecolor="white")
    ax.set_xlabel("# annotated statute excerpts per question")
    ax.set_ylabel("Count")
    ax.set_title("Housing QA — how many gold citations per row")
    save_fig(out / "housing_q_04_statute_excerpts_per_question.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(excerpt_lens_arr.clip(1, np.quantile(excerpt_lens_arr, 0.995)), bins=60, color="#553c9a", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Gold excerpt length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Housing QA — distribution of annotated excerpt lengths")
    save_fig(out / "housing_q_05_excerpt_length_histogram.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    el = excerpt_lens_arr[excerpt_lens_arr > 0]
    elp = np.log10(el.clip(1, np.quantile(el, 0.999)))
    if HAS_SNS:
        try:
            sns.kdeplot(x=elp, fill=True, color="#553c9a", alpha=0.55, ax=ax, warn_singular=False)
        except TypeError:
            sns.kdeplot(x=elp, fill=True, color="#553c9a", alpha=0.55, ax=ax)
    else:
        ax.hist(elp, bins=50, density=True, color="#553c9a", alpha=0.75, edgecolor="white")
    ax.set_xlabel("log10(excerpt chars)")
    ax.set_ylabel("density")
    ax.set_title("Housing QA — KDE of gold excerpt lengths (log scale)")
    save_fig(out / "housing_q_05b_excerpt_length_kde_log.png", dpi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    tmp = hq[["answer_norm", "n_statute_excerpts"]].copy()
    tmp = tmp[tmp["answer_norm"].isin(["Yes", "No"])]
    if HAS_SNS:
        sns.violinplot(
            data=tmp,
            x="answer_norm",
            y="n_statute_excerpts",
            ax=ax,
            order=["No", "Yes"],
            palette=["#c53030", "#276749"],
            inner="box",
        )
    else:
        data = [tmp.loc[tmp["answer_norm"] == "No", "n_statute_excerpts"], tmp.loc[tmp["answer_norm"] == "Yes", "n_statute_excerpts"]]
        ax.boxplot(data, tick_labels=["No", "Yes"], patch_artist=True)
    ax.set_title("Housing QA — # statute excerpts vs label")
    save_fig(out / "housing_q_06_violin_excerpts_by_label.png", dpi)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    qg = hq["question_group"].value_counts().head(35)
    ax.bar(range(len(qg)), qg.values, color=COLORS["warn"])
    ax.set_xticks(range(len(qg)))
    ax.set_xticklabels([str(i) for i in qg.index], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Housing QA — top 35 question_group IDs")
    save_fig(out / "housing_q_07_question_group_top35.png", dpi)

    rows_spread = []
    for g, sub in hq.groupby("question_group", observed=True):
        rates = sub.groupby("state")["answer_bin"].mean()
        rows_spread.append({"question_group": g, "std_yes": float(rates.std(ddof=0)), "n_states": int(rates.size)})
    qspread = pd.DataFrame(rows_spread).dropna().sort_values("std_yes", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(qspread["question_group"].astype(str), qspread["std_yes"], color="#654c8f")
    ax.set_xlabel("Std dev of Yes-rate across states (within group)")
    ax.set_title("Housing QA — most heterogeneous question templates by jurisdiction")
    save_fig(out / "housing_q_08_crossstate_heterogeneity.png", dpi)

    print("Loading housing statutes prefix …", file=sys.stderr)
    st = pd.read_csv(URLS["housing_statutes"], sep="\t", nrows=args.statute_rows)
    st["text_nchars"] = st["text"].map(_len_str)
    pool_ids = set(st["idx"].astype(int))
    inter = idx_in_labels & pool_ids

    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, f"HOUSING STATUTES (first {len(st):,} rows)")
        fh.write(describe_series(st["text_nchars"], "statute_text_nchars").to_string())
        fh.write("\n\ntop states in sequential sample (may be biased by file order):\n")
        fh.write(st["state"].astype(str).value_counts().head(15).to_string())
        log_section(fh, "HOUSING statute_idx coverage vs statutes prefix")
        fh.write(f"Unique statute_idx in question labels: {len(idx_in_labels):,}\n")
        fh.write(f"Overlap with loaded statute rows idx: {len(inter):,} ({100*len(inter)/max(len(idx_in_labels),1):.2f}%)\n")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sx, sy = ecdf(st["text_nchars"].values)
    ex, ey = ecdf(excerpt_lens_arr)
    ax.plot(np.log10(sx + 1), sy, lw=2.3, color=COLORS["accent"], label=f"full statute rows (n={len(st):,})")
    ax.plot(np.log10(ex + 1), ey, lw=2.3, color=COLORS["bad"], label="gold excerpts in questions")
    ax.set_xlabel("log10(chars + 1)")
    ax.set_ylabel("ECDF")
    ax.set_title("Housing — full statute text vs annotated excerpt lengths")
    ax.legend()
    save_fig(out / "housing_statutes_01_excerpt_vs_full_ecdf.png", dpi)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    stc = st["state"].astype(str).value_counts().head(15)
    ax.barh(stc.index[::-1], stc.values[::-1], color="#285e61")
    ax.set_xlabel("Rows in prefix sample")
    ax.set_title("Housing statutes — top 15 `state` in sequential prefix (see TXT caveat)")
    save_fig(out / "housing_statutes_02_state_counts_prefix.png", dpi)

    # Combined summary figure
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(cc.index, cc.values, color=[COLORS["accent"], COLORS["ok"], COLORS["warn"], "#805ad5"])
    ax1.set_title("Bar Exam — answer mix")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.pie(mx.values, labels=mx.index, autopct=lambda p: f"{p:.0f}%" if p > 8 else "")
    ax2.set_title("Passages — source mix")
    ax3 = fig.add_subplot(2, 2, 3)
    vc2 = hq["answer_norm"].value_counts()
    _yo = [x for x in ("No", "Yes") if x in vc2.index]
    vc2 = vc2.reindex(_yo)
    ax3.pie(
        vc2.values,
        labels=list(vc2.index),
        autopct=lambda p: f"{p:.0f}%",
        startangle=45,
        colors=[COLORS["bad"], COLORS["ok"]][: len(vc2)],
    )
    ax3.set_title("Housing — Yes/No")
    ax4 = fig.add_subplot(2, 2, 4)
    px, py = ecdf(psg["text_nchars"].values)
    ax4.plot(px, py, color=COLORS["accent"], label="passage pool")
    gx, gy = ecdf(bar["gold_passage_nchars"].values)
    ax4.plot(gx, gy, color=COLORS["bad"], label="bar gold")
    ax4.set_xscale("log")
    ax4.set_title("ECDF lengths (log-x): pool vs gold")
    ax4.legend(fontsize=8)
    fig.suptitle("REGLab datasets — executive summary panel", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(out / "summary_panel_executive.png", dpi)

    print("RAG chunk / retrieval report …", file=sys.stderr)
    rag_chunk_and_retrieval_report(
        out,
        txt_path,
        bar,
        psg,
        st,
        hq,
        excerpt_lens_arr,
        chunk_size=args.chunk_size,
        pool_hist_cap=max(10, args.chunk_hist_cap),
        dpi=dpi,
    )

    with open(txt_path, "a", encoding="utf-8") as fh:
        log_section(fh, "FILES WRITTEN")
        for p in sorted(out.glob("*.png")):
            fh.write(f"  {p.name}\n")

    print(f"Done. Figures + {txt_path.name} → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
