#!/usr/bin/env python3
"""
Reproduction of **Table 22** (Historical MBE subset, baseline rows) from:
  "A Reasoning-Focused Legal Retrieval Benchmark"
  Zheng et al., CS&Law 2025  (arXiv:2505.03970)

We target Table 22, NOT Table 21:
  • Table 21  = aggregate Bar Exam QA (Historical MBE + private BarBri subset).
  • Table 22  = disaggregated Historical MBE subset ONLY — this is the public
                data released on HuggingFace (reglab/barexam_qa).

Baseline retrievers (no query expansion):
  - BM25                    (BM25Okapi, rank_bm25)
  - E5-small-v2             (intfloat/e5-small-v2)
  - E5-base-v2              (intfloat/e5-base-v2)
  - E5-large-v2             (intfloat/e5-large-v2)
  - E5-mistral-7b-instruct  (intfloat/e5-mistral-7b-instruct)

Metrics: Recall@1, Recall@10, MRR@10, Recall@100, Recall@1000

Usage:
  python reproduce_table21_baseline.py                         # all models
  python reproduce_table21_baseline.py --models bm25 e5-small  # subset
  python reproduce_table21_baseline.py --model-idx 0            # BM25 (SLURM task 0)
  python reproduce_table21_baseline.py --model-idx 4            # E5-mistral (SLURM task 4)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Enforce offline mode: compute nodes on this cluster have no internet access.
# Models and datasets must already be cached on the login node before submission.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# datasets reads from parquet cache files, not the network.
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

RECALL_CUTOFFS = (1, 10, 100, 1000)
MRR_CUTOFF = 10

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "bm25": {
        "kind": "bm25",
        "display": "BM25",
    },
    "e5-small": {
        "kind": "dense",
        "hf_name": "intfloat/e5-small-v2",
        "display": "E5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 1024,
        "max_length": 512,
    },
    "e5-base": {
        "kind": "dense",
        "hf_name": "intfloat/e5-base-v2",
        "display": "E5-base-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 512,
        "max_length": 512,
    },
    "e5-large": {
        "kind": "dense",
        "hf_name": "intfloat/e5-large-v2",
        "display": "E5-large-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 256,
        "max_length": 512,
    },
    "e5-mistral": {
        "kind": "dense",
        "hf_name": "intfloat/e5-mistral-7b-instruct",
        "display": "E5-mistral-7b",
        # E5-mistral uses a task instruction prefix for queries; passages are untouched.
        "query_prefix": (
            "Instruct: Given a legal question, retrieve relevant legal passages "
            "that help answer the question.\nQuery: "
        ),
        "passage_prefix": "",
        "batch_size": 16,
        "max_length": 4096,
    },
}

# Model index → key mapping for SLURM array jobs (--model-idx 0..4)
MODEL_IDX_ORDER = ["bm25", "e5-small", "e5-base", "e5-large", "e5-mistral"]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_queries() -> tuple[list[str], list[str]]:
    """Return (query_texts, gold_passage_ids) from the HF qa config."""
    from datasets import load_dataset

    logger.info("Loading queries from reglab/barexam_qa (qa config)…")
    qa_ds = load_dataset("reglab/barexam_qa", "qa", trust_remote_code=True)

    queries: list[str] = []
    gold_ids: list[str] = []
    for split in ("train", "validation", "test"):
        if split not in qa_ds:
            continue
        for row in qa_ds[split]:
            prompt = (row.get("prompt") or "").strip()
            question = (row.get("question") or "").strip()
            qtext = f"{prompt}\n{question}" if prompt else question
            queries.append(qtext)
            gold_ids.append(str(row["gold_idx"]))

    logger.info("  Loaded %d queries", len(queries))
    return queries, gold_ids


def load_passages(cache_file: Path | None = None) -> tuple[list[str], list[str]]:
    """Return (passage_ids, passage_texts) from the HF passages config.

    Caches to ``cache_file`` (npz) if provided to avoid re-loading on re-runs.
    """
    if cache_file and cache_file.exists():
        logger.info("Loading passage cache from %s …", cache_file)
        data = np.load(str(cache_file), allow_pickle=True)
        return list(data["ids"]), list(data["texts"])

    from datasets import concatenate_datasets, load_dataset

    logger.info("Loading passages from reglab/barexam_qa (passages config)…")
    passage_ds = load_dataset("reglab/barexam_qa", "passages", trust_remote_code=True)

    parts = [passage_ds[s] for s in ("train", "validation", "test") if s in passage_ds]
    if len(parts) > 1:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(parts)
    else:
        ds = parts[0]

    ids: list[str] = []
    texts: list[str] = []
    for row in tqdm(ds, desc="Loading passages", unit="row"):
        ids.append(str(row["idx"]))
        texts.append(row["text"] or "")

    logger.info("  Loaded %d passages", len(ids))

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_file),
            ids=np.array(ids, dtype=object),
            texts=np.array(texts, dtype=object),
        )
        logger.info("  Saved passage cache → %s", cache_file)

    return ids, texts


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    top_indices: np.ndarray,
    passage_ids: list[str],
    gold_ids: list[str],
    ks: tuple[int, ...] = RECALL_CUTOFFS,
    mrr_cutoff: int = MRR_CUTOFF,
) -> dict[str, float]:
    """Compute Recall@k and MRR@cutoff from integer index rankings.

    ``top_indices[i]`` = ranked array of passage indices for query i,
    sorted best-first, with at least max(ks) entries.
    """
    n = len(gold_ids)
    recall_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    max_k = max(ks)

    for i, gid in enumerate(gold_ids):
        row = top_indices[i][:max_k]
        top_pids = [passage_ids[j] for j in row]

        for k in ks:
            if gid in set(top_pids[:k]):
                recall_sums[k] += 1.0

        for rank, pid in enumerate(top_pids[:mrr_cutoff], start=1):
            if pid == gid:
                mrr_sum += 1.0 / rank
                break

    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = 100.0 * recall_sums[k] / n
    result[f"mrr@{mrr_cutoff}"] = 100.0 * mrr_sum / n
    return result


# ── BM25 ─────────────────────────────────────────────────────────────────────

def run_bm25(
    queries: list[str],
    gold_ids: list[str],
    passage_ids: list[str],
    passage_texts: list[str],
    results_dir: Path,
) -> dict[str, float]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit(
            "rank_bm25 not importable. Install it on the LOGIN NODE before submitting:\n"
            "  pip install rank-bm25"
        )

    result_file = results_dir / "bm25_metrics.json"
    if result_file.exists():
        logger.info("BM25: loading cached metrics from %s", result_file)
        return json.loads(result_file.read_text())

    logger.info("BM25: tokenizing %d passages…", len(passage_texts))
    t0 = time.time()
    tokenized = [t.lower().split() for t in tqdm(passage_texts, desc="BM25 tokenize", unit="doc")]
    logger.info("  Tokenisation done in %.1fs", time.time() - t0)

    logger.info("BM25: building index…")
    t0 = time.time()
    bm25 = BM25Okapi(tokenized)
    logger.info("  Index built in %.1fs", time.time() - t0)

    max_k = max(RECALL_CUTOFFS)
    top_indices_list: list[np.ndarray] = []

    for q in tqdm(queries, desc="BM25 retrieval", unit="q"):
        scores = bm25.get_scores(q.lower().split())
        top_idx = np.argpartition(-scores, min(max_k, len(scores) - 1))[:max_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_indices_list.append(top_idx)

    top_indices = np.array(top_indices_list, dtype=np.int64)
    metrics = compute_metrics(top_indices, passage_ids, gold_ids)

    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(metrics, indent=2))
    logger.info("BM25 metrics saved → %s", result_file)
    return metrics


# ── Dense (E5) retrieval ─────────────────────────────────────────────────────

def _load_or_encode_passages(
    model_key: str,
    model_cfg: dict[str, Any],
    passage_texts: list[str],
    results_dir: Path,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return normalised float32 passage embeddings, loading from cache if possible."""
    emb_file = results_dir / f"{model_key}_passage_embs.npy"

    if emb_file.exists() and not force_recompute:
        logger.info("Loading cached passage embeddings: %s", emb_file)
        return np.load(str(emb_file))

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_name = model_cfg["hf_name"]
    logger.info("Loading model %s on %s…", hf_name, device)

    model = SentenceTransformer(hf_name, device=device)
    model.max_seq_length = model_cfg["max_length"]

    pfx = model_cfg["passage_prefix"]
    prefixed = [pfx + t for t in passage_texts] if pfx else passage_texts
    batch_size = model_cfg["batch_size"]

    logger.info(
        "Encoding %d passages (batch=%d, device=%s)…",
        len(prefixed), batch_size, device,
    )
    t0 = time.time()
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embs = embs.astype(np.float32)
    elapsed = time.time() - t0
    logger.info("  Encoded in %.1fs  shape=%s", elapsed, embs.shape)

    emb_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_file), embs)
    logger.info("  Saved → %s", emb_file)

    # Free GPU memory
    del model
    import gc; gc.collect()
    if device == "cuda":
        import torch; torch.cuda.empty_cache()

    return embs


def run_dense(
    model_key: str,
    queries: list[str],
    gold_ids: list[str],
    passage_ids: list[str],
    passage_texts: list[str],
    results_dir: Path,
    force_recompute: bool = False,
) -> dict[str, float]:
    model_cfg = MODEL_REGISTRY[model_key]

    result_file = results_dir / f"{model_key}_metrics.json"
    if result_file.exists() and not force_recompute:
        logger.info("%s: loading cached metrics from %s", model_cfg["display"], result_file)
        return json.loads(result_file.read_text())

    passage_embs = _load_or_encode_passages(
        model_key, model_cfg, passage_texts, results_dir, force_recompute
    )

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_name = model_cfg["hf_name"]
    logger.info("Encoding %d queries with %s…", len(queries), hf_name)

    model = SentenceTransformer(hf_name, device=device)
    model.max_seq_length = model_cfg["max_length"]
    q_pfx = model_cfg["query_prefix"]
    q_prefixed = [q_pfx + q for q in queries] if q_pfx else queries

    query_embs = model.encode(
        q_prefixed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    del model
    import gc; gc.collect()

    max_k = max(RECALL_CUTOFFS)
    logger.info("Searching top-%d passages…", max_k)

    try:
        import faiss
        dim = passage_embs.shape[1]
        logger.info("Building FAISS IndexFlatIP (dim=%d, N=%d)…", dim, len(passage_embs))
        index = faiss.IndexFlatIP(dim)
        index.add(passage_embs)
        _, top_indices = index.search(query_embs, max_k)
        logger.info("  FAISS search done")
    except ImportError:
        # faiss is optional; pure-numpy batched inner-product search is used instead.
        # For ~900K passages × ~1195 queries this is memory-efficient and fast enough.
        logger.info("faiss not available — using numpy batched search (this is fine)")
        chunk = 200
        top_indices_parts: list[np.ndarray] = []
        for start in tqdm(range(0, len(query_embs), chunk), desc="numpy search", unit="batch"):
            q_chunk = query_embs[start : start + chunk]
            sims = q_chunk @ passage_embs.T
            idx = np.argpartition(-sims, min(max_k, sims.shape[1] - 1), axis=1)[:, :max_k]
            # Sort within the partitioned top-k
            row_sims = sims[np.arange(len(q_chunk))[:, None], idx]
            order = np.argsort(-row_sims, axis=1)
            top_indices_parts.append(idx[np.arange(len(q_chunk))[:, None], order])
        top_indices = np.vstack(top_indices_parts)

    metrics = compute_metrics(top_indices, passage_ids, gold_ids)

    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(metrics, indent=2))
    logger.info("%s metrics saved → %s", model_cfg["display"], result_file)
    return metrics


# ── Output ────────────────────────────────────────────────────────────────────

def print_results_table(all_metrics: dict[str, dict[str, float]]) -> None:
    """Print reproduced vs. paper Table 22 side-by-side with absolute differences."""

    # Paper Table 22 (Historical MBE subset) — baseline rows
    # Order: (R@1, R@10, MRR@10, R@100, R@1000)
    paper_ref: dict[str, tuple[float, ...]] = {
        "bm25":       (0.25, 0.75,  0.37,  2.26,  8.79),
        "e5-small":   (0.08, 0.59,  0.18,  2.68,  9.29),
        "e5-base":    (0.25, 0.84,  0.39,  3.51, 11.21),
        "e5-large":   (0.17, 0.92,  0.34,  4.27, 12.30),
        "e5-mistral": (0.84, 3.26,  1.45,  9.71, 26.36),
    }

    col_hdr = f"{'Method':<22}  {'R@1':>6}  {'R@10':>6}  {'MRR@10':>7}  {'R@100':>7}  {'R@1000':>8}"
    sep = "─" * 74

    def _row(name: str, vals: tuple[float, ...]) -> str:
        return (
            f"  {name:<20}  "
            f"{vals[0]:6.2f}  {vals[1]:6.2f}  {vals[2]:7.2f}  {vals[3]:7.2f}  {vals[4]:8.2f}"
        )

    def _diff_row(name: str, ours: tuple[float, ...], ref: tuple[float, ...]) -> str:
        diffs = tuple(o - r for o, r in zip(ours, ref))
        def _fmt(d: float) -> str:
            return f"{d:+.2f}"
        return (
            f"  {name:<20}  "
            f"{_fmt(diffs[0]):>6}  {_fmt(diffs[1]):>6}  {_fmt(diffs[2]):>7}  "
            f"{_fmt(diffs[3]):>7}  {_fmt(diffs[4]):>8}"
        )

    print()
    print("=" * 74)
    print("  Table 22 Reproduction — Historical MBE Subset, Baseline Retrievers")
    print("  Zheng et al., CS&Law 2025 (arXiv:2505.03970)")
    print("=" * 74)

    # ── Reproduced ────────────────────────────────────────────────────────────
    print(f"\n  OUR RESULTS  (reglab/barexam_qa · Historical MBE · no query expansion)")
    print(f"  {col_hdr}")
    print(f"  {sep}")
    our_vals: dict[str, tuple[float, ...]] = {}
    for key, metrics in all_metrics.items():
        name = MODEL_REGISTRY[key]["display"]
        v = (
            metrics.get("recall@1",    0.0),
            metrics.get("recall@10",   0.0),
            metrics.get("mrr@10",      0.0),
            metrics.get("recall@100",  0.0),
            metrics.get("recall@1000", 0.0),
        )
        our_vals[key] = v
        print(_row(name, v))

    # ── Paper reference ───────────────────────────────────────────────────────
    print(f"\n  PAPER (Table 22)  (Zheng et al. 2025 — Historical MBE baseline rows)")
    print(f"  {col_hdr}")
    print(f"  {sep}")
    for key in all_metrics:
        if key in paper_ref:
            print(_row(MODEL_REGISTRY[key]["display"], paper_ref[key]))

    # ── Absolute difference (ours − paper) ───────────────────────────────────
    print(f"\n  ABSOLUTE DIFFERENCE  (ours − paper, positive = above paper)")
    print(f"  {col_hdr}")
    print(f"  {sep}")
    for key in all_metrics:
        if key in paper_ref:
            print(_diff_row(MODEL_REGISTRY[key]["display"], our_vals[key], paper_ref[key]))

    print()
    print("=" * 74)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce Table 21/22 baseline rows (Zheng et al. CS&Law 2025)"
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Which models to run (default: all). E.g. --models bm25 e5-small",
    )
    p.add_argument(
        "--model-idx",
        type=int,
        default=None,
        help=(
            "Run a single model by index (for SLURM array jobs). "
            "0=bm25 1=e5-small 2=e5-base 3=e5-large 4=e5-mistral"
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "table21_results",
        help="Directory for cached embeddings and metric JSONs.",
    )
    p.add_argument(
        "--passage-cache",
        type=Path,
        default=None,
        help="NPZ file to cache raw passage texts/IDs (avoids re-loading HF dataset).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings / metrics even if cached files exist.",
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write combined metrics to this JSON file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which model(s) to run
    if args.model_idx is not None:
        if args.model_idx < 0 or args.model_idx >= len(MODEL_IDX_ORDER):
            raise SystemExit(f"--model-idx must be 0–{len(MODEL_IDX_ORDER)-1}")
        model_keys = [MODEL_IDX_ORDER[args.model_idx]]
        logger.info("Running model [%d]: %s", args.model_idx, model_keys[0])
    else:
        model_keys = args.models

    # Load data
    passage_cache = args.passage_cache or args.results_dir / "passages_cache.npz"
    passage_ids, passage_texts = load_passages(passage_cache)
    queries, gold_ids = load_queries()

    logger.info(
        "Dataset: %d queries, %d passages (Historical MBE + full corpus)",
        len(queries), len(passage_ids),
    )

    # Run retrievers
    all_metrics: dict[str, dict[str, float]] = {}
    for key in model_keys:
        cfg = MODEL_REGISTRY[key]
        logger.info("\n=== Running: %s ===", cfg["display"])
        t0 = time.time()

        if cfg["kind"] == "bm25":
            metrics = run_bm25(queries, gold_ids, passage_ids, passage_texts, args.results_dir)
        else:
            metrics = run_dense(
                key, queries, gold_ids, passage_ids, passage_texts,
                args.results_dir, force_recompute=args.force,
            )

        elapsed = time.time() - t0
        logger.info("%s done in %.1fs", cfg["display"], elapsed)
        metrics["elapsed_s"] = round(elapsed, 1)
        all_metrics[key] = metrics

    # Print table
    print_results_table(all_metrics)

    # Merge and save summary JSON
    if args.summary_json:
        merged = {}
        # Load previously saved metrics for models not run this time
        for key in MODEL_REGISTRY:
            mf = args.results_dir / f"{key}_metrics.json"
            if key == "bm25":
                mf = args.results_dir / "bm25_metrics.json"
            if mf.exists():
                merged[key] = json.loads(mf.read_text())

        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(merged, indent=2))
        logger.info("Summary saved → %s", args.summary_json)
    elif all_metrics:
        # Always write partial summary to results dir
        partial = args.results_dir / "summary_metrics.json"
        # Load existing
        existing: dict[str, Any] = {}
        if partial.exists():
            try:
                existing = json.loads(partial.read_text())
            except Exception:
                pass
        existing.update(all_metrics)
        partial.write_text(json.dumps(existing, indent=2))
        logger.info("Partial summary → %s", partial)


if __name__ == "__main__":
    main(sys.argv[1:])
