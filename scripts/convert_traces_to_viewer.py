#!/usr/bin/env python3
"""
Convert our JSONL evaluation trace files to Tian's eval-viewer JSON format.

Outputs eval-viewer/data/{dataset}_{chunker}__{embedder}.json files
and updates eval-viewer/data/models.json.

Usage:
    python3 scripts/convert_traces_to_viewer.py [--out-dir eval-viewer/data]
"""

import argparse
import json
import os
import glob
import sys
from pathlib import Path

# ── configuration ──────────────────────────────────────────────────────────────
BAREXAM_RESULTS = "/scratch/ram112/reglab_eval/results"
HOUSING_RESULTS = "/scratch/ram112/reglab_eval/results_housing"

# Map subdirectory names → short model names used in output file names
DIR_TO_EMBED = {
    "clerc":      "clerc-ft",
    "legalbert":  "legalbert",
    "mpnet":      "mpnet",
    "legal-bge":  "legal-bge",
    "octen":      "octen",
    "qwen3":      "qwen3",
}

# (dataset_tag, chunker, results_root, glob_pattern)
TRACE_SPECS = [
    ("barexam", "hier", BAREXAM_RESULTS, "char_barexam_hier_*.jsonl"),
    ("barexam", "rec",  BAREXAM_RESULTS, "char_barexam_rec_*.jsonl"),
    ("housing", "hier", HOUSING_RESULTS, "char_housing_hier_*.jsonl"),
    ("housing", "rec",  HOUSING_RESULTS, "char_housing_rec_*.jsonl"),
]

# K values we want to emit in metrics_by_k
TARGET_KS = {2, 10, 20, 60}


# ── helpers ────────────────────────────────────────────────────────────────────

def select_best_file(paths):
    """Return the file with the most query records (break ties by mtime)."""
    best, best_count = None, -1
    for p in paths:
        count = 0
        try:
            with open(p) as f:
                for line in f:
                    if line.strip().startswith("{"):
                        count += 1
        except Exception:
            continue
        if count > best_count or (count == best_count and (best is None or os.path.getmtime(p) > os.path.getmtime(best))):
            best, best_count = p, count
    return best


def read_jsonl_trace(fpath):
    records = []
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
                if "metrics_by_k" in rec:
                    records.append(rec)
            except Exception:
                pass
    return records


def compute_gt_overlaps(chunk, ground_truth_raw):
    """Return gt_overlaps list for a single retrieved chunk."""
    overlaps = []
    c_start = int(chunk.get("char_start", 0))
    c_end   = int(chunk.get("char_end",   0))
    c_len   = c_end - c_start
    c_base  = os.path.basename(chunk.get("file", "")).replace(".txt", "")

    for gt in ground_truth_raw:
        gt_base = os.path.basename(gt.get("file", "")).replace(".txt", "")
        if gt_base != c_base:
            continue
        gt_start, gt_end = gt.get("span", [0, 0])
        gt_len = gt_end - gt_start

        ov_s = max(c_start, gt_start)
        ov_e = min(c_end,   gt_end)
        if ov_s >= ov_e:
            continue

        ov_chars = ov_e - ov_s
        overlaps.append({
            "overlap_span":        [ov_s, ov_e],
            "overlap_chars":       ov_chars,
            "overlap_pct_of_gt":   round(ov_chars / gt_len   * 100, 2) if gt_len   > 0 else 0.0,
            "overlap_pct_of_chunk": round(ov_chars / c_len   * 100, 2) if c_len    > 0 else 0.0,
        })
    return overlaps


def convert_query(rec, dataset_tag, top_k_display=20, max_answer_chars=500):
    """Convert one JSONL query record → Tian viewer format."""
    ground_truth_raw = rec.get("ground_truth", [])
    gt_files_needed  = {os.path.basename(g.get("file", "")).replace(".txt", "") for g in ground_truth_raw}

    ground_truth = [
        {
            "file_path": g.get("file", ""),
            "span":      g.get("span", [0, 0]),
            "answer":    (g.get("gt_text") or "")[:max_answer_chars],
        }
        for g in ground_truth_raw
    ]

    # Sort metrics by k so we can identify top-K retrieved list
    sorted_mbk = sorted(rec.get("metrics_by_k", []), key=lambda x: x["k"])

    # Build retrieved list from the largest available K
    retrieved      = []
    char_recall_max = 0.0
    n_gt_hit        = 0
    metrics_out     = []

    for mbk in sorted_mbk:
        k      = mbk["k"]
        cr     = mbk.get("char_recall", 0.0)
        cp     = mbk.get("char_precision", 0.0)
        char_recall_max = max(char_recall_max, cr)
        chunks = mbk.get("top_k_chunks", [])

        # Chunk-level metrics derived from gt_overlap boolean
        n_hit_chunks = sum(1 for c in chunks if str(c.get("gt_overlap")) in ("True", "true", "1"))
        chunk_precision = n_hit_chunks / k if k > 0 else 0.0

        gt_files_hit = set()
        for c in chunks:
            if str(c.get("gt_overlap")) in ("True", "true", "1"):
                gt_files_hit.add(os.path.basename(c.get("file", "")).replace(".txt", ""))
        chunk_recall = (
            len(gt_files_hit & gt_files_needed) / len(gt_files_needed)
            if gt_files_needed else 0.0
        )

        if k in TARGET_KS:
            metrics_out.append({
                "k":               k,
                "char_recall":     round(cr, 6),
                "char_precision":  round(cp, 6),
                "chunk_recall":    round(chunk_recall,    4),
                "chunk_precision": round(chunk_precision, 4),
            })

        # Build the display retrieved list from the largest K (capped to top_k_display)
        # Always include the first hit chunk if it falls outside the top_k_display window.
        if not sorted_mbk or k == sorted_mbk[-1]["k"]:
            hit_indices = [i for i, c in enumerate(chunks)
                           if str(c.get("gt_overlap")) in ("True", "true", "1")]
            first_hit_idx = hit_indices[0] if hit_indices else None

            display_indices = set(range(min(top_k_display, len(chunks))))
            if first_hit_idx is not None and first_hit_idx not in display_indices:
                display_indices.add(first_hit_idx)

            for i in sorted(display_indices):
                c = chunks[i]
                gt_overlaps = compute_gt_overlaps(c, ground_truth_raw)
                retrieved.append({
                    "rank":        int(c.get("rank", 0)),
                    "is_hit":      bool(gt_overlaps),
                    "file":        c.get("file", ""),
                    "char_start":  int(c.get("char_start", 0)),
                    "char_end":    int(c.get("char_end",   0)),
                    "score":       float(c.get("score", 0.0)),
                    "text":        c.get("chunk_text", ""),
                    "gt_overlaps": gt_overlaps,
                })

            # n_gt_hit based on the retrieved display list (consistent with viewer)
            display_files_hit = {
                os.path.basename(c.get("file", "")).replace(".txt", "")
                for c in retrieved if c["is_hit"]
            }
            n_gt_hit = len(display_files_hit & gt_files_needed)

    return {
        "idx":             rec.get("query_idx", 0),
        "dataset":         dataset_tag,
        "query":           rec.get("query", ""),
        "original_query":  rec.get("query", ""),
        "n_gt_snippets":   len(ground_truth),
        "n_gt_hit":        n_gt_hit,
        "char_recall_max": round(char_recall_max, 6),
        "ground_truth":    ground_truth,
        "metrics_by_k":    metrics_out,
        "retrieved":       retrieved,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL traces → eval-viewer JSON")
    parser.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent.parent / "eval-viewer" / "data"
    ))
    parser.add_argument(
        "--top-k-display", type=int, default=20,
        help="Max retrieved chunks stored per query for display (default 20). "
             "Metrics are still computed for all K values up to 60."
    )
    parser.add_argument(
        "--housing-top-k-display", type=int, default=5,
        help="Override top-k-display for housing_qa files (default 5, keeps files <50 MB)."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["barexam", "housing"],
        help="Only process these dataset(s). Default: all."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_map = {}   # model_key → [embedder_short, ...]
    generated  = []

    top_k_display         = args.top_k_display
    housing_top_k_display = args.housing_top_k_display
    filter_datasets       = set(args.datasets) if args.datasets else None

    for dataset_tag, chunker, results_root, pattern in TRACE_SPECS:
        if filter_datasets and dataset_tag not in filter_datasets:
            continue
        model_key = f"{dataset_tag}_{chunker}"

        for dir_name, embed_short in DIR_TO_EMBED.items():
            full_pattern = os.path.join(results_root, dir_name, pattern)
            candidates   = glob.glob(full_pattern)
            if not candidates:
                print(f"  [SKIP] no files for {model_key}/{embed_short}")
                continue

            best = select_best_file(candidates)
            print(f"  Converting {model_key}__{embed_short}  ← {os.path.basename(best)}")

            effective_top_k   = housing_top_k_display if dataset_tag == "housing" else top_k_display
            max_answer_chars  = 500 if dataset_tag == "housing" else 2000
            records  = read_jsonl_trace(best)
            queries  = [convert_query(r, f"{dataset_tag}_qa", effective_top_k, max_answer_chars) for r in records]

            out_file = out_dir / f"{model_key}__{embed_short}.json"
            with open(out_file, "w") as f:
                json.dump(
                    {"model": model_key, "embedder": embed_short, "queries": queries},
                    f, ensure_ascii=False, separators=(",", ":"),
                )

            size_mb = out_file.stat().st_size / 1_048_576
            print(f"    → {out_file.name}  ({len(queries)} queries, {size_mb:.1f} MB)")
            generated.append(out_file.name)

            models_map.setdefault(model_key, [])
            if embed_short not in models_map[model_key]:
                models_map[model_key].append(embed_short)

    # Write models.json
    models_file = out_dir / "models.json"
    # Preserve any existing models Tian already has
    if models_file.exists():
        try:
            existing = json.loads(models_file.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}
    existing.update(models_map)
    models_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(generated)} data files + models.json  →  {out_dir}")


if __name__ == "__main__":
    main()
