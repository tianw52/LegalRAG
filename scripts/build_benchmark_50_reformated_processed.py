#!/usr/bin/env python3
"""Build benchmark-style JSONs: same schema as benchmarks_50 but query = rewritten text.

Reads ground truth from ``benchmarks_50/*.json`` and rewrites from
``benchmark_50_reformated/<model>/*.json``. Rows are aligned by exact ``query`` ==
``original`` string (order may differ between files).

Output: ``benchmark_50_reformated_proccessed/<model>/{contractnli,cuad,maud,privacy_qa}.json``

Usage:
  python3 scripts/build_benchmark_50_reformated_processed.py
  python3 scripts/build_benchmark_50_reformated_processed.py --models mistral qwen72b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_50 = REPO_ROOT / "data" / "legalbenchrag-mini" / "benchmarks"
REFORMATED_ROOT = REPO_ROOT / "data/legalbenchrag-mini/benchmarks_gemini3_rewrite"
OUT_ROOT = REPO_ROOT / "data" / "legalbenchrag-mini" / "benchmarks_gemini3flash_rewrite_processed"
DATASETS = ("contractnli", "cuad", "maud", "privacy_qa")
DEFAULT_VARIANT = "v4_reddit_style"


def _is_bad_rewrite(text: str) -> bool:
    if not text or not str(text).strip():
        return True
    s = str(text).strip()
    if s.startswith("[ERROR"):
        return True
    return False


def _rewrite_map_from_reformated(reformated_path: Path, variant: str) -> dict[str, str]:
    data = json.loads(reformated_path.read_text(encoding="utf-8"))
    # Support both formats:
    #   dict with "results" key (mistral, qwen72b)
    #   flat list (gemini-3-flash)
    rows = data.get("results", data) if isinstance(data, dict) else data
    m: dict[str, str] = {}
    for row in rows:
        orig = row.get("original")
        if not orig:
            continue
        # Support both "rewrites" (plural) and "rewrite" (singular)
        rw_dict = row.get("rewrites") or row.get("rewrite") or {}
        rw = rw_dict.get(variant, "")
        if isinstance(rw, str) and not _is_bad_rewrite(rw):
            m[orig] = rw.strip()
    return m


def build_one_dataset(
    dataset: str,
    model_dir: Path,
    variant: str,
) -> tuple[list[dict], dict[str, int]]:
    bench_path = BENCHMARKS_50 / f"{dataset}.json"
    ref_path = model_dir / f"{dataset}.json"
    if not bench_path.exists():
        raise FileNotFoundError(bench_path)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    rw_map = _rewrite_map_from_reformated(ref_path, variant)

    stats = {"total": 0, "rewritten": 0, "fallback_original": 0}
    tests_out: list[dict] = []

    for t in bench.get("tests", []):
        stats["total"] += 1
        query = t["query"]
        new_query = rw_map.get(query)
        if new_query is None:
            new_query = query
            stats["fallback_original"] += 1
        else:
            stats["rewritten"] += 1

        # Deep copy snippets (keep answer, span, file_path as in source)
        snippets = [dict(s) for s in t.get("snippets", [])]
        out_test = {"query": new_query, "snippets": snippets}
        if "tags" in t:
            out_test["tags"] = t["tags"]
        tests_out.append(out_test)

    return tests_out, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subdirs under benchmark_50_reformated/ (default: all that exist).",
    )
    p.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=f"Rewrite key inside reformated JSON (default: {DEFAULT_VARIANT}).",
    )
    args = p.parse_args()

    if not BENCHMARKS_50.is_dir():
        print(f"ERROR: {BENCHMARKS_50} not found", file=sys.stderr)
        sys.exit(1)
    if not REFORMATED_ROOT.is_dir():
        print(f"ERROR: {REFORMATED_ROOT} not found", file=sys.stderr)
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        subdirs = sorted(
            d.name
            for d in REFORMATED_ROOT.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        # Fall back to treating REFORMATED_ROOT itself as a flat (no-subdir) model
        models = subdirs if subdirs else ["gemini3_flash"]

    grand = {m: {"total": 0, "rewritten": 0, "fallback_original": 0} for m in models}

    for model in models:
        model_dir = REFORMATED_ROOT / model
        # If no subdir exists, use REFORMATED_ROOT directly (flat layout)
        if not model_dir.is_dir():
            model_dir = REFORMATED_ROOT
        out_model_dir = OUT_ROOT / model
        out_model_dir.mkdir(parents=True, exist_ok=True)

        for ds in DATASETS:
            try:
                tests, st = build_one_dataset(ds, model_dir, args.variant)
            except FileNotFoundError as e:
                print(f"WARN: {e} — skipping {model}/{ds}", file=sys.stderr)
                continue
            for k in grand[model]:
                grand[model][k] += st[k]

            out_path = out_model_dir / f"{ds}.json"
            payload = {
                "metadata": {
                    "schema": "LegalBench-RAG benchmark (same as benchmarks_50)",
                    "source_benchmark_dir": "benchmarks_50",
                    "rewrite_dir": f"benchmark_50_reformated/{model}",
                    "rewrite_variant": args.variant,
                    "dataset": ds,
                    "model": model,
                    "stats": st,
                    "generator": "scripts/build_benchmark_50_reformated_processed.py",
                },
                "tests": tests,
            }
            out_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {out_path}  (rewritten {st['rewritten']}/{st['total']}, fallback {st['fallback_original']})")

    print("\nSummary per model:")
    for m, st in grand.items():
        if st["total"] == 0:
            continue
        print(f"  {m}: rewritten={st['rewritten']} fallback={st['fallback_original']} total_tests={st['total']}")


if __name__ == "__main__":
    main()
