#!/usr/bin/env python3
"""Build benchmark-style JSONs: same schema as benchmarks_50 but query = rewritten text.

Reads ground truth from ``data/LegalBenchRAG/benchmarks_50/*.json`` and rewrites from
``data/LegalBenchRAG/benchmark_50_reformated/<model>/*.json``. Rows are aligned by exact ``query`` ==
``original`` string (order may differ between files).

Output: ``data/LegalBenchRAG/benchmark_50_reformated_proccessed/<model>/{contractnli,cuad,maud,privacy_qa}.json``

legalbenchrag-mini (same schema as ``benchmarks_50`` tests)::

  python3 scripts/build_benchmark_50_reformated_processed.py --mini --models qwen35_9b

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
# Same tree as rewrite_benchmark_50_*.py and cluster layout under data/LegalBenchRAG/
DATA_LBR = REPO_ROOT / "data" / "LegalBenchRAG"
DATA_MINI = REPO_ROOT / "data" / "legalbenchrag-mini"
BENCHMARKS_50 = DATA_LBR / "benchmarks_50"
REFORMATED_ROOT = DATA_LBR / "benchmark_50_reformated"
OUT_ROOT = DATA_LBR / "benchmark_50_reformated_proccessed"
MINI_BENCHMARKS = DATA_MINI / "benchmarks"
MINI_REFORMATED = DATA_MINI / "benchmark_50_reformated"
MINI_OUT = DATA_MINI / "benchmark_50_reformated_proccessed"
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
    m: dict[str, str] = {}
    for row in data.get("results", []):
        orig = row.get("original")
        if not orig:
            continue
        rw = (row.get("rewrites") or {}).get(variant, "")
        if isinstance(rw, str) and not _is_bad_rewrite(rw):
            m[orig] = rw.strip()
    return m


def build_one_dataset(
    dataset: str,
    model: str,
    variant: str,
    *,
    bench_root: Path,
    reform_root: Path,
) -> tuple[list[dict], dict[str, int]]:
    bench_path = bench_root / f"{dataset}.json"
    ref_path = reform_root / model / f"{dataset}.json"
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
    p.add_argument(
        "--mini",
        action="store_true",
        help="Use data/legalbenchrag-mini/{benchmarks,benchmark_50_reformated,benchmark_50_reformated_proccessed}.",
    )
    p.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Ground-truth benchmarks directory (overrides --mini / default LegalBenchRAG).",
    )
    p.add_argument(
        "--reformated-root",
        type=Path,
        default=None,
        help="Parent of <model>/ dirs with reformated JSONs.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Parent output dir (writes <out-root>/<model>/).",
    )
    args = p.parse_args()

    if args.benchmark_dir is not None:
        bench_root = args.benchmark_dir.expanduser().resolve()
    elif args.mini:
        bench_root = MINI_BENCHMARKS
    else:
        bench_root = BENCHMARKS_50

    if args.reformated_root is not None:
        reform_root = args.reformated_root.expanduser().resolve()
    elif args.mini:
        reform_root = MINI_REFORMATED
    else:
        reform_root = REFORMATED_ROOT

    if args.out_root is not None:
        out_root = args.out_root.expanduser().resolve()
    elif args.mini:
        out_root = MINI_OUT
    else:
        out_root = OUT_ROOT

    try:
        bench_dir_str = str(bench_root.relative_to(REPO_ROOT))
    except ValueError:
        bench_dir_str = str(bench_root)
    try:
        reform_str_template = str(reform_root.relative_to(REPO_ROOT)) + "/{model}"
    except ValueError:
        reform_str_template = str(reform_root) + "/{model}"

    if not bench_root.is_dir():
        print(f"ERROR: {bench_root} not found", file=sys.stderr)
        sys.exit(1)
    if not reform_root.is_dir():
        print(f"ERROR: {reform_root} not found", file=sys.stderr)
        sys.exit(1)

    if args.models:
        models = args.models
    else:
        models = sorted(
            d.name
            for d in reform_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    if not models:
        print(f"ERROR: No model subdirs found under {reform_root}", file=sys.stderr)
        sys.exit(1)

    grand = {m: {"total": 0, "rewritten": 0, "fallback_original": 0} for m in models}

    for model in models:
        model_dir = reform_root / model
        if not model_dir.is_dir():
            print(f"WARN: skip missing model dir {model_dir}", file=sys.stderr)
            continue
        out_model_dir = out_root / model
        out_model_dir.mkdir(parents=True, exist_ok=True)

        for ds in DATASETS:
            try:
                tests, st = build_one_dataset(
                    ds,
                    model,
                    args.variant,
                    bench_root=bench_root,
                    reform_root=reform_root,
                )
            except FileNotFoundError as e:
                print(f"WARN: {e} — skipping {model}/{ds}", file=sys.stderr)
                continue
            for k in grand[model]:
                grand[model][k] += st[k]

            out_path = out_model_dir / f"{ds}.json"
            payload = {
                "metadata": {
                    "schema": "LegalBench-RAG benchmark (same as benchmarks_50)",
                    "source_benchmark_dir": bench_dir_str,
                    "rewrite_dir": reform_str_template.format(model=model),
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
