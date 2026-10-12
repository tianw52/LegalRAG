#!/usr/bin/env python3
"""
Pre-generate static data files for the Vercel-hosted eval viewer.

Usage:
    python scripts/build_static.py [--out-dir eval-viewer/data]

Outputs one JSON per (model × embedder) combo into --out-dir, plus models.json.
Commit the generated files and deploy eval-viewer/ to Vercel.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys

BASE          = pathlib.Path(__file__).resolve().parent.parent
LOGS_DIR      = BASE / "logs/eval/legalbenchrag-mini"
BENCH_DIR     = BASE / "data/legalbenchrag-mini/benchmarks"
CORPUS_DIR    = BASE / "data/legalbenchrag-mini/corpus"

_corpus: dict[str, str] = {}


def _read_corpus(file: str, s: int, e: int) -> str:
    if file not in _corpus:
        p = CORPUS_DIR / file
        try:
            _corpus[file] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _corpus[file] = ""
    return _corpus[file][s:e]


def build_combo(model: str, embedder: str, out_dir: pathlib.Path,
                sample: int | None = None, seed: int = 42) -> None:
    trace_path = LOGS_DIR / model / f"lbr_hier_{embedder}.jsonl"
    print(f"  reading {trace_path.relative_to(BASE)} …", end=" ", flush=True)

    rows: list[dict] = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "query" in obj:
                    rows.append(obj)
            except json.JSONDecodeError:
                pass

    # Per-dataset min query_idx → test array offset
    ds_min: dict[str, int] = {}
    for row in rows:
        ds = row.get("tags", ["unknown"])[0]
        qi = row.get("query_idx", 1)
        if ds not in ds_min or qi < ds_min[ds]:
            ds_min[ds] = qi

    # Benchmark tests (indexed list per dataset)
    gt_tests: dict[str, list] = {}
    for ds in ("contractnli", "cuad", "maud", "privacy_qa"):
        p = BENCH_DIR / f"{ds}.json"
        if p.exists():
            gt_tests[ds] = json.loads(p.read_text(encoding="utf-8"))["tests"]

    # Optionally downsample: pick `sample` entries per dataset (deterministic)
    if sample:
        rng = random.Random(seed)
        by_ds: dict[str, list] = {}
        for row in rows:
            ds = row.get("tags", ["unknown"])[0]
            by_ds.setdefault(ds, []).append(row)
        rows = []
        for ds_rows in by_ds.values():
            rows.extend(rng.sample(ds_rows, min(sample, len(ds_rows))))
        rows.sort(key=lambda r: r.get("query_idx", 0))

    queries = []
    for i, e in enumerate(rows):
        ds       = e.get("tags", ["unknown"])[0]
        qi       = e.get("query_idx", 1)
        test_idx = qi - ds_min.get(ds, 1)
        tests    = gt_tests.get(ds, [])
        test     = tests[test_idx] if 0 <= test_idx < len(tests) else {}
        snippets = test.get("snippets", [])

        mbk  = e.get("metrics_by_k", [])
        last = mbk[-1] if mbk else {}

        # Map chunk_id → hit info for quick lookup
        hit_map: dict[str, dict] = {
            c["chunk_id"]: c
            for c in last.get("chunk_hits", [])
        }

        retrieved = []
        for chunk in e.get("retrieved_all", []):
            cid   = chunk["chunk_id"]
            hinfo = hit_map.get(cid, {})
            is_hit = hinfo.get("is_chunk_hit", False)
            text = _read_corpus(chunk["file"], chunk["char_start"], chunk["char_end"])
            retrieved.append({
                "rank":        chunk["rank"],
                "file":        chunk["file"],
                "char_start":  chunk["char_start"],
                "char_end":    chunk["char_end"],
                "score":       chunk["score"],
                "is_hit":      is_hit,
                "gt_overlaps": hinfo.get("gt_overlaps", []) if is_hit else [],
                "text":        text,
            })

        queries.append({
            "idx":             i,
            "dataset":         ds,
            "query":           e["query"],
            "original_query":  test.get("query", ""),
            "n_gt_snippets":   last.get("n_gt_snippets", 0),
            "n_gt_hit":        last.get("n_gt_hit", 0),
            "char_recall_max": round(last.get("char_recall", 0), 4),
            "ground_truth":    snippets,
            "retrieved":       retrieved,
            # Slim metrics: only the 5 fields the charts need
            "metrics_by_k": [
                {
                    "k":               m["k"],
                    "char_recall":     m["char_recall"],
                    "char_precision":  m["char_precision"],
                    "chunk_recall":    m["chunk_recall"],
                    "chunk_precision": m["chunk_precision"],
                }
                for m in mbk
            ],
        })

    out_file = out_dir / f"{model}__{embedder}.json"
    out_file.write_text(
        json.dumps({"model": model, "embedder": embedder, "queries": queries},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    size_mb = out_file.stat().st_size / 1024 / 1024
    print(f"{len(queries)} queries → {out_file.name} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="eval-viewer/data",
                        help="Output directory for generated JSON files")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N queries per sub-dataset (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    out_dir = BASE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all (model, embedder) combinations
    models: dict[str, list[str]] = {}
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir():
            continue
        embs = sorted(
            f.stem.removeprefix("lbr_hier_")
            for f in d.glob("lbr_hier_*.jsonl")
        )
        if embs:
            models[d.name] = embs

    # Write models index
    models_file = out_dir / "models.json"
    models_file.write_text(json.dumps(models, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {models_file.relative_to(BASE)}")

    # Write one combo file per (model, embedder)
    total = sum(len(embs) for embs in models.values())
    done  = 0
    for model, embs in models.items():
        for emb in embs:
            done += 1
            print(f"[{done}/{total}] {model} / {emb}")
            build_combo(model, emb, out_dir, sample=args.sample, seed=args.seed)

    # Report total size
    total_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1024 / 1024
    print(f"\nDone. Total data size: {total_mb:.1f} MB in {out_dir.relative_to(BASE)}/")


if __name__ == "__main__":
    main()
