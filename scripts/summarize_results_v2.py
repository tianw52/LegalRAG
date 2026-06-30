#!/usr/bin/env python3
"""Merge paper-eval JSON files from results_v2 into a single 12-config summary table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

TASK_MAP: dict[int, tuple[str, str, str]] = {
    0: ("clerc", "hier", "jhu-clsp/BERT-DPR-CLERC-ft"),
    1: ("clerc", "rec", "jhu-clsp/BERT-DPR-CLERC-ft"),
    2: ("legalbert", "hier", "nlpaueb/legal-bert-base-uncased"),
    3: ("legalbert", "rec", "nlpaueb/legal-bert-base-uncased"),
    4: ("mpnet", "hier", "sentence-transformers/all-mpnet-base-v2"),
    5: ("mpnet", "rec", "sentence-transformers/all-mpnet-base-v2"),
    6: ("legal-bge", "hier", "axondendriteplus/Legal-Embed-bge-base-en-v1.5"),
    7: ("legal-bge", "rec", "axondendriteplus/Legal-Embed-bge-base-en-v1.5"),
    8: ("octen", "hier", "Octen/Octen-Embedding-0.6B"),
    9: ("octen", "rec", "Octen/Octen-Embedding-0.6B"),
    10: ("qwen3", "hier", "Qwen/Qwen3-Embedding-0.6B"),
    11: ("qwen3", "rec", "Qwen/Qwen3-Embedding-0.6B"),
}


def find_paper_json(results_dir: Path, model: str, chunk: str, task_id: int) -> Path | None:
    """Pick the newest paper JSON for a given task (supports reruns)."""
    pattern = f"paper_barexam_{chunk}_*_{task_id}.json"
    matches = sorted(
        (results_dir / model).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def load_row(task_id: int, path: Path | None) -> dict:
    model, chunk, embed = TASK_MAP[task_id]
    if path is None:
        return {
            "task": task_id,
            "model": model,
            "chunker": chunk,
            "embedding_model": embed,
            "status": "missing",
            "source_file": None,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    rec = data.get("recall_passage_percent", {})
    return {
        "task": task_id,
        "model": model,
        "chunker": chunk,
        "embedding_model": embed,
        "status": "ok",
        "source_file": str(path),
        "job_id": path.name.split("_")[-1].replace(".json", ""),
        "n_queries": data.get("n_queries"),
        "index_name": data.get("index_name"),
        "recall_at_1": rec.get("1"),
        "recall_at_10": rec.get("10"),
        "recall_at_100": rec.get("100"),
        "recall_at_1000": rec.get("1000"),
        "mrr_at_10": data.get("mrr_10_percent"),
    }


def main() -> None:
    results_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/scratch/ram112/reglab_eval/results_v2"
    )
    rows = []
    for task_id, (model, chunk, _) in TASK_MAP.items():
        path = find_paper_json(results_dir, model, chunk, task_id)
        rows.append(load_row(task_id, path))

    out_json = results_dir / "summary_paper_eval.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"BarExam paper-eval summary — {results_dir}")
    print(f"{'Task':>4} {'Model':<10} {'Chunk':<5} {'Status':<7} {'R@1':>6} {'R@10':>6} {'R@100':>6} {'MRR@10':>7}  Job/source")
    print("-" * 90)
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['task']:>4} {r['model']:<10} {r['chunker']:<5} {'MISSING':<7}")
            continue
        print(
            f"{r['task']:>4} {r['model']:<10} {r['chunker']:<5} {'OK':<7} "
            f"{r['recall_at_1']:6.2f} {r['recall_at_10']:6.2f} {r['recall_at_100']:6.2f} {r['mrr_at_10']:7.2f}  "
            f"{Path(r['source_file']).name}"
        )

    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda x: x["recall_at_10"] or 0)
        print(
            f"\nBest R@10: task {best['task']} ({best['model']} {best['chunker']}) "
            f"= {best['recall_at_10']:.2f}%"
        )
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
