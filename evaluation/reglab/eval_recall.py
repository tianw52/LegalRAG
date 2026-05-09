"""RegLab retrieval evaluation — same CharRecall@K / CharPrecision@K as LegalBench-RAG.

This CLI is a thin wrapper around :mod:`evaluation.LegalBenchRAG.eval_precision_recall`
with **K defaults aligned to CLAUDE.md** (20, 40, 60).  Pass ``--embedding-*`` and
``--index-name`` exactly as for LegalBench-RAG so results are comparable.

Typical flow::

    python -m evaluation.reglab.prepare barexam_qa --out-dir data/reglab_eval/barexam_qa

    python -m evaluation.LegalBenchRAG.ingest \\
        --data-dir data/reglab_eval/barexam_qa \\
        --index-name legalrag-reglab-barexam-mpnet \\
        --chunker hierarchical \\
        --parent-size 2048 \\
        --chunk-size 512 \\
        --chunk-overlap 64 \\
        --embedding-provider sentence_transformers \\
        --embedding-model sentence-transformers/all-mpnet-base-v2

    python -m evaluation.reglab.eval_recall \\
        --data-dir data/reglab_eval/barexam_qa \\
        --benchmarks barexam_qa \\
        --index-name legalrag-reglab-barexam-mpnet \\
        --embedding-provider sentence_transformers \\
        --embedding-model sentence-transformers/all-mpnet-base-v2 \\
        --trace-file logs/reglab/barexam_mpnet.jsonl
"""

from __future__ import annotations

import sys

from evaluation.LegalBenchRAG.eval_precision_recall import main as _lb_eval_main

# Full LegalBench-RAG runs (see CLAUDE.md)
_DEFAULT_KS = ("20", "40", "60")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    patched = list(args)
    if "--ks" not in patched:
        patched.extend(["--ks", *_DEFAULT_KS])
    _lb_eval_main(patched)


if __name__ == "__main__":
    main()
