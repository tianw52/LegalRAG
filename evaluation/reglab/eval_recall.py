"""RegLab retrieval evaluation — LegalBench-style CharRecall@K / CharPrecision@K.

For **passage-level** Recall@{1,10,100,1000} and MRR@10 matching Zheng et al.
(CS&Law 2025), use :mod:`evaluation.reglab.paper_eval` instead.
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
