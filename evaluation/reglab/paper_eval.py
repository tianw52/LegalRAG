"""Paper-aligned RegLab retrieval evaluation (Zheng et al., CS&Law 2025).

Computes passage-level **Recall@1, @10, @100, @1000** and **MRR@10** on the
ranked list of *unique* corpus passages obtained from LegalRAG chunk retrieval
(see :mod:`evaluation.reglab.paper_metrics`).

* **Bar Exam QA**: search the full passage index; one gold passage per question.
* **Housing Statute QA**: apply the **jurisdiction filter** (``court`` =
  query state) so retrieval matches Section 5.2; use ``--housing-recall``
  ``upper`` (main paper) or ``lower`` (Appendix H — all gold statutes in top-K).

Example::

    python -m evaluation.reglab.paper_eval \\
        --data-dir data/reglab_eval/barexam_qa \\
        --benchmark barexam_qa \\
        --index-name legalrag-reglab-barexam-mpnet \\
        --embedding-provider sentence_transformers \\
        --embedding-model sentence-transformers/all-mpnet-base-v2

Embedding provider/model must match the ingested index.  Re-ingest Housing
with :mod:`evaluation.reglab.ingest_reglab` so ``court`` is set on statute chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legalrag.core.config import settings
from legalrag.core.models import StructuredQuery
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME
from evaluation.reglab.paper_metrics import (
    mrr_at_cutoff,
    ranked_passage_citations,
    recall_at_k_lower,
    recall_at_k_upper,
)

logger = logging.getLogger(__name__)

# Table 21+ style cutoffs (percentages in the paper).
DEFAULT_RECALL_CUTOFFS = (1, 10, 100, 1000)
DEFAULT_MRR_CUTOFF = 10


@dataclass
class PaperEvalExample:
    query: str
    gold_paths: frozenset[str]
    jurisdiction: str | None


def load_paper_examples(benchmarks_dir: Path, benchmark: str) -> list[PaperEvalExample]:
    path = benchmarks_dir / f"{benchmark}.json"
    if not path.is_file():
        raise SystemExit(f"Benchmark file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    tests = raw.get("tests", [])
    out: list[PaperEvalExample] = []
    for t in tests:
        gold = frozenset(s["file_path"] for s in t.get("snippets", []))
        if not gold:
            continue
        out.append(
            PaperEvalExample(
                query=t["query"],
                gold_paths=gold,
                jurisdiction=t.get("jurisdiction"),
            )
        )
    return out


def build_retriever(
    top_k: int,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
) -> OpenSearchRetriever:
    cfg = settings.opensearch
    lb_cfg = OpenSearchSettings(
        **{
            "OPENSEARCH_HOST": cfg.host,
            "OPENSEARCH_PORT": cfg.port,
            "OPENSEARCH_USER": cfg.user,
            "OPENSEARCH_PASSWORD": cfg.password,
            "OPENSEARCH_USE_SSL": cfg.use_ssl,
            "OPENSEARCH_INDEX_NAME": index_name,
        }
    )
    embedder = build_embedder(model_name=embedding_model, provider=embedding_provider)
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    os_client._ensure_hybrid_pipeline()
    return OpenSearchRetriever(os_client, embedder, mode="hybrid", top_k=top_k)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="RegLab export root (contains benchmarks/).")
    p.add_argument(
        "--benchmark",
        required=True,
        choices=("barexam_qa", "housing_qa"),
        help="Which benchmark JSON to load.",
    )
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="OpenSearch index name.")
    p.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument(
        "--top-chunks",
        type=int,
        default=50_000,
        help=(
            "Retrieve this many chunks, then dedupe to passage ranking. "
            "Must be large enough to surface Recall@k unique passages (default 50000)."
        ),
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_RECALL_CUTOFFS),
        metavar="K",
        help="Recall@K passage cutoffs (default: 1 10 100 1000).",
    )
    p.add_argument(
        "--mrr-cutoff",
        type=int,
        default=DEFAULT_MRR_CUTOFF,
        help="MRR cutoff (paper uses 10).",
    )
    p.add_argument(
        "--housing-recall",
        choices=("upper", "lower"),
        default="upper",
        help="Housing only: upper = any gold passage (main paper); lower = all gold (Appendix H).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples (debug).",
    )
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Write aggregate metrics JSON to this path.",
    )
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    data_dir = Path(args.data_dir).resolve()
    examples = load_paper_examples(data_dir / "benchmarks", args.benchmark)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        print("No examples loaded.", file=sys.stderr)
        sys.exit(1)

    ks = sorted(set(args.ks))
    max_k = max(ks)
    recall_fn: Any
    if args.benchmark == "housing_qa" and args.housing_recall == "lower":
        recall_fn = recall_at_k_lower
        housing_mode = "lower"
    else:
        recall_fn = recall_at_k_upper
        housing_mode = "upper"

    retriever = build_retriever(
        top_k=args.top_chunks,
        index_name=args.index_name,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
    )

    # Accumulate sums for means (paper reports percentages).
    n = len(examples)
    recall_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0

    print(
        f"\nPaper-style passage retrieval — {args.benchmark}, n={n}, "
        f"top_chunks={args.top_chunks}, housing_recall={housing_mode if args.benchmark == 'housing_qa' else 'n/a'}\n"
    )

    for i, ex in enumerate(examples, 1):
        sq = StructuredQuery(
            raw_query=ex.query,
            reformulated_query=ex.query,
            court_filter=ex.jurisdiction if args.benchmark == "housing_qa" else None,
        )
        results = retriever.retrieve(sq)
        passage_rank = ranked_passage_citations(results)

        for k in ks:
            recall_sums[k] += recall_fn(passage_rank, set(ex.gold_paths), k)
        mrr_sum += mrr_at_cutoff(passage_rank, set(ex.gold_paths), args.mrr_cutoff)

        if i % 100 == 0 and args.log_level in ("DEBUG", "INFO"):
            logger.info("Processed %d / %d queries", i, n)

    recall_mean = {k: recall_sums[k] / n for k in ks}
    mrr_mean = mrr_sum / n

    # Print table (percentages like the paper).
    print("  Metric          " + "  ".join(f"@{k:4d}" for k in ks))
    print("  " + "─" * (18 + 7 * len(ks)))
    print(
        "  Recall (passage)"
        + "".join(f"  {100 * recall_mean[k]:5.2f}" for k in ks)
        + "   (%)"
    )
    print(f"  MRR@{args.mrr_cutoff:<8}" + f"{'':20s}" f"{100 * mrr_mean:5.2f}   (%)")
    print(f"\n  Index: {args.index_name}")

    if args.results_json:
        payload = {
            "benchmark": args.benchmark,
            "paper": "Zheng et al. CS&Law 2025",
            "n_queries": n,
            "top_chunks": args.top_chunks,
            "housing_recall_mode": housing_mode if args.benchmark == "housing_qa" else None,
            "recall_cutoffs": ks,
            "recall_passage_percent": {str(k): round(100 * recall_mean[k], 4) for k in ks},
            f"mrr_{args.mrr_cutoff}_percent": round(100 * mrr_mean, 4),
            "index_name": args.index_name,
            "embedding_model": args.embedding_model or "",
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n  Wrote {args.results_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
