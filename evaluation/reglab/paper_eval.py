"""Paper-aligned RegLab retrieval evaluation (Zheng et al., CS&Law 2025).

Computes passage-level **Recall@1, @10, @100, @1000** and **MRR@10** on the
ranked list of *unique* corpus passages obtained from LegalRAG chunk retrieval
(see :mod:`evaluation.reglab.paper_metrics`).

* **Bar Exam QA**: search the full passage index; one gold passage per question.
* **Housing Statute QA**: apply the **jurisdiction filter** (``court`` =
  query state) so retrieval matches Section 5.2; use ``--housing-recall``
  ``upper`` (main paper) or ``lower`` (Appendix H — all gold statutes in top-K).

Example (disk export)::

    python -m evaluation.reglab.paper_eval \\
        --source disk \\
        --data-dir data/reglab_eval/barexam_qa \\
        --benchmark barexam_qa \\
        --index-name legalrag-reglab-barexam-mpnet \\
        --embedding-provider sentence_transformers \\
        --embedding-model sentence-transformers/all-mpnet-base-v2

Example (Hugging Face, no disk export)::

    python -m evaluation.reglab.paper_eval \\
        --source hf \\
        --benchmark barexam_qa \\
        --index-name legalrag-reglab-barexam-hf

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
from typing import Callable

from legalrag.core.config import settings
from legalrag.core.models import StructuredQuery
from legalrag.ingestion.embedder import build_embedder
from legalrag.opensearch.client import OpenSearchClient, OpenSearchSettings
from legalrag.query.retriever import OpenSearchRetriever
from legalrag.utils.logging import configure_logging

from evaluation.LegalBenchRAG.pipeline import DEFAULT_INDEX_NAME
from evaluation.reglab.hf_reglab import (
    HFLoadStats,
    load_barexam_hf_benchmark_tests,
    load_housing_hf_benchmark_tests,
)
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


def _housing_court_for_filter(ex: PaperEvalExample) -> str | None:
    """Return ``court`` term matching the index (folder under ``statutes/``).

    Disk/HF benchmarks use Title Case ``jurisdiction`` while Parquet citations
    lower-case that folder segment; an exact OpenSearch filter on
    ``jurisdiction`` alone returns no hits.
    """
    if not ex.jurisdiction:
        return None
    for g in ex.gold_paths:
        parts = g.split("/")
        if len(parts) >= 3 and parts[0] == "statutes":
            return parts[1]
    j = ex.jurisdiction.strip()
    return j.casefold() if j else None


def load_paper_examples(benchmarks_dir: Path, benchmark: str) -> list[PaperEvalExample]:
    path = benchmarks_dir / f"{benchmark}.json"
    return load_paper_examples_from_file(path)


def load_paper_examples_from_file(path: Path) -> list[PaperEvalExample]:
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


def load_paper_examples_hf(
    benchmark: str,
    *,
    limit_queries: int | None = None,
) -> tuple[list[PaperEvalExample], HFLoadStats]:
    """Load examples from ``reglab/*`` on Hugging Face (no disk JSON)."""
    if benchmark == "barexam_qa":
        tests, st = load_barexam_hf_benchmark_tests(
            limit_queries=limit_queries,
            passage_map_limit=None,
        )
    else:
        tests, st = load_housing_hf_benchmark_tests(
            limit_queries=limit_queries,
            statute_map_limit=None,
        )
    out: list[PaperEvalExample] = []
    for t in tests:
        gold = frozenset(s.file_path for s in t.snippets)
        if not gold:
            continue
        out.append(
            PaperEvalExample(
                query=t.query,
                gold_paths=gold,
                jurisdiction=t.jurisdiction,
            )
        )
    return out, st


def build_retriever(
    top_k: int,
    index_name: str = DEFAULT_INDEX_NAME,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    retrieval_mode: str = "hybrid",
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
    # BM25-only: skip loading any embedding model; OpenSearchClient needs a
    # non-zero dim but we won't call knn_search / ensure_index here.
    if retrieval_mode == "lexical":
        from legalrag.ingestion.embedder import NullEmbedder
        embedder = NullEmbedder()
    else:
        embedder = build_embedder(model_name=embedding_model, provider=embedding_provider)
    os_client = OpenSearchClient(cfg=lb_cfg, embedding_dim=embedder.dim)
    if retrieval_mode in ("hybrid", "semantic"):
        os_client._ensure_hybrid_pipeline()
    return OpenSearchRetriever(
        os_client,
        embedder,
        mode=retrieval_mode,  # type: ignore[arg-type]
        top_k=top_k,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        "--reglab-source",
        dest="source",
        choices=("disk", "hf"),
        default="disk",
        help="Load queries from disk JSON (default) or Hugging Face ``reglab/*``.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="RegLab export root (contains benchmarks/). Required when --source disk.",
    )
    p.add_argument(
        "--benchmark",
        required=True,
        choices=("barexam_qa", "housing_qa"),
        help="Which benchmark JSON to load.",
    )
    p.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Override benchmark JSON path (e.g. housing_qa_expanded.json for query expansion).",
    )
    p.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="OpenSearch index name.")
    p.add_argument(
        "--embedding-provider",
        default=None,
        choices=["sentence_transformers", "huggingface", "openai"],
    )
    p.add_argument("--embedding-model", default=None)
    p.add_argument(
        "--retrieval-mode",
        choices=("semantic", "lexical", "hybrid"),
        default="hybrid",
        help="Retrieval mode: semantic (dense-only), lexical (BM25-only), hybrid (RRF).",
    )
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
        help=(
            "Cap examples: for --source disk, take first N after loading full JSON; "
            "for --source hf, pass N as limit_queries when loading (faster)."
        ),
    )
    p.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Write aggregate metrics JSON to this path.",
    )
    p.add_argument(
        "--min-index-docs",
        type=int,
        default=0,
        help=(
            "Require at least this many documents in the OpenSearch index before "
            "evaluating. If below threshold, write an invalid results JSON and exit 2."
        ),
    )
    p.add_argument(
        "--fail-on-zero-recall",
        action="store_true",
        help=(
            "If index is populated but Recall@1000 is 0%%, mark the run invalid "
            "(likely filter or gold-path mismatch)."
        ),
    )
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def _write_invalid_results(
    path: Path,
    *,
    reason: str,
    index_name: str,
    index_document_count: int,
    min_index_docs: int,
    benchmark: str,
    retrieval_mode: str,
    housing_recall_mode: str | None,
    embedding_model: str | None,
    n_queries: int,
    top_chunks: int,
) -> None:
    payload = {
        "status": "invalid",
        "invalid_reason": reason,
        "benchmark": benchmark,
        "source": "disk",
        "paper": "Zheng et al. CS&Law 2025",
        "n_queries": n_queries,
        "top_chunks": top_chunks,
        "retrieval_mode": retrieval_mode,
        "housing_recall_mode": housing_recall_mode,
        "jurisdiction_filter": "enabled" if benchmark == "housing_qa" else None,
        "recall_cutoffs": list(DEFAULT_RECALL_CUTOFFS),
        "recall_passage_percent": None,
        "mrr_10_percent": None,
        "index_name": index_name,
        "index_document_count": index_document_count,
        "min_index_docs_required": min_index_docs,
        "embedding_model": embedding_model or "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    hf_stats: HFLoadStats | None = None
    if args.source == "disk":
        if not args.data_dir:
            print("--data-dir is required when --source disk", file=sys.stderr)
            sys.exit(2)
        data_dir = Path(args.data_dir).resolve()
        if args.benchmark_file is not None:
            examples = load_paper_examples_from_file(args.benchmark_file.resolve())
        else:
            examples = load_paper_examples(data_dir / "benchmarks", args.benchmark)
    else:
        examples, hf_stats = load_paper_examples_hf(
            args.benchmark,
            limit_queries=args.limit,
        )
        logger.info(
            "paper_eval HF: queries_loaded=%d gold_snippets=%d dropped_queries=%d map_rows=%d",
            hf_stats.n_queries_loaded,
            hf_stats.n_gold_snippets,
            hf_stats.n_queries_missing_any_gold,
            hf_stats.n_corpus_rows_scanned_for_maps,
        )

    if args.source == "disk" and args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        print("No examples loaded.", file=sys.stderr)
        sys.exit(1)

    ks = sorted(set(args.ks))
    recall_fn: Callable[[list[str], set[str], int], float]
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
        retrieval_mode=args.retrieval_mode,
    )

    index_document_count = retriever._client.document_count()
    if args.min_index_docs > 0 and index_document_count < args.min_index_docs:
        msg = (
            f"Index '{args.index_name}' has {index_document_count} documents "
            f"(required >= {args.min_index_docs}). Marking run invalid."
        )
        print(msg, file=sys.stderr)
        if args.results_json:
            _write_invalid_results(
                args.results_json,
                reason="empty_or_underpopulated_index",
                index_name=args.index_name,
                index_document_count=index_document_count,
                min_index_docs=args.min_index_docs,
                benchmark=args.benchmark,
                retrieval_mode=args.retrieval_mode,
                housing_recall_mode=housing_mode if args.benchmark == "housing_qa" else None,
                embedding_model=args.embedding_model,
                n_queries=n,
                top_chunks=args.top_chunks,
            )
            print(f"\n  Wrote invalid results → {args.results_json}", file=sys.stderr)
        sys.exit(2)

    # Accumulate sums for means (paper reports percentages).
    n = len(examples)
    recall_sums = {k: 0.0 for k in ks}
    mrr_sum = 0.0

    print(
        f"\nPaper-style passage retrieval — {args.benchmark}, source={args.source}, n={n}, "
        f"top_chunks={args.top_chunks}, retrieval_mode={args.retrieval_mode}, "
        f"housing_recall={housing_mode if args.benchmark == 'housing_qa' else 'n/a'}\n"
    )

    for i, ex in enumerate(examples, 1):
        sq = StructuredQuery(
            raw_query=ex.query,
            reformulated_query=ex.query,
            court_filter=_housing_court_for_filter(ex) if args.benchmark == "housing_qa" else None,
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

    max_k = max(ks)
    if (
        args.fail_on_zero_recall
        and index_document_count > 0
        and recall_mean.get(max_k, 0.0) == 0.0
    ):
        msg = (
            f"Index '{args.index_name}' has {index_document_count} documents but "
            f"Recall@{max_k} is 0% — likely jurisdiction filter or gold-path mismatch."
        )
        print(msg, file=sys.stderr)
        if args.results_json:
            _write_invalid_results(
                args.results_json,
                reason="zero_recall_despite_populated_index",
                index_name=args.index_name,
                index_document_count=index_document_count,
                min_index_docs=args.min_index_docs,
                benchmark=args.benchmark,
                retrieval_mode=args.retrieval_mode,
                housing_recall_mode=housing_mode if args.benchmark == "housing_qa" else None,
                embedding_model=args.embedding_model,
                n_queries=n,
                top_chunks=args.top_chunks,
            )
            print(f"\n  Wrote invalid results → {args.results_json}", file=sys.stderr)
        sys.exit(2)

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
            "status": "valid",
            "benchmark": args.benchmark,
            "source": args.source,
            "paper": "Zheng et al. CS&Law 2025",
            "n_queries": n,
            "top_chunks": args.top_chunks,
            "retrieval_mode": args.retrieval_mode,
            "housing_recall_mode": housing_mode if args.benchmark == "housing_qa" else None,
            "jurisdiction_filter": "enabled" if args.benchmark == "housing_qa" else None,
            "recall_cutoffs": ks,
            "recall_passage_percent": {str(k): round(100 * recall_mean[k], 4) for k in ks},
            f"mrr_{args.mrr_cutoff}_percent": round(100 * mrr_mean, 4),
            "index_name": args.index_name,
            "index_document_count": index_document_count,
            "min_index_docs_required": args.min_index_docs,
            "embedding_model": args.embedding_model or "",
        }
        if hf_stats is not None:
            payload["hf_load_stats"] = {
                "n_queries_loaded": hf_stats.n_queries_loaded,
                "n_gold_snippets": hf_stats.n_gold_snippets,
                "n_queries_missing_any_gold": hf_stats.n_queries_missing_any_gold,
                "n_corpus_rows_scanned_for_maps": hf_stats.n_corpus_rows_scanned_for_maps,
            }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n  Wrote {args.results_json}")


if __name__ == "__main__":
    main(sys.argv[1:])
