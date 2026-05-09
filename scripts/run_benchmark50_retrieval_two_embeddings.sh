#!/usr/bin/env bash
# Retrieval eval: index corpus, then evaluate on benchmark_50 (reformatted queries).
#
#   Ingest (pick one):
#     - Default: --benchmarks-dir INGEST_BENCHMARKS_DIR (default: data/LegalBenchRAG/benchmarks)
#     - Full corpus: INGEST_ALL=1  →  ingest --all (every *.txt under corpus/; heaviest, complete)
#   Eval: --benchmarks-dir EVAL_BENCHMARKS_DIR (default: .../benchmark_50_reformated_proccessed/<variant>)
#
# Note: `benchmarks/` ingest = all files *referenced* by benchmark JSONs. `--all` = entire corpus tree.
#
# Prerequisites:
#   - OpenSearch on OPENSEARCH_HOST:OPENSEARCH_PORT
#   - corpus at DATA_DIR/corpus
#
# Usage:
#   ./scripts/run_benchmark50_retrieval_two_embeddings.sh
#
# Optional env:
#   INGEST_ALL=1                      (ingest entire corpus with --all; ignores INGEST_BENCHMARKS_DIR)
#   DATA_DIR                          (default: <repo>/data/LegalBenchRAG)
#   INGEST_BENCHMARKS_DIR             (default: $DATA_DIR/benchmarks; unused if INGEST_ALL=1)
#   EVAL_BENCHMARKS_DIR               (override eval dir entirely)
#   REWRITE_VARIANT=mistral|qwen72b   (only if EVAL_BENCHMARKS_DIR unset; default: mistral)
#   LIMIT=10                          (quick smoke: cap queries per benchmark)
#   KS="1 5 10 20"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/LegalBenchRAG}"
INGEST_ALL="${INGEST_ALL:-0}"
INGEST_BENCHMARKS_DIR="${INGEST_BENCHMARKS_DIR:-$DATA_DIR/benchmarks}"

REWRITE_VARIANT="${REWRITE_VARIANT:-mistral}"
EVAL_BENCHMARKS_DIR="${EVAL_BENCHMARKS_DIR:-$DATA_DIR/benchmark_50_reformated_proccessed/${REWRITE_VARIANT}}"

EMBED_DEFAULT="${EMBED_DEFAULT:-nlpaueb/legal-bert-base-uncased}"
EMBED_LEGAL="${EMBED_LEGAL:-axondendriteplus/Legal-Embed-bge-base-en-v1.5}"

INDEX_NAME="${INDEX_NAME:-legalrag-legalbenchrag}"
KS="${KS:-1 5 10 20}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-sentence_transformers}"

if [[ ! -d "$DATA_DIR/corpus" ]]; then
  echo "ERROR: corpus not found: $DATA_DIR/corpus"
  exit 1
fi
if [[ "$INGEST_ALL" != "1" ]]; then
  if [[ ! -f "$INGEST_BENCHMARKS_DIR/cuad.json" ]]; then
    echo "ERROR: ingest benchmarks not found (expected e.g. cuad.json): $INGEST_BENCHMARKS_DIR"
    echo "       Or set INGEST_ALL=1 to ingest the full corpus (--all)."
    exit 1
  fi
fi
if [[ ! -f "$EVAL_BENCHMARKS_DIR/cuad.json" ]]; then
  echo "ERROR: eval benchmarks not found: $EVAL_BENCHMARKS_DIR"
  exit 1
fi

OS_URL="http://${OPENSEARCH_HOST:-localhost}:${OPENSEARCH_PORT:-9200}"

run_one() {
  local NAME="$1"
  local MODEL="$2"
  echo ""
  echo "██████████████████████████████████████████████████████████████████████████████"
  echo "  EXPERIMENT: ${NAME}"
  echo "  EMBEDDING MODEL: ${MODEL}"
  echo "  data-dir:        ${DATA_DIR}"
  if [[ "$INGEST_ALL" == "1" ]]; then
    echo "  INGEST (index):  --all  (entire corpus/*.txt)"
  else
    echo "  INGEST (index):  benchmarks-dir=${INGEST_BENCHMARKS_DIR}"
  fi
  echo "  EVAL (queries):  ${EVAL_BENCHMARKS_DIR}"
  echo "██████████████████████████████████████████████████████████████████████████████"
  echo ""

  echo ">>> Dropping index ${INDEX_NAME} (required when embedding dim/model changes)..."
  curl -sS -X DELETE "${OS_URL}/${INDEX_NAME}" >/dev/null || true

  if [[ "$INGEST_ALL" == "1" ]]; then
    echo ">>> Ingesting FULL corpus (--all; every *.txt under corpus/)..."
    python3 -m evaluation.LegalBenchRAG.ingest \
      --data-dir "$DATA_DIR" \
      --all \
      --embedding-model "$MODEL" \
      --index-name "$INDEX_NAME"
  else
    echo ">>> Ingesting corpus (files referenced by INGEST benchmarks)..."
    python3 -m evaluation.LegalBenchRAG.ingest \
      --data-dir "$DATA_DIR" \
      --benchmarks-dir "$INGEST_BENCHMARKS_DIR" \
      --embedding-model "$MODEL" \
      --index-name "$INDEX_NAME"
  fi

  echo ">>> eval_precision_recall (all queries in eval JSONs; no limit unless LIMIT is set)..."
  python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir "$DATA_DIR" \
    --benchmarks-dir "$EVAL_BENCHMARKS_DIR" \
    --index-name "$INDEX_NAME" \
    --embedding-model "$MODEL" \
    --ks $KS \
    --log-level INFO \
    "${LIMIT_ARGS[@]}"
}

run_one "1_DEFAULT_CODE_EMBEDDING" "$EMBED_DEFAULT"
run_one "2_LEGAL_EMBED_BGE" "$EMBED_LEGAL"

echo ""
echo "=== All experiments finished. ==="
