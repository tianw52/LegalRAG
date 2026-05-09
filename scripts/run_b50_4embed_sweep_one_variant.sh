#!/usr/bin/env bash
# 12-run sweep: hierarchical × 6 embeddings + recursive × 6.
# HF ids (verified): LegalBERT, SBERT mpnet, Legal-Embed, DPR-CLERC, Qwen3, Octen.
# Note: thenlper/gte-large is a different model from all-mpnet-base-v2 (see script comments).
# Expects OpenSearch at localhost:9200 (start it in Slurm or locally).
#
# Env:
#   LEGALRAG_ROOT
#   REWRITE_VARIANT   — mistral | qwen72b | qwen35_9b (folder under benchmark_50_reformated_proccessed/)
#   EVAL_BENCHMARK_ROOT — default: <data-dir parent> = .../LegalBenchRAG
#   RESULT_ROOT       — default: results/${REWRITE_VARIANT}_4embed_sweep
#   KS                — default: 2 4 6 10 15 20 40 60
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
cd "$LEGALRAG_ROOT"

DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
REWRITE_VARIANT="${REWRITE_VARIANT:-mistral}"
EVAL_BENCHMARK_ROOT="${EVAL_BENCHMARK_ROOT:-$DATA_DIR}"
EVAL_BENCHMARKS_DIR="$EVAL_BENCHMARK_ROOT/benchmark_50_reformated_proccessed/${REWRITE_VARIANT}"
RESULT_ROOT="${RESULT_ROOT:-$LEGALRAG_ROOT/results/${REWRITE_VARIANT}_4embed_sweep}"
METRICS_DIR="$RESULT_ROOT/metrics"
KS="${KS:-2 4 6 10 15 20 40 60}"

case "$REWRITE_VARIANT" in
  mistral)
    PLOT_TITLE_SUFFIX="${PLOT_TITLE_SUFFIX:-Mistral reformulated queries}"
    ;;
  qwen72b)
    PLOT_TITLE_SUFFIX="${PLOT_TITLE_SUFFIX:-Qwen 2.5 72B reformulated queries}"
    ;;
  qwen35_9b)
    PLOT_TITLE_SUFFIX="${PLOT_TITLE_SUFFIX:-Qwen 3.5 9B reformulated queries}"
    ;;
  *)
    PLOT_TITLE_SUFFIX="${PLOT_TITLE_SUFFIX:-${REWRITE_VARIANT} reformulated queries}"
    ;;
esac

[[ -d "$DATA_DIR/corpus" ]] || { echo "ERROR: missing corpus: $DATA_DIR/corpus"; exit 1; }
[[ -f "$EVAL_BENCHMARKS_DIR/cuad.json" ]] || {
  echo "ERROR: missing eval JSONs: $EVAL_BENCHMARKS_DIR"
  exit 1
}

mkdir -p "$METRICS_DIR"

step() {
  local CHUNKER="$1" INDEX="$2" EMODEL="$3" MLABEL="$4" METRICS_SLUG="$5"
  local CTAG
  if [[ "$CHUNKER" == "hierarchical" ]]; then
    CTAG="hierarchical"
  else
    CTAG="recursive"
  fi

  echo ""
  echo "================================================================================"
  echo "  chunker=$CHUNKER  index=$INDEX"
  echo "  embedding=$EMODEL"
  echo "================================================================================"

  curl -sS -X DELETE "http://localhost:9200/${INDEX}" >/dev/null || true
  # Let OpenSearch finish shard cleanup so the next ensure_index() sees a clean slate.
  sleep 3

  python3 -m evaluation.LegalBenchRAG.ingest \
    --data-dir "$DATA_DIR" \
    --all \
    --chunker "$CHUNKER" \
    --embedding-model "$EMODEL" \
    --index-name "$INDEX" \
    --log-level INFO

  python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
    --data-dir "$DATA_DIR" \
    --benchmarks-dir "$EVAL_BENCHMARKS_DIR" \
    --index-name "$INDEX" \
    --embedding-model "$EMODEL" \
    --ks $KS \
    --metrics-json-out "$METRICS_DIR/metrics_${METRICS_SLUG}.json" \
    --metrics-model-label "$MLABEL" \
    --metrics-chunker-tag "$CTAG" \
    --log-level INFO
}

# --- hierarchical (6) — order matches aggregate MODEL_ORDER ---
# 1 LegalBERT — https://huggingface.co/nlpaueb/legal-bert-base-uncased
step hierarchical lbr-hier-legalbert \
  "nlpaueb/legal-bert-base-uncased" \
  "LegalBERT" \
  hier_legalbert

# 2 SBERT — https://huggingface.co/sentence-transformers/all-mpnet-base-v2 (not thenlper/gte-large)
step hierarchical lbr-hier-mpnet \
  "sentence-transformers/all-mpnet-base-v2" \
  "SBERT (all-mpnet-base-v2)" \
  hier_mpnet

# 3 Legal-Embed — https://huggingface.co/axondendriteplus/Legal-Embed-bge-base-en-v1.5
step hierarchical lbr-hier-legal-embed \
  "axondendriteplus/Legal-Embed-bge-base-en-v1.5" \
  "Legal-Embed-bge-base" \
  hier_legal-embed

# 4 DPR CLERC — https://huggingface.co/jhu-clsp/BERT-DPR-CLERC-ft
step hierarchical lbr-hier-dpr-clerc \
  "jhu-clsp/BERT-DPR-CLERC-ft" \
  "BERT-DPR-CLERC-ft" \
  hier_dpr-clerc

# 5 Qwen3 — https://huggingface.co/Qwen/Qwen3-Embedding-0.6B  (needs recent transformers / ST)
step hierarchical lbr-hier-qwen3 \
  "Qwen/Qwen3-Embedding-0.6B" \
  "Qwen3" \
  hier_qwen3

# 6 Octen — https://huggingface.co/Octen/Octen-Embedding-0.6B
step hierarchical lbr-hier-octen \
  "Octen/Octen-Embedding-0.6B" \
  "Octen" \
  hier_octen

# --- recursive (6) ---
step recursive lbr-rec-legalbert \
  "nlpaueb/legal-bert-base-uncased" \
  "LegalBERT" \
  rec_legalbert

step recursive lbr-rec-mpnet \
  "sentence-transformers/all-mpnet-base-v2" \
  "SBERT (all-mpnet-base-v2)" \
  rec_mpnet

step recursive lbr-rec-legal-embed \
  "axondendriteplus/Legal-Embed-bge-base-en-v1.5" \
  "Legal-Embed-bge-base" \
  rec_legal-embed

step recursive lbr-rec-dpr-clerc \
  "jhu-clsp/BERT-DPR-CLERC-ft" \
  "BERT-DPR-CLERC-ft" \
  rec_dpr-clerc

step recursive lbr-rec-qwen3 \
  "Qwen/Qwen3-Embedding-0.6B" \
  "Qwen3" \
  rec_qwen3

step recursive lbr-rec-octen \
  "Octen/Octen-Embedding-0.6B" \
  "Octen" \
  rec_octen

OUT_PREFIX="$RESULT_ROOT/${REWRITE_VARIANT}_reform_4embed"
echo ""
echo ">>> Aggregate + plots → $OUT_PREFIX*"
python3 scripts/aggregate_plot_mistral_4embed_sweep.py \
  --metrics-dir "$METRICS_DIR" \
  --out-prefix "$OUT_PREFIX" \
  --plot-title-suffix "$PLOT_TITLE_SUFFIX"

echo ""
echo "=== Done 6-embed ×2 chunker sweep for REWRITE_VARIANT=$REWRITE_VARIANT ==="
echo "    Metrics: $METRICS_DIR"
echo "    Plots/CSV prefix: $OUT_PREFIX"
