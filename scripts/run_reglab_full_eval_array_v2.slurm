#!/bin/bash
#SBATCH --job-name=rgl-eval-v2
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/rgl_eval_v2_%A_%a.out
#SBATCH --error=slurm_logs/rgl_eval_v2_%A_%a.err
#
# Step 2 of 2: Re-ingest FULL barexam_qa corpus (900K passages) + eval.
#
# Differences from rgl_eval_60986758 (v1):
#   1. PARQUET_META_DIR now points to passages_meta_full (full ~857K corpus)
#   2. Chunker bug is fixed (HierarchicalChunker warns+skips empty docs)
#   3. Results go to $SCRATCH/reglab_eval/results_v2/<model>/
#
# Run AFTER run_reglab_build_barexam_full.slurm completes.
#
# Optional env overrides:
#   PARQUET_META_DIR  — path to the NEW full Parquet dir (default below)
#   BAREXAM_DIR       — path to barexam_qa export dir (for benchmarks JSON)
#   RUN_PAPER_EVAL    — 0 to skip paper_eval, 1 (default) to include it
#   TOP_CHUNKS        — passed to paper_eval --top-chunks (default 10000)
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

# ← POINT AT THE NEW FULL PARQUET DIR from run_reglab_build_barexam_full.slurm
PARQUET_META_DIR="${PARQUET_META_DIR:-${SCRATCH}/reglab_eval/barexam_qa/passages_meta_full}"
BAREXAM_DIR="${BAREXAM_DIR:-${SCRATCH}/reglab_eval/barexam_qa}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/reglab_eval/results_v2}"
RUN_PAPER_EVAL="${RUN_PAPER_EVAL:-1}"

ST_ENCODE_BATCH="${ST_ENCODE_BATCH:-256}"
CHILD_CHUNK_BATCH="${CHILD_CHUNK_BATCH:-1024}"
TOP_CHUNKS="${TOP_CHUNKS:-10000}"

# ---- Embedding table (identical to v1) ----------------------------------------
EMBED_SHORTS=(clerc legalbert mpnet legal-bge octen qwen3)
EMBED_PROVIDERS=(
    huggingface
    huggingface
    sentence_transformers
    sentence_transformers
    sentence_transformers
    sentence_transformers
)
EMBED_MODELS=(
    "jhu-clsp/BERT-DPR-CLERC-ft"
    "nlpaueb/legal-bert-base-uncased"
    "sentence-transformers/all-mpnet-base-v2"
    "axondendriteplus/Legal-Embed-bge-base-en-v1.5"
    "Octen/Octen-Embedding-0.6B"
    "Qwen/Qwen3-Embedding-0.6B"
)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
T_MODEL=$((TASK_ID / 2))
T_CHUNK=$((TASK_ID % 2))

if [[ "$T_CHUNK" -eq 0 ]]; then
    CHUNK_TAG="hier"
    CHUNKER="hierarchical"
    PARENT_ARGS=(--parent-size 2048)
else
    CHUNK_TAG="rec"
    CHUNKER="recursive"
    PARENT_ARGS=()
fi

EMBED_SHORT="${EMBED_SHORTS[$T_MODEL]}"
EMBED_PROVIDER="${EMBED_PROVIDERS[$T_MODEL]}"
EMBED_MODEL="${EMBED_MODELS[$T_MODEL]}"
INDEX_NAME="rgl-v2-${CHUNK_TAG}-${EMBED_SHORT}"

echo "=== RegLab eval v2 (full ~857K corpus) — array task ${TASK_ID} ==="
echo "  MODEL_SLOT      = $T_MODEL ($EMBED_SHORT)"
echo "  CHUNK           = $CHUNK_TAG / $CHUNKER"
echo "  EMBED_PROVIDER  = $EMBED_PROVIDER"
echo "  EMBED_MODEL     = $EMBED_MODEL"
echo "  INDEX_NAME      = $INDEX_NAME"
echo "  PARQUET_META    = $PARQUET_META_DIR"
echo "  BAREXAM_DIR     = $BAREXAM_DIR"
echo "  RUN_PAPER_EVAL  = $RUN_PAPER_EVAL"
date

mkdir -p slurm_logs
mkdir -p "$RESULTS_DIR/$EMBED_SHORT"

[[ -d "$PARQUET_META_DIR" ]] && compgen -G "$PARQUET_META_DIR/passages_part_*.parquet" >/dev/null \
    || { echo "ERROR: No passages_part_*.parquet under $PARQUET_META_DIR"; exit 1; }

[[ -d "$BAREXAM_DIR/benchmarks" ]] \
    || { echo "ERROR: Missing $BAREXAM_DIR/benchmarks (run build job first)"; exit 1; }

# ---- Find OpenSearch -------------------------------------------------------
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    for _cand in "${SCRATCH}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
        [[ -f "$_cand/bin/opensearch" ]] && { OPENSEARCH_DIR="$_cand"; break; }
    done
fi
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "ERROR: OpenSearch not found"; exit 1; }

OS_CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
grep -q "plugins.security.disabled" "$OS_CONFIG" \
    || echo "plugins.security.disabled: true" >> "$OS_CONFIG"

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 cuda python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env 2>/dev/null || true
grep -q "OPENSEARCH_PORT" .env \
    && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env \
    || echo "OPENSEARCH_PORT=9200" >> .env

echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

OS_DATA_DIR="${SLURM_TMPDIR}/opensearch-data-${TASK_ID}"
OS_LOGS_DIR="${SLURM_TMPDIR}/opensearch-logs-${TASK_ID}"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"
OPENSEARCH_STAMP="${SLURM_TMPDIR}/opensearch_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"

cleanup_os() {
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
}
trap cleanup_os EXIT

echo "--- Starting OpenSearch ---"
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
    -Epath.data="$OS_DATA_DIR" -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
curl -s http://localhost:9200/_cluster/health | grep -q '"status"' \
    || { echo "ERROR: OpenSearch failed to start"; exit 1; }
echo "OpenSearch ready."

if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-knn" ]]; then
    for i in $(seq 1 30); do
        _ST=$(curl -s "http://localhost:9200/_cluster/health" \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
        [[ "$_ST" == "yellow" || "$_ST" == "green" ]] && break
        sleep 5
    done
    sleep 10
fi

echo "=== Fast ingest from Parquet (full corpus) → $INDEX_NAME ($CHUNKER) ==="
curl -s -X DELETE "http://localhost:9200/$INDEX_NAME" 2>/dev/null || true

python3 tools/full_corpus_fast/fast_ingest.py \
    --parquet "$PARQUET_META_DIR" \
    --index-name "$INDEX_NAME" \
    --embedding-provider "$EMBED_PROVIDER" \
    --embedding-model "$EMBED_MODEL" \
    --chunker "$CHUNKER" \
    "${PARENT_ARGS[@]}" \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --st-encode-batch "$ST_ENCODE_BATCH" \
    --child-chunk-batch "$CHILD_CHUNK_BATCH" \
    --log-level WARNING

echo "=== Ingest done ==="
date

TRACE_CHAR="$RESULTS_DIR/$EMBED_SHORT/char_barexam_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.jsonl"
echo "=== Char eval (Tian Ks) → $TRACE_CHAR ==="

python3 -m evaluation.reglab.eval_recall \
    --data-dir "$BAREXAM_DIR" \
    --benchmarks barexam_qa \
    --index-name "$INDEX_NAME" \
    --embedding-provider "$EMBED_PROVIDER" \
    --embedding-model "$EMBED_MODEL" \
    --trace-file "$TRACE_CHAR" \
    --ks 2 4 6 10 15 20 40 60

echo "=== Char eval done ==="
date

if [[ "$RUN_PAPER_EVAL" == "1" ]]; then
    RESULTS_PAPER="$RESULTS_DIR/$EMBED_SHORT/paper_barexam_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.json"
    echo "=== Paper eval (Zheng Recall@K / MRR) → $RESULTS_PAPER ==="
    python3 -m evaluation.reglab.paper_eval \
        --source disk \
        --data-dir "$BAREXAM_DIR" \
        --benchmark barexam_qa \
        --index-name "$INDEX_NAME" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --top-chunks "$TOP_CHUNKS" \
        --ks 1 10 100 1000 \
        --results-json "$RESULTS_PAPER" \
        --log-level WARNING
    echo "=== Paper eval done ==="
    date
else
    echo "RUN_PAPER_EVAL=0 — skipped"
fi

echo "=== Task ${TASK_ID} ($EMBED_SHORT $CHUNK_TAG) complete. Results: $RESULTS_DIR/$EMBED_SHORT/ ==="
