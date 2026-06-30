#!/bin/bash
#SBATCH --job-name=rgl-retry
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-2
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/rgl_retry_%A_%a.out
#SBATCH --error=slurm_logs/rgl_retry_%A_%a.err
#
# Retry three tasks that failed in job 60986758:
#
#   0 → legal-bge hier (task 6): cause: OpenSearch JVM heap ~1GB → increased to 16G
#   1 → legal-bge rec  (task 7): ingest+char OK; run paper_eval only
#   2 → qwen3 rec      (task11): ingest+char OK; run paper_eval only
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

PARQUET_META_DIR="${PARQUET_META_DIR:-${SCRATCH}/reglab_eval/barexam_qa/passages_meta}"
BAREXAM_DIR="${BAREXAM_DIR:-${SCRATCH}/reglab_eval/barexam_qa}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/reglab_eval/results}"
TOP_CHUNKS="${TOP_CHUNKS:-10000}"
ST_ENCODE_BATCH="${ST_ENCODE_BATCH:-256}"
CHILD_CHUNK_BATCH="${CHILD_CHUNK_BATCH:-1024}"

T="${SLURM_ARRAY_TASK_ID}"

# ---- Task definitions --------------------------------------------------------
case "$T" in
  0)
    EMBED_SHORT="legal-bge"
    EMBED_PROVIDER="sentence_transformers"
    EMBED_MODEL="axondendriteplus/Legal-Embed-bge-base-en-v1.5"
    CHUNK_TAG="hier"
    CHUNKER="hierarchical"
    PARENT_ARGS=(--parent-size 2048)
    SKIP_INGEST=0
    ;;
  1)
    EMBED_SHORT="legal-bge"
    EMBED_PROVIDER="sentence_transformers"
    EMBED_MODEL="axondendriteplus/Legal-Embed-bge-base-en-v1.5"
    CHUNK_TAG="rec"
    CHUNKER="recursive"
    PARENT_ARGS=()
    SKIP_INGEST=1   # ingest was already OK
    ;;
  2)
    EMBED_SHORT="qwen3"
    EMBED_PROVIDER="sentence_transformers"
    EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"
    CHUNK_TAG="rec"
    CHUNKER="recursive"
    PARENT_ARGS=()
    SKIP_INGEST=1   # ingest was already OK
    ;;
  *)
    echo "Unknown task $T"; exit 1 ;;
esac

INDEX_NAME="rgl-${CHUNK_TAG}-${EMBED_SHORT}"

echo "=== rgl-retry task ${T} ==="
echo "  EMBED=$EMBED_SHORT  CHUNKER=$CHUNKER  SKIP_INGEST=$SKIP_INGEST"
echo "  INDEX=$INDEX_NAME"
date

mkdir -p "$LEGALRAG_ROOT/slurm_logs"
mkdir -p "$RESULTS_DIR/$EMBED_SHORT"

OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    for _cand in "${SCRATCH}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
        if [[ -n "$_cand" && -f "$_cand/bin/opensearch" ]]; then
            OPENSEARCH_DIR="$_cand"
            break
        fi
    done
fi
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "ERROR: OpenSearch not found"; exit 1; }

OS_CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-security" ]]; then
    grep -q "plugins.security.disabled" "$OS_CONFIG" \
        || echo "plugins.security.disabled: true" >> "$OS_CONFIG"
fi

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

OS_DATA_DIR="${SLURM_TMPDIR}/opensearch-data-${T}"
OS_LOGS_DIR="${SLURM_TMPDIR}/opensearch-logs-${T}"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"
OPENSEARCH_STAMP="${SLURM_TMPDIR}/opensearch_${SLURM_ARRAY_JOB_ID}_${T}.pid"

cleanup_os() {
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
}
trap cleanup_os EXIT

echo "--- Starting OpenSearch (heap=16g) ---"
cd "$OPENSEARCH_DIR"
OPENSEARCH_JAVA_OPTS="-Xms16g -Xmx16g" \
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
    -Epath.data="$OS_DATA_DIR" -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' \
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

# ---- Ingest (task 0 only) ---------------------------------------------
if [[ "$SKIP_INGEST" -eq 0 ]]; then
    echo "=== Ingest → $INDEX_NAME ($CHUNKER) ==="
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
        --log-level INFO
    echo "=== Ingest done ===" && date

    # Char eval
    TRACE_CHAR="$RESULTS_DIR/$EMBED_SHORT/char_barexam_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${T}.jsonl"
    echo "=== Char eval → $TRACE_CHAR ==="
    python3 -m evaluation.reglab.eval_recall \
        --data-dir "$BAREXAM_DIR" \
        --benchmarks barexam_qa \
        --index-name "$INDEX_NAME" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$TRACE_CHAR" \
        --ks 2 4 6 10 15 20 40 60
    echo "=== Char eval done ===" && date
else
    echo "=== SKIP_INGEST=1: loading index from previous job (must also exist in this job's OS) ==="
    echo "WARNING: index lives in SLURM_TMPDIR of the previous job and is not accessible now — must re-ingest."
    # Index was in previous job's SLURM_TMPDIR and is now gone — must re-ingest
    echo "=== (Auto-corrected) Ingest → $INDEX_NAME ($CHUNKER) ==="
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
        --log-level INFO
    echo "=== Ingest done ===" && date

    # Char eval (for completeness; result should match the previous job)
    TRACE_CHAR="$RESULTS_DIR/$EMBED_SHORT/char_barexam_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${T}.jsonl"
    echo "=== Char eval → $TRACE_CHAR ==="
    python3 -m evaluation.reglab.eval_recall \
        --data-dir "$BAREXAM_DIR" \
        --benchmarks barexam_qa \
        --index-name "$INDEX_NAME" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$TRACE_CHAR" \
        --ks 2 4 6 10 15 20 40 60
    echo "=== Char eval done ===" && date
fi

# ---- Paper eval -----------------------------------------------------------
RESULTS_PAPER="$RESULTS_DIR/$EMBED_SHORT/paper_barexam_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${T}.json"
echo "=== Paper eval → $RESULTS_PAPER ==="
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
echo "=== Paper eval done ===" && date

echo "=== retry task ${T} ($EMBED_SHORT $CHUNK_TAG) complete ==="
