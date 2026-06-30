#!/bin/bash
#SBATCH --job-name=rgl-hou-nf
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/rgl_hou_nf_%A_%a.out
#SBATCH --error=slurm_logs/rgl_hou_nf_%A_%a.err
#
# HousingQA full-corpus ablation — same 6 embeddings × (hier | rec) as the
# jurisdiction-filtered housing eval, but with --no-court-filter so that
# retrieval searches the entire index (no per-state restriction).
#
# This produces an apples-to-apples comparison with BarExam full-corpus retrieval.
# Existing jurisdiction-filtered results are NOT touched.
#
# Results are written to:
#   $SCRATCH/reglab_eval/results_housing_nofilter/<model>/
#   char_housing_nf_<chunk>_<jobid>_<taskid>.jsonl
#
# Prerequisites (index already built from filtered run — same Parquet/index):
#   Parquet:  sbatch tools/full_corpus_fast/run_build_housing_parquet.slurm
#   Index is rebuilt per-task (same as filtered run) so both sets are independent.
#
# Env overrides:
#   HOUSING_DIR        — default: $SCRATCH/reglab_eval/housing_qa
#   PARQUET_META_DIR   — default: $HOUSING_DIR/passages_meta
#   RESULTS_DIR        — default: $SCRATCH/reglab_eval/results_housing_nofilter
#   OPENSEARCH_JAVA_OPTS="-Xms16g -Xmx16g"  — if OOM on large models
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

HOUSING_DIR="${HOUSING_DIR:-${SCRATCH}/reglab_eval/housing_qa}"
PARQUET_META_DIR="${PARQUET_META_DIR:-${HOUSING_DIR}/passages_meta}"
# Separate results directory — never overwrites jurisdiction-filtered results
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH}/reglab_eval/results_housing_nofilter}"

ST_ENCODE_BATCH="${ST_ENCODE_BATCH:-256}"
CHILD_CHUNK_BATCH="${CHILD_CHUNK_BATCH:-1024}"

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
# Separate index name so it does not collide with the filtered-run index
INDEX_NAME="rglhounf-${CHUNK_TAG}-${EMBED_SHORT}"

echo "=== HousingQA no-filter ablation — task ${TASK_ID} ==="
echo "  MODEL_SLOT      = $T_MODEL ($EMBED_SHORT)"
echo "  CHUNK           = $CHUNK_TAG / $CHUNKER"
echo "  INDEX_NAME      = $INDEX_NAME"
echo "  PARQUET_META    = $PARQUET_META_DIR"
echo "  HOUSING_DIR     = $HOUSING_DIR"
echo "  RESULTS_DIR     = $RESULTS_DIR"
echo "  court_filter    = DISABLED (--no-court-filter)"
date

mkdir -p "$LEGALRAG_ROOT/slurm_logs"
mkdir -p "$RESULTS_DIR/$EMBED_SHORT"

[[ -d "$PARQUET_META_DIR" ]] && compgen -G "$PARQUET_META_DIR/passages_part_*.parquet" >/dev/null \
    || { echo "ERROR: No passages_part_*.parquet under $PARQUET_META_DIR"; exit 1; }

[[ -f "$HOUSING_DIR/benchmarks/housing_qa.json" ]] \
    || { echo "ERROR: Missing $HOUSING_DIR/benchmarks/housing_qa.json"; exit 1; }

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

# Unique port per task — avoids collisions when multiple tasks share a node.
# Tasks 0-11 → ports 9200-9211.
OS_PORT=$((9200 + TASK_ID))
export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT="${OS_PORT}"
echo "  OS_PORT         = $OS_PORT  (unique per task)"

[[ ! -f .env ]] && cp .env.example .env 2>/dev/null || true
grep -q "OPENSEARCH_PORT" .env \
    && sed -i "s/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=${OS_PORT}/" .env \
    || echo "OPENSEARCH_PORT=${OS_PORT}" >> .env
grep -q "OPENSEARCH_HOST" .env \
    && sed -i "s/OPENSEARCH_HOST=.*/OPENSEARCH_HOST=localhost/" .env \
    || echo "OPENSEARCH_HOST=localhost" >> .env

echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

OS_DATA_DIR="${SLURM_TMPDIR}/opensearch-hounf-${TASK_ID}"
OS_LOGS_DIR="${SLURM_TMPDIR}/opensearch-hounf-logs-${TASK_ID}"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"
OPENSEARCH_STAMP="${SLURM_TMPDIR}/opensearch_hounf_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.pid"

cleanup_os() {
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
}
trap cleanup_os EXIT

echo "--- Starting OpenSearch on port ${OS_PORT} (OPENSEARCH_JAVA_OPTS=${OPENSEARCH_JAVA_OPTS:-<default>}) ---"
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
    -Epath.data="$OS_DATA_DIR" -Epath.logs="$OS_LOGS_DIR" \
    -Ehttp.port="${OS_PORT}" \
    -Etransport.port="$((OS_PORT + 100))"
cd "$LEGALRAG_ROOT"

# Wait up to 10 minutes (120 × 5 s) for OpenSearch to become healthy.
for i in $(seq 1 120); do
    curl -s "http://localhost:${OS_PORT}/_cluster/health" 2>/dev/null | grep -q '"status"' && break
    sleep 5
done
curl -s "http://localhost:${OS_PORT}/_cluster/health" 2>/dev/null | grep -q '"status"' \
    || { echo "ERROR: OpenSearch failed to start on port ${OS_PORT}"; exit 1; }
echo "OpenSearch ready on port ${OS_PORT}."

if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-knn" ]]; then
    for i in $(seq 1 30); do
        _ST=$(curl -s "http://localhost:${OS_PORT}/_cluster/health" \
            | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
        [[ "$_ST" == "yellow" || "$_ST" == "green" ]] && break
        sleep 5
    done
    sleep 10
fi

echo "=== Fast ingest from Parquet → $INDEX_NAME ($CHUNKER) ==="
curl -s -X DELETE "http://localhost:${OS_PORT}/$INDEX_NAME" 2>/dev/null || true

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

echo "=== Ingest done ==="
date

TRACE_CHAR="$RESULTS_DIR/$EMBED_SHORT/char_housing_nf_${CHUNK_TAG}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.jsonl"
echo "=== Char eval (no court filter) → $TRACE_CHAR ==="

python3 -m evaluation.reglab.eval_recall \
    --data-dir "$HOUSING_DIR" \
    --benchmarks housing_qa \
    --index-name "$INDEX_NAME" \
    --embedding-provider "$EMBED_PROVIDER" \
    --embedding-model "$EMBED_MODEL" \
    --trace-file "$TRACE_CHAR" \
    --ks 2 4 6 10 15 20 40 60 \
    --no-court-filter

echo "=== Char eval done ==="
date
echo "=== Task ${TASK_ID} ($EMBED_SHORT $CHUNK_TAG) complete. Results: $RESULTS_DIR/$EMBED_SHORT/ ==="
