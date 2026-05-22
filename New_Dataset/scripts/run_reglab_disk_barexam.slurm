#!/bin/bash
#SBATCH --job-name=reglab-disk-barexam
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/reglab_disk_barexam_%j.out
#SBATCH --error=slurm_logs/reglab_disk_barexam_%j.err

# Evaluate barexam_qa from disk files (corpus/*.txt + benchmarks/*.json)
# Data: $SCRATCH/reglab_eval/barexam_qa
#
# Optional env overrides:
#   BAREXAM_DIR       — data root (default: $SCRATCH/reglab_eval/barexam_qa)
#   INDEX_BAREXAM     — index name (default: legalrag-reglab-barexam-disk)
#   EMBED_PROVIDER / EMBED_MODEL
#   SKIP_INGEST=1     — skip ingest if index already exists
#   KS="20 40 60"     — K cutoffs for char eval
#   LIMIT=N           — restrict number of queries for quick testing
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
REGLAB_LOGS="${REGLAB_LOGS:-${SCRATCH:-$HOME}/legalrag_reglab_logs}"
BAREXAM_DIR="${BAREXAM_DIR:-${SCRATCH:-$HOME}/reglab_eval/barexam_qa}"

EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
INDEX_BAREXAM="${INDEX_BAREXAM:-legalrag-reglab-barexam-disk}"
TOP_CHUNKS="${TOP_CHUNKS:-5000}"
# Number of samples for ingest and eval (default 200)
N_SAMPLES="${N_SAMPLES:-200}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== RegLab disk eval: barexam_qa ==="
echo "  BAREXAM_DIR   = $BAREXAM_DIR"
echo "  INDEX         = $INDEX_BAREXAM"
echo "  EMBED_MODEL   = $EMBED_MODEL"
echo "  SKIP_INGEST   = ${SKIP_INGEST:-0}"
echo "  N_SAMPLES     = $N_SAMPLES"
date

mkdir -p "$LEGALRAG_ROOT/slurm_logs" "$REGLAB_LOGS"

# ── Modules ────────────────────────────────────────────────────────────────────
module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 cuda python/3.12 2>/dev/null || true

source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

python3 -c "from legalrag.core.config import settings; from legalrag.ingestion.embedder import build_embedder; print('legalrag OK')" || {
    echo "ERROR: legalrag or dependencies not installed. On login node: pip install -e '.[eval]'"
    exit 1
}

# ── Data validation ────────────────────────────────────────────────────────────────
if [[ ! -f "$BAREXAM_DIR/benchmarks/barexam_qa.json" ]]; then
    echo "ERROR: $BAREXAM_DIR/benchmarks/barexam_qa.json not found."
    exit 1
fi
if [[ ! -d "$BAREXAM_DIR/corpus/passages" ]]; then
    echo "ERROR: $BAREXAM_DIR/corpus/passages not found."
    exit 1
fi
CORPUS_COUNT=$(find "$BAREXAM_DIR/corpus/passages" -name "*.txt" | wc -l)
echo "Corpus OK: $CORPUS_COUNT .txt files"
BENCH_TESTS=$(python3 -c "import json; d=json.load(open('$BAREXAM_DIR/benchmarks/barexam_qa.json')); print(len(d['tests']))")
echo "Benchmark OK: $BENCH_TESTS tests"

# ── OpenSearch ─────────────────────────────────────────────────────────────────
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    for cand in "${SCRATCH:-}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
        if [[ -n "$cand" && -f "$cand/bin/opensearch" ]]; then
            OPENSEARCH_DIR="$cand"
            break
        fi
    done
fi
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch not found. Set OPENSEARCH_TARBALL_DIR."
    exit 1
fi
echo "OpenSearch: $OPENSEARCH_DIR"

# Security plugin → disabled: true  (without the plugin this setting is unknown)
OS_CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-security" ]]; then
    grep -q "plugins.security.disabled" "$OS_CONFIG" \
        || echo "plugins.security.disabled: true" >> "$OS_CONFIG"
    echo "Security plugin found → disabled"
else
    sed -i '/plugins\.security/d' "$OS_CONFIG"
    echo "No security plugin → setting removed"
fi

# .env
[[ ! -f .env ]] && cp .env.example .env 2>/dev/null || true
grep -q "OPENSEARCH_PORT" .env \
    && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env \
    || echo "OPENSEARCH_PORT=9200" >> .env

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
# CUDA_VISIBLE_DEVICES is managed by Slurm — do not set manually

OPENSEARCH_PID_FILE="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-manual}.pid"

# Each job writes its data/logs to SLURM_TMPDIR (node-local, not shared)
# This prevents lock contention between concurrent jobs on the same scratch filesystem
OS_DATA_DIR="${SLURM_TMPDIR:-/tmp/os_${SLURM_JOB_ID:-manual}}/opensearch-data"
OS_LOGS_DIR="${SLURM_TMPDIR:-/tmp/os_${SLURM_JOB_ID:-manual}}/opensearch-logs"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"

echo "Starting OpenSearch (data=$OS_DATA_DIR)..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_PID_FILE" \
    -Epath.data="$OS_DATA_DIR" \
    -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

# Wait for cluster health
echo "Waiting for OpenSearch cluster health..."
for i in $(seq 1 60); do
    if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
        echo "Cluster health OK (try $i)."
        break
    fi
    sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
    echo "ERROR: OpenSearch failed to start."
    [[ -f "$OPENSEARCH_PID_FILE" ]] && kill "$(cat "$OPENSEARCH_PID_FILE")" 2>/dev/null || true
    exit 1
fi

# Wait for KNN plugin
if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-knn" ]]; then
    echo "Waiting for KNN plugin initialization..."
    for i in $(seq 1 24); do
        if curl -s "http://localhost:9200/_cat/plugins?v" 2>/dev/null | grep -q "opensearch-knn"; then
            echo "KNN plugin ready (try $i)."
            break
        fi
        sleep 5
    done
    sleep 10
fi

cleanup_os() {
    echo "Shutting down OpenSearch..."
    [[ -f "$OPENSEARCH_PID_FILE" ]] && kill "$(cat "$OPENSEARCH_PID_FILE")" 2>/dev/null || true
}
trap cleanup_os EXIT

# ── Ingest ─────────────────────────────────────────────────────────────────────
if [[ -z "${SKIP_INGEST:-}" ]]; then
    echo ""
    echo "=== Ingest: barexam_qa → $INDEX_BAREXAM ==="
    curl -s -X DELETE "http://localhost:9200/$INDEX_BAREXAM" || true

    # Only ingest files referenced by the first N_SAMPLES tests (not all 686k)
    # Chunker settings exactly matching Tian's scripts
    INGEST_ARGS=(
        --data-dir "$BAREXAM_DIR"
        --benchmarks barexam_qa
        --limit "$N_SAMPLES"
        --index-name "$INDEX_BAREXAM"
        --embedding-provider "$EMBED_PROVIDER"
        --embedding-model   "$EMBED_MODEL"
        --chunker hierarchical
        --parent-size 2048
        --chunk-size  512
        --chunk-overlap 64
        --log-level INFO
    )
    python3 -m evaluation.LegalBenchRAG.ingest "${INGEST_ARGS[@]}"
    echo "Ingest complete."
else
    echo "=== SKIP_INGEST=1: using existing index ==="
fi

# ── Char Recall Eval ───────────────────────────────────────────────────────────
echo ""
echo "=== Char eval: barexam_qa (disk) ==="

CHAR_ARGS=(
    --data-dir "$BAREXAM_DIR"
    --benchmarks barexam_qa
    --limit "$N_SAMPLES"
    --index-name "$INDEX_BAREXAM"
    --embedding-provider "$EMBED_PROVIDER"
    --embedding-model   "$EMBED_MODEL"
    --trace-file "$REGLAB_LOGS/barexam_char_disk_${SLURM_JOB_ID:-manual}.jsonl"
    --log-level INFO
)
[[ -n "${KS:-}" ]] && CHAR_ARGS+=(--ks $KS)

python3 -m evaluation.reglab.eval_recall "${CHAR_ARGS[@]}"

# ── Paper Eval ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Paper eval: barexam_qa (disk) ==="

PAPER_ARGS=(
    --source disk
    --data-dir "$BAREXAM_DIR"
    --benchmark barexam_qa
    --limit "$N_SAMPLES"
    --index-name "$INDEX_BAREXAM"
    --embedding-provider "$EMBED_PROVIDER"
    --embedding-model   "$EMBED_MODEL"
    --top-chunks "$TOP_CHUNKS"
    --results-json "$REGLAB_LOGS/paper_barexam_disk_${SLURM_JOB_ID:-manual}.json"
    --log-level INFO
)

python3 -m evaluation.reglab.paper_eval "${PAPER_ARGS[@]}"

echo ""
echo "=== Done: RegLab disk eval barexam_qa ==="
echo "Results in: $REGLAB_LOGS"
date
