#!/bin/bash
#SBATCH --job-name=reglab-paper
#SBATCH --output=slurm_logs/reglab_paper_eval_%j.out
#SBATCH --error=slurm_logs/reglab_paper_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# RegLab datasets — passage-level metrics (Recall@1,10,100,1000; MRR@10) per Zheng et al. CS&Law 2025.
# Housing uses jurisdiction filtering (ingest_reglab + court metadata).
#
# Prerequisites:
#   - data/reglab_eval/barexam_qa and data/reglab_eval/housing_qa with statutes/<State>/ layout for housing
#     (re-run prepare if you have an older flat corpus)
#
# Usage:
#   sbatch scripts/run_reglab_paper_eval.slurm
#
# Optional exports:
#   SKIP_INGEST=1
#   EMBED_PROVIDER / EMBED_MODEL — must match index if SKIP_INGEST=1
#   INDEX_BAREXAM / INDEX_HOUSING — default *-paper so char and paper jobs do not clobber each other
#   TOP_CHUNKS=50000 — chunks retrieved per query before passage dedupe (raise if Recall@1000 is saturated)
#   HOUSING_RECALL=upper|lower — upper = main paper; lower = Appendix H (all gold statutes in top-K)
#   RUN_BAREXAM=0 / RUN_HOUSING=0
#   LIMIT=N — passthrough --limit to paper_eval (debug)
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
BAREXAM_DIR="$LEGALRAG_ROOT/data/reglab_eval/barexam_qa"
HOUSING_DIR="$LEGALRAG_ROOT/data/reglab_eval/housing_qa"

EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
INDEX_BAREXAM="${INDEX_BAREXAM:-legalrag-reglab-barexam-paper}"
INDEX_HOUSING="${INDEX_HOUSING:-legalrag-reglab-housing-paper}"
TOP_CHUNKS="${TOP_CHUNKS:-50000}"
HOUSING_RECALL="${HOUSING_RECALL:-upper}"
RUN_BAREXAM="${RUN_BAREXAM:-1}"
RUN_HOUSING="${RUN_HOUSING:-1}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e ".[eval]" -q

export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs logs/reglab

if [[ "$RUN_BAREXAM" == "1" ]] && [[ ! -d "$BAREXAM_DIR/corpus" ]]; then
    echo "ERROR: Missing $BAREXAM_DIR/corpus"
    exit 1
fi
if [[ "$RUN_HOUSING" == "1" ]] && [[ ! -d "$HOUSING_DIR/corpus" ]]; then
    echo "ERROR: Missing $HOUSING_DIR/corpus"
    exit 1
fi

if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
    echo "ERROR: OpenSearch not at $OPENSEARCH_DIR"
    exit 1
fi

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
    echo "" >> "$CONFIG"
    echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_STAMP="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
    ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
    if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
        echo "OpenSearch ready."
        break
    fi
    sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
    echo "ERROR: OpenSearch failed to start."
    [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    exit 1
fi

INGEST_ARGS=(
    --chunker hierarchical
    --parent-size 2048
    --chunk-size 512
    --chunk-overlap 64
    --embedding-provider "$EMBED_PROVIDER"
    --embedding-model "$EMBED_MODEL"
)

if [[ -z "${SKIP_INGEST:-}" ]]; then
    if [[ "$RUN_BAREXAM" == "1" ]]; then
        echo "=== Ingest barexam_qa (standard loader) → $INDEX_BAREXAM ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_BAREXAM" 2>/dev/null || true
        python3 -m evaluation.LegalBenchRAG.ingest \
            --data-dir "$BAREXAM_DIR" \
            --index-name "$INDEX_BAREXAM" \
            "${INGEST_ARGS[@]}"
    fi

    if [[ "$RUN_HOUSING" == "1" ]]; then
        echo "=== Ingest housing_qa (RegLab loader) → $INDEX_HOUSING ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_HOUSING" 2>/dev/null || true
        python3 -m evaluation.reglab.ingest_reglab \
            --data-dir "$HOUSING_DIR" \
            --index-name "$INDEX_HOUSING" \
            "${INGEST_ARGS[@]}"
    fi
else
    echo "=== SKIP_INGEST set — using existing indices ==="
fi

PAPER_EXTRA=(--top-chunks "$TOP_CHUNKS")
if [[ -n "${LIMIT:-}" ]]; then
    PAPER_EXTRA+=(--limit "$LIMIT")
fi

if [[ "$RUN_BAREXAM" == "1" ]]; then
    echo "=== Paper metrics: barexam_qa ==="
    python3 -m evaluation.reglab.paper_eval \
        --data-dir "$BAREXAM_DIR" \
        --benchmark barexam_qa \
        --index-name "$INDEX_BAREXAM" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --results-json "$LEGALRAG_ROOT/logs/reglab/paper_barexam_${SLURM_JOB_ID}.json" \
        "${PAPER_EXTRA[@]}"
fi

if [[ "$RUN_HOUSING" == "1" ]]; then
    echo "=== Paper metrics: housing_qa ($HOUSING_RECALL) ==="
    python3 -m evaluation.reglab.paper_eval \
        --data-dir "$HOUSING_DIR" \
        --benchmark housing_qa \
        --index-name "$INDEX_HOUSING" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --housing-recall "$HOUSING_RECALL" \
        --results-json "$LEGALRAG_ROOT/logs/reglab/paper_housing_${HOUSING_RECALL}_${SLURM_JOB_ID}.json" \
        "${PAPER_EXTRA[@]}"
fi

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_STAMP" ]]; then
    kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    rm -f "$OPENSEARCH_STAMP"
fi

echo "=== RegLab paper eval done ==="
