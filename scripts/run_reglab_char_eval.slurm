#!/bin/bash
#SBATCH --job-name=reglab-char
#SBATCH --output=slurm_logs/reglab_char_eval_%j.out
#SBATCH --error=slurm_logs/reglab_char_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# RegLab datasets — LegalBench-style CharRecall@K / CharPrecision@K (eval_recall, default K=20,40,60).
#
# Prerequisites:
#   - Prepared exports: data/reglab_eval/barexam_qa and data/reglab_eval/housing_qa
#     (python -m evaluation.reglab.prepare ...)
#   - OpenSearch tarball (see run_benchmark_50_eval.slurm)
#
# Usage:
#   sbatch scripts/run_reglab_char_eval.slurm
#
# Optional exports:
#   SKIP_INGEST=1              — do not delete/re-ingest (indices must already exist on this node’s OpenSearch)
#   EMBED_PROVIDER=sentence_transformers
#   EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
#   INDEX_BAREXAM=legalrag-reglab-barexam-mpnet
#   INDEX_HOUSING=legalrag-reglab-housing-mpnet
#   KS="20 40 60"               — passed to eval (default: eval_recall adds 20 40 60 if you omit --ks)
#   RUN_BAREXAM=0 / RUN_HOUSING=0 — skip one dataset
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
BAREXAM_DIR="$LEGALRAG_ROOT/data/reglab_eval/barexam_qa"
HOUSING_DIR="$LEGALRAG_ROOT/data/reglab_eval/housing_qa"

EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
INDEX_BAREXAM="${INDEX_BAREXAM:-legalrag-reglab-barexam-mpnet}"
INDEX_HOUSING="${INDEX_HOUSING:-legalrag-reglab-housing-mpnet}"
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
    echo "ERROR: Missing $BAREXAM_DIR/corpus — run: python -m evaluation.reglab.prepare barexam_qa --out-dir $BAREXAM_DIR"
    exit 1
fi
if [[ "$RUN_HOUSING" == "1" ]] && [[ ! -d "$HOUSING_DIR/corpus" ]]; then
    echo "ERROR: Missing $HOUSING_DIR/corpus — run: python -m evaluation.reglab.prepare housing_qa --out-dir $HOUSING_DIR"
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
        echo "=== Ingest barexam_qa → $INDEX_BAREXAM ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_BAREXAM" 2>/dev/null || true
        python3 -m evaluation.LegalBenchRAG.ingest \
            --data-dir "$BAREXAM_DIR" \
            --index-name "$INDEX_BAREXAM" \
            "${INGEST_ARGS[@]}"
    fi

    if [[ "$RUN_HOUSING" == "1" ]]; then
        echo "=== Ingest housing_qa → $INDEX_HOUSING ==="
        curl -s -X DELETE "http://localhost:9200/$INDEX_HOUSING" 2>/dev/null || true
        python3 -m evaluation.LegalBenchRAG.ingest \
            --data-dir "$HOUSING_DIR" \
            --index-name "$INDEX_HOUSING" \
            "${INGEST_ARGS[@]}"
    fi
else
    echo "=== SKIP_INGEST set — using existing indices ==="
fi

EVAL_EXTRA=()
if [[ -n "${KS:-}" ]]; then
    # shellcheck disable=2206
    EVAL_EXTRA+=(--ks $KS)
fi

if [[ "$RUN_BAREXAM" == "1" ]]; then
    echo "=== Char eval: barexam_qa ==="
    python3 -m evaluation.reglab.eval_recall \
        --data-dir "$BAREXAM_DIR" \
        --benchmarks barexam_qa \
        --index-name "$INDEX_BAREXAM" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$LEGALRAG_ROOT/logs/reglab/barexam_char_${SLURM_JOB_ID}.jsonl" \
        "${EVAL_EXTRA[@]}"
fi

if [[ "$RUN_HOUSING" == "1" ]]; then
    echo "=== Char eval: housing_qa ==="
    python3 -m evaluation.reglab.eval_recall \
        --data-dir "$HOUSING_DIR" \
        --benchmarks housing_qa \
        --index-name "$INDEX_HOUSING" \
        --embedding-provider "$EMBED_PROVIDER" \
        --embedding-model "$EMBED_MODEL" \
        --trace-file "$LEGALRAG_ROOT/logs/reglab/housing_char_${SLURM_JOB_ID}.jsonl" \
        "${EVAL_EXTRA[@]}"
fi

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_STAMP" ]]; then
    kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    rm -f "$OPENSEARCH_STAMP"
fi

echo "=== RegLab char eval done ==="
