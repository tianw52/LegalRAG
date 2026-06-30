#!/bin/bash
#SBATCH --job-name=reglab-char
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Narval / Alliance: full A100 (see sinfo / docs). For MIG slices use e.g. gpu:a100_4g.20gb:1
#SBATCH --gres=gpu:a100:1
# Writable cwd recommended: cd "${SCRATCH:-$HOME}" && sbatch .../run_reglab_char_eval.slurm
#SBATCH --output=reglab_char_eval_%j.out
#SBATCH --error=reglab_char_eval_%j.err

# RegLab datasets — LegalBench-style CharRecall@K / CharPrecision@K (eval_recall, default K=20,40,60).
#
# Prerequisites:
#   - Prepared HF exports (corpus/ + benchmarks/). On a LOGIN node run:
#       ./scripts/run_reglab_prepare_data.sh
#     Or set REGLAB_ROOT if data lives elsewhere (e.g. $SCRATCH/reglab_eval).
#   - OpenSearch tarball (see run_benchmark_50_eval.slurm)
#
# Usage:
#   sbatch scripts/run_reglab_char_eval.slurm
#
# Optional exports:
#   RUN_PIP_INSTALL=1          — run pip install -e ".[eval]" on the compute node (needs outbound network; most CC nodes have none)
#   LEGALRAG_VENV              — venv with legalrag installed (default: ../PyTorch)
#   REGLAB_ROOT                — parent folder containing barexam_qa/ and housing_qa/ (default: <repo>/data/reglab_eval)
#   SKIP_INGEST=1              — do not delete/re-ingest (indices must already exist on this node’s OpenSearch)
#   EMBED_PROVIDER=sentence_transformers
#   EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
#   INDEX_BAREXAM=legalrag-reglab-barexam-mpnet
#   INDEX_HOUSING=legalrag-reglab-housing-mpnet
#   KS="20 40 60"               — passed to eval (default: eval_recall adds 20 40 60 if you omit --ks)
#   RUN_BAREXAM=0 / RUN_HOUSING=0 — skip one dataset
#   REGLAB_REGLOG=/path       — trace JSONL (default: $SCRATCH/legalrag_reglab_logs or $HOME/...)
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"
REGLAB_ROOT="${REGLAB_ROOT:-$LEGALRAG_ROOT/data/reglab_eval}"
if [[ "${REGLAB_ROOT}" == "$LEGALRAG_ROOT/data/reglab_eval" ]] && [[ ! -d "$REGLAB_ROOT/barexam_qa/corpus" ]] && [[ -n "${SCRATCH:-}" ]]; then
  if [[ -d "${SCRATCH}/reglab_eval/barexam_qa/corpus" ]]; then
    REGLAB_ROOT="${SCRATCH}/reglab_eval"
    echo "Using REGLAB_ROOT=$REGLAB_ROOT (data found on SCRATCH)"
  fi
fi
BAREXAM_DIR="$REGLAB_ROOT/barexam_qa"
HOUSING_DIR="$REGLAB_ROOT/housing_qa"
REGLAB_REGLOG="${REGLAB_REGLOG:-${SCRATCH:-$HOME}/legalrag_reglab_logs}"

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
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"

cd "$LEGALRAG_ROOT"
if [[ "${RUN_PIP_INSTALL:-0}" == "1" ]]; then
    pip uninstall legalrag -y 2>/dev/null || true
    pip install -e ".[eval]" -q
else
    echo "RUN_PIP_INSTALL not set — skipping pip (compute nodes often have no network). Using: $LEGALRAG_VENV"
fi
python3 -c "import legalrag" || {
    echo "ERROR: legalrag not importable in venv. On a login node:"
    echo "  source $LEGALRAG_VENV/bin/activate && cd $LEGALRAG_ROOT && pip install -e '.[eval]'"
    exit 1
}

export CUDA_VISIBLE_DEVICES=0
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p "$REGLAB_REGLOG"

if [[ "$RUN_BAREXAM" == "1" ]] && [[ ! -d "$BAREXAM_DIR/corpus" ]]; then
    echo "ERROR: Missing $BAREXAM_DIR/corpus"
    echo "  On a login node: $LEGALRAG_ROOT/scripts/run_reglab_prepare_data.sh"
    echo "  Or: REGLAB_ROOT=/path/to/parent sbatch --export=ALL,REGLAB_ROOT=... scripts/run_reglab_char_eval.slurm"
    exit 1
fi
if [[ "$RUN_HOUSING" == "1" ]] && [[ ! -d "$HOUSING_DIR/corpus" ]]; then
    echo "ERROR: Missing $HOUSING_DIR/corpus"
    echo "  On a login node: $LEGALRAG_ROOT/scripts/run_reglab_prepare_data.sh"
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
        --trace-file "$REGLAB_REGLOG/barexam_char_${SLURM_JOB_ID}.jsonl" \
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
        --trace-file "$REGLAB_REGLOG/housing_char_${SLURM_JOB_ID}.jsonl" \
        "${EVAL_EXTRA[@]}"
fi

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_STAMP" ]]; then
    kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
    rm -f "$OPENSEARCH_STAMP"
fi

echo "=== RegLab char eval done ==="
