#!/bin/bash
#SBATCH --job-name=table21-baseline
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Reproduce Table 21 / Table 22 (Historical MBE subset) BASELINE rows from:
#   "A Reasoning-Focused Legal Retrieval Benchmark" (Zheng et al., CS&Law 2025)
#
# ARRAY TASK → MODEL:
#   0 = BM25          (CPU only, no GPU needed — fast)
#   1 = E5-small-v2   (~30 min on A100)
#   2 = E5-base-v2    (~60 min on A100)
#   3 = E5-large-v2   (~2 hr  on A100)
#   4 = E5-mistral-7b (~8 hr  on A100)
#
# Prerequisites:
#   1. Run download_e5_models.slurm first to cache all four E5 models.
#   2. The reglab/barexam_qa dataset should already be cached in HF_HOME.
#
# Submit all five jobs in parallel:
#   sbatch reproduction/run_table21_baseline.slurm
#
# To run a single model (e.g. BM25 only):
#   sbatch --array=0 reproduction/run_table21_baseline.slurm
#   sbatch --array=3 reproduction/run_table21_baseline.slurm  # E5-large only
#
# Results are written to:
#   ${RESULTS_DIR}/{bm25,e5-*}_metrics.json   (per-model)
#   ${RESULTS_DIR}/summary_metrics.json       (merged after all jobs complete)
#
set -euo pipefail

REPO_ROOT="/home/ram112/projects/def-jieliang/ram112"
LEGALRAG_VENV="${LEGALRAG_VENV:-${REPO_ROOT}/PyTorch}"
SCRIPT="${REPO_ROOT}/reproduction/reproduce_table21_baseline.py"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/reproduction/table21_results}"
PASSAGE_CACHE="${RESULTS_DIR}/passages_cache.npz"

# Use SCRATCH for large embedding arrays (each E5-large embedding file ~3.7 GB)
EMB_DIR="${SCRATCH:-${RESULTS_DIR}}/table21_embeddings"
RESULTS_DIR_FULL="${RESULTS_DIR}"

module --force purge
module load StdEnv/2023 gcc cuda/12.2 python/3.12 2>/dev/null || true
source "$LEGALRAG_VENV/bin/activate"

echo "=== Table 22 Baseline Reproduction (Historical MBE subset) ==="
echo "  SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}"
echo "  HOST                : $(hostname)"
echo "  GPU                 : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "  Python              : $(python3 --version)"
echo "  Results dir         : ${RESULTS_DIR_FULL}"
echo ""

# Ensure extra packages are installed (idempotent)
pip install --quiet rank-bm25 faiss-gpu 2>/dev/null || \
    pip install --quiet rank-bm25 faiss-cpu 2>/dev/null || true

mkdir -p "${RESULTS_DIR_FULL}"
mkdir -p "${EMB_DIR}"

# Symlink embeddings dir into results so the script finds them
if [[ "${EMB_DIR}" != "${RESULTS_DIR_FULL}" ]]; then
    ln -sfn "${EMB_DIR}" "${RESULTS_DIR_FULL}/emb_symlink" 2>/dev/null || true
fi

# Run only the model corresponding to this array task
python3 "${SCRIPT}" \
    --model-idx "${SLURM_ARRAY_TASK_ID}" \
    --results-dir "${RESULTS_DIR_FULL}" \
    --passage-cache "${PASSAGE_CACHE}"

echo ""
echo "=== Task ${SLURM_ARRAY_TASK_ID} complete ==="
