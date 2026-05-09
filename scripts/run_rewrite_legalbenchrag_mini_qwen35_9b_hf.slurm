#!/bin/bash
#SBATCH --job-name=lbrag-mini-q35-9b
#SBATCH --output=slurm_logs/rewrite_legalbenchrag_mini_qwen35_9b_%j.out
#SBATCH --error=slurm_logs/rewrite_legalbenchrag_mini_qwen35_9b_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Rewrite legalbenchrag-mini benchmarks (194 queries × 4 datasets) with Qwen3.5-9B (HF Router).
# Output: data/legalbenchrag-mini/benchmark_50_reformated/qwen35_9b/
#
# Submit:
#   mkdir -p slurm_logs
#   sbatch scripts/run_rewrite_legalbenchrag_mini_qwen35_9b_hf.slurm
#
# Resume after interruption:
#   REWRITE_EXTRA_ARGS="--resume" sbatch scripts/run_rewrite_legalbenchrag_mini_qwen35_9b_hf.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
REWRITE_EXTRA_ARGS="${REWRITE_EXTRA_ARGS:---resume}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
export LEGALRAG_ROOT="$(pwd -P)"

PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")"
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip uninstall legalrag -y 2>/dev/null || true
  pip install -e . -q
) 9>"$PIP_LOCK"

HF_ENV="${LEGALRAG_ROOT}/.env.hf_rewrite"
[[ -f "$HF_ENV" ]] || { echo "ERROR: missing $HF_ENV"; exit 1; }

mkdir -p slurm_logs slurm_logs/results
RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/rewrite_legalbenchrag_mini_qwen35_9b_${SLURM_JOB_ID:-local}.txt"

echo "================================================================================"
echo "  legalbenchrag-mini Qwen3.5-9B (HF) — Job ${SLURM_JOB_ID:-local}"
echo "  Repo:           $LEGALRAG_ROOT"
echo "  Source:         data/legalbenchrag-mini/benchmarks/"
echo "  Output:         data/legalbenchrag-mini/benchmark_50_reformated/qwen35_9b/"
echo "  Extra args:     $REWRITE_EXTRA_ARGS"
echo "================================================================================"

[[ -f "$LEGALRAG_ROOT/data/legalbenchrag-mini/benchmarks/cuad.json" ]] || { echo "ERROR: mini benchmarks missing"; exit 1; }

export HF_REWRITE_ENV_FILE="$HF_ENV"
python3 scripts/rewrite_benchmark_50_qwen35_9b_hf.py --mini $REWRITE_EXTRA_ARGS 2>&1 | tee "$RESULT_FILE"

echo ""
echo "=== Next: python3 scripts/build_benchmark_50_reformated_processed.py --mini --models qwen35_9b ==="
