#!/bin/bash
#SBATCH --job-name=lbrag-pip-local-rewrite
#SBATCH --output=slurm_logs/pip_install_local_rewrite_%j.out
#SBATCH --error=slurm_logs/pip_install_local_rewrite_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# فقط نصب وابستگی‌های requirements-local-rewrite.txt روی همان venv پایتون ۳.۱۲ + PyTorch.
# برای وقتی که روی نود لاگین «pip install -r ...» خطای ResolutionImpossible / numpy می‌دهد.
#
#   mkdir -p slurm_logs
#   sbatch scripts/run_pip_install_local_rewrite.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
PYTORCH_VENV="${PYTORCH_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
# shellcheck source=/dev/null
source "$PYTORCH_VENV"

cd "$LEGALRAG_ROOT"
export LEGALRAG_ROOT="$(pwd -P)"

echo "================================================================================"
echo "  pip install local-rewrite deps — Job ${SLURM_JOB_ID:-local}"
echo "  Python: $(python3 -c 'import sys; print(sys.executable)')"
echo "  requirements: ${LEGALRAG_ROOT}/requirements-local-rewrite.txt"
echo "================================================================================"

PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")"
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip install --upgrade pip setuptools wheel
  pip install --upgrade 'numpy>=1.17'
  pip install --upgrade -r "${LEGALRAG_ROOT}/requirements-local-rewrite.txt"
) 9>"$PIP_LOCK"

echo "OK. سپس: sbatch scripts/run_rewrite_benchmark_50_qwen35_9b_local.slurm"
