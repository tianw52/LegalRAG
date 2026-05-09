#!/bin/bash
#SBATCH --job-name=lbrag-b50-q35-local
#SBATCH --output=slurm_logs/rewrite_benchmark_50_qwen35_9b_local_%j.out
#SBATCH --error=slurm_logs/rewrite_benchmark_50_qwen35_9b_local_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Benchmark-50 rewrite با Qwen3.5-9B روی GPU لوکال (transformers) — بدون HF/Together API.
#
# پیش‌نیاز: وزن مدل (Hub کش یا مسیر لوکال در LOCAL_MODEL_ID داخل .env.local_rewrite)
#   cp .env.local_rewrite.example .env.local_rewrite
# اگر روی لاگین pip خطای numpy/transformers داد، فقط نصب وابستگی را بفرست:
#   sbatch scripts/run_pip_install_local_rewrite.slurm
#
# Submit:
#   mkdir -p slurm_logs slurm_logs/results
#   sbatch scripts/run_rewrite_benchmark_50_qwen35_9b_local.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
REWRITE_EXTRA_ARGS="${REWRITE_EXTRA_ARGS:-}"
# همان venv پایتون ۳.۱۲ + PyTorch که روی GPU جاب‌ها استفاده می‌شود (torch از اینجا می‌آید)
PYTORCH_VENV="${PYTORCH_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
# بدون این، pip روی پایتون ۳.۹ لاگین است و با numpy/transformers تعارض می‌گیرد
# shellcheck source=/dev/null
source "$PYTORCH_VENV"

cd "$LEGALRAG_ROOT"
export LEGALRAG_ROOT="$(pwd -P)"

echo "Python: $(python3 -c 'import sys; print(sys.executable, sys.version)')"

PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")"
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip install --upgrade pip setuptools wheel -q
  pip install --upgrade 'numpy>=1.17' -q
  pip install --upgrade -r "${LEGALRAG_ROOT}/requirements-local-rewrite.txt" -q
) 9>"$PIP_LOCK"

LOCAL_ENV="${LEGALRAG_ROOT}/.env.local_rewrite"
if [[ -f "$LOCAL_ENV" ]]; then
  export LOCAL_REWRITE_ENV_FILE="$LOCAL_ENV"
fi

mkdir -p slurm_logs slurm_logs/results
RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/rewrite_benchmark_50_qwen35_9b_local_${SLURM_JOB_ID:-local}.txt"

echo "================================================================================"
echo "  Benchmark-50 Qwen3.5-9B LOCAL (transformers) — Job ${SLURM_JOB_ID:-local}"
echo "  Repo:           $LEGALRAG_ROOT"
echo "  Optional env:   $LOCAL_ENV (LOCAL_MODEL_ID, LOCAL_REWRITE_*)"
echo "  Output:         data/LegalBenchRAG/benchmark_50_reformated/qwen35_9b/"
echo "  Log:            $RESULT_FILE"
echo "================================================================================"

[[ -f "$LEGALRAG_ROOT/data/LegalBenchRAG/benchmarks_50/cuad.json" ]] || { echo "ERROR: benchmarks_50 missing"; exit 1; }

python3 scripts/rewrite_benchmark_50_qwen35_9b_local.py $REWRITE_EXTRA_ARGS 2>&1 | tee "$RESULT_FILE"

echo ""
echo "=== Next: python3 scripts/build_benchmark_50_reformated_processed.py --models qwen35_9b ==="
