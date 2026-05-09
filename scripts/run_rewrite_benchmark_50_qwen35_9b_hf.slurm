#!/bin/bash
#SBATCH --job-name=lbrag-b50-q35-9b-hf
#SBATCH --output=slurm_logs/rewrite_benchmark_50_qwen35_9b_%j.out
#SBATCH --error=slurm_logs/rewrite_benchmark_50_qwen35_9b_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Rewrite benchmark_50 (200 queries) with Qwen3.5-9B via Hugging Face Router.
# Uses ONLY LegalRAG/.env.hf_rewrite (HF_TOKEN) — not the main .env used for Gemini.
#
# Setup once:
#   cp .env.hf_rewrite.example .env.hf_rewrite
#   # put HF_TOKEN=hf_... in .env.hf_rewrite
#
# Submit:
#   mkdir -p slurm_logs
#   sbatch scripts/run_rewrite_benchmark_50_qwen35_9b_hf.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"

REWRITE_EXTRA_ARGS="${REWRITE_EXTRA_ARGS:-}"

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
[[ -f "$HF_ENV" ]] || { echo "ERROR: missing $HF_ENV — copy from .env.hf_rewrite.example and set HF_TOKEN"; exit 1; }
if ! grep -qE '^[[:space:]]*(export[[:space:]]+)?(HF_TOKEN|LLM_API_KEY)[[:space:]]*=' "$HF_ENV"; then
  echo "ERROR: $HF_ENV must contain HF_TOKEN= or LLM_API_KEY=" >&2
  exit 1
fi

mkdir -p slurm_logs slurm_logs/results
RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/rewrite_benchmark_50_qwen35_9b_${SLURM_JOB_ID:-local}.txt"

echo "================================================================================"
echo "  Benchmark-50 Qwen3.5-9B (HF) — Job ${SLURM_JOB_ID:-local}"
echo "  Repo:           $LEGALRAG_ROOT"
echo "  HF env file:    $HF_ENV"
echo "  Output:         data/LegalBenchRAG/benchmark_50_reformated/qwen35_9b/"
echo "  Log:            $RESULT_FILE"
echo "================================================================================"

[[ -f "$LEGALRAG_ROOT/data/LegalBenchRAG/benchmarks_50/cuad.json" ]] || { echo "ERROR: benchmarks_50 missing"; exit 1; }

export HF_REWRITE_ENV_FILE="$HF_ENV"
python3 scripts/rewrite_benchmark_50_qwen35_9b_hf.py $REWRITE_EXTRA_ARGS 2>&1 | tee "$RESULT_FILE"

echo ""
echo "=== Next: python3 scripts/build_benchmark_50_reformated_processed.py --models qwen35_9b ==="
