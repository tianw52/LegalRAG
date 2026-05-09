#!/bin/bash
#SBATCH --job-name=lbrag-b50-gemini-rewrite
#SBATCH --output=slurm_logs/rewrite_benchmark_50_gemini_%j.out
#SBATCH --error=slurm_logs/rewrite_benchmark_50_gemini_%j.err
#SBATCH --time=72:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Rewrite all 200 LegalBench benchmark_50 queries with Gemini (google-genai).
# No GPU required (API-only). Output:
#   $DATA_DIR/benchmark_50_reformated/gemini25_flash_lite/*.json
#
# Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in LegalRAG/.env
#
# Submit (default: --resume only fills missing / failed rows; safe for first run too):
#   mkdir -p slurm_logs
#   sbatch scripts/run_rewrite_benchmark_50_gemini.slurm
#
# Force rewrite every row from scratch:
#   REWRITE_EXTRA_ARGS="--all" sbatch scripts/run_rewrite_benchmark_50_gemini.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"

# Default --resume: skip rows that already have a non-empty v4_reddit_style (not [ERROR])
REWRITE_EXTRA_ARGS="${REWRITE_EXTRA_ARGS:---resume}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
# Match Python's idea of repo root (symlink home → /project) so LEGALRAG_ROOT and .env align
export LEGALRAG_ROOT="$(pwd -P)"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
OUT_REFORM="$DATA_DIR/benchmark_50_reformated/gemini25_flash_lite"

PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")"
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip uninstall legalrag -y 2>/dev/null || true
  pip install -e . -q
  pip install 'google-genai>=1.0.0' -q
) 9>"$PIP_LOCK"

[[ ! -f .env ]] && { echo "ERROR: missing $LEGALRAG_ROOT/.env (add GOOGLE_API_KEY)"; exit 1; }
if ! grep -qE '^[[:space:]]*(export[[:space:]]+)?(GOOGLE_API_KEY|GEMINI_API_KEY)[[:space:]]*=' .env; then
  echo "ERROR: .env must contain GOOGLE_API_KEY= or GEMINI_API_KEY=" >&2
  exit 1
fi

mkdir -p slurm_logs "$OUT_REFORM"

RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/rewrite_benchmark_50_gemini_${SLURM_JOB_ID:-local}.txt"
mkdir -p slurm_logs/results

echo "================================================================================"
echo "  Benchmark-50 Gemini rewrite — Job ${SLURM_JOB_ID:-local}"
echo "  Repo:              $LEGALRAG_ROOT"
echo "  Source queries:      $DATA_DIR/benchmarks_50/*.json"
echo "  Output rewrites:     $OUT_REFORM/"
echo "  Model (env):         ${LLM_MODEL:-gemini-2.5-flash-lite}"
echo "  Extra args:          ${REWRITE_EXTRA_ARGS:-<none>}"
echo "  Log copy:            $RESULT_FILE"
echo "================================================================================"

[[ -f "$DATA_DIR/benchmarks_50/cuad.json" ]] || { echo "ERROR: missing benchmarks_50"; exit 1; }

python3 scripts/rewrite_benchmark_50_gemini.py $REWRITE_EXTRA_ARGS 2>&1 | tee "$RESULT_FILE"

echo ""
echo "=== Rewrite step finished. Next (on login node or new job): ==="
echo "  python3 scripts/build_benchmark_50_reformated_processed.py --models gemini25_flash_lite"
echo "=== Done. ==="
