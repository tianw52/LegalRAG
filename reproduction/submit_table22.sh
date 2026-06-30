#!/bin/bash
# Submit Table 22 baseline evaluation jobs.
#
# This script:
#   1. Runs the pre-flight check to verify all deps + models are cached
#   2. Submits the 5-task array job (BM25 + 4 E5 models)
#   3. Prints monitoring and summary commands
#
# Usage:
#   bash reproduction/submit_table22.sh
#
set -euo pipefail

REPO_ROOT="/home/ram112/projects/def-jieliang/ram112"
VENV="${REPO_ROOT}/PyTorch"
cd "${REPO_ROOT}"

echo "========================================="
echo "  Table 22 Baseline — Submission Script"
echo "========================================="
echo ""

# ── Step 1: Pre-flight check ─────────────────────────────────────────────────
echo "--- Step 1: Pre-flight check ---"
# arrow/17.0.0 provides pyarrow needed by datasets and sentence-transformers
module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true
source "${VENV}/bin/activate"
if ! python3 reproduction/preflight_check.py; then
    echo ""
    echo "ERROR: Pre-flight check failed. Fix issues above, then re-run this script."
    exit 1
fi

# ── Step 2: Submit evaluation array ──────────────────────────────────────────
echo "--- Step 2: Submitting SLURM array job (tasks 0–4) ---"
EVAL_JOB=$(sbatch --parsable reproduction/run_table22_baseline.slurm)
echo "  Eval array job ID: ${EVAL_JOB}"
echo ""

# ── Step 3: Instructions ─────────────────────────────────────────────────────
echo "========================================="
echo "  Jobs submitted!"
echo "========================================="
echo ""
echo "  Array tasks:"
echo "    ${EVAL_JOB}_0  → BM25"
echo "    ${EVAL_JOB}_1  → E5-small-v2"
echo "    ${EVAL_JOB}_2  → E5-base-v2"
echo "    ${EVAL_JOB}_3  → E5-large-v2"
echo "    ${EVAL_JOB}_4  → E5-mistral-7b-instruct"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER -o '%.10i %.22j %.8T %.12M %.12l %R'"
echo ""
echo "  Per-task logs:"
echo "    tail -f table22-baseline_${EVAL_JOB}_0.out   # BM25"
echo "    tail -f table22-baseline_${EVAL_JOB}_4.out   # E5-mistral"
echo ""
echo "  Results (as tasks finish):"
echo "    ls ${REPO_ROOT}/reproduction/table21_results/"
echo ""
echo "  Print comparison table when all done:"
echo "    python3 reproduction/print_table21_summary.py"
echo ""
