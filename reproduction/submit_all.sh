#!/bin/bash
# One-shot submit script for Table 21 baseline reproduction.
# Run this from the project root:
#   bash reproduction/submit_all.sh
#
# Step 1: Download E5 models (needs internet; login node or compute node).
# Step 2: Run evaluation array (5 tasks: BM25 + 4 E5 models in parallel).
# Step 3: After all jobs complete, print the summary table.
set -euo pipefail

REPO_ROOT="/home/ram112/projects/def-jieliang/ram112"
cd "${REPO_ROOT}"

echo "=== Step 1: Submit E5 model download job ==="
DOWNLOAD_JOB=$(sbatch --parsable reproduction/download_e5_models.slurm)
echo "  Download job ID: ${DOWNLOAD_JOB}"

echo ""
echo "=== Step 2: Submit evaluation array (depends on download job) ==="
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:"${DOWNLOAD_JOB}" \
    reproduction/run_table21_baseline.slurm)
echo "  Eval array job ID: ${EVAL_JOB}"

echo ""
echo "=== Submitted! ==="
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER -o '%.10i %.15j %.8T %.12M %.12l %R'"
echo ""
echo "Once all jobs are done, print the results table:"
echo "  source ${REPO_ROOT}/PyTorch/bin/activate"
echo "  python3 reproduction/print_table21_summary.py"
echo ""
echo "Individual job logs:"
echo "  table21-baseline_<ARRAY_ID>_0.out  ← BM25"
echo "  table21-baseline_<ARRAY_ID>_1.out  ← E5-small"
echo "  table21-baseline_<ARRAY_ID>_2.out  ← E5-base"
echo "  table21-baseline_<ARRAY_ID>_3.out  ← E5-large"
echo "  table21-baseline_<ARRAY_ID>_4.out  ← E5-mistral"
