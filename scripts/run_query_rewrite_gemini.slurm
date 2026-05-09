#!/bin/bash
#SBATCH --job-name=query_rewrite_gemini
#SBATCH --output=slurm_logs/query_rewrite_gemini_%j.out
#SBATCH --error=slurm_logs/query_rewrite_gemini_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Runs query_rewrite.py --mode compare using Google Gemini (free tier).
# No GPU needed — API calls only. Requires outbound internet on compute node.
#
# Setup (one-time):
#   1. Add GOOGLE_API_KEY to .env (get free key at https://aistudio.google.com/app/apikey)
#
# Submit:
#   cd /home/ram112/projects/def-jieliang/ram112/LegalRAG
#   sbatch scripts/run_query_rewrite_gemini.slurm
#
# If compute nodes have no internet, run on login node instead:
#   nohup bash scripts/run_query_rewrite_gemini.sh > slurm_logs/gemini_$(date +%Y%m%d_%H%M%S).out 2>&1 &

set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
QUERY_SCRIPT="$LEGALRAG_ROOT/scripts/query_rewrite.py"

echo "================================================================"
echo " LegalRAG Query Rewrite (Gemini) — SLURM job $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo "================================================================"

cd "$LEGALRAG_ROOT"
mkdir -p slurm_logs

module --force purge 2>/dev/null || true
module load StdEnv/2023 python/3.12 2>/dev/null || true

source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate 2>/dev/null || \
source /project/6006955/ram112/PyTorch/bin/activate 2>/dev/null || true

pip install google-genai -q 2>/dev/null || true

# Load .env and set Gemini
set -a
[[ -f .env ]] && source .env
set +a
export LLM_PROVIDER=gemini
if [[ "$LLM_MODEL" != gemini-* ]]; then
    export LLM_MODEL="gemini-2.5-flash-lite"
fi

if [[ -z "$GOOGLE_API_KEY" && -z "$GEMINI_API_KEY" ]]; then
    echo "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set in .env"
    exit 1
fi

echo ""
echo "Model: $LLM_MODEL | Provider: gemini | Mode: compare"
echo ""

python3 -u "$QUERY_SCRIPT" --mode compare --n 4

echo ""
echo "Finished: $(date)"
echo "Results in $LEGALRAG_ROOT/results/query_rewrites_compare_2vars_${LLM_MODEL}.json"
