#!/bin/bash
#SBATCH --job-name=reglab-prep
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/reglab_prepare_data_%j.out
#SBATCH --error=slurm_logs/reglab_prepare_data_%j.err
#
# Downloads reglab/barexam_qa + reglab/housing_qa via HuggingFace and writes
# data/reglab_eval/{barexam_qa,housing_qa} (CPU + disk; no GPU).
#
# Submit from any directory:
#   sbatch /home/ram112/projects/def-jieliang/ram112/LegalRAG/scripts/run_reglab_prepare_data.slurm
#
# Optional (sbatch --export=ALL,...):
#   REGLAB_ROOT=/path         — output parent (default: <repo>/data/reglab_eval)
#   MAX_CORPUS_DOCS=5000      — smoke subset for both datasets
#   LEGALRAG_VENV=/path      — Python venv (default: ../PyTorch)
#   SKIP_PIP_INSTALL=1        — skip pip install -e ".[eval]" if deps already in venv
#
# Note: Compute nodes may have no internet; if pip/HF fails, run the .sh on a login node
# or ask your site which Slurm partition allows outbound HTTP(S).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGALRAG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$LEGALRAG_ROOT"

mkdir -p slurm_logs

echo "=== reglab-prep host=$(hostname) JOB=${SLURM_JOB_ID:-local} LEGALRAG_ROOT=$LEGALRAG_ROOT ==="
echo "REGLAB_ROOT=${REGLAB_ROOT:-<default \$LEGALRAG_ROOT/data/reglab_eval>} MAX_CORPUS_DOCS=${MAX_CORPUS_DOCS:-<full>}"

chmod +x scripts/run_reglab_prepare_data.sh
bash scripts/run_reglab_prepare_data.sh

echo "=== reglab prepare Slurm job finished ==="
