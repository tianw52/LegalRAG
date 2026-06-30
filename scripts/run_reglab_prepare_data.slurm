#!/bin/bash
#SBATCH --job-name=reglab-prep
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
# Logs go to the directory you run sbatch from (use e.g. cd "$SCRATCH" first — project dirs are often read-only on compute nodes).
#SBATCH --output=reglab_prepare_data_%j.out
#SBATCH --error=reglab_prepare_data_%j.err
#
# Downloads reglab/barexam_qa + reglab/housing_qa via HuggingFace and writes
# REGLAB_ROOT/{barexam_qa,housing_qa} (CPU + disk; no GPU). Default REGLAB_ROOT:
# $SCRATCH/reglab_eval on Slurm if unset, else <repo>/data/reglab_eval.
#
# Submit from a writable directory (recommended):
#   cd "${SCRATCH:-$HOME}" && sbatch /home/ram112/projects/def-jieliang/ram112/LegalRAG/scripts/run_reglab_prepare_data.slurm
#
# Optional (sbatch --export=ALL,...):
#   REGLAB_ROOT=/path         — output parent (Slurm default if unset: $SCRATCH/reglab_eval)
#   MAX_CORPUS_DOCS=5000      — smoke subset for both datasets
#   LEGALRAG_VENV=/path      — Python venv (default: ../PyTorch)
#   SKIP_PIP_INSTALL=0        — try pip install -e ".[eval]" on compute (not recommended: no network)
#
# Note: Compute nodes may have no internet; if pip/HF fails, run the .sh on a login node
# or ask your site which Slurm partition allows outbound HTTP(S).
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
cd "$LEGALRAG_ROOT"

module --force purge
module load StdEnv/2023
module load gcc
module load arrow/24.0.0
module load python/3.12

# Large HF downloads: default to SCRATCH on clusters when REGLAB_ROOT is not set (project FS may be read-only on compute).
if [[ -z "${REGLAB_ROOT:-}" ]] && [[ -n "${SCRATCH:-}" ]]; then
  export REGLAB_ROOT="${SCRATCH}/reglab_eval"
  echo "REGLAB_ROOT unset — using SCRATCH: $REGLAB_ROOT"
fi

echo "=== reglab-prep host=$(hostname) JOB=${SLURM_JOB_ID:-local} LEGALRAG_ROOT=$LEGALRAG_ROOT ==="
echo "REGLAB_ROOT=${REGLAB_ROOT:-<default \$LEGALRAG_ROOT/data/reglab_eval>} MAX_CORPUS_DOCS=${MAX_CORPUS_DOCS:-<full>}"
export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"

chmod +x scripts/run_reglab_prepare_data.sh
bash scripts/run_reglab_prepare_data.sh

echo "=== reglab prepare Slurm job finished ==="
