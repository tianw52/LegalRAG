#!/bin/bash
#SBATCH --job-name=lbrag-b50-4embed-all
#SBATCH --output=slurm_logs/b50_4embed_sweep_%A_%a.out
#SBATCH --error=slurm_logs/b50_4embed_sweep_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-2
#
# One array task = one REWRITE_VARIANT (mistral, qwen72b, qwen35_9b).
# Each task runs the full 12-config ingest+eval sweep + aggregate plots.
#
#   mkdir -p slurm_logs
#   sbatch --export=HF_TOKEN=hf_... scripts/run_b50_4embed_sweep_all_variants.slurm
#
# Override account/time/gpu if your site differs.
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) export REWRITE_VARIANT=mistral ;;
  1) export REWRITE_VARIANT=qwen72b ;;
  2) export REWRITE_VARIANT=qwen35_9b ;;
  *)
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Reuse the same OpenSearch + pip + sweep driver as the single-variant Slurm script.
bash "$LEGALRAG_ROOT/scripts/run_b50_4embed_sweep_one_variant.slurm"
