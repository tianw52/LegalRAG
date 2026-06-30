#!/bin/bash
#SBATCH --job-name=rgl-hou-char
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=32:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=8,10
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/rgl_housing_char_retry_%A_%a.out
#SBATCH --error=slurm_logs/rgl_housing_char_retry_%A_%a.err
#
# Retry HousingQA tasks that failed during ingest (T8 octen-hier, T10 qwen3-hier).
# Tian metrics only: ingest + CharRecall/CharPrecision — no paper_eval.
#
# Submit:
#   sbatch scripts/run_reglab_housing_char_retry.slurm
#
set -euo pipefail

export RUN_PAPER_EVAL=0
export OPENSEARCH_JAVA_OPTS="${OPENSEARCH_JAVA_OPTS:--Xms16g -Xmx16g}"

exec bash scripts/run_reglab_housing_full_eval_array.slurm
