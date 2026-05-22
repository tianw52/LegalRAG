#!/bin/bash
#SBATCH --job-name=housing-parquet
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Build reglab/housing_qa directly into sharded Parquet + benchmarks JSON.
# No intermediate .txt files are created (avoids ~1.7M inode usage).
#
# HuggingFace dataset is read from the shared-filesystem cache populated by
# the login-node pre-download step.  Set HF_DATASETS_OFFLINE=1 if you are
# certain the cache is complete and want to avoid any outbound requests.
#
# Submit from the LegalRAG repo root, e.g.:
#   cd /home/ram112/projects/def-jieliang/ram112/LegalRAG
#   sbatch scripts/run_housing_direct_parquet.slurm
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="/home/ram112/projects/def-jieliang/ram112/PyTorch"

# Output locations — must match eval script defaults:
#   PARQUET_META_DIR = $HOUSING_DIR/passages_meta
#   BENCHMARKS_DIR   = $HOUSING_DIR/benchmarks
HOUSING_DIR="${HOUSING_DIR:-${SCRATCH}/reglab_eval/housing_qa}"
PARQUET_DIR="${PARQUET_DIR:-${HOUSING_DIR}/passages_meta}"
BENCHMARKS_DIR="${BENCHMARKS_DIR:-${HOUSING_DIR}/benchmarks}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-200000}"

# ---- environment -----------------------------------------------------------
module --force purge
module load StdEnv/2023
module load gcc
module load arrow/17.0.0
module load python/3.11

export PYTHONPATH="$LEGALRAG_ROOT:${PYTHONPATH:-}"
# Use the default HF cache (~/.cache/huggingface) where the login-node download placed data.
# Do NOT override HF_HOME to a scratch path (would lose the cached Arrow files).
export HF_DATASETS_OFFLINE=1   # dataset is fully cached — no internet needed

echo "=== housing-parquet ==="
echo "  host        : $(hostname)"
echo "  JOB         : ${SLURM_JOB_ID:-local}"
echo "  PARQUET_DIR : $PARQUET_DIR"
echo "  BENCHMARKS  : $BENCHMARKS_DIR"
echo "  shard size  : $ROWS_PER_SHARD"
echo "  HF_OFFLINE  : $HF_DATASETS_OFFLINE"
echo "  Python      : $("$LEGALRAG_VENV/bin/python3" --version)"
date

cd "$LEGALRAG_ROOT"

"$LEGALRAG_VENV/bin/python3" tools/full_corpus_fast/build_housing_direct_parquet.py \
    --parquet-dir   "$PARQUET_DIR" \
    --benchmarks-dir "$BENCHMARKS_DIR" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --log-level INFO

echo "=== DONE ==="
date
ls -lh "$PARQUET_DIR"/passages_part_*.parquet 2>/dev/null || echo "(no shards yet)"
ls -lh "$BENCHMARKS_DIR"/housing_qa.json 2>/dev/null || echo "(no benchmark file yet)"
