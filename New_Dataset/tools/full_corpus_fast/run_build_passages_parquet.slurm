#!/bin/bash
#SBATCH --job-name=reglab-parquet
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/reglab_build_parquet_%j.out
#SBATCH --error=slurm_logs/reglab_build_parquet_%j.err
#
# Step 1: convert corpus/passages/*.txt to Parquet (no GPU).
# Default: passages_part_XXX.parquet shards under passages_meta/ (resume + TIMEOUT safety).
#
# Optional env:
#   CORPUS_ROOT      — default: $SCRATCH/reglab_eval/barexam_qa/corpus
#   PARQUET_META_DIR — default: $SCRATCH/reglab_eval/barexam_qa/passages_meta
#   SHARD_ROWS       — default: 50000 (rows per part)
#   FLUSH_ROWS       — default: 50000 (in-memory batch inside each part)
#   RESUME=1         — continue from valid existing parts
#
# Legacy single-file (optional):
#   SINGLE_PARQUET_OUT=/path/passages.parquet  — if set, write one file instead of shards
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

CORPUS_ROOT="${CORPUS_ROOT:-${SCRATCH}/reglab_eval/barexam_qa/corpus}"
PARQUET_META_DIR="${PARQUET_META_DIR:-${SCRATCH}/reglab_eval/barexam_qa/passages_meta}"
SHARD_ROWS="${SHARD_ROWS:-50000}"
FLUSH_ROWS="${FLUSH_ROWS:-50000}"

echo "=== RegLab: build passages Parquet ==="
echo "  CORPUS_ROOT      = $CORPUS_ROOT"
echo "  PARQUET_META_DIR = $PARQUET_META_DIR"
echo "  SHARD_ROWS       = $SHARD_ROWS"
echo "  FLUSH_ROWS       = $FLUSH_ROWS"
echo "  RESUME           = ${RESUME:-0}"
date

mkdir -p "$LEGALRAG_ROOT/slurm_logs"

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

[[ -d "$CORPUS_ROOT" ]] || { echo "ERROR: CORPUS_ROOT not found: $CORPUS_ROOT"; exit 1; }

if [[ -n "${SINGLE_PARQUET_OUT:-}" ]]; then
  mkdir -p "$(dirname "$SINGLE_PARQUET_OUT")"
  python3 tools/full_corpus_fast/build_passages_parquet.py \
    --corpus-root "$CORPUS_ROOT" \
    --out "$SINGLE_PARQUET_OUT" \
    --flush-rows "$FLUSH_ROWS" \
    --compression zstd
  echo "=== Done Parquet build (single file) ==="
  ls -lh "$SINGLE_PARQUET_OUT"
else
  mkdir -p "$PARQUET_META_DIR"
  EXTRA=()
  if [[ "${RESUME:-0}" == "1" ]]; then
    EXTRA+=(--resume)
  fi
  python3 tools/full_corpus_fast/build_passages_parquet.py \
    --corpus-root "$CORPUS_ROOT" \
    --output-dir "$PARQUET_META_DIR" \
    --shard-rows "$SHARD_ROWS" \
    --flush-rows "$FLUSH_ROWS" \
    --compression zstd \
    "${EXTRA[@]}"
  echo "=== Done Parquet build (shards) ==="
  ls -lh "$PARQUET_META_DIR"/passages_part_*.parquet 2>/dev/null | head -20
  ls -lh "$PARQUET_META_DIR"/passages_part_*.parquet 2>/dev/null | tail -5
fi

date
