#!/bin/bash
#SBATCH --job-name=reglab-housing-parquet
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/reglab_build_housing_parquet_%j.out
#SBATCH --error=slurm_logs/reglab_build_housing_parquet_%j.err
#
# Turn reglab/housing_qa disk export (after prepare) into Parquet shards.
# ~1.7M statute .txt files — long wall time; use RESUME=1 after failures.
#
# Prerequisite on login (or Slurm with HF/cache):
#   REGLAB_ROOT=$SCRATCH/reglab_eval  bash scripts/run_reglab_prepare_data.sh
#   (or only:  python -m evaluation.reglab.prepare housing_qa --out-dir "$REGLAB_ROOT/housing_qa")
#
# Env (defaults for Narval scratch):
#   CORPUS_ROOT      — $SCRATCH/reglab_eval/housing_qa/corpus
#   PARQUET_META_DIR — $SCRATCH/reglab_eval/housing_qa/passages_meta
#   SHARD_ROWS, FLUSH_ROWS, RESUME  — same as run_build_passages_parquet.slurm
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

CORPUS_ROOT="${CORPUS_ROOT:-${SCRATCH}/reglab_eval/housing_qa/corpus}"
PARQUET_META_DIR="${PARQUET_META_DIR:-${SCRATCH}/reglab_eval/housing_qa/passages_meta}"
SHARD_ROWS="${SHARD_ROWS:-50000}"
FLUSH_ROWS="${FLUSH_ROWS:-50000}"

echo "=== RegLab housing: build statute Parquet ==="
echo "  CORPUS_ROOT      = $CORPUS_ROOT"
echo "  PARQUET_META_DIR = $PARQUET_META_DIR"
echo "  RESUME           = ${RESUME:-0}"
date

mkdir -p "$LEGALRAG_ROOT/slurm_logs"

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

[[ -d "$CORPUS_ROOT/statutes" ]] || {
  echo "ERROR: expected statutes/ under CORPUS_ROOT: $CORPUS_ROOT"
  exit 1
}

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

echo "=== Done ==="
ls -lh "$PARQUET_META_DIR"/passages_part_*.parquet 2>/dev/null | head -5
ls -lh "$PARQUET_META_DIR"/passages_part_*.parquet 2>/dev/null | tail -3
date
