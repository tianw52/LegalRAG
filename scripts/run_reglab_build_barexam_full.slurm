#!/bin/bash
#SBATCH --job-name=barexam-build
#SBATCH --account=def-jieliang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/barexam_build_%j.out
#SBATCH --error=slurm_logs/barexam_build_%j.err
#
# Step 1 of 2: Build a COMPLETE barexam_qa Parquet corpus from HuggingFace.
#
# Full HF corpus = train + validation + test (~856,830 passages).
# This job writes to passages_meta_full (old train-only shards removed on fresh run).
#
# After this job completes, submit run_reglab_full_eval_array_v2.slurm
# which points PARQUET_META_DIR at passages_meta_full.
#
# Optional env overrides:
#   ROWS_PER_SHARD   — default 50000
#   MAX_CORPUS_DOCS  — default unset (full corpus)
#   RESUME           — set to 1 to skip already-written shards (safe to re-run)
#
set -euo pipefail

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"

PARQUET_OUT="${PARQUET_OUT:-${SCRATCH}/reglab_eval/barexam_qa/passages_meta_full}"
BENCHMARKS_OUT="${BENCHMARKS_OUT:-${SCRATCH}/reglab_eval/barexam_qa/benchmarks}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-50000}"
RESUME="${RESUME:-0}"

mkdir -p slurm_logs

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

echo "=== barexam-build  JOB=${SLURM_JOB_ID}  host=$(hostname) ==="
echo "  PARQUET_OUT     = $PARQUET_OUT"
echo "  BENCHMARKS_OUT  = $BENCHMARKS_OUT"
echo "  ROWS_PER_SHARD  = $ROWS_PER_SHARD"
echo "  RESUME          = $RESUME"
date

EXTRA_ARGS=()
[[ -n "${MAX_CORPUS_DOCS:-}" ]] && EXTRA_ARGS+=(--max-corpus-docs "$MAX_CORPUS_DOCS")
[[ "$RESUME" == "1" ]] && EXTRA_ARGS+=(--resume)

python3 tools/full_corpus_fast/build_barexam_direct_parquet.py \
    --parquet-dir    "$PARQUET_OUT" \
    --benchmarks-dir "$BENCHMARKS_OUT" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --hf-offline \
    "${EXTRA_ARGS[@]}"

echo "=== Parquet build done ==="
ls -lh "$PARQUET_OUT"
wc -l "$BENCHMARKS_OUT/barexam_qa.json" || true
date
echo "=== Submit eval array next: ==="
echo "  sbatch --export=ALL,PARQUET_META_DIR=$PARQUET_OUT,BAREXAM_DIR=${SCRATCH}/reglab_eval/barexam_qa \\"
echo "         scripts/run_reglab_full_eval_array_v2.slurm"
