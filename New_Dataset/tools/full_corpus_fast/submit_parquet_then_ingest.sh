#!/bin/bash
# Submit chain: (1) build Parquet  (2) full ingest with OpenSearch + GPU
# Run from LegalRAG root:
#   bash tools/full_corpus_fast/submit_parquet_then_ingest.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p slurm_logs

BUILD_JOB="$(sbatch --parsable "${ROOT}/tools/full_corpus_fast/run_build_passages_parquet.slurm")"
INGEST_JOB="$(sbatch --parsable --dependency=afterok:"${BUILD_JOB}" "${ROOT}/tools/full_corpus_fast/run_full_corpus_ingest.slurm")"

echo "Submitted:"
echo "  (1) build Parquet    JOBID=$BUILD_JOB"
echo "  (2) full fast ingest JOBID=$INGEST_JOB  (starts after 1 succeeds)"
echo "Logs: ${ROOT}/slurm_logs/reglab_build_parquet_${BUILD_JOB}.{out,err}"
echo "      ${ROOT}/slurm_logs/reglab_full_ingest_${INGEST_JOB}.{out,err}"
