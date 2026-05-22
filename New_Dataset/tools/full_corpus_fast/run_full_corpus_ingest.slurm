#!/bin/bash
#SBATCH --job-name=reglab-full-ingest
#SBATCH --account=def-jieliang_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/ram112/projects/def-jieliang/ram112/LegalRAG
#SBATCH --output=slurm_logs/reglab_full_ingest_%j.out
#SBATCH --error=slurm_logs/reglab_full_ingest_%j.err
#
# Ingest the **full** barexam corpus (or any RegLab disk export) via the fast path.
#
# Data source (set one):
#   USE_PARQUET=1   PARQUET_FILE=/scratch/.../passages.parquet  (single file)
#   or PARQUET_FILE=/scratch/.../passages_meta  (directory of passages_part_*.parquet)
#   or
#   STAGE_TAR=/scratch/.../barexam_passages.tar   (tar root contains passages/)
#   or
#   CORPUS_DIR=/scratch/.../barexam_qa/corpus   (read directly from scratch — slower)
#
# Common variables:
#   INDEX_NAME=legalrag-reglab-barexam-full
#   ST_ENCODE_BATCH=256
#   CHILD_CHUNK_BATCH=1024
#
set -e

LEGALRAG_ROOT="/home/ram112/projects/def-jieliang/ram112/LegalRAG"
LEGALRAG_VENV="${LEGALRAG_VENV:-/home/ram112/projects/def-jieliang/ram112/PyTorch}"
SCRATCH="${SCRATCH:-/scratch/ram112}"
INDEX_NAME="${INDEX_NAME:-legalrag-reglab-barexam-full}"
ST_ENCODE_BATCH="${ST_ENCODE_BATCH:-256}"
CHILD_CHUNK_BATCH="${CHILD_CHUNK_BATCH:-1024}"
# Default: shard directory (after new build job); otherwise legacy single file
PARQUET_DIR_DEFAULT="${SCRATCH}/reglab_eval/barexam_qa/passages_meta"
PARQUET_FILE_LEGACY="${SCRATCH}/reglab_eval/barexam_qa/passages_meta/passages.parquet"
if [[ -z "${PARQUET_FILE:-}" ]]; then
  if [[ -d "$PARQUET_DIR_DEFAULT" ]] && compgen -G "$PARQUET_DIR_DEFAULT/passages_part_*.parquet" >/dev/null; then
    PARQUET_FILE="$PARQUET_DIR_DEFAULT"
  else
    PARQUET_FILE="$PARQUET_FILE_LEGACY"
  fi
fi
# If USE_PARQUET is unset and Parquet (file or shard dir) exists, enable it
if [[ -z "${USE_PARQUET:-}" ]]; then
  if [[ -f "$PARQUET_FILE" ]] || { [[ -d "$PARQUET_FILE" ]] && compgen -G "$PARQUET_FILE/passages_part_*.parquet" >/dev/null; }; then
    USE_PARQUET=1
  else
    USE_PARQUET=0
  fi
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p "$LEGALRAG_ROOT/slurm_logs"

module --force purge
module load StdEnv/2023 gcc arrow/17.0.0 cuda python/3.12 2>/dev/null || true
# shellcheck source=/dev/null
source "$LEGALRAG_VENV/bin/activate"
cd "$LEGALRAG_ROOT"

EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-mpnet-base-v2}"
EMBED_PROVIDER="${EMBED_PROVIDER:-sentence_transformers}"

# OpenSearch — same pattern as reglab eval jobs
OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-}"
if [[ -z "$OPENSEARCH_DIR" || ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
  for cand in "${SCRATCH:-}/opensearch-2.14.0" "$HOME/opensearch-2.14.0"; do
    if [[ -n "$cand" && -f "$cand/bin/opensearch" ]]; then
      OPENSEARCH_DIR="$cand"
      break
    fi
  done
fi
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "OpenSearch not found"; exit 1; }

OS_CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-security" ]]; then
  grep -q "plugins.security.disabled" "$OS_CONFIG" || echo "plugins.security.disabled: true" >> "$OS_CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env 2>/dev/null || true
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OS_DATA_DIR="${SLURM_TMPDIR}/opensearch-data"
OS_LOGS_DIR="${SLURM_TMPDIR}/opensearch-logs"
mkdir -p "$OS_DATA_DIR" "$OS_LOGS_DIR"
OPENSEARCH_STAMP="${SLURM_TMPDIR}/opensearch.pid"

cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
  ./bin/opensearch --daemonize -p "$OPENSEARCH_STAMP" \
  -Epath.data="$OS_DATA_DIR" -Epath.logs="$OS_LOGS_DIR"
cd "$LEGALRAG_ROOT"

for i in $(seq 1 60); do
  curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && break
  sleep 5
done
curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' || { echo "OpenSearch failed"; exit 1; }

if [[ -d "$OPENSEARCH_DIR/plugins/opensearch-knn" ]]; then
  for i in $(seq 1 30); do
    STATUS=$(curl -s "http://localhost:9200/_cluster/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    [[ "$STATUS" == "yellow" || "$STATUS" == "green" ]] && break
    sleep 5
  done
  sleep 15
fi

cleanup_os() {
  [[ -f "$OPENSEARCH_STAMP" ]] && kill "$(cat "$OPENSEARCH_STAMP")" 2>/dev/null || true
}
trap cleanup_os EXIT

echo "=== Data mode: USE_PARQUET=$USE_PARQUET  PARQUET_FILE=$PARQUET_FILE ==="
if [[ "$USE_PARQUET" == "1" ]]; then
  if [[ -f "$PARQUET_FILE" ]]; then
    :
  elif [[ -d "$PARQUET_FILE" ]] && compgen -G "$PARQUET_FILE/passages_part_*.parquet" >/dev/null; then
    :
  else
    echo "ERROR: Parquet expected but missing (file or passages_part_*.parquet dir): $PARQUET_FILE"
    echo "  Run first: sbatch tools/full_corpus_fast/run_build_passages_parquet.slurm"
    exit 1
  fi
fi

curl -s -X DELETE "http://localhost:9200/$INDEX_NAME" || true

INGEST_BASE=(
  tools/full_corpus_fast/fast_ingest.py
  --index-name "$INDEX_NAME"
  --embedding-provider "$EMBED_PROVIDER"
  --embedding-model "$EMBED_MODEL"
  --chunker hierarchical
  --parent-size 2048
  --chunk-size 512
  --chunk-overlap 64
  --st-encode-batch "$ST_ENCODE_BATCH"
  --child-chunk-batch "$CHILD_CHUNK_BATCH"
  --log-level INFO
)

if [[ "${USE_PARQUET:-0}" == "1" && -n "${PARQUET_FILE:-}" ]]; then
  echo "=== Fast ingest from Parquet: $PARQUET_FILE ==="
  python3 "${INGEST_BASE[@]}" --parquet "$PARQUET_FILE"
elif [[ -n "${STAGE_TAR:-}" ]]; then
  echo "=== Extract corpus tar → \$SLURM_TMPDIR/staged ==="
  mkdir -p "$SLURM_TMPDIR/staged"
  tar -xf "$STAGE_TAR" -C "$SLURM_TMPDIR/staged"
  # Tar layout must expose a corpus/ directory (containing passages/) under staged/.
  if [[ -n "${STAGE_CORPUS_ROOT:-}" ]]; then
    CORP_FOR_INGEST="$SLURM_TMPDIR/staged/$STAGE_CORPUS_ROOT"
  elif [[ -d "$SLURM_TMPDIR/staged/corpus" ]]; then
    CORP_FOR_INGEST="$SLURM_TMPDIR/staged/corpus"
  elif [[ -d "$SLURM_TMPDIR/staged/passages" ]]; then
    CORP_FOR_INGEST="$SLURM_TMPDIR/staged"
  else
    echo "ERROR: corpus directory not found after extract. Build tar from barexam_qa root: tar -cf c.tar corpus"
    echo "  Or set STAGE_CORPUS_ROOT=relative/path under staged/."
    exit 1
  fi
  echo "=== Fast ingest from staged corpus-dir: $CORP_FOR_INGEST ==="
  python3 "${INGEST_BASE[@]}" --corpus-dir "$CORP_FOR_INGEST"
else
  CORP_FOR_INGEST="${CORPUS_DIR:-${SCRATCH:-}/reglab_eval/barexam_qa/corpus}"
  echo "=== Fast ingest from CORPUS_DIR: $CORP_FOR_INGEST ==="
  python3 "${INGEST_BASE[@]}" --corpus-dir "$CORP_FOR_INGEST"
fi

echo "=== Done full ingest → index $INDEX_NAME ==="
date
