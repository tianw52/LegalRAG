#!/bin/bash
#SBATCH --job-name=lbrag-b50raw-dpr
#SBATCH --output=slurm_logs/benchmarks50_raw_dpr_clerc_%j.out
#SBATCH --error=slurm_logs/benchmarks50_raw_dpr_clerc_%j.err
#SBATCH --time=18:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Full-corpus ingest + retrieval eval using **non-reformulated** benchmark JSONs:
#   $LEGALRAG_ROOT/data/LegalBenchRAG/benchmarks_50/
#
# Model: jhu-clsp/BERT-DPR-CLERC-ft
# OpenSearch index: **legalrag-legalbenchrag-b50raw-dpr-clerc** (isolated from
# reformatted runs that use legalrag-legalbenchrag-exp2-dpr-clerc).
#
# Logs / snapshots use the `benchmarks50_raw_*` prefix so nothing overwrites
# prior benchmark50_* jobs.
#
# Submit:
#   mkdir -p slurm_logs/results
#   sbatch scripts/run_benchmarks50_raw_dpr_clerc.slurm
#
# Optional:
#   sbatch --export=HF_TOKEN=hf_...,BENCHMARKS50_DIR=/path scripts/run_benchmarks50_raw_dpr_clerc.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
BENCHMARKS50_DIR="${BENCHMARKS50_DIR:-$DATA_DIR/benchmarks_50}"

EMBED_MODEL="${EMBED_MODEL:-jhu-clsp/BERT-DPR-CLERC-ft}"
INDEX_NAME="${INDEX_NAME:-legalrag-legalbenchrag-b50raw-dpr-clerc}"
KS="${KS:-1 5 10 20}"

export OPENSEARCH_TIMEOUT="${OPENSEARCH_TIMEOUT:-600}"
export OPENSEARCH_JAVA_OPTS="${OPENSEARCH_JAVA_OPTS:--Xms4g -Xmx10g}"

OPENSEARCH_DIR="${OPENSEARCH_TARBALL_DIR:-$HOME/opensearch-2.14.0}"

module --force purge
module load StdEnv/2023
module load gcc
module load cuda
module load python/3.12
source /home/ram112/projects/def-jieliang/ram112/PyTorch/bin/activate

cd "$LEGALRAG_ROOT"
pip uninstall legalrag -y 2>/dev/null || true
pip install -e . -q

export CUDA_VISIBLE_DEVICES=0
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-sentence_transformers}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/benchmarks50_raw_${SLURM_JOB_ID:-local}_DPR_CLERC_FT.txt"

echo "================================================================================"
echo "  benchmarks_50 (RAW) — BERT-DPR-CLERC-ft — Job ${SLURM_JOB_ID:-local}"
echo "  LEGALRAG_ROOT:     $LEGALRAG_ROOT"
echo "  Corpus:            $DATA_DIR/corpus  [--all]"
echo "  Eval JSONs:        $BENCHMARKS50_DIR"
echo "  Embedding:         $EMBED_MODEL"
echo "  INDEX_NAME:        $INDEX_NAME  (isolated from reformatted runs)"
echo "  OPENSEARCH_TIMEOUT / JAVA_OPTS: ${OPENSEARCH_TIMEOUT} / ${OPENSEARCH_JAVA_OPTS}"
echo "  Eval snapshot:     $RESULT_FILE"
echo "================================================================================"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
mkdir -p slurm_logs/results

if [[ ! -d "$DATA_DIR/corpus" ]]; then
  echo "ERROR: corpus not found: $DATA_DIR/corpus"
  exit 1
fi
if [[ ! -f "$BENCHMARKS50_DIR/cuad.json" ]]; then
  echo "ERROR: eval JSONs not found under: $BENCHMARKS50_DIR"
  exit 1
fi
if [[ ! -f "$OPENSEARCH_DIR/bin/opensearch" ]]; then
  echo "ERROR: OpenSearch not at $OPENSEARCH_DIR. Run: ./scripts/setup_opensearch_tarball.sh"
  exit 1
fi

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
  echo "" >> "$CONFIG"
  echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost
export OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-0}.pid"
echo "Starting OpenSearch..."
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" \
DISABLE_INSTALL_DEMO_CONFIG=true \
./bin/opensearch --daemonize -p "$OPENSEARCH_PID"
cd "$LEGALRAG_ROOT"

echo "Waiting for OpenSearch..."
for _i in $(seq 1 60); do
  if curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"'; then
    echo "OpenSearch ready."
    break
  fi
  sleep 5
done
if ! curl -s http://localhost:9200/_cluster/health | grep -q '"status"'; then
  echo "ERROR: OpenSearch failed to start."
  [[ -f "$OPENSEARCH_PID" ]] && kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
  exit 1
fi

echo ""
echo "██████████████████████████████████████████████████████████████████████████████"
echo "  EXPERIMENT: benchmarks_50 RAW — BERT-DPR-CLERC-ft"
echo "  EMBEDDING: $EMBED_MODEL"
echo "  INDEX:     $INDEX_NAME"
echo "██████████████████████████████████████████████████████████████████████████████"
echo ""

echo ">>> Dropping index ${INDEX_NAME} if it exists..."
curl -sS -X DELETE "http://localhost:9200/${INDEX_NAME}" >/dev/null || true

echo ">>> Ingest: full corpus (--all) ..."
python3 -m evaluation.LegalBenchRAG.ingest \
  --data-dir "$DATA_DIR" \
  --all \
  --embedding-model "$EMBED_MODEL" \
  --index-name "$INDEX_NAME" \
  --log-level INFO

echo ""
echo ">>> Eval: retrieval — EMBEDDING=${EMBED_MODEL} INDEX=${INDEX_NAME}"
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
  --data-dir "$DATA_DIR" \
  --benchmarks-dir "$BENCHMARKS50_DIR" \
  --index-name "$INDEX_NAME" \
  --embedding-model "$EMBED_MODEL" \
  --ks $KS \
  --log-level INFO 2>&1 | tee "$RESULT_FILE"

echo ""
echo ">>> Finished benchmarks_50 RAW — BERT-DPR-CLERC-ft."

echo "Stopping OpenSearch..."
if [[ -f "$OPENSEARCH_PID" ]]; then
  kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
  rm -f "$OPENSEARCH_PID"
fi

echo "=== Done (benchmarks_50 raw — DPR-CLERC). ==="
