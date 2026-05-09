#!/bin/bash
#SBATCH --job-name=lbrag-b50rec-fl-le
#SBATCH --output=slurm_logs/b50_recursive_b50flat_legal_embed_%j.out
#SBATCH --error=slurm_logs/b50_recursive_b50flat_legal_embed_%j.err
#SBATCH --time=18:00:00
#SBATCH --account=def-jieliang_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# exp3 — Legal-Embed + **recursive** chunking + flat benchmarks_50 (non-reformulated)
#
# Eval:  $BENCHMARKS50_DIR (default: .../data/LegalBenchRAG/benchmarks_50)
# Index: legalrag-legalbenchrag-recursive-b50flat-legal-embed
#
# Submit:
#   mkdir -p slurm_logs/results
#   sbatch scripts/run_b50_recursive_b50flat_legal_embed.slurm
#
set -euo pipefail

LEGALRAG_ROOT="${LEGALRAG_ROOT:-/home/ram112/projects/def-jieliang/ram112/LegalRAG}"
DATA_DIR="$LEGALRAG_ROOT/data/LegalBenchRAG"
BENCHMARKS50_DIR="${BENCHMARKS50_DIR:-$DATA_DIR/benchmarks_50}"

EMBED_MODEL="${EMBED_MODEL:-axondendriteplus/Legal-Embed-bge-base-en-v1.5}"
INDEX_NAME="${INDEX_NAME:-legalrag-legalbenchrag-recursive-b50flat-legal-embed}"
KS="${KS:-1 5 10 20}"
CHUNKER="${CHUNKER:-recursive}"

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
PIP_LOCK="${HOME}/.cache/locks/legalrag_pip_install.lock"
mkdir -p "$(dirname "$PIP_LOCK")"
(
  flock -w 7200 9 || { echo "ERROR: pip lock wait timeout"; exit 1; }
  pip uninstall legalrag -y 2>/dev/null || true
  pip install -e . -q
) 9>"$PIP_LOCK"

export CUDA_VISIBLE_DEVICES=0
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-sentence_transformers}"
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

RESULT_FILE="$LEGALRAG_ROOT/slurm_logs/results/b50_recursive_b50flat_${SLURM_JOB_ID:-local}_LEGAL_EMBED.txt"

echo "================================================================================"
echo "  exp3 — recursive — benchmarks_50 (flat) — Legal-Embed — Job ${SLURM_JOB_ID:-local}"
echo "  CHUNKER:           $CHUNKER"
echo "  Corpus:            $DATA_DIR/corpus  [--all]"
echo "  Eval JSONs:        $BENCHMARKS50_DIR"
echo "  Embedding:         $EMBED_MODEL"
echo "  INDEX_NAME:        $INDEX_NAME"
echo "  Snapshot:          $RESULT_FILE"
echo "================================================================================"
mkdir -p slurm_logs/results

[[ -d "$DATA_DIR/corpus" ]] || { echo "ERROR: missing corpus"; exit 1; }
[[ -f "$BENCHMARKS50_DIR/cuad.json" ]] || { echo "ERROR: missing eval JSONs: $BENCHMARKS50_DIR"; exit 1; }
[[ -f "$OPENSEARCH_DIR/bin/opensearch" ]] || { echo "ERROR: OpenSearch not at $OPENSEARCH_DIR"; exit 1; }

CONFIG="$OPENSEARCH_DIR/config/opensearch.yml"
if [[ -f "$CONFIG" ]] && ! grep -q "plugins.security.disabled" "$CONFIG"; then
  echo "" >> "$CONFIG"
  echo "plugins.security.disabled: true" >> "$CONFIG"
fi

export OPENSEARCH_HOST=localhost OPENSEARCH_PORT=9200
[[ ! -f .env ]] && cp .env.example .env
grep -q "OPENSEARCH_PORT" .env && sed -i 's/OPENSEARCH_PORT=.*/OPENSEARCH_PORT=9200/' .env || echo "OPENSEARCH_PORT=9200" >> .env

OPENSEARCH_PID="${SLURM_TMPDIR:-/tmp}/opensearch_${SLURM_JOB_ID:-0}.pid"
OPENSEARCH_DATA_DIR="${SLURM_TMPDIR:-/tmp}/opensearch_data_${SLURM_JOB_ID:-0}"
mkdir -p "$OPENSEARCH_DATA_DIR"
echo "OpenSearch path.data (job-local): $OPENSEARCH_DATA_DIR"
cd "$OPENSEARCH_DIR"
OPENSEARCH_PATH_CONF="$OPENSEARCH_DIR/config" DISABLE_INSTALL_DEMO_CONFIG=true \
  ./bin/opensearch --daemonize -p "$OPENSEARCH_PID" \
  -E "path.data=$OPENSEARCH_DATA_DIR" \
  -E "cluster.name=legalrag-os-${SLURM_JOB_ID:-local}"
cd "$LEGALRAG_ROOT"

for _i in $(seq 1 60); do
  curl -s http://localhost:9200/_cluster/health 2>/dev/null | grep -q '"status"' && { echo "OpenSearch ready."; break; }
  sleep 5
done
curl -s http://localhost:9200/_cluster/health | grep -q '"status"' || { echo "ERROR: OpenSearch failed"; exit 1; }

curl -sS -X DELETE "http://localhost:9200/${INDEX_NAME}" >/dev/null || true

echo ">>> Ingest: --all --chunker $CHUNKER ..."
python3 -m evaluation.LegalBenchRAG.ingest \
  --data-dir "$DATA_DIR" \
  --all \
  --chunker "$CHUNKER" \
  --embedding-model "$EMBED_MODEL" \
  --index-name "$INDEX_NAME" \
  --log-level INFO

echo ">>> Eval ..."
python3 -m evaluation.LegalBenchRAG.eval_precision_recall \
  --data-dir "$DATA_DIR" \
  --benchmarks-dir "$BENCHMARKS50_DIR" \
  --index-name "$INDEX_NAME" \
  --embedding-model "$EMBED_MODEL" \
  --ks $KS \
  --log-level INFO 2>&1 | tee "$RESULT_FILE"

[[ -f "$OPENSEARCH_PID" ]] && kill "$(cat "$OPENSEARCH_PID")" 2>/dev/null || true
rm -f "$OPENSEARCH_PID"
echo "=== Done exp3 (recursive + benchmarks_50 + Legal-Embed). ==="
